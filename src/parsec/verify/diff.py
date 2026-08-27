"""Claim-level session diff (M14 phase 1: diachronic research).

Two recorded sessions of the same question, taken at different times, are
two frozen observations of a moving world. This module compares them
mechanically — no model, no network: the diff is a pure function of the
two stored evidence graphs and each session's recorded config, so the
same pair of sessions always yields the same report (T4). No session
state is written; credence is recomputed per side with persist=False.
(One deliberate exception: the default embedder is the shared
content-addressed embedding memo, so the `embeddings` table may gain
rows — a pure-function cache that can't affect any outcome, and the
same behavior `parsec verify` already has.)

Claim identity is a ladder (T9: the exact tiers are the floor, the fuzzy
tier is advisory and always labeled with its similarity):

  id    — identical node_id: same sentence AND same cited evidence.
          Node IDs are content-derived (ids.node_id), so identity holds
          across sessions by construction.
  text  — same normalized sentence; the evidence behind it may differ.
  fuzzy — hashed-n-gram cosine >= CLAIM_MATCH_COSINE (the syndication
          embedder): a lightly reworded claim, never silently treated
          as exact.

Each matched claim is then classified by its recomputed credence, each
side scored under its own session's recorded config:

  superseded    — B's support rests on a premise newer evidence replaced
                  (M10 supersession) where A's did not
  strengthened  — credence rose by >= epsilon
  weakened      — credence fell by >= epsilon
  held          — neither

plus the unmatched remainders: retracted (claimed in A, absent from B)
and new (claimed in B, absent from A). For every change the drivers name
the premise-level evidence delta behind it — premises gained, lost,
moved, or superseded, matched across sessions by normalized text.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Callable

from parsec.config import RunConfig
from parsec.retrieval.embeddings import cosine
from parsec.verify.credence import annotate, compute_credences, render_tier

# Fuzzy-match floor on character-3-gram cosine: a lightly reworded claim
# stays above this; two different claims about the same topic fall below.
# Deliberately stricter than "same topic", looser than syndication's 0.9
# near-dup bar — a rewritten sentence is not a republished document.
CLAIM_MATCH_COSINE = 0.8

# Display/sort order: changes first, background last.
STATUS_ORDER = ("superseded", "weakened", "strengthened", "new", "retracted", "held")

Embed = Callable[[list[str]], list[list[float]]]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _short(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class ClaimDelta:
    status: str                  # one of STATUS_ORDER
    text: str                    # B's wording when matched (the current claim), else the only side's
    match: str | None            # "id" | "text" | "fuzzy"; None for new/retracted
    similarity: float | None     # fuzzy matches only
    a_id: str | None
    b_id: str | None
    credence_a: float | None
    credence_b: float | None
    provenance_a: str | None     # annotate() register: tier + uncertainty provenance
    provenance_b: str | None
    drivers: list[str] = field(default_factory=list)

    def to_payload(self) -> dict:
        return {
            "status": self.status,
            "text": self.text,
            "match": self.match,
            "similarity": None if self.similarity is None else round(self.similarity, 6),
            "a_id": self.a_id,
            "b_id": self.b_id,
            "credence_a": None if self.credence_a is None else round(self.credence_a, 6),
            "credence_b": None if self.credence_b is None else round(self.credence_b, 6),
            "provenance_a": self.provenance_a,
            "provenance_b": self.provenance_b,
            "drivers": self.drivers,
        }


@dataclass
class DocDelta:
    url: str
    status: str                  # "changed" | "added" | "dropped"
    hashes_a: list[str]
    hashes_b: list[str]

    def to_payload(self) -> dict:
        return {
            "url": self.url, "status": self.status,
            "hashes_a": self.hashes_a, "hashes_b": self.hashes_b,
        }


@dataclass
class DiffReport:
    session_a: str
    session_b: str
    query_a: str
    query_b: str
    config_skew: bool            # credence-relevant config differs between sessions
    claims: list[ClaimDelta]
    documents: list[DocDelta]

    @property
    def counts(self) -> dict[str, int]:
        out = {status: 0 for status in STATUS_ORDER}
        for c in self.claims:
            out[c.status] += 1
        return out

    @property
    def unchanged(self) -> bool:
        """No material claim-level change (document churn alone is not one)."""
        return all(c.status == "held" for c in self.claims)

    def to_payload(self) -> dict:
        return {
            "session_a": self.session_a,
            "session_b": self.session_b,
            "query_a": self.query_a,
            "query_b": self.query_b,
            "same_query": self.query_a == self.query_b,
            "config_skew": self.config_skew,
            "unchanged": self.unchanged,
            "counts": self.counts,
            "claims": [c.to_payload() for c in self.claims],
            "documents": [d.to_payload() for d in self.documents],
        }


@dataclass
class _SessionGraph:
    session_id: str
    query: str
    config: RunConfig
    claims: dict[str, dict]           # non-narrative ReportClaim node_id -> payload
    premises: dict[str, dict]         # Premise node_id -> payload
    spans: dict[str, dict]            # SourceSpan node_id -> payload
    evidence_of: dict[str, list[str]] # src (derived) -> dst (evidence), contradicts excluded
    credence: "object"                # CredenceReport


def _load(conn: sqlite3.Connection, session_id: str, embed: Embed) -> _SessionGraph:
    row = conn.execute(
        "SELECT query, config_json FROM sessions WHERE session_id=?", (session_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown session: {session_id}")
    config = RunConfig.model_validate(json.loads(row["config_json"]))

    claims: dict[str, dict] = {}
    premises: dict[str, dict] = {}
    spans: dict[str, dict] = {}
    for n in conn.execute(
        "SELECT node_id, node_type, payload_json FROM nodes WHERE session_id=?", (session_id,)
    ):
        payload = json.loads(n["payload_json"])
        if n["node_type"] == "ReportClaim" and not payload.get("narrative"):
            claims[n["node_id"]] = payload
        elif n["node_type"] == "Premise":
            premises[n["node_id"]] = payload
        elif n["node_type"] == "SourceSpan":
            spans[n["node_id"]] = payload

    evidence_of: dict[str, list[str]] = {}
    for e in conn.execute(
        "SELECT src_node_id, dst_node_id, edge_type FROM edges WHERE session_id=?", (session_id,)
    ):
        if e["edge_type"] != "contradicts":
            evidence_of.setdefault(e["src_node_id"], []).append(e["dst_node_id"])

    credence = compute_credences(
        conn,
        session_id,
        source_tiers=config.source_tiers,
        stakes_threshold=config.stakes_threshold,
        volatile_penalty=config.volatile_penalty,
        persist=False,
        volatile_half_life_days=config.volatile_half_life_days,
        slow_half_life_days=config.slow_half_life_days,
        learned_reliability=config.learned_reliability,
        embed=embed,
    )
    return _SessionGraph(session_id, row["query"], config, claims, premises, spans, evidence_of, credence)


def _support_premises(graph: _SessionGraph, claim_id: str) -> list[str]:
    """Premise node IDs a claim ultimately rests on (claim → finding → premise
    walk down derivation edges), sorted for determinism."""
    seen: set[str] = set()
    stack = [claim_id]
    out: set[str] = set()
    while stack:
        nid = stack.pop()
        for parent in graph.evidence_of.get(nid, []):
            if parent in seen:
                continue
            seen.add(parent)
            if parent in graph.premises:
                out.add(parent)
            else:
                stack.append(parent)
    return sorted(out)


def _credence_config(config: RunConfig) -> tuple:
    return (
        config.source_tiers,
        config.stakes_threshold,
        config.volatile_penalty,
        config.volatile_half_life_days,
        config.slow_half_life_days,
        config.learned_reliability,
    )


def _match_claims(
    a: _SessionGraph, b: _SessionGraph, embed: Embed
) -> tuple[list[tuple[str, str, str, float | None]], list[str], list[str]]:
    """Deterministic three-tier matching. Returns (matches, a_only, b_only)
    where matches are (a_id, b_id, tier, similarity)."""
    matches: list[tuple[str, str, str, float | None]] = []

    # Tier 1: content-derived node identity (same sentence, same evidence).
    for nid in sorted(set(a.claims) & set(b.claims)):
        matches.append((nid, nid, "id", None))
    a_rem = sorted(set(a.claims) - set(b.claims))
    b_rem = sorted(set(b.claims) - set(a.claims))

    # Tier 2: normalized text. Duplicate texts pair off in sorted-id order.
    b_by_text: dict[str, list[str]] = {}
    for nid in b_rem:
        b_by_text.setdefault(_norm(b.claims[nid]["text"]), []).append(nid)
    still_a: list[str] = []
    for nid in a_rem:
        bucket = b_by_text.get(_norm(a.claims[nid]["text"]))
        if bucket:
            matches.append((nid, bucket.pop(0), "text", None))
        else:
            still_a.append(nid)
    a_rem = still_a
    b_rem = [nid for nid in b_rem if nid in {x for bucket in b_by_text.values() for x in bucket}]

    # Tier 3: fuzzy (advisory). Greedy best-cosine, ties broken by id.
    if a_rem and b_rem:
        vec_a = embed([a.claims[nid]["text"] for nid in a_rem])
        vec_b = embed([b.claims[nid]["text"] for nid in b_rem])
        candidates = sorted(
            (
                (-round(cosine(vec_a[i], vec_b[j]), 6), a_rem[i], b_rem[j])
                for i in range(len(a_rem))
                for j in range(len(b_rem))
                if cosine(vec_a[i], vec_b[j]) >= CLAIM_MATCH_COSINE
            ),
        )
        used_a: set[str] = set()
        used_b: set[str] = set()
        for neg_sim, aid, bid in candidates:
            if aid in used_a or bid in used_b:
                continue
            used_a.add(aid)
            used_b.add(bid)
            matches.append((aid, bid, "fuzzy", -neg_sim))
        a_rem = [nid for nid in a_rem if nid not in used_a]
        b_rem = [nid for nid in b_rem if nid not in used_b]

    return matches, a_rem, b_rem


def _premise_deltas(
    a: _SessionGraph, b: _SessionGraph, a_claim: str, b_claim: str, epsilon: float
) -> tuple[list[str], bool, bool]:
    """Drivers for one matched claim: the premise-level evidence delta,
    matched by normalized text. Returns (drivers, a_superseded, b_superseded)."""
    def by_text(graph: _SessionGraph, claim_id: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for pid in _support_premises(graph, claim_id):
            out.setdefault(_norm(graph.premises[pid]["text"]), pid)
        return out

    prem_a, prem_b = by_text(a, a_claim), by_text(b, b_claim)
    a_sup = any(a.credence.nodes[p].superseded_by for p in prem_a.values())
    b_sup = any(b.credence.nodes[p].superseded_by for p in prem_b.values())

    drivers: list[str] = []
    for t in sorted(prem_b):
        if b.credence.nodes[prem_b[t]].superseded_by:
            drivers.append(f"evidence superseded: {_short(b.premises[prem_b[t]]['text'])}")
    moved: list[tuple[float, str]] = []
    for t in sorted(prem_a.keys() & prem_b.keys()):
        ca = a.credence.nodes[prem_a[t]].credence
        cb = b.credence.nodes[prem_b[t]].credence
        if abs(cb - ca) >= epsilon and not b.credence.nodes[prem_b[t]].superseded_by:
            verb = "strengthened" if cb > ca else "weakened"
            moved.append((
                -abs(cb - ca),
                f"premise {verb}: {_short(b.premises[prem_b[t]]['text'])} "
                f"({render_tier(ca)}→{render_tier(cb)})",
            ))
    drivers += [d for _, d in sorted(moved)]
    for t in sorted(prem_a.keys() - prem_b.keys()):
        drivers.append(f"premise lost: {_short(a.premises[prem_a[t]]['text'])}")
    for t in sorted(prem_b.keys() - prem_a.keys()):
        drivers.append(f"premise gained: {_short(b.premises[prem_b[t]]['text'])}")
    return drivers, a_sup, b_sup


def _document_deltas(a: _SessionGraph, b: _SessionGraph) -> list[DocDelta]:
    def by_url(graph: _SessionGraph) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for payload in graph.spans.values():
            out.setdefault(payload["url"], set()).add(payload["doc_hash"])
        return out

    urls_a, urls_b = by_url(a), by_url(b)
    deltas: list[DocDelta] = []
    for url in sorted(urls_a.keys() | urls_b.keys()):
        ha, hb = sorted(urls_a.get(url, ())), sorted(urls_b.get(url, ()))
        if not ha:
            deltas.append(DocDelta(url, "added", ha, hb))
        elif not hb:
            deltas.append(DocDelta(url, "dropped", ha, hb))
        elif ha != hb:
            deltas.append(DocDelta(url, "changed", ha, hb))
    return deltas


def diff_sessions(
    conn: sqlite3.Connection,
    session_a: str,
    session_b: str,
    epsilon: float = 0.05,
    embed: Embed | None = None,
) -> DiffReport:
    """Mechanical claim-level diff of two recorded sessions (A = earlier,
    B = later). Writes no session state: credence is recomputed per side
    without persisting (the default embedder's pure-function memo table is
    the one sanctioned write — see the module docstring)."""
    if embed is None:
        from parsec.retrieval.embeddings import EmbeddingCache, HashedNgramEmbedder

        embed = EmbeddingCache(conn, HashedNgramEmbedder()).embed

    a = _load(conn, session_a, embed)
    b = _load(conn, session_b, embed)

    claims: list[ClaimDelta] = []
    matches, a_only, b_only = _match_claims(a, b, embed)
    for a_id, b_id, tier, similarity in matches:
        ca = a.credence.nodes[a_id].credence
        cb = b.credence.nodes[b_id].credence
        drivers, a_sup, b_sup = _premise_deltas(a, b, a_id, b_id, epsilon)
        if b_sup and not a_sup:
            status = "superseded"
        elif cb - ca >= epsilon:
            status = "strengthened"
        elif ca - cb >= epsilon:
            status = "weakened"
        else:
            status = "held"
        claims.append(ClaimDelta(
            status=status,
            text=b.claims[b_id]["text"],
            match=tier,
            similarity=similarity,
            a_id=a_id,
            b_id=b_id,
            credence_a=ca,
            credence_b=cb,
            provenance_a=annotate(a.credence.nodes[a_id]),
            provenance_b=annotate(b.credence.nodes[b_id]),
            drivers=drivers if status != "held" else [],
        ))
    for a_id in a_only:
        claims.append(ClaimDelta(
            status="retracted", text=a.claims[a_id]["text"], match=None, similarity=None,
            a_id=a_id, b_id=None,
            credence_a=a.credence.nodes[a_id].credence, credence_b=None,
            provenance_a=annotate(a.credence.nodes[a_id]), provenance_b=None,
        ))
    for b_id in b_only:
        claims.append(ClaimDelta(
            status="new", text=b.claims[b_id]["text"], match=None, similarity=None,
            a_id=None, b_id=b_id,
            credence_a=None, credence_b=b.credence.nodes[b_id].credence,
            provenance_a=None, provenance_b=annotate(b.credence.nodes[b_id]),
        ))
    claims.sort(key=lambda c: (STATUS_ORDER.index(c.status), c.text))

    return DiffReport(
        session_a=session_a,
        session_b=session_b,
        query_a=a.query,
        query_b=b.query,
        config_skew=_credence_config(a.config) != _credence_config(b.config),
        claims=claims,
        documents=_document_deltas(a, b),
    )
