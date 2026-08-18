"""Credence model 2.0 (§2.1, T3, T10): premises carry credence, never
presumption — and the number must survive contact with correlated,
conflicting, and stale evidence.

Root prior per Premise = f(source_tier, corroboration, mutability):
  - source_tier: a small static domain table (overridable per run) — no
    source is presumed true, the tier is a prior, not a verdict. With
    `learned_reliability` on, truth-discovery iteration (verify/reliability)
    adjusts the prior ±cap from observed agreement, provenance-stamped.
  - corroboration: independent CONTENT clusters asserting the premise
    (verify/syndication — M10). v1 clustered by URL domain, which counts
    twelve syndicated copies of one wire story as twelve confirmations;
    now near-identical text merges into one cluster regardless of domain.
  - mutability (M10): FreshQA-style classes. `volatile` keeps the v1 flat
    penalty as a floor AND decays with evidence age; `slow` decays gently;
    `stable` never. Age is clock-free: report time is the newest evidence
    timestamp in the corpus, both recorded — replay-identical (T4).

Conflict-aware aggregation (M10): `contradicts` edges finally participate.
Each side's independent support becomes belief b; opposing support becomes
disbelief d; final credence = b(1-d)/(1-b·d) — degrades to plain noisy-OR
when d=0, and two strong disagreeing sources land near 0.5 ("genuinely
uncertain"), strictly lower than either alone. v1 could not express this.
On a MUTABLE claim, newer evidence contradicting older evidence
SUPERSEDES instead: the older node is marked superseded and discounted;
the newer one is not dragged down by it.

Propagation is unchanged from v1: min(parents) × edge_penalty along one
derivation path, noisy-OR across independent paths.

Numbers stay internal (§10.3): render_tier()/annotate() map credence to the
labels users see — plus, once `parsec calibrate` has run, quantified ranges
("high (72–96%)") and per-node uncertainty provenance ("conflicting
sources", "possibly stale", "superseded").
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable
from urllib.parse import urlsplit

from parsec.verify.syndication import cluster_spans

# Domain suffix -> prior. Longest matching suffix wins; config merges on top.
DEFAULT_SOURCE_TIERS: dict[str, float] = {
    ".gov": 0.9,
    ".edu": 0.9,
    "arxiv.org": 0.85,
    "nature.com": 0.85,
    "wikipedia.org": 0.8,
    "reuters.com": 0.8,
    "apnews.com": 0.8,
    "medium.com": 0.4,
    "blogspot.com": 0.4,
    "wordpress.com": 0.4,
    "substack.com": 0.4,
    "reddit.com": 0.4,
}
DEFAULT_TIER = 0.6

EDGE_PENALTY: dict[str, float] = {
    "extracts": 1.0,    # mechanically checked at record time
    "deduces": 0.95,
    "induces": 0.85,    # generalization is the leakiest derivation
    "temporal": 0.9,
    "aggregates": 1.0,  # structural merge, no new inference
}

HIGH, MODERATE = 0.8, 0.6

VOLATILE_HALF_LIFE_DAYS = 30.0
SLOW_HALF_LIFE_DAYS = 365.0


def render_tier(credence: float) -> str:
    """The user-facing register (§10.3): tiers, never raw numbers."""
    if credence >= HIGH:
        return "high"
    if credence >= MODERATE:
        return "moderate"
    return "low"


def domain_of(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


def source_tier(url: str, overrides: dict[str, float] | None = None) -> float:
    table = dict(DEFAULT_SOURCE_TIERS)
    if overrides:
        table.update(overrides)
    domain = domain_of(url)
    best_len, best = -1, DEFAULT_TIER
    for suffix, tier in table.items():
        s = suffix.lower().lstrip()
        if (domain == s.lstrip(".") or domain.endswith(s if s.startswith(".") else "." + s)) and len(s) > best_len:
            best_len, best = len(s), tier
    return best


def noisy_or(values: list[float]) -> float:
    p = 1.0
    for v in values:
        p *= 1.0 - v
    return 1.0 - p


def conflict_discount(belief: float, disbelief: float) -> float:
    """Subjective-logic-style renormalized conflict: b(1-d)/(1-b·d).
    d=0 degrades to plain belief; b=d=0.9 lands at ~0.47 — strong
    disagreement reads as genuine uncertainty, not near-certain falsehood."""
    denominator = 1.0 - belief * disbelief
    if denominator < 1e-9:
        return 0.5
    return belief * (1.0 - disbelief) / denominator


def parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


@dataclass
class NodeCredence:
    credence: float
    # One representative domain per independent CONTENT cluster (M10) — so
    # single_source means "one independent story", not "one URL domain".
    domains: frozenset[str]
    disbelief: float = 0.0          # noisy-OR of live contradicting evidence
    conflicted: bool = False        # unresolved contradiction lowered this
    superseded_by: str | None = None  # newer evidence replaced this mutable fact
    stale: bool = False             # mutable and decayed with evidence age

    @property
    def single_source(self) -> bool:
        return len(self.domains) <= 1


def annotate(nc: NodeCredence, ranges: dict[str, tuple[int, int]] | None = None) -> str:
    """The full user-facing register: tier (with calibrated range once one
    exists) + uncertainty provenance straight from the graph."""
    label = render_tier(nc.credence)
    if ranges and label in ranges:
        lo, hi = ranges[label]
        label = f"{label} ({lo}–{hi}%)"
    parts = [label]
    if nc.superseded_by:
        parts.append("superseded")
    if nc.conflicted:
        parts.append("conflicting sources")
    if nc.single_source:
        parts.append("single source")
    if nc.stale:
        parts.append("possibly stale")
    return ", ".join(parts)


@dataclass
class CredenceReport:
    nodes: dict[str, NodeCredence] = field(default_factory=dict)
    flagged_claims: list[str] = field(default_factory=list)  # below stakes threshold
    source_reliability: dict[str, str] = field(default_factory=dict)  # domain -> provenance

    def tier(self, node_id: str) -> str:
        return render_tier(self.nodes[node_id].credence)


def compute_credences(
    conn: sqlite3.Connection,
    session_id: str,
    source_tiers: dict[str, float] | None = None,
    stakes_threshold: float = 0.7,
    volatile_penalty: float = 0.85,
    persist: bool = True,
    volatile_half_life_days: float = VOLATILE_HALF_LIFE_DAYS,
    slow_half_life_days: float = SLOW_HALF_LIFE_DAYS,
    learned_reliability: bool = False,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
) -> CredenceReport:
    """Recompute credence over the whole session graph (§6 stage 3),
    tier-by-tier bottom-up with conflict resolution applied per tier, and
    (optionally) persist values onto the nodes."""
    if embed is None:
        from parsec.retrieval.embeddings import EmbeddingCache, HashedNgramEmbedder

        embed = EmbeddingCache(conn, HashedNgramEmbedder()).embed

    nodes: dict[str, dict] = {}
    for row in conn.execute(
        "SELECT node_id, node_type, tier, payload_json FROM nodes WHERE session_id=?",
        (session_id,),
    ):
        nodes[row["node_id"]] = {
            "type": row["node_type"],
            "tier": row["tier"],
            "payload": json.loads(row["payload_json"]),
        }
    out_edges: dict[str, list[dict]] = {}
    conflict_pairs: set[tuple[str, str]] = set()
    for row in conn.execute(
        "SELECT src_node_id, dst_node_id, edge_type FROM edges WHERE session_id=?",
        (session_id,),
    ):
        out_edges.setdefault(row["src_node_id"], []).append(dict(row))
        if row["edge_type"] == "contradicts":
            a, b = sorted((row["src_node_id"], row["dst_node_id"]))
            if a != b and a in nodes and b in nodes:
                conflict_pairs.add((a, b))

    report = CredenceReport()
    tiers = dict(source_tiers or {})
    if learned_reliability:
        from parsec.verify.reliability import learn_source_reliability

        for dom, est in sorted(learn_source_reliability(conn, session_id, source_tiers).items()):
            tiers[dom] = est.adjusted
            report.source_reliability[dom] = est.provenance

    # span node -> metadata the premise pass needs for clustering and decay
    span_meta: dict[str, dict] = {}
    for nid in sorted(nodes):
        node = nodes[nid]
        if node["type"] == "SourceSpan":
            p = node["payload"]
            span_meta[nid] = {
                "domain": domain_of(p["url"]),
                "text": p["text"],
                "credence": source_tier(p["url"], tiers),
                "ts": parse_ts(p.get("fetched_ts")),
            }
    # Clock-free "now" (T4): the newest evidence timestamp in the corpus.
    corpus_dt = max((m["ts"] for m in span_meta.values() if m["ts"]), default=None)

    # premise -> (claim_class, newest evidence ts): supersession inputs
    premise_class: dict[str, str] = {}
    premise_ts: dict[str, datetime | None] = {}

    by_tier: dict[int, list[str]] = {}
    for nid in sorted(nodes, key=lambda n: (nodes[n]["tier"], n)):
        by_tier.setdefault(nodes[nid]["tier"], []).append(nid)

    for tier_level in sorted(by_tier):
        for nid in by_tier[tier_level]:
            node = nodes[nid]
            if node["type"] == "SourceSpan":
                m = span_meta[nid]
                report.nodes[nid] = NodeCredence(m["credence"], frozenset([m["domain"]]))
            elif node["type"] == "Premise":
                report.nodes[nid] = _premise_credence(
                    nid, node, out_edges, span_meta, corpus_dt,
                    volatile_penalty, volatile_half_life_days, slow_half_life_days, embed,
                )
                premise_class[nid] = node["payload"].get("claim_class", "stable")
                premise_ts[nid] = _newest_evidence_ts(nid, out_edges, span_meta)
            elif node["type"] == "Finding":
                report.nodes[nid] = _derived_credence(nid, out_edges, report, joint=True)
            elif node["type"] in ("Synthesis", "ReportClaim"):
                # cited refs are independent supporting paths -> noisy-OR
                report.nodes[nid] = _derived_credence(nid, out_edges, report, joint=False)
        _resolve_conflicts(
            tier_level, conflict_pairs, nodes, report, premise_class, premise_ts
        )

    for nid, node in nodes.items():
        if node["type"] == "ReportClaim" and not node["payload"].get("narrative"):
            if report.nodes[nid].credence < stakes_threshold:
                report.flagged_claims.append(nid)
    report.flagged_claims.sort()

    if persist:
        conn.executemany(
            "UPDATE nodes SET credence=? WHERE session_id=? AND node_id=?",
            [(round(nc.credence, 6), session_id, nid) for nid, nc in report.nodes.items()],
        )
    return report


def _newest_evidence_ts(nid: str, out_edges: dict, span_meta: dict) -> datetime | None:
    stamps = [
        span_meta[e["dst_node_id"]]["ts"]
        for e in out_edges.get(nid, [])
        if e["edge_type"] == "extracts"
        and e["dst_node_id"] in span_meta
        and span_meta[e["dst_node_id"]]["ts"]
    ]
    return max(stamps, default=None)


def _premise_credence(
    nid: str,
    node: dict,
    out_edges: dict,
    span_meta: dict,
    corpus_dt: datetime | None,
    volatile_penalty: float,
    volatile_half_life_days: float,
    slow_half_life_days: float,
    embed,
) -> NodeCredence:
    """Noisy-OR over independent CONTENT clusters; mutability-classified decay."""
    spans = [
        span_meta[e["dst_node_id"]]
        for e in out_edges.get(nid, [])
        if e["edge_type"] == "extracts" and e["dst_node_id"] in span_meta
    ]
    if not spans:
        return NodeCredence(0.0, frozenset())
    spans.sort(key=lambda m: (m["domain"], m["text"]))
    clusters = cluster_spans(spans, embed)
    strengths = sorted(max(spans[i]["credence"] for i in cluster) for cluster in clusters)
    domains = frozenset(min(spans[i]["domain"] for i in cluster) for cluster in clusters)
    credence = noisy_or(strengths)

    claim_class = node["payload"].get("claim_class", "stable")
    stale = False
    if claim_class == "volatile":
        credence *= volatile_penalty  # v1 floor: mutability risk even at age 0
    if claim_class in ("volatile", "slow") and corpus_dt is not None:
        newest = _newest_evidence_ts(nid, out_edges, span_meta)
        if newest is not None:
            age_days = int((corpus_dt - newest).total_seconds() // 86400)
            if age_days > 0:
                half_life = (
                    volatile_half_life_days if claim_class == "volatile" else slow_half_life_days
                )
                credence *= 0.5 ** (age_days / half_life)
                stale = True
    return NodeCredence(credence, domains, stale=stale)


def _derived_credence(
    nid: str, out_edges: dict, report: CredenceReport, joint: bool
) -> NodeCredence:
    """joint=True: one derivation over all parents (min × penalty).
    joint=False: each ref is an independent path (noisy-OR of per-path values).
    Uncertainty provenance flags propagate up from the parents."""
    paths: list[float] = []
    domains: set[str] = set()
    penalties: list[float] = []
    conflicted = stale = False
    for e in out_edges.get(nid, []):
        if e["edge_type"] == "contradicts" or e["dst_node_id"] not in report.nodes:
            continue
        parent = report.nodes[e["dst_node_id"]]
        penalty = EDGE_PENALTY.get(e["edge_type"], 0.9)
        paths.append(parent.credence * penalty)
        penalties.append(penalty)
        domains.update(parent.domains)
        conflicted = conflicted or parent.conflicted
        stale = stale or parent.stale
    if not paths:
        return NodeCredence(0.0, frozenset())
    if joint:
        # single derivation path: min of parents, one penalty application
        raw_parents = [p / pen for p, pen in zip(paths, penalties)]
        credence = min(raw_parents) * min(penalties)
    else:
        credence = noisy_or(sorted(paths))
    return NodeCredence(credence, frozenset(domains), conflicted=conflicted, stale=stale)


def _resolve_conflicts(
    tier_level: int,
    conflict_pairs: set[tuple[str, str]],
    nodes: dict,
    report: CredenceReport,
    premise_class: dict[str, str],
    premise_ts: dict,
) -> None:
    """Apply contradicts edges whose higher endpoint lives at this tier, so
    both sides have base values. Supersession first (mutable + dated evidence:
    newer replaces older); every remaining opponent contributes disbelief and
    the conflict discount renormalizes. All decisions read the tier's
    PRE-conflict snapshot — symmetric and order-independent."""
    pairs = [
        (a, b)
        for a, b in sorted(conflict_pairs)
        if max(nodes[a]["tier"], nodes[b]["tier"]) == tier_level
        and a in report.nodes
        and b in report.nodes
    ]
    if not pairs:
        return
    snapshot = {nid: report.nodes[nid].credence for pair in pairs for nid in pair}

    superseded_by: dict[str, str] = {}
    for a, b in pairs:
        winner_loser = _supersession(a, b, premise_class, premise_ts)
        if winner_loser is None:
            continue
        winner, loser = winner_loser
        current = superseded_by.get(loser)
        # several superseders: the newest evidence wins, ties break by id
        if current is None or (premise_ts[winner], winner) > (premise_ts[current], current):
            superseded_by[loser] = winner

    opponents: dict[str, set[str]] = {}
    for a, b in pairs:
        opponents.setdefault(a, set()).add(b)
        opponents.setdefault(b, set()).add(a)

    for nid in sorted(opponents):
        nc = report.nodes[nid]
        if nid in superseded_by:
            winner = superseded_by[nid]
            discount = 1.0 - snapshot.get(winner, report.nodes[winner].credence)
            report.nodes[nid] = NodeCredence(
                nc.credence * discount, nc.domains,
                disbelief=snapshot.get(winner, 0.0),
                superseded_by=winner, stale=nc.stale,
            )
            continue
        live = sorted(o for o in opponents[nid] if o not in superseded_by)
        if not live:
            continue  # every opponent was superseded away
        disbelief = noisy_or([snapshot[o] for o in live])
        report.nodes[nid] = NodeCredence(
            conflict_discount(nc.credence, disbelief), nc.domains,
            disbelief=disbelief, conflicted=True, stale=nc.stale,
        )


def _supersession(
    a: str, b: str, premise_class: dict[str, str], premise_ts: dict
) -> tuple[str, str] | None:
    """(winner, loser) when newer evidence supersedes older on a mutable
    claim (M10): both endpoints are premises with dated evidence, the dates
    differ, and at least one side is mutable. Else None -> symmetric conflict."""
    if a not in premise_class or b not in premise_class:
        return None
    ts_a, ts_b = premise_ts.get(a), premise_ts.get(b)
    if ts_a is None or ts_b is None or ts_a == ts_b:
        return None
    if premise_class[a] == "stable" and premise_class[b] == "stable":
        return None
    return (a, b) if ts_a > ts_b else (b, a)
