"""Stage-1/2 mechanical verification (§6): no judgment.

Walks the persisted Evidence DAG of a session and mechanically checks:
  acyclic          — the DAG is actually a DAG
  claim-path       — every non-narrative ReportClaim reaches ≥1 SourceSpan
  tier-integrity   — every Premise has ≥1 extracts edge; every Finding has
                     ≥1 premise edge; edge types connect the tiers they may
  corpus-integrity — every SourceSpan node still matches its span row, its
                     document row, and the verbatim slice of the stored text
  containment      — numbers/quotes in every Premise re-checked against its
                     spans (transform_note on the edge exempts numbers)
  temporal-order   — stage 2 (M9): ordering findings on `temporal` edges
                     checked against evidence date intervals; definite
                     contradictions are violations

Advisory checks (M9, T9 — recorded and surfaced, but they never flip `ok`):
  premise-support  — grounded-NLI tier: does the evidence actually support
                     each premise beyond exact-match containment?
  premise-lint     — Claimify-style ambiguity/decontextualization lints,
                     re-run over recorded premises (hard-gated for NEW
                     premises in record_premises; advisory here so sessions
                     recorded before M9 stay verifiable)
  temporal         — temporal findings that could not be mechanically decided

Runs at the end of every session and on demand via `parsec verify` — the
latter is what catches corpus corruption after the fact.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from parsec.store.blobs import BlobStore
from parsec.verify.containment import check_containment
from parsec.verify.lints import lint_premise
from parsec.verify.nli import GroundedChecker, LexicalGroundedChecker
from parsec.verify.temporal import check_temporal_findings

# edge type -> (allowed src node_type(s), allowed dst node_type(s))
_EDGE_RULES: dict[str, tuple[set[str], set[str]]] = {
    "extracts": ({"Premise"}, {"SourceSpan"}),
    "deduces": ({"Finding"}, {"Premise", "Finding"}),
    "induces": ({"Finding"}, {"Premise", "Finding"}),
    "temporal": ({"Finding"}, {"Premise", "Finding"}),
    "aggregates": ({"Synthesis", "ReportClaim"}, {"Premise", "Finding", "Synthesis"}),
    "contradicts": (
        {"Premise", "Finding", "Synthesis"},
        {"Premise", "Finding", "Synthesis"},
    ),
}


@dataclass
class Violation:
    check: str
    subject: str  # node/edge id
    detail: str


@dataclass
class VerificationReport:
    violations: list[Violation] = field(default_factory=list)
    # Advisory tier (T9): recorded verdicts that inform but never gate — a
    # session with only advisories is still `ok`.
    advisories: list[Violation] = field(default_factory=list)
    checked_claims: int = 0
    checked_premises: int = 0
    checked_spans: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_payload(self) -> dict:
        return {
            "ok": self.ok,
            "checked": {
                "claims": self.checked_claims,
                "premises": self.checked_premises,
                "spans": self.checked_spans,
            },
            "violations": [
                {"check": v.check, "subject": v.subject, "detail": v.detail}
                for v in self.violations
            ],
            "advisories": [
                {"check": v.check, "subject": v.subject, "detail": v.detail}
                for v in self.advisories
            ],
        }


# Sentinel default: callers that pass nothing get the always-on lexical
# tier; passing None disables grounded support checking entirely.
_DEFAULT_NLI: GroundedChecker = LexicalGroundedChecker()


def verify_session(
    conn: sqlite3.Connection,
    blobs: BlobStore,
    session_id: str,
    nli_checker: GroundedChecker | None = _DEFAULT_NLI,
) -> VerificationReport:
    report = VerificationReport()
    # ORDER BY id, not insertion: concurrent subagents (M11) interleave row
    # creation nondeterministically, and violation/advisory ordering must
    # replay byte-identically.
    nodes: dict[str, dict] = {}
    for row in conn.execute(
        "SELECT node_id, node_type, payload_json FROM nodes WHERE session_id=? ORDER BY node_id",
        (session_id,),
    ):
        nodes[row["node_id"]] = {
            "type": row["node_type"],
            "payload": json.loads(row["payload_json"]),
        }
    edges = [
        dict(row)
        for row in conn.execute(
            "SELECT edge_id, src_node_id, dst_node_id, edge_type, payload_json"
            " FROM edges WHERE session_id=? ORDER BY edge_id",
            (session_id,),
        )
    ]

    _check_acyclic(nodes, edges, report)
    _check_edge_rules(nodes, edges, report)
    _check_claim_paths(nodes, edges, report)
    _check_tier_integrity(nodes, edges, report)
    _check_corpus_integrity(conn, blobs, nodes, report)
    _check_containment(nodes, edges, report)
    _check_temporal(nodes, edges, report)
    _check_premise_support(nodes, edges, report, nli_checker)
    _lint_premises(nodes, report)
    _flag_dependent_claims(nodes, edges, report)
    return report


def _out_edges(edges: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for e in edges:
        out.setdefault(e["src_node_id"], []).append(e)
    return out


def _check_acyclic(nodes, edges, report: VerificationReport) -> None:
    out = _out_edges(edges)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {nid: WHITE for nid in nodes}
    for start in nodes:
        if color[start] != WHITE:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        color[start] = GRAY
        while stack:
            nid, i = stack[-1]
            targets = [e["dst_node_id"] for e in out.get(nid, []) if e["dst_node_id"] in nodes]
            if i < len(targets):
                stack[-1] = (nid, i + 1)
                nxt = targets[i]
                if color[nxt] == GRAY:
                    report.violations.append(
                        Violation("acyclic", nxt, f"cycle detected via {nid}")
                    )
                elif color[nxt] == WHITE:
                    color[nxt] = GRAY
                    stack.append((nxt, 0))
            else:
                color[nid] = BLACK
                stack.pop()


def _check_edge_rules(nodes, edges, report: VerificationReport) -> None:
    for e in edges:
        src = nodes.get(e["src_node_id"])
        dst = nodes.get(e["dst_node_id"])
        if src is None or dst is None:
            report.violations.append(
                Violation("tier-integrity", e["edge_id"], "edge references a missing node")
            )
            continue
        rule = _EDGE_RULES.get(e["edge_type"])
        if rule is None:
            report.violations.append(
                Violation("tier-integrity", e["edge_id"], f"unknown edge type {e['edge_type']!r}")
            )
            continue
        allowed_src, allowed_dst = rule
        if src["type"] not in allowed_src or dst["type"] not in allowed_dst:
            report.violations.append(
                Violation(
                    "tier-integrity",
                    e["edge_id"],
                    f"{e['edge_type']} edge may not connect {src['type']} -> {dst['type']}",
                )
            )


def _check_claim_paths(nodes, edges, report: VerificationReport) -> None:
    out = _out_edges(edges)
    for nid, node in nodes.items():
        if node["type"] != "ReportClaim":
            continue
        if node["payload"].get("narrative"):
            continue
        report.checked_claims += 1
        # BFS: does any path from the claim reach a SourceSpan?
        seen, queue, reached = {nid}, [nid], False
        while queue and not reached:
            cur = queue.pop(0)
            for e in out.get(cur, []):
                dst = e["dst_node_id"]
                if dst in seen or dst not in nodes:
                    continue
                if nodes[dst]["type"] == "SourceSpan":
                    reached = True
                    break
                seen.add(dst)
                queue.append(dst)
        if not reached:
            report.violations.append(
                Violation(
                    "claim-path",
                    nid,
                    f"claim {node['payload'].get('text', '')[:80]!r} has no path to any SourceSpan",
                )
            )


def _check_tier_integrity(nodes, edges, report: VerificationReport) -> None:
    out = _out_edges(edges)
    for nid, node in nodes.items():
        if node["type"] == "Premise":
            if not any(e["edge_type"] == "extracts" for e in out.get(nid, [])):
                report.violations.append(
                    Violation("tier-integrity", nid, "Premise has no extracts edge to any span")
                )
        elif node["type"] == "Finding":
            targets = [
                nodes[e["dst_node_id"]]["type"]
                for e in out.get(nid, [])
                if e["dst_node_id"] in nodes
            ]
            if "Premise" not in targets:
                report.violations.append(
                    Violation("tier-integrity", nid, "Finding has no edge to any Premise")
                )


def _check_corpus_integrity(conn, blobs: BlobStore, nodes, report: VerificationReport) -> None:
    for nid, node in nodes.items():
        if node["type"] != "SourceSpan":
            continue
        report.checked_spans += 1
        p = node["payload"]
        span_row = conn.execute(
            "SELECT * FROM spans WHERE span_id=?", (p["span_id"],)
        ).fetchone()
        if span_row is None:
            report.violations.append(
                Violation("corpus-integrity", nid, f"span row {p['span_id']} missing")
            )
            continue
        if span_row["text"] != p["text"]:
            report.violations.append(
                Violation(
                    "corpus-integrity",
                    nid,
                    f"span row text diverges from DAG node text for {p['span_id']}",
                )
            )
        doc = conn.execute(
            "SELECT * FROM documents WHERE doc_hash=?", (span_row["doc_hash"],)
        ).fetchone()
        if doc is None:
            report.violations.append(
                Violation("corpus-integrity", nid, f"document {span_row['doc_hash'][:12]} missing")
            )
            continue
        if not blobs.exists(doc["raw_blob"]) or not blobs.exists(doc["text_blob"]):
            report.violations.append(
                Violation("corpus-integrity", nid, "document blobs missing from blob store")
            )
            continue
        text = blobs.get_text(doc["text_blob"])
        if text[span_row["char_start"] : span_row["char_end"]] != span_row["text"]:
            report.violations.append(
                Violation(
                    "corpus-integrity",
                    nid,
                    f"span {p['span_id']} is not the verbatim slice of its document",
                )
            )


def _flag_dependent_claims(nodes, edges, report: VerificationReport) -> None:
    """Walk violations back up the DAG: every ReportClaim whose evidence path
    touches a violated node is itself flagged — a corrupted span condemns the
    claims that rest on it, mechanically."""
    violated = {
        v.subject
        for v in report.violations
        if v.subject in nodes
        and v.check in ("corpus-integrity", "containment", "tier-integrity", "temporal-order")
    }
    if not violated:
        return
    reverse: dict[str, list[str]] = {}
    for e in edges:
        reverse.setdefault(e["dst_node_id"], []).append(e["src_node_id"])
    for start in sorted(violated):
        seen, queue = {start}, [start]
        while queue:
            cur = queue.pop(0)
            for src in reverse.get(cur, []):
                if src in seen or src not in nodes:
                    continue
                seen.add(src)
                if nodes[src]["type"] == "ReportClaim":
                    report.violations.append(
                        Violation(
                            "dependent-claim",
                            src,
                            f"claim {nodes[src]['payload'].get('text', '')[:80]!r} "
                            f"depends on violated node {start}",
                        )
                    )
                else:
                    queue.append(src)


def _check_containment(nodes, edges, report: VerificationReport) -> None:
    out = _out_edges(edges)
    for nid, node in nodes.items():
        if node["type"] != "Premise":
            continue
        report.checked_premises += 1
        span_texts: list[str] = []
        transform_note: str | None = None
        for e in out.get(nid, []):
            if e["edge_type"] != "extracts" or e["dst_node_id"] not in nodes:
                continue
            span_texts.append(nodes[e["dst_node_id"]]["payload"]["text"])
            note = json.loads(e["payload_json"] or "{}").get("transform_note")
            if note:
                transform_note = note
        if not span_texts:
            continue  # already flagged by tier-integrity
        for problem in check_containment(node["payload"]["text"], span_texts, transform_note):
            report.violations.append(Violation("containment", nid, problem))


def _check_temporal(nodes, edges, report: VerificationReport) -> None:
    """Stage 2 (§6): temporal edges checked against evidence timestamps
    mechanically, before any judge sees them. Definite ordering
    contradictions are violations; undecidable findings are advisories."""
    out = _out_edges(edges)
    violations, advisories = check_temporal_findings(nodes, out)
    for subject, detail in violations:
        report.violations.append(Violation("temporal-order", subject, detail))
    for subject, detail in advisories:
        report.advisories.append(Violation("temporal", subject, detail))


def _check_premise_support(
    nodes, edges, report: VerificationReport, checker: GroundedChecker | None
) -> None:
    """Grounded-NLI tier (M9, T9): flags premises whose spans pass exact-match
    containment but do not appear to carry the premise's content. Advisory —
    NLI error rates are real, so this never sole-gates a premise."""
    if checker is None:
        return
    out = _out_edges(edges)
    for nid in sorted(nodes):
        node = nodes[nid]
        if node["type"] != "Premise":
            continue
        span_texts = sorted(
            nodes[e["dst_node_id"]]["payload"]["text"]
            for e in out.get(nid, [])
            if e["edge_type"] == "extracts" and e["dst_node_id"] in nodes
        )
        if not span_texts:
            continue  # already flagged by tier-integrity
        verdict = checker.check(node["payload"]["text"], span_texts)
        if verdict.flagged:
            report.advisories.append(Violation("premise-support", nid, verdict.describe()))


def _lint_premises(nodes, report: VerificationReport) -> None:
    for nid in sorted(nodes):
        node = nodes[nid]
        if node["type"] != "Premise":
            continue
        for reason in lint_premise(node["payload"]["text"]):
            report.advisories.append(Violation("premise-lint", nid, reason))
