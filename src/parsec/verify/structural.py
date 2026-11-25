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
  finding-support  — (derivations >= 2) every number in a Finding's text
                     appears in the premises it cites, or is the output of
                     its own operation
  computation      — (derivations >= 2) every `computes` operation is
                     re-run with exact decimal arithmetic: operands must be
                     numbers their premises state, and the output must match
  observation-date — (recency >= 2) a date a premise gave for a fact
                     (`as_of`) is stated, at least as precisely, by the span
                     it is attached to
  sentence-support — (sentence_check >= 1) every number and quotation in a
                     ReportClaim's text appears in the text of the premises
                     and findings it cites; a narrative claim carries neither.
                     The same check the writer's citation check runs, so a
                     wrong figure cannot pass one and fail the other

Advisory checks (M9, T9 — recorded and surfaced, but they never flip `ok`):
  premise-support  — grounded-NLI tier: does the evidence actually support
                     each premise beyond exact-match containment?
  premise-lint     — Claimify-style ambiguity/decontextualization lints,
                     re-run over recorded premises (hard-gated for NEW
                     premises in record_premises; advisory here so sessions
                     recorded before M9 stay verifiable)
  temporal         — temporal findings that could not be mechanically decided
  unverified-derivation — a derived figure the harness cannot recompute: a
                     free-text `computes` operation, or (when an older
                     session is re-scored) a premise whose numbers rest on a
                     transform_note

Runs at the end of every session and on demand via `parsec verify` — the
latter is what catches corpus corruption after the fact.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from parsec.store.blobs import BlobStore
from parsec.verify.computations import UNVERIFIED, check_finding, evidence_text
from parsec.verify.containment import check_containment, check_sentence, has_checkable_content
from parsec.verify.dating import DATE_SOURCE_AS_OF, check_date_containment, stated_years_text
from parsec.verify.lints import lint_premise
from parsec.verify.nli import GroundedChecker, LexicalGroundedChecker
from parsec.verify.temporal import check_temporal_findings

# edge type -> (allowed src node_type(s), allowed dst node_type(s))
_EDGE_RULES: dict[str, tuple[set[str], set[str]]] = {
    "extracts": ({"Premise"}, {"SourceSpan"}),
    "deduces": ({"Finding"}, {"Premise", "Finding"}),
    "induces": ({"Finding"}, {"Premise", "Finding"}),
    "temporal": ({"Finding"}, {"Premise", "Finding"}),
    "computes": ({"Finding"}, {"Premise", "Finding"}),
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
    sentence_check: bool = False,
    derivations: int = 1,
    recency: int = 1,
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
    _check_containment(nodes, edges, report, verbatim=derivations >= 2)
    if derivations >= 2:
        _check_findings(nodes, report)
    if recency >= 2:
        _check_observation_dates(nodes, edges, report)
    _check_temporal(nodes, edges, report)
    if sentence_check:
        _check_sentence_support(nodes, edges, report)
    _check_premise_support(nodes, edges, report, nli_checker)
    _lint_premises(nodes, report)
    _flag_dependent_claims(nodes, edges, report)
    return report


def _out_edges(edges: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for e in edges:
        out.setdefault(e["src_node_id"], []).append(e)
    return out


def _check_acyclic(
    nodes: dict[str, dict], edges: list[dict], report: VerificationReport
) -> None:
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


def _check_edge_rules(
    nodes: dict[str, dict], edges: list[dict], report: VerificationReport
) -> None:
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


def _check_claim_paths(
    nodes: dict[str, dict], edges: list[dict], report: VerificationReport
) -> None:
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


def _check_tier_integrity(
    nodes: dict[str, dict], edges: list[dict], report: VerificationReport
) -> None:
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


def _check_corpus_integrity(
    conn: sqlite3.Connection,
    blobs: BlobStore,
    nodes: dict[str, dict],
    report: VerificationReport,
) -> None:
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


def _flag_dependent_claims(
    nodes: dict[str, dict], edges: list[dict], report: VerificationReport
) -> None:
    """Walk violations back up the DAG: every ReportClaim whose evidence path
    touches a violated node is itself flagged — a corrupted span condemns the
    claims that rest on it, mechanically."""
    violated = {
        v.subject
        for v in report.violations
        if v.subject in nodes
        and v.check
        in (
            "corpus-integrity", "containment", "tier-integrity", "temporal-order",
            "finding-support", "computation", "observation-date",
        )
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


def _check_observation_dates(
    nodes: dict[str, dict], edges: list[dict], report: VerificationReport
) -> None:
    """recency >= 2: re-run the date check intake ran. A date sourced from
    the premise must still be found in the span on the other end of the edge."""
    for e in edges:
        if e["edge_type"] != "extracts":
            continue
        payload = json.loads(e["payload_json"] or "{}")
        if payload.get("date_source") != DATE_SOURCE_AS_OF:
            continue
        span = nodes.get(e["dst_node_id"])
        observed = payload.get("observed_at", "")
        if span is None:
            continue  # already flagged by tier-integrity
        period, _ = check_date_containment(
            observed, span["payload"].get("text", ""), allow_interval=True
        )
        if period is None:
            report.violations.append(
                Violation(
                    "observation-date", e["src_node_id"],
                    f"premise is dated {observed} but its span {span['payload'].get('span_id')} "
                    "does not state that date",
                )
            )


def _check_findings(nodes: dict[str, dict], report: VerificationReport) -> None:
    """derivations >= 2: re-run, from the graph alone, what the harness
    checked when the finding was submitted."""
    for nid in sorted(nodes):
        node = nodes[nid]
        if node["type"] != "Finding":
            continue
        payload = node["payload"]
        texts = {
            pid: nodes[pid]["payload"].get("text", "")
            for pid in payload.get("premise_ids", [])
            if pid in nodes
        }
        edge_type = payload.get("edge_type", "")
        check = check_finding(payload.get("text", ""), edge_type, payload.get("operation"), texts)
        kind = "computation" if edge_type == "computes" else "finding-support"
        for problem in check.problems:
            report.violations.append(Violation(kind, nid, problem))
        if check.ok and check.status != payload.get("status"):
            report.violations.append(
                Violation(
                    "computation", nid,
                    f"stored status {payload.get('status')!r} but the operation "
                    f"recomputes as {check.status!r}",
                )
            )
        elif check.ok and check.status == UNVERIFIED:
            report.advisories.append(
                Violation(
                    "unverified-derivation", nid,
                    f"derived figure the harness cannot recompute: {payload.get('text', '')[:80]!r}",
                )
            )


def _check_containment(
    nodes: dict[str, dict], edges: list[dict], report: VerificationReport, verbatim: bool = False
) -> None:
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
        if verbatim and transform_note:
            # Only reachable when an older session is re-scored: sessions
            # recorded at derivations >= 2 cannot carry a note. The premise
            # was accepted under the old rule, so this is reported as what
            # it now is — an unverified derivation — not as corruption.
            for problem in check_containment(node["payload"]["text"], span_texts, None):
                report.advisories.append(
                    Violation(
                        "unverified-derivation", nid,
                        f"premise rests on a transform note ({transform_note!r}): {problem}",
                    )
                )
            continue
        for problem in check_containment(node["payload"]["text"], span_texts, transform_note):
            report.violations.append(Violation("containment", nid, problem))


def _check_sentence_support(
    nodes: dict[str, dict], edges: list[dict], report: VerificationReport
) -> None:
    """Re-run the writer's sentence check over the stored claims: the text a
    reader saw against the text of what it cites, from the graph alone."""
    out = _out_edges(edges)

    def evidence(node_id: str) -> str:
        # as the writer's check builds it: text, plus the years of any date
        # the premise gave and its span states
        years = stated_years_text(
            [
                json.loads(e["payload_json"] or "{}")
                for e in out.get(node_id, [])
                if e["edge_type"] == "extracts"
            ]
        )
        text = evidence_text(nodes[node_id]["payload"])
        return f"{text} {years}" if years else text

    def cited_texts(claim_id: str) -> list[str]:
        return [
            evidence(e["dst_node_id"])
            for e in out.get(claim_id, [])
            if e["edge_type"] == "aggregates" and e["dst_node_id"] in nodes
        ]

    # What a heading may draw on: evidence cited by any claim in the answer.
    cited_anywhere = [
        text
        for nid in sorted(nodes)
        if nodes[nid]["type"] == "ReportClaim"
        for text in cited_texts(nid)
    ]
    for nid, node in nodes.items():
        if node["type"] != "ReportClaim":
            continue
        text = node["payload"].get("text", "")
        if node["payload"].get("heading"):
            for problem in check_sentence(text, cited_anywhere):
                report.violations.append(
                    Violation(
                        "sentence-support", nid,
                        f"heading {text[:80]!r}: {problem} anywhere in the answer",
                    )
                )
            continue
        if node["payload"].get("narrative"):
            if has_checkable_content(text):
                report.violations.append(
                    Violation(
                        "sentence-support", nid,
                        f"narrative sentence carries a number or quotation: {text[:80]!r}",
                    )
                )
            continue
        cited = cited_texts(nid)
        if not cited:
            continue  # already flagged by claim-path
        for problem in check_sentence(text, cited):
            report.violations.append(
                Violation("sentence-support", nid, f"{problem}: {text[:80]!r}")
            )


def _check_temporal(
    nodes: dict[str, dict], edges: list[dict], report: VerificationReport
) -> None:
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
    nodes: dict[str, dict],
    edges: list[dict],
    report: VerificationReport,
    checker: GroundedChecker | None,
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


def _lint_premises(nodes: dict[str, dict], report: VerificationReport) -> None:
    for nid in sorted(nodes):
        node = nodes[nid]
        if node["type"] != "Premise":
            continue
        for reason in lint_premise(node["payload"]["text"]):
            report.advisories.append(Violation("premise-lint", nid, reason))
