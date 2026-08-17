"""Claim-support checking: does the cited evidence actually support the claim?

The 2026 audits' recurring finding is a 20–50 point gap between "link
works" and "evidence supports the claim." This module grades every
non-narrative ReportClaim against the verbatim spans behind it — from the
frozen fetch cache, so support checking is fully reproducible.

`SupportChecker` is the seam: the default `MechanicalSupportChecker` grades
with exact number/quote containment plus content-word overlap (no model,
deterministic). `GroundedSupportChecker` (M9) keeps the exact-match floor
and grades the prose with the grounded-NLI tier from `parsec.verify.nli` —
select it with `parsec eval run --support-checker grounded`.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Literal, Protocol

from parsec.verify.containment import extract_numbers, extract_quotes
from parsec.verify.nli import GroundedChecker, LexicalGroundedChecker

Grade = Literal["full", "partial", "none"]

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were with".split()
)
OVERLAP_FULL = 0.5
OVERLAP_PARTIAL = 0.2


class SupportChecker(Protocol):
    name: str

    def grade(self, claim_text: str, evidence_texts: list[str]) -> Grade: ...


class MechanicalSupportChecker:
    """Deterministic, model-free grading:
    - any number/quote in the claim missing from all evidence -> none
    - content-word overlap >= OVERLAP_FULL -> full
    - overlap >= OVERLAP_PARTIAL -> partial, else none
    """

    name = "mechanical-v1"

    def grade(self, claim_text: str, evidence_texts: list[str]) -> Grade:
        if not evidence_texts:
            return "none"
        if _hard_mismatch(claim_text, evidence_texts):
            return "none"

        claim_words = _content_words(claim_text)
        evidence_blob = " ".join(evidence_texts)
        if not claim_words:
            return "partial"
        evidence_words = _content_words(evidence_blob)
        overlap = len(claim_words & evidence_words) / len(claim_words)
        if overlap >= OVERLAP_FULL:
            return "full"
        if overlap >= OVERLAP_PARTIAL:
            return "partial"
        return "none"


def _hard_mismatch(claim_text: str, evidence_texts: list[str]) -> bool:
    """The exact-match floor (§10.2: never let NLI override it): a number or
    quote in the claim that appears in NO evidence text is a hard fail."""
    evidence_blob = " ".join(evidence_texts)
    evidence_numbers = set(extract_numbers(evidence_blob))
    for num in extract_numbers(claim_text):
        if num not in evidence_numbers:
            return True
    normalized_evidence = " ".join(evidence_blob.split())
    return any(quote not in normalized_evidence for quote in extract_quotes(claim_text))


class GroundedSupportChecker:
    """M9: the grounded-NLI tier behind the M8 seam. Exact-match floor first
    (numbers/quotes must appear, non-negotiable), then the grounded checker's
    verdict grades the prose: supported -> full, uncertain -> partial,
    unsupported/contradicted -> none."""

    def __init__(self, checker: GroundedChecker | None = None):
        self.checker = checker or LexicalGroundedChecker()
        self.name = f"grounded-{self.checker.name}"

    def grade(self, claim_text: str, evidence_texts: list[str]) -> Grade:
        if not evidence_texts:
            return "none"
        if _hard_mismatch(claim_text, evidence_texts):
            return "none"
        verdict = self.checker.check(claim_text, evidence_texts)
        if verdict.verdict == "supported":
            return "full"
        if verdict.verdict == "uncertain":
            return "partial"
        return "none"


def make_support_checker(name: str) -> SupportChecker:
    """CLI seam for `parsec eval run --support-checker`."""
    if name == "mechanical":
        return MechanicalSupportChecker()
    if name == "grounded":
        return GroundedSupportChecker()
    raise ValueError(f"unknown support checker {name!r}; expected mechanical | grounded")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


@dataclass
class ClaimSupport:
    claim_id: str
    claim_text: str
    grade: Grade
    evidence_spans: int


def score_claim_support(
    conn: sqlite3.Connection,
    session_id: str,
    checker: SupportChecker,
) -> tuple[float | None, list[ClaimSupport]]:
    """Grade every non-narrative claim against the span texts reachable
    through its evidence chain (claim -> finding/premise -> spans).
    Score = (full + 0.5*partial) / claims."""
    nodes: dict[str, dict] = {}
    for row in conn.execute(
        "SELECT node_id, node_type, payload_json FROM nodes WHERE session_id=?", (session_id,)
    ):
        nodes[row["node_id"]] = {
            "type": row["node_type"],
            "payload": json.loads(row["payload_json"]),
        }
    out_edges: dict[str, list[str]] = {}
    for row in conn.execute(
        "SELECT src_node_id, dst_node_id, edge_type FROM edges WHERE session_id=?", (session_id,)
    ):
        if row["edge_type"] != "contradicts":
            out_edges.setdefault(row["src_node_id"], []).append(row["dst_node_id"])

    details: list[ClaimSupport] = []
    for nid in sorted(nodes):
        node = nodes[nid]
        if node["type"] != "ReportClaim" or node["payload"].get("narrative"):
            continue
        span_texts: list[str] = []
        seen, stack = {nid}, [nid]
        while stack:
            cur = stack.pop()
            for dst in out_edges.get(cur, []):
                if dst in seen or dst not in nodes:
                    continue
                seen.add(dst)
                if nodes[dst]["type"] == "SourceSpan":
                    span_texts.append(nodes[dst]["payload"]["text"])
                else:
                    stack.append(dst)
        span_texts.sort()
        grade = checker.grade(node["payload"]["text"], span_texts)
        details.append(ClaimSupport(nid, node["payload"]["text"], grade, len(span_texts)))

    if not details:
        return None, []
    score = sum(1.0 if d.grade == "full" else 0.5 if d.grade == "partial" else 0.0 for d in details)
    return score / len(details), details
