"""Eval scoring axes, ordered by trustworthiness:

1. citation faithfulness — MECHANICAL: fraction of non-narrative claims
   untouched by structural-verification violations. No model.
2. coverage — MECHANICAL: fraction of gold must-find items present in the
   run's claim texts (substring, or "re:" regex). No model.
3. nugget recall — MECHANICAL (M8): weighted binary rubric items graded
   supported / partial / missing against claim texts, with contradiction
   patterns catching reports that assert the opposite of the gold.
4. claim support — MECHANICAL-ish (M8): every claim graded against the
   verbatim spans behind it, from the frozen cache (SupportChecker seam;
   grounded-NLI implementation arrives in M9).
5. synthesis — JUDGE (least trusted, advisory only, §6): a different model
   family scores the prose. Never a gate; nullable when no judge is wired.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field

from parsec.evals.case import Nugget
from parsec.evals.support import MechanicalSupportChecker, SupportChecker, score_claim_support
from parsec.store.blobs import BlobStore
from parsec.verify.structural import verify_session

OKAY_WEIGHT = 0.5


@dataclass
class AxisScores:
    citation_faithfulness: float | None  # None when the run made no claims
    coverage: float | None               # None when the case has no must_find list
    synthesis: float | None = None       # judge score normalized to [0,1]; None without a judge
    nugget_recall: float | None = None   # None when the case has no nuggets
    claim_support: float | None = None   # None when the run made no claims
    claims_total: int = 0
    claims_faithful: int = 0
    must_find_hits: list[str] = None  # type: ignore[assignment]
    must_find_misses: list[str] = None  # type: ignore[assignment]
    nugget_hits: list[str] = field(default_factory=list)
    nugget_misses: list[str] = field(default_factory=list)
    nugget_contradictions: list[str] = field(default_factory=list)
    claim_support_grades: dict[str, int] = field(default_factory=dict)  # grade -> count

    def to_payload(self) -> dict:
        return {
            "citation_faithfulness": self.citation_faithfulness,
            "coverage": self.coverage,
            "synthesis": self.synthesis,
            "nugget_recall": self.nugget_recall,
            "claim_support": self.claim_support,
            "claims_total": self.claims_total,
            "claims_faithful": self.claims_faithful,
            "must_find_hits": self.must_find_hits or [],
            "must_find_misses": self.must_find_misses or [],
            "nugget_hits": self.nugget_hits,
            "nugget_misses": self.nugget_misses,
            "nugget_contradictions": self.nugget_contradictions,
            "claim_support_grades": self.claim_support_grades,
        }


def claim_texts(conn: sqlite3.Connection, session_id: str) -> list[str]:
    texts = []
    for row in conn.execute(
        "SELECT payload_json FROM nodes WHERE session_id=? AND tier=4 ORDER BY created_seq",
        (session_id,),
    ):
        payload = json.loads(row["payload_json"])
        if not payload.get("narrative"):
            texts.append(payload["text"])
    return texts


def score_citation_faithfulness(
    conn: sqlite3.Connection, blobs: BlobStore, session_id: str
) -> tuple[float | None, int, int]:
    """(score, total, faithful): claims implicated in ANY structural violation
    (as subject of claim-path or dependent-claim) count as unfaithful."""
    claims = {
        row["node_id"]
        for row in conn.execute(
            "SELECT node_id, payload_json FROM nodes WHERE session_id=? AND tier=4",
            (session_id,),
        )
        if not json.loads(row["payload_json"]).get("narrative")
    }
    if not claims:
        return None, 0, 0
    report = verify_session(conn, blobs, session_id)
    implicated = {v.subject for v in report.violations if v.subject in claims}
    faithful = len(claims) - len(implicated)
    return faithful / len(claims), len(claims), faithful


def score_coverage(
    conn: sqlite3.Connection, session_id: str, must_find: list[str]
) -> tuple[float | None, list[str], list[str]]:
    """(score, hits, misses) against the run's non-narrative claim texts."""
    if not must_find:
        return None, [], []
    texts = [t.lower() for t in claim_texts(conn, session_id)]
    hits, misses = [], []
    for item in must_find:
        if item.startswith("re:"):
            pattern = re.compile(item[3:], re.IGNORECASE)
            found = any(pattern.search(t) for t in texts)
        else:
            found = any(item.lower() in t for t in texts)
        (hits if found else misses).append(item)
    return len(hits) / len(must_find), hits, misses


def _matches(pattern: str, texts: list[str]) -> bool:
    if pattern.startswith("re:"):
        compiled = re.compile(pattern[3:], re.IGNORECASE)
        return any(compiled.search(t) for t in texts)
    return any(pattern.lower() in t for t in texts)


def score_nuggets(
    conn: sqlite3.Connection, session_id: str, nuggets: list[Nugget]
) -> tuple[float | None, list[str], list[str], list[str]]:
    """Weighted nugget recall over claim texts (vital=1.0, okay=0.5).
    A contradicted nugget scores zero AND is reported separately — asserting
    the opposite of the gold is worse than silence."""
    if not nuggets:
        return None, [], [], []
    texts = [t.lower() for t in claim_texts(conn, session_id)]
    hits, misses, contradictions = [], [], []
    earned, possible = 0.0, 0.0
    for nugget in nuggets:
        weight = 1.0 if nugget.weight == "vital" else OKAY_WEIGHT
        possible += weight
        if any(_matches(p, texts) for p in nugget.contradiction_patterns):
            contradictions.append(nugget.text)
        elif any(_matches(p, texts) for p in nugget.patterns):
            hits.append(nugget.text)
            earned += weight
        else:
            misses.append(nugget.text)
    return earned / possible, hits, misses, contradictions


def score_session(
    conn: sqlite3.Connection,
    blobs: BlobStore,
    session_id: str,
    must_find: list[str],
    synthesis: float | None = None,
    nuggets: list[Nugget] | None = None,
    support_checker: SupportChecker | None = None,
) -> AxisScores:
    faith, total, faithful = score_citation_faithfulness(conn, blobs, session_id)
    coverage, hits, misses = score_coverage(conn, session_id, must_find)
    nugget_recall, n_hits, n_misses, n_contra = score_nuggets(conn, session_id, nuggets or [])
    support, support_details = score_claim_support(
        conn, session_id, support_checker or MechanicalSupportChecker()
    )
    grade_counts: dict[str, int] = {}
    for d in support_details:
        grade_counts[d.grade] = grade_counts.get(d.grade, 0) + 1
    return AxisScores(
        citation_faithfulness=faith,
        coverage=coverage,
        synthesis=synthesis,
        nugget_recall=nugget_recall,
        claim_support=support,
        claims_total=total,
        claims_faithful=faithful,
        must_find_hits=hits,
        must_find_misses=misses,
        nugget_hits=n_hits,
        nugget_misses=n_misses,
        nugget_contradictions=n_contra,
        claim_support_grades=grade_counts,
    )
