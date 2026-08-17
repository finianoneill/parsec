"""3-axis eval scoring (§11 M5), ordered by trustworthiness:

1. citation faithfulness — MECHANICAL: fraction of non-narrative claims
   untouched by structural-verification violations. No model.
2. coverage — MECHANICAL: fraction of gold must-find items present in the
   run's claim texts (substring, or "re:" regex). No model.
3. synthesis — JUDGE (least trusted, advisory only, §6): a different model
   family scores the prose. Never a gate; nullable when no judge is wired.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass

from parsec.store.blobs import BlobStore
from parsec.verify.structural import verify_session


@dataclass
class AxisScores:
    citation_faithfulness: float | None  # None when the run made no claims
    coverage: float | None               # None when the case has no must_find list
    synthesis: float | None = None       # judge score normalized to [0,1]; None without a judge
    claims_total: int = 0
    claims_faithful: int = 0
    must_find_hits: list[str] = None  # type: ignore[assignment]
    must_find_misses: list[str] = None  # type: ignore[assignment]

    def to_payload(self) -> dict:
        return {
            "citation_faithfulness": self.citation_faithfulness,
            "coverage": self.coverage,
            "synthesis": self.synthesis,
            "claims_total": self.claims_total,
            "claims_faithful": self.claims_faithful,
            "must_find_hits": self.must_find_hits or [],
            "must_find_misses": self.must_find_misses or [],
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


def score_session(
    conn: sqlite3.Connection,
    blobs: BlobStore,
    session_id: str,
    must_find: list[str],
    synthesis: float | None = None,
) -> AxisScores:
    faith, total, faithful = score_citation_faithfulness(conn, blobs, session_id)
    coverage, hits, misses = score_coverage(conn, session_id, must_find)
    return AxisScores(
        citation_faithfulness=faith,
        coverage=coverage,
        synthesis=synthesis,
        claims_total=total,
        claims_faithful=faithful,
        must_find_hits=hits,
        must_find_misses=misses,
    )
