"""Eval scoring functions, ordered by trustworthiness:

1. citation faithfulness — MECHANICAL: fraction of non-narrative claims
   untouched by structural-verification violations. No model.
2. coverage — MECHANICAL: fraction of gold must-find items present in the
   run's claim texts (substring, or "re:" regex). No model.
3. nugget recall — MECHANICAL (M8): weighted binary rubric items graded
   supported / partial / missing against claim texts, with contradiction
   patterns catching reports that assert the opposite of the gold.
4. claim support — MECHANICAL-ish (M8/M9): every claim graded against the
   verbatim spans behind it, from the frozen cache (SupportChecker seam in
   `parsec.evals.support`).
5. synthesis — JUDGE (least trusted, advisory only, §6): a different model
   family scores the prose. Never a gate; nullable when no judge is wired.

These are the primitives. The axis objects that wrap them — and the tier
tag that decides whether an axis may gate — live in `parsec.evals.metrics`
(M15.1); `parsec.evals.losses` reuses the same matchers to say WHY a gold
item went missing.
"""

from __future__ import annotations

import json
import re
import sqlite3

from parsec.evals.case import Nugget
from parsec.store.blobs import BlobStore
from parsec.verify.structural import verify_session

OKAY_WEIGHT = 0.5


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


def matches(pattern: str, texts: list[str]) -> bool:
    """One gold matcher against lowercased texts: case-insensitive substring,
    or a regex when prefixed "re:"."""
    if pattern.startswith("re:"):
        compiled = re.compile(pattern[3:], re.IGNORECASE)
        return any(compiled.search(t) for t in texts)
    return any(pattern.lower() in t for t in texts)


def matches_any(patterns: list[str], texts: list[str]) -> bool:
    return any(matches(p, texts) for p in patterns)


def score_coverage(
    conn: sqlite3.Connection, session_id: str, must_find: list[str]
) -> tuple[float | None, list[str], list[str]]:
    """(score, hits, misses) against the run's non-narrative claim texts."""
    if not must_find:
        return None, [], []
    texts = [t.lower() for t in claim_texts(conn, session_id)]
    hits, misses = [], []
    for item in must_find:
        (hits if matches(item, texts) else misses).append(item)
    return len(hits) / len(must_find), hits, misses


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
        if matches_any(nugget.contradiction_patterns, texts):
            contradictions.append(nugget.text)
        elif matches_any(nugget.patterns, texts):
            hits.append(nugget.text)
            earned += weight
        else:
            misses.append(nugget.text)
    return earned / possible, hits, misses, contradictions
