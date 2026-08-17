"""Trajectory/process metrics — how the agent worked, not just what it said.

Computed from the event log and ledger (deterministic to log, nearly free
given the harness records everything). Reported alongside outcome scores in
the regression diff so efficiency regressions surface even when scores hold.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from parsec.canonical import hash_obj
from parsec.models.events import EventType
from parsec.retrieval.fetcher import canonicalize_url
from parsec.retrieval.search_provider import normalize_query
from parsec.store.event_log import EventLog
from parsec.store.ledger import Ledger


@dataclass
class TrajectoryMetrics:
    searches: int = 0
    fetches: int = 0
    redundant_searches: int = 0        # same normalized query re-issued
    repeated_tool_calls: int = 0       # identical (tool, input) re-issued — loop smell
    tool_errors: int = 0
    gold_fetch_fraction: float | None = None       # fetched URLs that were gold docs
    distractor_fetch_fraction: float | None = None # fetched URLs that were planted distractors
    fetches_to_first_gold: int | None = None       # 1-based; None = never hit gold
    tokens: int = 0
    usd: float = 0.0

    def to_payload(self) -> dict:
        return {
            "searches": self.searches,
            "fetches": self.fetches,
            "redundant_searches": self.redundant_searches,
            "repeated_tool_calls": self.repeated_tool_calls,
            "tool_errors": self.tool_errors,
            "gold_fetch_fraction": self.gold_fetch_fraction,
            "distractor_fetch_fraction": self.distractor_fetch_fraction,
            "fetches_to_first_gold": self.fetches_to_first_gold,
            "tokens": self.tokens,
            "usd": round(self.usd, 6),
        }


def compute_trajectory(
    conn: sqlite3.Connection,
    event_log: EventLog,
    ledger: Ledger,
    session_id: str,
    gold_docs: list[str],
    distractor_docs: list[str],
) -> TrajectoryMetrics:
    m = TrajectoryMetrics()
    gold = {canonicalize_url(u) for u in gold_docs}
    distractors = {canonicalize_url(u) for u in distractor_docs}
    seen_queries: set[str] = set()
    seen_intents: set[str] = set()
    fetched_urls: list[str] = []

    for ev in event_log.read(session_id):
        if ev.event_type == EventType.TOOL_INTENT:
            intent_key = hash_obj({"tool": ev.payload["tool_name"], "input": ev.payload["input"]})
            if intent_key in seen_intents:
                m.repeated_tool_calls += 1
            seen_intents.add(intent_key)
            if ev.payload["tool_name"] in ("search_broad", "search_within"):
                m.searches += 1
                q = normalize_query(ev.payload["input"].get("query", ""))
                if q in seen_queries:
                    m.redundant_searches += 1
                seen_queries.add(q)
        elif ev.event_type == EventType.FETCH_PERFORMED:
            m.fetches += 1
            fetched_urls.append(canonicalize_url(ev.payload["url"]))
        elif ev.event_type == EventType.TOOL_RESULT and not ev.payload.get("ok", True):
            m.tool_errors += 1

    if fetched_urls and gold:
        hits = [u for u in fetched_urls if u in gold]
        m.gold_fetch_fraction = round(len(hits) / len(fetched_urls), 4)
        for i, u in enumerate(fetched_urls):
            if u in gold:
                m.fetches_to_first_gold = i + 1
                break
    if fetched_urls and distractors:
        m.distractor_fetch_fraction = round(
            len([u for u in fetched_urls if u in distractors]) / len(fetched_urls), 4
        )

    totals = ledger.totals(session_id)
    m.tokens = int(
        sum(totals.get(c, 0.0) for c in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens"))
    )
    m.usd = totals.get("usd", 0.0)
    return m
