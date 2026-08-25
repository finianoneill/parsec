"""Append-only event log and the replay-comparison projection (T4, T8).

"Byte-identical replay" is defined as: the projection of the replayed
session's event stream equals the original's, byte for byte, AND the
final answer blobs are identical. The projection strips only fields that
are legitimately volatile across runs: timestamps and wall-clock ledger
amounts. Token and USD debits are kept — they derive from recorded usage
and must reproduce.

M11 (T8 — concurrency is recorded, not forbidden): every event carries a
STREAM coordinate. A stream is one logical sequential actor — the
orchestrator, or one subagent — identified by `CURRENT_STREAM`, a
contextvar that asyncio tasks inherit at creation, so a concurrent
subagent's gateway calls, tool events, and DAG writes land in its own
stream without threading a parameter through every call site. Within a
stream, (stream_idx) ordering is deterministic; ACROSS streams, arrival
interleaving (the global idx) is genuinely nondeterministic under
concurrency. The projection therefore compares per-stream event sequences,
and the orchestrator's recorded SUBAGENT_JOINED events pin the one
cross-stream fact that matters: the merge order.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from parsec.canonical import canonical_json
from parsec.config import Clock
from parsec.models.events import Event, EventType

ORCHESTRATOR_STREAM = "orchestrator"

CURRENT_STREAM: ContextVar[str] = ContextVar("parsec_current_stream", default=ORCHESTRATOR_STREAM)


@contextmanager
def stream_scope(stream_id: str) -> Iterator[None]:
    """Scope all events (and gateway calls, tool events, debits) emitted
    inside to the given stream. Works both for inline awaits and for
    asyncio tasks (each task copies the context at creation)."""
    token = CURRENT_STREAM.set(stream_id)
    try:
        yield
    finally:
        CURRENT_STREAM.reset(token)


class EventLog:
    def __init__(self, conn: sqlite3.Connection, clock: Clock):
        self.conn = conn
        self.clock = clock
        # Display-only tap (CLI activity view): called after each append with
        # (event_type, payload, stream_id). Best-effort; never affects the log.
        self.listener = None

    def append(
        self,
        session_id: str,
        actor: str,
        event_type: EventType,
        payload: dict,
        parent_seq: int | None = None,
    ) -> int:
        """Append one event to the current stream; returns its global seq.
        Synchronous end-to-end, so concurrent asyncio tasks cannot interleave
        mid-append — both ordinals are race-free on one connection."""
        stream_id = CURRENT_STREAM.get()
        idx = self.conn.execute(
            "SELECT COALESCE(MAX(idx)+1, 0) FROM events WHERE session_id=?", (session_id,)
        ).fetchone()[0]
        stream_idx = self.conn.execute(
            "SELECT COALESCE(MAX(stream_idx)+1, 0) FROM events WHERE session_id=? AND stream_id=?",
            (session_id, stream_id),
        ).fetchone()[0]
        cur = self.conn.execute(
            "INSERT INTO events (session_id, idx, stream_id, stream_idx, ts, actor, event_type,"
            " payload_json, parent_seq) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                session_id,
                idx,
                stream_id,
                stream_idx,
                self.clock.now_iso(),
                actor,
                event_type.value,
                canonical_json(payload),
                parent_seq,
            ),
        )
        if self.listener is not None:
            try:
                self.listener(event_type, payload, stream_id)
            except Exception:  # noqa: BLE001 — display must never break the log
                pass
        return cur.lastrowid

    def read(self, session_id: str) -> list[Event]:
        rows = self.conn.execute(
            "SELECT idx, stream_id, stream_idx, ts, session_id, actor, event_type, payload_json,"
            " parent_seq FROM events WHERE session_id=? ORDER BY idx",
            (session_id,),
        ).fetchall()
        import json

        return [
            Event(
                idx=r["idx"],
                stream_id=r["stream_id"],
                stream_idx=r["stream_idx"],
                ts=r["ts"],
                session_id=r["session_id"],
                actor=r["actor"],
                event_type=EventType(r["event_type"]),
                payload=json.loads(r["payload_json"]),
                parent_seq=r["parent_seq"],
            )
            for r in rows
        ]

    def projection(self, session_id: str) -> bytes:
        """Deterministic byte view of a session's trajectory for replay
        comparison: per-stream event sequences (streams sorted by id, events
        by stream ordinal). Cross-stream arrival order is volatile under
        concurrency and deliberately absent; the recorded join order lives in
        the orchestrator stream's SUBAGENT_JOINED events, so it IS compared."""
        streams: dict[str, list[Event]] = {}
        for ev in self.read(session_id):
            streams.setdefault(ev.stream_id, []).append(ev)
        lines: list[str] = []
        for stream_id in sorted(streams):
            for ev in sorted(streams[stream_id], key=lambda e: e.stream_idx):
                payload = _strip_volatile(ev.event_type, dict(ev.payload))
                if payload is None:
                    continue
                lines.append(
                    canonical_json(
                        {
                            "stream": stream_id,
                            "idx": ev.stream_idx,
                            "actor": ev.actor,
                            "event_type": ev.event_type.value,
                            "payload": payload,
                        }
                    )
                )
        return ("\n".join(lines) + "\n").encode("utf-8")


def _strip_volatile(event_type: EventType, payload: dict) -> dict | None:
    """Remove legitimately-volatile fields; return None to drop the event entirely.

    stream_idx is NOT renumbered after drops — dropped event types must
    therefore be dropped identically in both runs (they are: wall_ms debits
    occur at the same points in an identical per-stream trajectory).
    """
    payload.pop("ts", None)
    if event_type == EventType.BUDGET_DEBIT and payload.get("category") == "wall_ms":
        return None
    if event_type == EventType.SESSION_STARTED:
        # session ids differ between original and replay by construction
        payload.pop("session_id", None)
        config = payload.get("config")
        if isinstance(config, dict):
            config = dict(config)
            config.pop("session_id", None)
            config.pop("adapter", None)  # replay swaps anthropic->replay
            config.pop("cache_mode", None)  # replay forces replay mode
            # Transport/provenance fields no recorded trajectory depends on;
            # stripping keeps pre-Phase-0 recordings replayable. model_retry's
            # effects ARE the journaled LLM_RETRY events (which are compared),
            # so the knob itself is likewise volatile; same argument for
            # max_context_tokens (its effects are the CONTEXT_COMPACTED
            # events and the prompt hashes).
            config.pop("model_max_retries", None)
            config.pop("model_retry", None)
            config.pop("max_context_tokens", None)
            config.pop("parsec_version", None)
            payload["config"] = config
    if event_type == EventType.FETCH_PERFORMED:
        payload.pop("from_cache", None)  # original may live-fetch; replay always hits cache
        payload.pop("mode", None)
    return payload
