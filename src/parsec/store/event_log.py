"""Append-only event log and the replay-comparison projection (T4).

"Byte-identical replay" is defined as: the projection of the replayed
session's event stream equals the original's, byte for byte, AND the
final answer blobs are identical. The projection strips only fields that
are legitimately volatile across runs: timestamps and wall-clock ledger
amounts. Token and USD debits are kept — they derive from recorded usage
and must reproduce.
"""

from __future__ import annotations

import sqlite3

from parsec.canonical import canonical_json
from parsec.config import Clock
from parsec.models.events import Event, EventType


class EventLog:
    def __init__(self, conn: sqlite3.Connection, clock: Clock):
        self.conn = conn
        self.clock = clock

    def append(
        self,
        session_id: str,
        actor: str,
        event_type: EventType,
        payload: dict,
        parent_seq: int | None = None,
    ) -> int:
        """Append one event; returns its global seq."""
        row = self.conn.execute(
            "SELECT COALESCE(MAX(idx)+1, 0) FROM events WHERE session_id=?", (session_id,)
        ).fetchone()
        idx = row[0]
        cur = self.conn.execute(
            "INSERT INTO events (session_id, idx, ts, actor, event_type, payload_json, parent_seq)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                session_id,
                idx,
                self.clock.now_iso(),
                actor,
                event_type.value,
                canonical_json(payload),
                parent_seq,
            ),
        )
        return cur.lastrowid

    def read(self, session_id: str) -> list[Event]:
        rows = self.conn.execute(
            "SELECT idx, ts, session_id, actor, event_type, payload_json, parent_seq"
            " FROM events WHERE session_id=? ORDER BY idx",
            (session_id,),
        ).fetchall()
        import json

        return [
            Event(
                idx=r["idx"],
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
        """Deterministic byte view of a session's trajectory for replay comparison."""
        lines: list[str] = []
        for ev in self.read(session_id):
            payload = _strip_volatile(ev.event_type, dict(ev.payload))
            if payload is None:
                continue
            lines.append(
                canonical_json(
                    {
                        "idx": ev.idx,
                        "actor": ev.actor,
                        "event_type": ev.event_type.value,
                        "payload": payload,
                    }
                )
            )
        return ("\n".join(lines) + "\n").encode("utf-8")


def _strip_volatile(event_type: EventType, payload: dict) -> dict | None:
    """Remove legitimately-volatile fields; return None to drop the event entirely.

    idx is NOT renumbered after drops — dropped event types must therefore be
    dropped identically in both runs (they are: wall_ms debits occur at the
    same points in an identical trajectory).
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
            payload["config"] = config
    if event_type == EventType.FETCH_PERFORMED:
        payload.pop("from_cache", None)  # original may live-fetch; replay always hits cache
        payload.pop("mode", None)
    return payload
