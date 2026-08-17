"""Coverage ledger (§2.3): the subquestion tree, outside any model context.

The primary omission defense: the writer refuses to run while `open` items
exist — every subquestion must end answered, partial, blocked, or dropped,
each of the latter with an explicit reason. Every status change is an event.
"""

from __future__ import annotations

import sqlite3

from parsec.models.events import EventType
from parsec.store.event_log import EventLog

STATUSES = ("open", "partial", "answered", "blocked", "dropped")
RESOLVED_STATUSES = ("partial", "answered", "blocked", "dropped")


class CoverageLedger:
    def __init__(self, conn: sqlite3.Connection, event_log: EventLog):
        self.conn = conn
        self.event_log = event_log

    def create(self, session_id: str, sq_id: str, question: str) -> None:
        seq = self.event_log.append(
            session_id,
            "harness",
            EventType.COVERAGE_UPDATED,
            {"sq_id": sq_id, "question": question, "status": "open", "reason": None},
        )
        self.conn.execute(
            "INSERT INTO coverage (session_id, sq_id, question, status, created_seq, updated_seq)"
            " VALUES (?,?,?,?,?,?)",
            (session_id, sq_id, question, "open", seq, seq),
        )

    def set_status(self, session_id: str, sq_id: str, status: str, reason: str | None = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown coverage status: {status}")
        if status in ("blocked", "dropped") and not reason:
            raise ValueError(f"status {status!r} requires an explicit reason")
        seq = self.event_log.append(
            session_id,
            "harness",
            EventType.COVERAGE_UPDATED,
            {"sq_id": sq_id, "status": status, "reason": reason},
        )
        self.conn.execute(
            "UPDATE coverage SET status=?, reason=?, updated_seq=? WHERE session_id=? AND sq_id=?",
            (status, reason, seq, session_id, sq_id),
        )

    def all(self, session_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM coverage WHERE session_id=? ORDER BY sq_id", (session_id,)
        ).fetchall()

    def open_items(self, session_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM coverage WHERE session_id=? AND status='open' ORDER BY sq_id",
            (session_id,),
        ).fetchall()

    def summary(self, session_id: str) -> dict[str, str]:
        return {r["sq_id"]: r["status"] for r in self.all(session_id)}
