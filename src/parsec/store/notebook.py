"""Notebook (§2.2): append-only markdown, one per session.

The compaction handoff object and human-legible debugging surface: distilled
findings with node IDs, open questions, dead ends. Entries are never
mutated or deleted; the event log carries a hash of each entry so replay
compares notebook content too.
"""

from __future__ import annotations

import sqlite3

from parsec.canonical import sha256_hex
from parsec.config import Clock
from parsec.models.events import EventType
from parsec.store.event_log import EventLog


class Notebook:
    def __init__(self, conn: sqlite3.Connection, event_log: EventLog, clock: Clock):
        self.conn = conn
        self.event_log = event_log
        self.clock = clock

    def append(self, session_id: str, author: str, md_text: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(entry_idx)+1, 0) FROM notebook WHERE session_id=?",
            (session_id,),
        ).fetchone()
        entry_idx = row[0]
        self.conn.execute(
            "INSERT INTO notebook (session_id, entry_idx, ts, author, md_text) VALUES (?,?,?,?,?)",
            (session_id, entry_idx, self.clock.now_iso(), author, md_text),
        )
        self.event_log.append(
            session_id,
            author,
            EventType.NOTEBOOK_APPENDED,
            {"entry_idx": entry_idx, "md_hash": sha256_hex(md_text)},
        )
        return entry_idx

    def render(self, session_id: str) -> str:
        rows = self.conn.execute(
            "SELECT * FROM notebook WHERE session_id=? ORDER BY entry_idx", (session_id,)
        ).fetchall()
        return "\n\n".join(r["md_text"] for r in rows)
