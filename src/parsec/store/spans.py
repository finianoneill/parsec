"""Span rows: the tier-0 addressable units every claim ultimately cites."""

from __future__ import annotations

import sqlite3


class SpanStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def put_spans(
        self,
        doc_hash: str,
        spans: list[tuple[str, int, int, str]],  # (span_id, start, end, text)
        created_seq: int | None = None,
    ) -> None:
        self.conn.executemany(
            "INSERT OR IGNORE INTO spans (span_id, doc_hash, char_start, char_end, text, created_seq)"
            " VALUES (?,?,?,?,?,?)",
            [(sid, doc_hash, s, e, t, created_seq) for sid, s, e, t in spans],
        )

    def get(self, span_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM spans WHERE span_id=?", (span_id,)).fetchone()

    def for_doc(self, doc_hash: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM spans WHERE doc_hash=? ORDER BY char_start", (doc_hash,)
        ).fetchall()
