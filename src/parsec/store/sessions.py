"""Session rows: creation, status transitions, lookup."""

from __future__ import annotations

import json
import sqlite3

from parsec.canonical import canonical_json
from parsec.config import Clock, RunConfig


class SessionStore:
    def __init__(self, conn: sqlite3.Connection, clock: Clock):
        self.conn = conn
        self.clock = clock

    def create(self, config: RunConfig, parent_session_id: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO sessions (session_id, created_ts, query, config_json, status, parent_session_id)"
            " VALUES (?,?,?,?,?,?)",
            (
                config.session_id,
                self.clock.now_iso(),
                config.query,
                canonical_json(config.to_json_dict()),
                "running",
                parent_session_id,
            ),
        )

    def finish(self, session_id: str, status: str, answer_blob: str | None) -> None:
        self.conn.execute(
            "UPDATE sessions SET status=?, answer_blob=?, finished_ts=? WHERE session_id=?",
            (status, answer_blob, self.clock.now_iso(), session_id),
        )

    def get(self, session_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()

    def get_config(self, session_id: str) -> RunConfig:
        row = self.get(session_id)
        if row is None:
            raise KeyError(f"unknown session: {session_id}")
        return RunConfig.model_validate(json.loads(row["config_json"]))

    def list(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM sessions ORDER BY created_ts DESC"
        ).fetchall()
