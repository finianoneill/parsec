"""open_db() must migrate databases created by older parsec versions.

Regression for the Phase-0 ledger.stream_id column: schema.sql carried an
index on the new column, which made executescript() fail on any pre-change
database BEFORE _migrate() could add the column — every fresh-DB test
passed while the checked-in data/parsec.db crashed the CLI. Indexes on
migrated columns belong in _migrate() only."""

from __future__ import annotations

import sqlite3

from parsec.db.connection import open_db

# The pre-Phase-0 ledger shape (no stream_id).
_OLD_LEDGER = """
CREATE TABLE ledger (
  entry_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id   TEXT NOT NULL,
  ts           TEXT NOT NULL,
  category     TEXT NOT NULL,
  amount       REAL NOT NULL,
  actor        TEXT NOT NULL,
  ref_seq      INTEGER,
  note         TEXT
);
"""

# The pre-M11 events shape (no stream coordinates) — the older migration
# this file also locks in.
_OLD_EVENTS = """
CREATE TABLE events (
  seq          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id   TEXT NOT NULL,
  idx          INTEGER NOT NULL,
  ts           TEXT NOT NULL,
  actor        TEXT NOT NULL,
  event_type   TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  parent_seq   INTEGER,
  UNIQUE(session_id, idx)
);
"""


def _cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _indexes(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        if row[0]
    }


def test_open_db_migrates_pre_phase0_ledger(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(_OLD_LEDGER + _OLD_EVENTS)
    old.execute(
        "INSERT INTO ledger (session_id, ts, category, amount, actor)"
        " VALUES ('s-old', 't', 'usd', 0.01, 'gateway:m')"
    )
    old.commit()
    old.close()

    conn = open_db(path)  # must not raise
    assert "stream_id" in _cols(conn, "ledger")
    assert "stream_id" in _cols(conn, "events") and "stream_idx" in _cols(conn, "events")
    assert {"ix_ledger_stream", "ix_events_stream"} <= _indexes(conn)
    # pre-existing rows default into the orchestrator stream
    row = conn.execute("SELECT stream_id, amount FROM ledger WHERE session_id='s-old'").fetchone()
    assert row["stream_id"] == "orchestrator" and row["amount"] == 0.01


def test_open_db_is_idempotent_on_current_schema(tmp_path):
    path = tmp_path / "new.db"
    open_db(path).close()
    conn = open_db(path)  # second open: schema + migrations re-apply cleanly
    assert "stream_id" in _cols(conn, "ledger")
