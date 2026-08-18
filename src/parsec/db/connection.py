"""SQLite connection management: WAL, foreign keys, idempotent schema apply."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Iterator


def open_db(path: Path | str) -> sqlite3.Connection:
    """Open (creating if needed) the parsec database with required pragmas."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)  # autocommit; explicit txns via helper
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    schema = resources.files("parsec.db").joinpath("schema.sql").read_text(encoding="utf-8")
    conn.executescript(schema)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent column migrations for databases created before the current
    schema. CREATE TABLE IF NOT EXISTS never alters an existing table, so new
    columns are added here."""
    event_cols = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    if "stream_id" not in event_cols:
        # Pre-M11 events all belonged to the single global stream; their
        # global idx doubles as the stream ordinal.
        conn.execute("ALTER TABLE events ADD COLUMN stream_id TEXT NOT NULL DEFAULT 'orchestrator'")
        conn.execute("ALTER TABLE events ADD COLUMN stream_idx INTEGER NOT NULL DEFAULT 0")
        conn.execute("UPDATE events SET stream_idx = idx")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_events_stream ON events(session_id, stream_id, stream_idx)"
    )


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
