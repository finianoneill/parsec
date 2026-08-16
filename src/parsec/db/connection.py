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
    return conn


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
