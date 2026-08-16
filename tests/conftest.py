from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from parsec.config import Budgets, CacheMode, RunConfig
from parsec.db.connection import open_db
from parsec.store.blobs import BlobStore
from parsec.store.event_log import EventLog
from parsec.store.ledger import Ledger
from parsec.store.sessions import SessionStore


class FrozenClock:
    """Deterministic clock for tests: fixed timestamp, manually advanced monotonic."""

    def __init__(self):
        self.mono = 0.0

    def now_iso(self) -> str:
        return "2026-01-01T00:00:00.000+00:00"

    def monotonic(self) -> float:
        return self.mono

    async def sleep(self, seconds: float) -> None:
        self.mono += seconds


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock()


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    return open_db(tmp_path / "parsec.db")


@pytest.fixture
def blobs(tmp_path: Path) -> BlobStore:
    return BlobStore(tmp_path / "blobs")


@pytest.fixture
def event_log(db, clock) -> EventLog:
    return EventLog(db, clock)


@pytest.fixture
def ledger(db, clock) -> Ledger:
    return Ledger(db, clock)


@pytest.fixture
def sessions(db, clock) -> SessionStore:
    return SessionStore(db, clock)


def make_config(tmp_path: Path, session_id: str = "s-test", **overrides) -> RunConfig:
    defaults = dict(
        session_id=session_id,
        query="test query",
        model="fake-model",
        adapter="fake",
        cache_mode=CacheMode.RECORD,
        budgets=Budgets(),
        data_dir=tmp_path,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


@pytest.fixture
def config(tmp_path: Path) -> RunConfig:
    return make_config(tmp_path)
