"""Per-subagent event streams (M11, T8): stream scoping, per-stream
ordinals, interleaving-invariant projections, and the schema migration."""

import asyncio
import sqlite3

from parsec.db.connection import open_db
from parsec.models.events import EventType
from parsec.store.event_log import ORCHESTRATOR_STREAM, EventLog, stream_scope
from tests.conftest import FrozenClock, make_config


def _mine(event_log, sid):
    return [ev for ev in event_log.read(sid) if ev.event_type == EventType.NOTEBOOK_APPENDED]


def test_default_stream_and_per_stream_ordinals(event_log, sessions, config):
    sessions.create(config)
    sid = config.session_id
    event_log.append(sid, "harness", EventType.NOTEBOOK_APPENDED, {"n": 0})
    with stream_scope("sq-1"):
        event_log.append(sid, "harness", EventType.NOTEBOOK_APPENDED, {"n": 1})
        event_log.append(sid, "harness", EventType.NOTEBOOK_APPENDED, {"n": 2})
    event_log.append(sid, "harness", EventType.NOTEBOOK_APPENDED, {"n": 3})

    coords = [(ev.stream_id, ev.stream_idx, ev.payload["n"]) for ev in _mine(event_log, sid)]
    orch = [(i, n) for s, i, n in coords if s == ORCHESTRATOR_STREAM]
    sq1 = [(i, n) for s, i, n in coords if s == "sq-1"]
    # orchestrator ordinals are contiguous around the scoped block; the
    # subagent stream numbers independently from zero
    assert [n for _, n in orch] == [0, 3]
    assert orch[1][0] == orch[0][0] + 1
    assert [i for i, _ in sq1] == [0, 1] and [n for _, n in sq1] == [1, 2]


async def test_asyncio_tasks_carry_their_own_stream(event_log, sessions, config):
    sessions.create(config)
    sid = config.session_id

    async def worker(stream: str, values: list[int]):
        with stream_scope(stream):
            for v in values:
                event_log.append(sid, "harness", EventType.NOTEBOOK_APPENDED, {"n": v})
                await asyncio.sleep(0)  # force interleaving with the sibling

    await asyncio.gather(worker("sq-1", [10, 11, 12]), worker("sq-2", [20, 21, 22]))
    by_stream: dict[str, list[tuple[int, int]]] = {}
    for ev in _mine(event_log, sid):
        by_stream.setdefault(ev.stream_id, []).append((ev.stream_idx, ev.payload["n"]))
    # interleaved arrival, but each stream's ordinals are dense and in order
    assert by_stream["sq-1"] == [(0, 10), (1, 11), (2, 12)]
    assert by_stream["sq-2"] == [(0, 20), (1, 21), (2, 22)]


def test_projection_is_invariant_to_cross_stream_interleaving(db, event_log, sessions, clock, tmp_path):
    """The load-bearing M11 property: two runs whose per-stream sequences
    match but whose arrival interleavings differ project identically."""
    store_sessions = sessions
    a, b = make_config(tmp_path, "s-aaa"), make_config(tmp_path, "s-bbb")
    store_sessions.create(a)
    store_sessions.create(b)

    def emit(sid, stream, n):
        with stream_scope(stream):
            event_log.append(sid, "harness", EventType.NOTEBOOK_APPENDED, {"n": n})

    # session a: sq-1 first, then sq-2, interleaved one way
    emit(a.session_id, "sq-1", 0)
    emit(a.session_id, "sq-2", 0)
    emit(a.session_id, "sq-1", 1)
    emit(a.session_id, "sq-2", 1)
    # session b: same per-stream sequences, opposite interleaving
    emit(b.session_id, "sq-2", 0)
    emit(b.session_id, "sq-2", 1)
    emit(b.session_id, "sq-1", 0)
    emit(b.session_id, "sq-1", 1)

    assert event_log.projection(a.session_id) == event_log.projection(b.session_id)


def test_projection_lines_carry_stream_coordinates(event_log, sessions, config):
    sessions.create(config)
    with stream_scope("sq-1"):
        event_log.append(config.session_id, "harness", EventType.NOTEBOOK_APPENDED, {"n": 1})
    lines = event_log.projection(config.session_id).decode().splitlines()
    assert any('"stream":"sq-1"' in line for line in lines)


def test_migration_backfills_pre_m11_databases(tmp_path):
    """A database created before M11 gains stream columns, with old events
    backfilled into the orchestrator stream keyed by their global idx."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, created_ts TEXT NOT NULL,"
        " query TEXT NOT NULL, config_json TEXT NOT NULL, status TEXT NOT NULL,"
        " answer_blob TEXT, parent_session_id TEXT, finished_ts TEXT)"
    )
    conn.execute(
        "CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,"
        " idx INTEGER NOT NULL, ts TEXT NOT NULL, actor TEXT NOT NULL, event_type TEXT NOT NULL,"
        " payload_json TEXT NOT NULL, parent_seq INTEGER, UNIQUE(session_id, idx))"
    )
    conn.execute(
        "INSERT INTO sessions VALUES ('old', 't', 'q', '{}', 'done', NULL, NULL, NULL)"
    )
    for i in range(3):
        conn.execute(
            "INSERT INTO events (session_id, idx, ts, actor, event_type, payload_json)"
            " VALUES ('old', ?, 't', 'harness', 'notebook_appended', '{}')",
            (i,),
        )
    conn.commit()
    conn.close()

    migrated = open_db(path)
    events = EventLog(migrated, FrozenClock()).read("old")
    assert [(ev.stream_id, ev.stream_idx) for ev in events] == [
        (ORCHESTRATOR_STREAM, 0),
        (ORCHESTRATOR_STREAM, 1),
        (ORCHESTRATOR_STREAM, 2),
    ]
