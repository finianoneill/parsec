"""M14.3 exit tests: `parsec watch` chains refreshes off the newest session,
reports only material rounds, labels the prior run's claims held/overturned
as (credence, outcome) pairs `parsec calibrate` reads, sleeps the schedule
between rounds, and every round is itself a replayable recording."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

import parsec.cli as cli
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.models.events import EventType
from parsec.replay import run_replay
from parsec.store.event_log import EventLog
from parsec.verify.diff import ClaimDelta, DiffReport
from parsec.watch import (
    append_labels,
    format_duration,
    label_outcomes,
    parse_duration,
    read_labels,
    run_watch,
)
from tests.integration.m14_corpus import SENT_A, SENT_B2, URL_B, _premise_id, _span


def _watch_script(answer: str) -> list:
    """The seeded refresh: sq-2 re-fetches the price page, records the new
    price, submits; the writer emits `answer`."""
    return [
        scripted_response(
            [{"type": "tool_use", "id": "tu_wf", "name": "fetch", "input": {"url": URL_B}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_wr", "name": "record_premises",
              "input": {"premises": [{"text": SENT_B2, "span_refs": [_span(SENT_B2)],
                                      "claim_class": "volatile"}]}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_ws", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use"),
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn"),
    ]


def _height_only_answer() -> str:
    """Drops the price claim: against the parent that reads as retracted."""
    return f"Only the height is settled. [narrative]\n{SENT_A} [{_premise_id(SENT_A)}]"


def _delta(status, credence_a=0.8, a_id="claim:a", text="c") -> ClaimDelta:
    return ClaimDelta(
        status=status, text=text, match="id" if a_id and status not in ("new", "retracted") else None,
        similarity=None, a_id=a_id, b_id="claim:b" if status != "retracted" else None,
        credence_a=credence_a, credence_b=0.7, provenance_a=None, provenance_b=None,
    )


def test_label_outcomes_maps_time_to_outcomes():
    report = DiffReport(
        "s-a", "s-b", "q", "q", False,
        claims=[
            _delta("held", 0.9, "c1"), _delta("strengthened", 0.6, "c2"),
            _delta("weakened", 0.7, "c3"), _delta("superseded", 0.85, "c4"),
            _delta("retracted", 0.5, "c5"), _delta("new", None, None),
        ],
        documents=[],
    )
    done = label_outcomes(report, "done")
    assert [(lb["claim_id"], lb["credence"], lb["label"], lb["outcome"]) for lb in done] == [
        ("c1", 0.9, 1, "held"), ("c2", 0.6, 1, "held"), ("c3", 0.7, 1, "held"),
        ("c4", 0.85, 0, "overturned"), ("c5", 0.5, 0, "overturned"),
    ]
    assert all(lb["session"] == "s-a" and lb["refreshed"] == "s-b" for lb in done)
    # A partial refresh may simply not have re-researched the claim: skipped.
    partial = label_outcomes(report, "partial")
    assert [lb["claim_id"] for lb in partial] == ["c1", "c2", "c3", "c4"]


def test_duration_parsing():
    assert parse_duration("30m") == 1800 and parse_duration("6h") == 21600
    assert parse_duration("1d") == 86400 and parse_duration("45") == 45
    assert parse_duration("1.5h") == 5400
    for bad in ("", "0", "-5m", "6x", "soon"):
        with pytest.raises(ValueError):
            parse_duration(bad)
    assert format_duration(21600) == "6h" and format_duration(90) == "90s"


def test_concurrent_watches_never_lose_labels(tmp_path):
    """Separate processes appending to one labels file serialize on the
    file lock and land by atomic rename: nothing is lost, nothing is torn."""
    import subprocess
    import sys

    path = tmp_path / "shared.json"
    per_proc, procs = 25, 4
    code = (
        "import sys; from pathlib import Path; from parsec.watch import append_labels\n"
        "p, who = Path(sys.argv[1]), sys.argv[2]\n"
        f"for i in range({per_proc}): append_labels(p, [{{'credence': 0.5, 'label': 1, 'who': who, 'i': i}}])\n"
    )
    workers = [
        subprocess.Popen([sys.executable, "-c", code, str(path), f"w{n}"])
        for n in range(procs)
    ]
    assert [w.wait(timeout=120) for w in workers] == [0] * procs
    labels = read_labels(path)
    assert len(labels) == per_proc * procs
    assert sorted((lb["who"], lb["i"]) for lb in labels) == sorted(
        (f"w{n}", i) for n in range(procs) for i in range(per_proc)
    )
    # No stray temp files; the lock sidecar is the only companion.
    assert sorted(q.name for q in tmp_path.iterdir()) == ["shared.json", "shared.json.lock"]


def test_lock_backend_covers_windows_and_refuses_unlocked_writes(tmp_path, monkeypatch):
    """Without fcntl the lock comes from msvcrt: contention (EACCES) polls,
    any other lock failure propagates instead of spinning forever; with
    neither module, append_labels refuses rather than racing."""
    import errno
    import sys

    calls: list[int] = []

    class FakeMsvcrt:
        LK_NBLCK, LK_UNLCK = 2, 0
        failures = [OSError(errno.EACCES, "locked by another process")]

        @staticmethod
        def locking(fd, mode, nbytes):
            calls.append(mode)
            if mode == FakeMsvcrt.LK_NBLCK and FakeMsvcrt.failures:
                raise FakeMsvcrt.failures.pop()

    monkeypatch.setitem(sys.modules, "fcntl", None)  # `import fcntl` -> ImportError
    monkeypatch.setitem(sys.modules, "msvcrt", FakeMsvcrt)
    monkeypatch.setattr("time.sleep", lambda s: None)
    path = tmp_path / "labels.json"
    assert append_labels(path, [{"credence": 0.5, "label": 1}]) == 1
    assert calls == [2, 2, 0]  # busy once, acquired, released
    assert read_labels(path) == [{"credence": 0.5, "label": 1}]

    # A permanent failure (bad descriptor) is not contention: one attempt, raised.
    calls.clear()
    FakeMsvcrt.failures = [OSError(errno.EBADF, "bad file descriptor")]
    with pytest.raises(OSError) as info:
        append_labels(path, [{"credence": 0.5, "label": 1}])
    assert info.value.errno == errno.EBADF and calls == [2]
    assert read_labels(path) == [{"credence": 0.5, "label": 1}]

    monkeypatch.setitem(sys.modules, "msvcrt", None)
    with pytest.raises(RuntimeError, match="process lock"):
        append_labels(path, [{"credence": 0.5, "label": 1}])
    assert read_labels(path) == [{"credence": 0.5, "label": 1}]  # nothing written unlocked


@pytest.mark.skipif(sys.platform != "win32", reason="native msvcrt lock semantics")
def test_windows_lock_propagates_non_contention_errors(tmp_path):
    """Native regression: msvcrt.locking on an invalid descriptor raises
    EBADF, which must surface immediately rather than be retried."""
    import errno
    import os

    from parsec.watch import _lock_backend

    acquire, _release = _lock_backend()
    fd = os.open(tmp_path / "lock", os.O_RDWR | os.O_CREAT)
    os.close(fd)

    class Closed:
        def fileno(self):
            return fd  # no longer valid

    with pytest.raises(OSError) as info:
        acquire(Closed())
    assert info.value.errno == errno.EBADF


def test_labels_file_round_trips_in_calibrate_shape(tmp_path):
    path = tmp_path / "labels.json"
    assert read_labels(path) == []
    assert append_labels(path, [{"credence": 0.8, "label": 1}]) == 1
    assert append_labels(path, [{"credence": 0.3, "label": 0}]) == 2
    assert cli._extract_label_pairs(json.loads(path.read_text())) == [(0.8, 1), (0.3, 0)]


async def test_watch_chains_refreshes_reports_material_only_and_labels(
    parent, db, blobs, clock, pages, transport, tmp_path
):
    pages[URL_B] = SENT_B2
    # Round 1 drops the price claim (material: retracted); round 2 repeats
    # the same answer against round 1's session (quiet: every claim held).
    scripts = iter([_watch_script(_height_only_answer()), _watch_script(_height_only_answer())])
    labels_path = tmp_path / "labels.json"
    seen = []

    summary = await run_watch(
        db, blobs, clock, parent, lambda config: FakeAdapter(next(scripts)),
        fetch_transport=transport, every_s=3600, rounds=2,
        labels_path=labels_path, on_round=seen.append,
    )

    assert summary.error is None and len(summary.rounds) == 2 and seen == summary.rounds
    r1, r2 = summary.rounds
    # Chained off the newest observation, one schedule sleep between rounds.
    assert r1.parent_session_id == parent and r1.session_id.startswith(f"{parent}-refresh-")
    assert r2.parent_session_id == r1.session_id
    assert clock.mono == 3600
    # Material-only: round 1's diff carries the retraction, round 2 is quiet.
    assert r1.material and r1.status == "done"
    assert r1.report.counts["retracted"] == 1 and r1.report.counts["held"] == 1
    assert not r2.material and r2.report.unchanged
    assert r1.to_payload()["diff"]["counts"]["retracted"] == 1
    assert r2.to_payload()["diff"] is None
    assert summary.material and summary.latest_session_id == r2.session_id
    # Labels: the parent's two claims scored by round 1 (height held, price
    # overturned), round 1's one claim scored by round 2 — credence as the
    # prior run recorded it, in the shape `parsec calibrate` reads.
    # (Diff order is changes-first, so the retraction's label leads.)
    assert sorted((lb["outcome"], lb["session"]) for lb in r1.labels) == [
        ("held", parent), ("overturned", parent)
    ]
    assert [(lb["outcome"], lb["session"]) for lb in r2.labels] == [("held", r1.session_id)]
    assert summary.labels_total == 3
    pairs = cli._extract_label_pairs(json.loads(labels_path.read_text()))
    assert sorted(label for _, label in pairs) == [0, 1, 1]
    assert all(0 < credence < 1 for credence, _ in pairs)
    # Every round is a first-class recording: the chain's tail replays.
    outcome = await run_replay(db, blobs, clock, r2.session_id)
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match


async def test_single_round_without_schedule_never_sleeps(
    parent, db, blobs, clock, pages, transport, tmp_path
):
    pages[URL_B] = SENT_B2
    summary = await run_watch(
        db, blobs, clock, parent, lambda config: FakeAdapter(_watch_script(_height_only_answer())),
        fetch_transport=transport,
    )
    assert len(summary.rounds) == 1 and clock.mono == 0
    assert summary.labels_path is None and summary.labels_total == 0
    with pytest.raises(KeyError):
        await run_watch(db, blobs, clock, "s-missing", lambda config: FakeAdapter([]))


async def test_cli_watch_json_lines_and_exit_codes(
    parent, db, blobs, clock, pages, transport, tmp_path, capsys
):
    pages[URL_B] = SENT_B2
    db.close()
    labels_path = tmp_path / "labels.json"
    prev = (cli.adapter_factory, cli.fetch_transport)
    cli.adapter_factory = lambda config: FakeAdapter(_watch_script(_height_only_answer()))
    cli.fetch_transport = transport
    try:
        code = await asyncio.to_thread(
            cli.main,
            ["watch", parent, "--data-dir", str(tmp_path), "--json", "--rounds", "1",
             "--labels", str(labels_path)],
        )
    finally:
        cli.adapter_factory, cli.fetch_transport = prev
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [line["type"] for line in lines] == ["round", "summary"]
    rnd, summary = lines
    assert rnd["material"] and rnd["diff"]["counts"]["retracted"] == 1
    assert sorted(lb["label"] for lb in rnd["labels"]) == [0, 1]
    assert summary["rounds"] == 1 and summary["material_rounds"] == 1
    assert summary["labels_total"] == 2 and summary["labels_path"] == str(labels_path)
    assert code == cli.EXIT_PARTIAL

    # Human mode: the material round renders the diff and the label tally;
    # the chain continues from round 1's session.
    cli.adapter_factory = lambda config: FakeAdapter(_watch_script(_height_only_answer()))
    cli.fetch_transport = transport
    try:
        code = await asyncio.to_thread(
            cli.main, ["watch", rnd["session_id"], "--data-dir", str(tmp_path),
                       "--labels", str(labels_path)],
        )
    finally:
        cli.adapter_factory, cli.fetch_transport = prev
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK  # same answer again: every claim held
    assert "unchanged (1 held)" in out and "labels: +1 (1 held, 0 overturned)" in out
    assert "3 labels in" in out and "parsec calibrate" in out

    code = await asyncio.to_thread(cli.main, ["watch", "s-nope", "--data-dir", str(tmp_path)])
    assert code == cli.EXIT_USAGE
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["watch", "s", "--every", "soon"])
    assert cli.build_parser().parse_args(["watch", "s", "--every", "6h"]).every == 21600


async def test_cli_watch_halted_refresh_keeps_json_stdout_parseable(
    parent, db, tmp_path, capsys, monkeypatch
):
    """A refresh that halts stops the watch with exit 4; in --json mode the
    error rides in the summary line and nothing else touches stdout."""
    import parsec.watch as watch_mod
    from parsec.watch import WatchSummary

    async def halted(conn, blobs, clock, session_id, make_adapter, **kw):
        return WatchSummary(session_id, [], kw.get("labels_path"), 0,
                            error=f"refresh {session_id}-refresh-1 ended halted_budget")

    monkeypatch.setattr(watch_mod, "run_watch", halted)
    db.close()
    code = await asyncio.to_thread(
        cli.main, ["watch", parent, "--data-dir", str(tmp_path), "--json"]
    )
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert code == cli.EXIT_ERROR
    assert [line["type"] for line in lines] == ["summary"]
    assert lines[0]["error"].endswith("ended halted_budget") and lines[0]["rounds"] == 0

    code = await asyncio.to_thread(cli.main, ["watch", parent, "--data-dir", str(tmp_path)])
    assert code == cli.EXIT_ERROR
    assert "ended halted_budget — watch stopped" in capsys.readouterr().out


async def test_watch_rounds_are_recordings(parent, db, blobs, clock, pages, transport, tmp_path):
    pages[URL_B] = SENT_B2
    summary = await run_watch(
        db, blobs, clock, parent, lambda config: FakeAdapter(_watch_script(_height_only_answer())),
        fetch_transport=transport,
    )
    sid = summary.rounds[0].session_id
    events = EventLog(db, clock).read(sid)
    (seeded,) = [ev for ev in events if ev.event_type == EventType.REFRESH_SEEDED]
    assert [c["sq_id"] for c in seeded.payload["carried"]] == ["sq-1"]
