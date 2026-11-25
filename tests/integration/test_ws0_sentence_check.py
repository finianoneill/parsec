"""WS0 end to end, on the offline demo with a writer that gets a number wrong:
the citation check catches it, repair fixes it or the sentence is marked, and
the session — repair round, marker and all — still replays byte-identically."""

from __future__ import annotations

import asyncio
import json

import pytest

import parsec.cli as cli
from parsec import config as config_module
from parsec import demo
from parsec.config import RealClock
from parsec.db.connection import open_db
from parsec.gateway.fake_adapter import scripted_response
from parsec.loop.citations import UNVERIFIED_TAG
from parsec.replay import run_replay
from parsec.store.blobs import BlobStore


@pytest.fixture
def run_demo(tmp_path, monkeypatch, capsys):
    """Run the demo with the writer's drafts replaced. Each draft is the real
    answer with its figure rewritten; `drafts` lists what the writer says on
    its first call, then on each repair call."""
    data_dir = tmp_path / "data"
    real_factory = demo.demo_adapter_factory

    def _run(drafts: list[str]) -> dict:
        def factory(config):
            adapter = real_factory(config)
            good = adapter._responses[-1].text
            adapter._responses[-1:] = [
                scripted_response(
                    [{"type": "text", "text": good.replace("3.26", figure)}], index=90 + i
                )
                for i, figure in enumerate(drafts)
            ]
            return adapter

        monkeypatch.setattr(demo, "demo_adapter_factory", factory)
        code = cli.main(["demo", "--data-dir", str(data_dir)])
        capsys.readouterr()
        conn = open_db(data_dir / "parsec.db")
        row = conn.execute(
            "SELECT session_id, status, answer_blob FROM sessions WHERE parent_session_id IS NULL"
        ).fetchone()
        blobs = BlobStore(data_dir / "blobs")
        writer_calls = conn.execute(
            "SELECT COUNT(*) FROM events WHERE session_id=? AND event_type='llm_request'"
            " AND stream_id='orchestrator'",
            (row["session_id"],),
        ).fetchone()[0] - 1  # minus the decomposer
        return {
            "code": code, "session_id": row["session_id"], "status": row["status"],
            "answer": blobs.get_text(row["answer_blob"]), "writer_calls": writer_calls,
            "conn": conn, "blobs": blobs, "data_dir": data_dir,
        }

    return _run


def _verify(run: dict, capsys, *extra: str) -> tuple[int, list[dict]]:
    code = cli.main(
        ["verify", run["session_id"], "--data-dir", str(run["data_dir"]), "--json", *extra]
    )
    return code, json.loads(capsys.readouterr().out)["violations"]


def test_repair_fixes_a_wrong_number(run_demo, capsys):
    run = run_demo(["32.6", "3.26"])  # wrong draft, then a correct repair
    assert run["status"] == "done" and run["code"] == cli.EXIT_OK
    assert run["writer_calls"] == 2
    assert "3.26 light-years" in run["answer"] and "32.6" not in run["answer"]
    assert UNVERIFIED_TAG not in run["answer"]
    assert "Unverified sentences" not in run["answer"]
    assert _verify(run, capsys) == (0, [])
    # the repair prompt named the figure, so the model could find it
    repair_request = run["conn"].execute(
        "SELECT payload_json FROM events WHERE session_id=? AND event_type='llm_request'"
        " AND stream_id='orchestrator' ORDER BY stream_idx DESC LIMIT 1",
        (run["session_id"],),
    ).fetchone()
    body = run["blobs"].get_text(json.loads(repair_request["payload_json"])["request_blob"])
    assert "'32.6'" in body and "does not appear in the cited evidence" in body


def test_unrepaired_sentence_is_marked_and_the_run_is_partial(run_demo, capsys):
    run = run_demo(["32.6", "32.6"])  # repair does not fix it
    assert run["status"] == "partial" and run["code"] == cli.EXIT_PARTIAL
    answer, appendix = run["answer"].split("--- end of answer ---")
    lines = answer.splitlines()
    assert "32.6 light-years" in lines[0] and lines[0].endswith(UNVERIFIED_TAG)
    assert UNVERIFIED_TAG not in lines[1]  # the sentence that was right is left alone
    assert "Unverified sentences" in appendix and "'32.6'" in appendix

    code, violations = _verify(run, capsys)
    assert code != 0
    assert [v["check"] for v in violations] == ["sentence-support"]
    assert "'32.6'" in violations[0]["detail"]


@pytest.mark.parametrize("drafts", [["32.6", "3.26"], ["32.6", "32.6"]])
def test_a_session_with_repair_and_markers_replays_byte_identically(run_demo, drafts):
    run = run_demo(drafts)
    outcome = asyncio.run(run_replay(run["conn"], run["blobs"], RealClock(), run["session_id"]))
    assert outcome.verified, outcome.first_divergence


def test_an_older_session_is_verified_as_recorded_unless_rescored(run_demo, capsys, monkeypatch):
    """A session recorded before the check keeps the verdict it was given;
    --rescore-with current applies today's check without touching it."""
    with monkeypatch.context() as recording_era:
        recording_era.setitem(config_module.CURRENT_BEHAVIOUR, "sentence_check", 0)
        run = run_demo(["32.6"])  # no repair call: the old check sees nothing wrong
    assert run["status"] == "done" and run["writer_calls"] == 1
    assert UNVERIFIED_TAG not in run["answer"]

    before = run["conn"].execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert _verify(run, capsys) == (0, [])
    code, violations = _verify(run, capsys, "--rescore-with", "current")
    assert code != 0 and "'32.6'" in violations[0]["detail"]
    assert run["conn"].execute("SELECT COUNT(*) FROM events").fetchone()[0] == before
