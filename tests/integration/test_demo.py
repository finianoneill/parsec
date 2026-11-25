"""The built-in offline demo: a full recorded run with no keys and no network,
replayable byte-identically, passing verification with no advisories."""

from __future__ import annotations

import asyncio

import parsec.cli as cli
from parsec.config import RealClock
from parsec.db.connection import open_db
from parsec.replay import run_replay
from parsec.store.blobs import BlobStore


def test_demo_runs_offline_and_replays(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    data_dir = tmp_path / "data"

    exit_code = cli.main(["demo", "--data-dir", str(data_dir)])
    assert exit_code == 0
    # the seams are restored, not left pointing at the demo script
    assert cli.adapter_factory is None
    assert cli.fetch_transport is None

    conn = open_db(data_dir / "parsec.db")
    session_id = conn.execute("SELECT session_id FROM sessions").fetchone()["session_id"]

    # a real recording: replays byte-identically offline
    blobs = BlobStore(data_dir / "blobs")
    outcome = asyncio.run(run_replay(conn, blobs, RealClock(), session_id))
    assert outcome.result.status == "done"
    assert outcome.verified

    # robots-blocked demo URL surfaced as a typed outcome, not a fetch
    row = conn.execute(
        "SELECT meta_json FROM documents WHERE url LIKE '%/blocked/%'"
    ).fetchone()
    assert row is not None


def test_demo_verification_is_clean(tmp_path, capsys):
    data_dir = tmp_path / "data"
    assert cli.main(["demo", "--data-dir", str(data_dir)]) == 0
    conn = open_db(data_dir / "parsec.db")
    session_id = conn.execute("SELECT session_id FROM sessions").fetchone()["session_id"]
    capsys.readouterr()

    assert cli.main(["verify", session_id, "--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "verification passed" in out
    assert "advisory" not in out


def test_verify_never_rewrites_recorded_credences(tmp_path, capsys):
    """`parsec verify` inspects a recording; the credences the run persisted
    (which calibration labels read back) must survive it untouched."""
    data_dir = tmp_path / "data"
    assert cli.main(["demo", "--data-dir", str(data_dir)]) == 0
    conn = open_db(data_dir / "parsec.db")
    session_id = conn.execute("SELECT session_id FROM sessions").fetchone()["session_id"]
    # A sentinel no credence model would produce: any rewrite is visible.
    conn.execute("UPDATE nodes SET credence=0.123 WHERE session_id=?", (session_id,))
    conn.close()
    capsys.readouterr()

    assert cli.main(["verify", session_id, "--data-dir", str(data_dir)]) == 0
    assert cli.main(["verify", session_id, "--data-dir", str(data_dir), "--json"]) == 0

    conn = open_db(data_dir / "parsec.db")
    values = {
        row["credence"]
        for row in conn.execute("SELECT credence FROM nodes WHERE session_id=?", (session_id,))
    }
    assert values == {0.123}
