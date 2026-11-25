"""The frozen recordings must replay byte-identically under every build.

Each directory under tests/fixtures/recordings/ holds sessions recorded by a
past release and exported as a text corpus: four by the code at commit
366f1f6 (0.1.0, before the remediation work) and one by 0.2.0. Replaying
them here is the oracle for "we did not break old recordings": the answer
bytes and the per-stream event projection must both match what that
release produced.

A same-build record-then-replay test cannot show this; these can, because the
recording side never changes. If one of these fails, a behaviour that older
recordings depend on changed in place instead of behind a behaviour version
(see parsec.config.LEGACY_BEHAVIOUR). Fix the code. Never re-record.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from parsec.config import BEHAVIOUR_FIELDS, RealClock
from parsec.db.connection import open_db
from parsec.replay import run_replay
from parsec.store.blobs import BlobStore
from parsec.store.portable import export_corpus, import_corpus
from parsec.verify.structural import verify_session

RECORDINGS = Path(__file__).parent.parent / "fixtures" / "recordings"
NAMES = sorted(p.parent.name for p in RECORDINGS.glob("*/recording.json"))


def _meta(name: str) -> dict:
    return json.loads((RECORDINGS / name / "recording.json").read_text(encoding="utf-8"))


@pytest.fixture
def restore(tmp_path, monkeypatch):
    """Rebuild a recording in a scratch directory and stand in it: recorded
    configs name their search fixtures by relative path."""

    def _restore(name: str) -> Path:
        data_dir = tmp_path / "data"
        import_corpus(RECORDINGS / name / "corpus", data_dir)
        for source, destination in _meta(name)["aux"].items():
            target = tmp_path / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(RECORDINGS / name / "aux" / source, target)
        monkeypatch.chdir(tmp_path)
        return data_dir

    return _restore


V0_1_0 = [name for name in NAMES if _meta(name)["parsec_version"] == "0.1.0"]


def test_the_expected_recordings_are_present():
    assert NAMES == ["demo", "full", "parallel", "refresh", "v020"]
    assert V0_1_0 == ["demo", "full", "parallel", "refresh"]


@pytest.mark.parametrize("name", NAMES)
def test_frozen_recording_replays_byte_identically(name, restore):
    data_dir = restore(name)
    conn = open_db(data_dir / "parsec.db")
    blobs = BlobStore(data_dir / "blobs")
    meta = _meta(name)
    for session_id in meta["sessions"]:
        outcome = asyncio.run(run_replay(conn, blobs, RealClock(), session_id))
        assert outcome.answers_match, f"{name}/{session_id}: answer bytes differ"
        assert outcome.projections_match, f"{name}/{session_id}: {outcome.first_divergence}"
        assert outcome.result.status == meta["status"][session_id]


@pytest.mark.parametrize("name", NAMES)
def test_frozen_recording_still_verifies(name, restore):
    data_dir = restore(name)
    conn = open_db(data_dir / "parsec.db")
    blobs = BlobStore(data_dir / "blobs")
    for session_id in _meta(name)["sessions"]:
        report = verify_session(conn, blobs, session_id)
        assert report.ok, [f"{v.check}: {v.detail}" for v in report.violations]


@pytest.mark.parametrize("name", V0_1_0)
def test_frozen_recordings_predate_the_behaviour_registry(name, restore):
    """Guards the fixtures themselves: a recording made with newer code would
    carry behaviour keys in its stored config, and would no longer be a test
    of anything the old code did."""
    data_dir = restore(name)
    conn = open_db(data_dir / "parsec.db")
    for session_id in _meta(name)["sessions"]:
        stored = json.loads(
            conn.execute(
                "SELECT config_json FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()["config_json"]
        )
        assert not set(BEHAVIOUR_FIELDS) & set(stored)
        assert stored["parsec_version"] == "0.1.0"


@pytest.mark.parametrize("name", NAMES)
def test_committed_corpus_is_in_canonical_export_form(name, restore, tmp_path):
    """Import then export must reproduce the committed files exactly, so a
    hand-edited or stale corpus cannot sit in the repository unnoticed."""
    data_dir = restore(name)
    again = tmp_path / "again"
    export_corpus(data_dir, again)

    def files(root: Path) -> dict[str, bytes]:
        return {
            str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
        }

    assert files(again) == files(RECORDINGS / name / "corpus")


def test_full_recording_covers_the_paths_the_remediation_changes(restore):
    """The oracle is only as good as what it exercises: pin that the `full`
    recording really contains each path a correctness fix will touch."""
    data_dir = restore("full")
    conn = open_db(data_dir / "parsec.db")

    def count(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    def events(event_type: str, stream: str | None = None) -> int:
        clause = f" AND stream_id='{stream}'" if stream else ""
        return count(f"SELECT COUNT(*) FROM events WHERE event_type='{event_type}'{clause}")

    # writer: draft, repair after the uncited sentence, rewrite after gap-fill
    assert events("llm_request", "orchestrator") == 4
    assert events("gap_fill_started") == 1
    assert events("llm_request", "sq-gap-1") > 0
    edge_types = {r[0] for r in conn.execute("SELECT DISTINCT edge_type FROM edges")}
    assert edge_types == {"extracts", "deduces", "temporal", "contradicts", "aggregates"}
    # a transform_note premise (the WS3 bypass) and a recorded conflict
    payloads = [r[0] for r in conn.execute("SELECT payload_json FROM edges")]
    assert any("transform_note" in p for p in payloads)
    assert any('"note"' in p and "transform_note" not in p for p in payloads)
    # near-duplicates attached as corroboration: two premises with two spans
    assert count(
        "SELECT COUNT(*) FROM (SELECT src_node_id FROM edges WHERE edge_type='extracts'"
        " GROUP BY src_node_id HAVING COUNT(*) = 2)"
    ) == 2


def test_parallel_recording_joined_out_of_dispatch_order(restore):
    data_dir = restore("parallel")
    conn = open_db(data_dir / "parsec.db")
    joined = [
        json.loads(r[0])["sq_id"]
        for r in conn.execute(
            "SELECT payload_json FROM events WHERE event_type='subagent_joined' ORDER BY idx"
        )
    ]
    assert joined == ["sq-3", "sq-2", "sq-1"]


def test_rescoring_an_old_recording_reports_its_transform_note_as_unverified(restore, capsys):
    """The `full` recording holds a premise accepted on a transform_note.
    Verified as recorded it is clean; under today's checks the premise is
    reported for what it is — a derivation nobody recomputed — as an
    advisory, not as corruption, and the recording itself is not touched."""
    import parsec.cli as cli

    data_dir = restore("full")

    def verify(*extra: str) -> tuple[int, dict]:
        code = cli.main(["verify", "frozen-full", "--data-dir", str(data_dir), "--json", *extra])
        return code, json.loads(capsys.readouterr().out)

    code, recorded = verify()
    assert code == 0
    assert not [a for a in recorded["advisories"] if a["check"] == "unverified-derivation"]

    code, rescored = verify("--rescore-with", "current")
    assert code == 0 and rescored["violations"] == []
    derivations = [a for a in rescored["advisories"] if a["check"] == "unverified-derivation"]
    assert len(derivations) == 1
    assert "transform note" in derivations[0]["detail"] and "'13.6'" in derivations[0]["detail"]


def test_the_0_2_0_recording_carries_its_releases_behaviour_versions(restore):
    """Recorded by 0.2.0 with released (integer) versions — never -dev ones,
    which carry no replay guarantee."""
    data_dir = restore("v020")
    conn = open_db(data_dir / "parsec.db")
    stored = json.loads(
        conn.execute("SELECT config_json FROM sessions WHERE session_id='frozen-v020'").fetchone()[0]
    )
    meta = _meta("v020")
    assert stored["parsec_version"] == "0.2.0" == meta["parsec_version"]
    assert {key: stored[key] for key in meta["behaviour"]} == meta["behaviour"]
    assert all(isinstance(value, int) for value in meta["behaviour"].values())
    assert stored["as_of_ts"]


def test_the_0_2_0_recording_covers_what_0_2_0_added(restore):
    data_dir = restore("v020")
    conn = open_db(data_dir / "parsec.db")
    blobs = BlobStore(data_dir / "blobs")

    def events(event_type: str, stream: str | None = None) -> list[dict]:
        clause = "AND stream_id=?" if stream else ""
        args = (event_type, stream) if stream else (event_type,)
        return [
            json.loads(r[0])
            for r in conn.execute(
                f"SELECT payload_json FROM events WHERE event_type=? {clause} ORDER BY idx", args
            )
        ]

    # WS0: draft, then one repair round after the sentence check
    assert len(events("llm_request", "orchestrator")) == 3
    # WS3: a computes edge, and a first report that was rejected (4 calls, not 3)
    assert {r[0] for r in conn.execute("SELECT DISTINCT edge_type FROM edges")} >= {"computes"}
    assert len(events("llm_request", "sq-1")) == 4
    statuses = [
        json.loads(r[0]).get("status") for r in conn.execute("SELECT payload_json FROM nodes WHERE tier=2")
    ]
    assert statuses == ["verified"]
    # WS2: target times in the plan, a dated edge in the journal, a premise about 2019
    planned = events("subquestions_planned")[0]["subquestions"]
    assert [s["target_time"] for s in planned] == ["now", "2019-01-01/2019-12-31"]
    dated = [e["payload"] for e in events("edge_added") if "payload" in e]
    assert {"observed_at": "2019-01-01/2019-12-31", "date_source": "as_of"} in dated
    # and it passes verification under the checks it was recorded with
    report = verify_session(conn, blobs, "frozen-v020", sentence_check=True, derivations=2, recency=2)
    assert report.ok, [f"{v.check}: {v.detail}" for v in report.violations]
