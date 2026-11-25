"""Portable corpus: a recorded data directory survives the trip to text and
back — row for row, blob for blob — and the rebuilt recording still replays."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import parsec.cli as cli
from parsec.config import RealClock
from parsec.db.connection import open_db
from parsec.evals.case import copy_corpus
from parsec.replay import run_replay
from parsec.store.blobs import BlobStore
from parsec.store.portable import (
    DERIVED_TABLES,
    PORTABLE_TABLES,
    PortableCorpusError,
    _user_tables,
    export_corpus,
    import_corpus,
)


@pytest.fixture
def recorded(tmp_path) -> Path:
    """A data dir holding one real recorded session (the offline demo)."""
    data_dir = tmp_path / "data"
    assert cli.main(["demo", "--data-dir", str(data_dir)]) == 0
    return data_dir


def _dump(data_dir: Path) -> dict[str, list[tuple]]:
    conn = open_db(data_dir / "parsec.db")
    try:
        return {
            table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
            for table in PORTABLE_TABLES
        }
    finally:
        conn.close()


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _session_id(data_dir: Path) -> str:
    conn = open_db(data_dir / "parsec.db")
    try:
        return conn.execute(
            "SELECT session_id FROM sessions WHERE parent_session_id IS NULL"
        ).fetchone()["session_id"]
    finally:
        conn.close()


def test_every_table_is_exported_or_declared_derived(tmp_path):
    """A table added to the schema must be a conscious decision here, never
    silently missing from committed corpora."""
    conn = open_db(tmp_path / "parsec.db")
    assert _user_tables(conn) == set(PORTABLE_TABLES) | DERIVED_TABLES


def test_round_trip_preserves_every_row_and_blob(recorded, tmp_path):
    corpus, rebuilt = tmp_path / "corpus", tmp_path / "rebuilt"
    summary = export_corpus(recorded, corpus)
    assert summary.tables["sessions"] == 1 and summary.tables["events"] > 0 and summary.blobs > 0

    assert import_corpus(corpus, rebuilt).tables == summary.tables
    assert _dump(rebuilt) == _dump(recorded)
    assert _files(rebuilt / "blobs") == _files(recorded / "blobs")


def test_export_is_deterministic_and_stable_across_a_round_trip(recorded, tmp_path):
    first, again, rebuilt, third = (tmp_path / n for n in ("c1", "c2", "rebuilt", "c3"))
    export_corpus(recorded, first)
    export_corpus(recorded, again)
    import_corpus(first, rebuilt)
    export_corpus(rebuilt, third)
    assert _files(first) == _files(again) == _files(third)


def test_rebuilt_recording_replays_byte_identically(recorded, tmp_path):
    corpus, rebuilt = tmp_path / "corpus", tmp_path / "rebuilt"
    export_corpus(recorded, corpus)
    import_corpus(corpus, rebuilt)

    session_id = _session_id(rebuilt)
    conn = open_db(rebuilt / "parsec.db")
    outcome = asyncio.run(run_replay(conn, BlobStore(rebuilt / "blobs"), RealClock(), session_id))
    assert outcome.verified, outcome.first_divergence
    conn.close()
    # ...and structural verification still finds every span intact
    assert cli.main(["verify", session_id, "--data-dir", str(rebuilt)]) == 0


def test_search_index_is_rebuilt_on_import(recorded, tmp_path):
    corpus, rebuilt = tmp_path / "corpus", tmp_path / "rebuilt"
    export_corpus(recorded, corpus)
    assert not (corpus / "spans_fts.jsonl").exists()
    import_corpus(corpus, rebuilt)

    def hits(data_dir: Path) -> list[str]:
        conn = open_db(data_dir / "parsec.db")
        try:
            return [
                row["span_id"]
                for row in conn.execute(
                    "SELECT span_id FROM spans_fts WHERE spans_fts MATCH 'parsec' ORDER BY rowid"
                )
            ]
        finally:
            conn.close()

    assert hits(rebuilt) and hits(rebuilt) == hits(recorded)


def test_json_columns_nest_only_when_lossless(recorded, tmp_path):
    """Canonical object payloads are written nested (diffable); a stored
    string that is not canonical JSON must come back byte for byte."""
    odd = '{"b": 1,  "a": 2}'  # valid JSON, not canonical
    conn = open_db(recorded / "parsec.db")
    doc_hash = conn.execute("SELECT doc_hash FROM documents").fetchone()["doc_hash"]
    conn.execute("UPDATE documents SET meta_json=? WHERE doc_hash=?", (odd, doc_hash))
    conn.close()

    corpus, rebuilt = tmp_path / "corpus", tmp_path / "rebuilt"
    export_corpus(recorded, corpus)
    lines = [json.loads(x) for x in (corpus / "documents.jsonl").read_text().splitlines()]
    by_hash = {line["doc_hash"]: line["meta_json"] for line in lines}
    assert by_hash[doc_hash] == odd                      # kept as the stored string
    assert any(isinstance(v, dict) for v in by_hash.values())  # the rest are nested
    events = [json.loads(x) for x in (corpus / "events.jsonl").read_text().splitlines()]
    assert all(isinstance(e["payload_json"], dict) for e in events)

    import_corpus(corpus, rebuilt)
    assert _dump(rebuilt) == _dump(recorded)


def test_import_refuses_an_existing_database(recorded, tmp_path):
    corpus = tmp_path / "corpus"
    export_corpus(recorded, corpus)
    with pytest.raises(PortableCorpusError, match="already exists"):
        import_corpus(corpus, recorded)


def test_import_rejects_a_tampered_blob_and_leaves_no_database(recorded, tmp_path):
    corpus, rebuilt = tmp_path / "corpus", tmp_path / "rebuilt"
    export_corpus(recorded, corpus)
    victim = next(p for p in (corpus / "blobs").glob("*/*"))
    victim.write_bytes(victim.read_bytes() + b"x")
    with pytest.raises(PortableCorpusError, match="content hash"):
        import_corpus(corpus, rebuilt)
    assert not (rebuilt / "parsec.db").exists()


def test_import_rejects_missing_rows_and_unknown_columns(recorded, tmp_path):
    corpus = tmp_path / "corpus"
    export_corpus(recorded, corpus)
    events = corpus / "events.jsonl"
    original = events.read_text()

    events.write_text("".join(original.splitlines(keepends=True)[:-1]))
    with pytest.raises(PortableCorpusError, match="rows for events"):
        import_corpus(corpus, tmp_path / "short")
    assert not (tmp_path / "short" / "parsec.db").exists()

    lines = [json.loads(x) for x in original.splitlines()]
    for line in lines:
        line["not_a_column"] = 1
    events.write_text("".join(json.dumps(line) + "\n" for line in lines))
    with pytest.raises(PortableCorpusError, match="not_a_column"):
        import_corpus(corpus, tmp_path / "extra")


def test_export_will_not_replace_a_directory_that_is_not_a_corpus(recorded, tmp_path):
    precious = tmp_path / "notes"
    precious.mkdir()
    (precious / "keep.txt").write_text("mine")
    with pytest.raises(PortableCorpusError, match="already exists"):
        export_corpus(recorded, precious)
    with pytest.raises(PortableCorpusError, match="not a parsec corpus"):
        export_corpus(recorded, precious, force=True)
    assert (precious / "keep.txt").read_text() == "mine"


def test_case_with_only_a_text_corpus_can_be_forked(recorded, tmp_path):
    case_dir = tmp_path / "case"
    export_corpus(recorded, case_dir / "corpus")
    forked = copy_corpus(case_dir, tmp_path / "work")
    assert _dump(forked) == _dump(recorded)
    with pytest.raises(FileNotFoundError):
        copy_corpus(tmp_path / "empty-case", tmp_path / "work2")


def test_cli_export_then_import(recorded, tmp_path, capsys):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    recorded.rename(case_dir / "data")
    assert cli.main(["eval", "export-case", str(case_dir)]) == cli.EXIT_OK
    assert (case_dir / "corpus" / "manifest.json").is_file()
    # refuses to clobber without --force, replaces with it
    assert cli.main(["eval", "export-case", str(case_dir)]) == cli.EXIT_USAGE
    assert cli.main(["eval", "export-case", str(case_dir), "--force"]) == cli.EXIT_OK

    before = _dump(case_dir / "data")
    assert cli.main(["eval", "import-case", str(case_dir)]) == cli.EXIT_USAGE  # data/ exists
    rebuilt = tmp_path / "rebuilt"
    assert (
        cli.main(["eval", "import-case", str(case_dir), "--data-dir", str(rebuilt)])
        == cli.EXIT_OK
    )
    assert _dump(rebuilt) == before
    assert "blobs" in capsys.readouterr().out
