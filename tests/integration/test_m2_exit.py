"""M2 exit test (§11): corrupt a span → the dependent claim is mechanically
flagged by stage-1 structural verification, with no model involved.
"""

from __future__ import annotations

import json

import pytest

import parsec.cli as cli
from parsec.db.connection import open_db
from tests.integration.test_m1_exit import (  # shared scripted session
    PAGE_A,
    fixtures_path,
    page_span_ids,
    run_ask,
    scripted_adapter,
    transport,
)


def _verify(data_dir, session_id, capsys):
    code = cli.main(["verify", session_id, "--data-dir", str(data_dir), "--json"])
    return code, json.loads(capsys.readouterr().out)


def test_corrupt_span_flags_dependent_claim(tmp_path, transport, fixtures_path, scripted_adapter, capsys):
    data_dir = tmp_path / "data"
    session_id = "m2-exit-session"

    assert run_ask(data_dir, fixtures_path, session_id) == 0
    capsys.readouterr()

    # Clean corpus verifies clean.
    code, report = _verify(data_dir, session_id, capsys)
    assert code == 0
    assert report["ok"] is True
    assert report["checked"] == {"claims": 2, "premises": 2, "spans": 2}

    # Corrupt the cited span from page A after the fact.
    conn = open_db(data_dir / "parsec.db")
    victim = page_span_ids(PAGE_A)[0]
    conn.execute("UPDATE spans SET text='TAMPERED EVIDENCE' WHERE span_id=?", (victim,))
    conn.close()

    code, report = _verify(data_dir, session_id, capsys)
    assert code == 3
    assert report["ok"] is False
    checks = {v["check"] for v in report["violations"]}
    assert "corpus-integrity" in checks
    # The claim that rests on the corrupted span is itself flagged, by ID,
    # with no model involved.
    dependent = [v for v in report["violations"] if v["check"] == "dependent-claim"]
    assert len(dependent) == 1
    assert dependent[0]["subject"].startswith("reportclaim:")
    assert "100 degrees Celsius" in dependent[0]["detail"]


def test_deleted_document_blob_flagged(tmp_path, transport, fixtures_path, scripted_adapter, capsys):
    data_dir = tmp_path / "data"
    session_id = "m2-blob-session"

    assert run_ask(data_dir, fixtures_path, session_id) == 0
    capsys.readouterr()

    # Remove page A's raw blob from the content-addressed store.
    conn = open_db(data_dir / "parsec.db")
    from parsec import ids as _ids

    doc_hash = _ids.doc_hash(PAGE_A)
    conn.close()
    blob_path = data_dir / "blobs" / doc_hash[:2] / doc_hash
    blob_path.unlink()

    code, report = _verify(data_dir, session_id, capsys)
    assert code == 3
    assert any(
        v["check"] == "corpus-integrity" and "blobs missing" in v["detail"]
        for v in report["violations"]
    )
    assert any(v["check"] == "dependent-claim" for v in report["violations"])
