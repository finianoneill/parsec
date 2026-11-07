"""Orchestrator exit tests (M1 + M3 criteria, §11), driven through the real
CLI entrypoint with a scripted adapter and a mock HTTP transport.

M1: one query → cited answer where every claim traces to a cached span,
replayable byte-identically with zero HTTP calls.
M3: multi-part question → coverage ledger fully resolved or explicitly
blocked; the orchestrator's own model calls never contain a raw document.
"""

from __future__ import annotations

import asyncio
import json

import parsec.cli as cli
from parsec.config import RealClock
from parsec.db.connection import open_db
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.models.events import EventType
from parsec.replay import run_replay
from parsec.store.blobs import BlobStore
from parsec.store.event_log import EventLog
from tests.integration.m3_corpus import (
    PAGE_A,
    PREMISE_A_TEXT,
    RAW_DOC_SENTINEL,
    SQ1,
    URL_A,
    page_span_ids,
    premise_id,
    run_ask,
)


def test_orchestrator_exit(tmp_path, m3_transport, m3_fixtures_path, m3_scripted_adapter, capsys):
    data_dir = tmp_path / "data"
    session_id = "m3-exit-session"

    exit_code = run_ask(data_dir, m3_fixtures_path, session_id)
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "done"
    assert out["claims_total"] == 2
    assert out["unresolved"] == []
    assert out["violations"] == []

    # --- M3: coverage ledger fully resolved ---
    assert out["coverage"] == {"sq-1": "answered", "sq-2": "answered"}

    conn = open_db(data_dir / "parsec.db")
    blobs = BlobStore(data_dir / "blobs")

    # Full chain exists: claim -> (finding ->) premise -> span, spans verbatim.
    edge_types = {
        row["edge_type"]
        for row in conn.execute("SELECT edge_type FROM edges WHERE session_id=?", (session_id,))
    }
    assert edge_types == {"extracts", "aggregates", "deduces"}
    tiers = {
        row["tier"]: row["n"]
        for row in conn.execute(
            "SELECT tier, COUNT(*) AS n FROM nodes WHERE session_id=? GROUP BY tier", (session_id,)
        )
    }
    assert tiers == {0: 2, 1: 2, 2: 1, 4: 2}

    for span in conn.execute(
        "SELECT s.* FROM spans s JOIN nodes n ON n.session_id=? AND n.tier=0"
        " AND json_extract(n.payload_json,'$.span_id')=s.span_id",
        (session_id,),
    ):
        doc = conn.execute("SELECT * FROM documents WHERE doc_hash=?", (span["doc_hash"],)).fetchone()
        text = blobs.get_text(doc["text_blob"])
        assert text[span["char_start"]:span["char_end"]] == span["text"]

    # --- M3: orchestrator context never contains a raw document (T6) ---
    event_log = EventLog(conn, RealClock())
    requests = []
    for ev in event_log.read(session_id):
        if ev.event_type == EventType.LLM_REQUEST:
            body = blobs.get_text(ev.payload["request_blob"])
            requests.append((ev.payload["call_index"], body))
    requests.sort()
    decomposer_body = requests[0][1]
    writer_body = requests[-1][1]
    subagent_bodies = [b for _, b in requests[1:-1]]
    assert any(RAW_DOC_SENTINEL in b for b in subagent_bodies)  # sentinel is real
    assert RAW_DOC_SENTINEL not in decomposer_body
    assert RAW_DOC_SENTINEL not in writer_body

    # Notebook distills the run.
    notebook_md = "\n".join(
        r["md_text"] for r in conn.execute(
            "SELECT md_text FROM notebook WHERE session_id=? ORDER BY entry_idx", (session_id,)
        )
    )
    assert "## Plan" in notebook_md
    assert "sq-1" in notebook_md and "sq-2" in notebook_md
    assert "Status: answered" in notebook_md or "**Status:** answered" in notebook_md

    # --- M1: replay against the frozen corpus, byte-identical, zero HTTP ---
    calls_before = m3_transport["calls"]
    outcome = asyncio.run(run_replay(conn, blobs, RealClock(), session_id))
    assert outcome.result.status == "done"
    assert m3_transport["calls"] == calls_before
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match
    assert outcome.result.coverage == {"sq-1": "answered", "sq-2": "answered"}


def test_blocked_subquestion_is_explicit(tmp_path, m3_transport, m3_fixtures_path, monkeypatch, capsys):
    """A subagent that finds nothing submits blocked with dead_ends; the
    ledger records it and the writer acknowledges the gap."""
    span_a = page_span_ids(PAGE_A)[0]
    p_a = premise_id(PREMISE_A_TEXT, span_a)
    answer = (
        f"Water boils at 100 degrees Celsius at sea level. [{p_a}] "
        "No usable sources were found for the Everest part of the question. [narrative]"
    )
    responses = [
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
              "input": {"subquestions": [SQ1, "boiling point in the mariana trench"]}}],
            stop_reason="tool_use", index=0,
        ),
        scripted_response(
            [{"type": "tool_use", "id": "tu_s1", "name": "search_broad", "input": {"query": SQ1, "k": 5}}],
            stop_reason="tool_use", index=1,
        ),
        scripted_response(
            [{"type": "tool_use", "id": "tu_f1", "name": "fetch", "input": {"url": URL_A}}],
            stop_reason="tool_use", index=2,
        ),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r1", "name": "record_premises",
              "input": {"premises": [{"text": PREMISE_A_TEXT, "span_refs": [span_a]}]}}],
            stop_reason="tool_use", index=3,
        ),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub1", "name": "submit_report",
              "input": {"status": "answered"}}],
            stop_reason="tool_use", index=4,
        ),
        # subagent sq-2: search finds nothing, reports blocked
        scripted_response(
            [{"type": "tool_use", "id": "tu_s2", "name": "search_broad",
              "input": {"query": "boiling point in the mariana trench", "k": 5}}],
            stop_reason="tool_use", index=5,
        ),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub2", "name": "submit_report",
              "input": {"status": "blocked",
                        "dead_ends": ["no search results for mariana trench boiling point"]}}],
            stop_reason="tool_use", index=6,
        ),
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=7),
    ]
    monkeypatch.setattr(cli, "adapter_factory", lambda config: FakeAdapter(responses))

    data_dir = tmp_path / "data"
    exit_code = run_ask(data_dir, m3_fixtures_path, "m3-blocked-session")
    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["status"] == "done"  # blocked-with-reason is a resolved ledger state
    assert out["coverage"] == {"sq-1": "answered", "sq-2": "blocked"}

    conn = open_db(data_dir / "parsec.db")
    row = conn.execute(
        "SELECT * FROM coverage WHERE session_id=? AND sq_id='sq-2'", ("m3-blocked-session",)
    ).fetchone()
    assert row["status"] == "blocked"
    assert "mariana trench" in row["reason"]
