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

import httpx
import pytest

import parsec.cli as cli
from parsec import ids
from parsec.config import RealClock
from parsec.db.connection import open_db
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.models.events import EventType
from parsec.replay import run_replay
from parsec.retrieval.extract import extract_text
from parsec.retrieval.span_indexer import index_spans
from parsec.store.blobs import BlobStore
from parsec.store.event_log import EventLog

QUERY = "what temperature does water boil at sea level, and what about on mount everest"
SQ1 = "boiling point at sea level"
SQ2 = "boiling point on mount everest"

# "(212 degrees Fahrenheit)" sits inside the first span's 200-char preview:
# it marks raw-document content that subagent contexts DO contain and
# orchestrator contexts must NOT.
RAW_DOC_SENTINEL = "212 degrees Fahrenheit"

PAGE_A = (
    "<html><head><title>Boiling Point</title></head><body>"
    "<p>Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard "
    "atmospheric pressure of one atmosphere, which is the pressure found at sea level. "
    "This has been the accepted reference value in thermodynamics for over a century and "
    "is used to calibrate thermometers worldwide. The Celsius scale itself was originally "
    "defined with the boiling point of water fixed at one hundred degrees exactly.</p>"
    "</body></html>"
).encode()

PAGE_B = (
    "<html><head><title>Altitude Effects</title></head><body>"
    "<p>At higher altitudes the boiling point of water decreases because atmospheric "
    "pressure is lower. In Denver, Colorado, at about 1600 meters of elevation, water "
    "boils at roughly 95 degrees Celsius. On the summit of Mount Everest water boils at "
    "only about 70 degrees Celsius, which makes cooking food by boiling significantly "
    "slower and less effective at extreme elevations.</p>"
    "</body></html>"
).encode()

URL_A = "https://example.test/boiling"
URL_B = "https://example.test/altitude"

PREMISE_A_TEXT = "Water boils at 100 degrees Celsius at standard atmospheric pressure at sea level."
PREMISE_B_TEXT = "On the summit of Mount Everest water boils at about 70 degrees Celsius."
FINDING_A_TEXT = "The sea-level boiling point of water is 100 degrees Celsius."


def page_span_ids(page: bytes) -> list[str]:
    """Compute span IDs exactly as the fetch pipeline will."""
    text, _, _ = extract_text(page, "text/html")
    h = ids.doc_hash(page)
    return [ids.span_id(h, s, e) for s, e in index_spans(text)]


def premise_id(text: str, span_ref: str) -> str:
    return ids.node_id("Premise", {"text": text, "span_refs": [span_ref], "claim_class": "stable"})


def finding_id(text: str, premise_ids: list[str], edge_type: str = "deduces") -> str:
    return ids.node_id("Finding", {"text": text, "premise_ids": premise_ids, "edge_type": edge_type})


@pytest.fixture
def transport(monkeypatch):
    pages = {URL_A: PAGE_A, URL_B: PAGE_B}
    counter = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["calls"] += 1
        page = pages.get(str(request.url).rstrip("/")) or pages.get(str(request.url))
        if page is None:
            return httpx.Response(404, content=b"not found")
        return httpx.Response(200, content=page, headers={"content-type": "text/html"})

    monkeypatch.setattr(cli, "fetch_transport", httpx.MockTransport(handler))
    return counter


@pytest.fixture
def fixtures_path(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps(
            {
                SQ1: [{"title": "Boiling Point", "url": URL_A, "snippet": "100C at sea level"}],
                SQ2: [{"title": "Altitude Effects", "url": URL_B, "snippet": "lower at altitude"}],
            }
        )
    )
    return path


@pytest.fixture
def scripted_adapter(monkeypatch):
    span_a = page_span_ids(PAGE_A)[0]
    span_b = page_span_ids(PAGE_B)[0]
    p_a = premise_id(PREMISE_A_TEXT, span_a)
    p_b = premise_id(PREMISE_B_TEXT, span_b)
    f_a = finding_id(FINDING_A_TEXT, [p_a])
    answer = (
        "Here is what the sources say. [narrative]\n"
        f"Water boils at 100 degrees Celsius at sea level. [{f_a}] "
        f"On the summit of Mount Everest it boils at about 70 degrees Celsius. [{p_b}]"
    )
    responses = [
        # orchestrator: decomposition
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
              "input": {"subquestions": [SQ1, SQ2]}}],
            stop_reason="tool_use", index=0,
        ),
        # subagent sq-1
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
              "input": {
                  "status": "answered",
                  "findings": [{"text": FINDING_A_TEXT, "premise_ids": [p_a], "edge_type": "deduces"}],
                  "summary": "Sea-level boiling point established from one source.",
              }}],
            stop_reason="tool_use", index=4,
        ),
        # subagent sq-2
        scripted_response(
            [{"type": "tool_use", "id": "tu_s2", "name": "search_broad", "input": {"query": SQ2, "k": 5}}],
            stop_reason="tool_use", index=5,
        ),
        scripted_response(
            [{"type": "tool_use", "id": "tu_f2", "name": "fetch", "input": {"url": URL_B}}],
            stop_reason="tool_use", index=6,
        ),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r2", "name": "record_premises",
              "input": {"premises": [{"text": PREMISE_B_TEXT, "span_refs": [span_b]}]}}],
            stop_reason="tool_use", index=7,
        ),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub2", "name": "submit_report",
              "input": {"status": "answered"}}],
            stop_reason="tool_use", index=8,
        ),
        # orchestrator: writer
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=9),
        # spare for a repair round in negative scenarios
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=10),
    ]
    monkeypatch.setattr(cli, "adapter_factory", lambda config: FakeAdapter(responses))
    return answer


def run_ask(data_dir, fixtures_path, session_id):
    return cli.main(
        [
            "ask",
            QUERY,
            "--session-id", session_id,
            "--adapter", "fake",
            "--model", "fake-model",
            "--cache-mode", "record",
            "--data-dir", str(data_dir),
            "--search-fixtures", str(fixtures_path),
            "--json",
        ]
    )


def test_orchestrator_exit(tmp_path, transport, fixtures_path, scripted_adapter, capsys):
    data_dir = tmp_path / "data"
    session_id = "m3-exit-session"

    exit_code = run_ask(data_dir, fixtures_path, session_id)
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
    calls_before = transport["calls"]
    outcome = asyncio.run(run_replay(conn, blobs, RealClock(), session_id))
    assert outcome.result.status == "done"
    assert transport["calls"] == calls_before
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match
    assert outcome.result.coverage == {"sq-1": "answered", "sq-2": "answered"}


def test_blocked_subquestion_is_explicit(tmp_path, transport, fixtures_path, monkeypatch, capsys):
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
    exit_code = run_ask(data_dir, fixtures_path, "m3-blocked-session")
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
