"""M7 exit tests (v2 plan): a live-provider query end-to-end, then replayed
byte-identically offline; robots/402-gated URLs surface as typed outcomes
that also replay; search_within works over the fetched corpus.

"Live" here = a mock SearXNG instance + mock web served through the CLI's
transport seam — the real network path, no real network.
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

QUERY = "at what temperature does water boil"
SEARX_URL = "http://searx.local"

PAGE = (
    "<html><head><title>Boiling Point Reference</title></head><body>"
    "<article><p>Water boils at 100 degrees Celsius at standard atmospheric pressure, "
    "which is the pressure found at sea level. This value has served as a fixed "
    "calibration point for thermometers for well over a century and remains the "
    "reference taught in every introductory chemistry curriculum worldwide.</p>"
    "<p>At higher altitudes the boiling point decreases measurably; on the summit of "
    "Mount Everest water boils at only about 70 degrees Celsius, which slows cooking "
    "and is a standard example of pressure dependence in phase transitions.</p></article>"
    "</body></html>"
).encode()

PAGE_URL = "https://example.test/boiling"
BLOCKED_URL = "https://example.test/blocked/secret"
PAYWALL_URL = "https://paywall.test/article"

ROBOTS = "User-agent: *\nDisallow: /blocked/\nLicense: https://example.test/license.xml\n"

PREMISE_TEXT = "Water boils at 100 degrees Celsius at standard atmospheric pressure at sea level."


def page_span_ids() -> list[str]:
    text, _, _ = extract_text(PAGE, "text/html")
    h = ids.doc_hash(PAGE)
    return [ids.span_id(h, s, e) for s, e in index_spans(text)]


def premise_id(span_ref: str) -> str:
    return ids.node_id(
        "Premise", {"text": PREMISE_TEXT, "span_refs": [span_ref], "claim_class": "stable"}
    )


@pytest.fixture
def transport(monkeypatch):
    counter = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["calls"] += 1
        url = str(request.url)
        if url.startswith(f"{SEARX_URL}/search"):
            return httpx.Response(
                200,
                json={"results": [
                    {"title": "Boiling Point Reference", "url": PAGE_URL, "content": "100C at sea level"},
                ]},
            )
        if url == "https://example.test/robots.txt":
            return httpx.Response(200, text=ROBOTS)
        if url == "https://paywall.test/robots.txt":
            return httpx.Response(404, text="not found")
        if url.rstrip("/") == PAGE_URL:
            return httpx.Response(200, content=PAGE, headers={"content-type": "text/html"})
        if url.rstrip("/") == PAYWALL_URL:
            return httpx.Response(402, text="payment required")
        return httpx.Response(404, content=b"not found")

    monkeypatch.setattr(cli, "fetch_transport", httpx.MockTransport(handler))
    return counter


@pytest.fixture
def scripted_adapter(monkeypatch):
    span = page_span_ids()[0]
    p_id = premise_id(span)
    answer = (
        "According to the fetched reference: [narrative]\n"
        f"Water boils at 100 degrees Celsius at sea level. [{p_id}]"
    )
    responses = [
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
              "input": {"subquestions": [QUERY]}}], stop_reason="tool_use", index=0),
        scripted_response(
            [{"type": "tool_use", "id": "tu_s", "name": "search_broad",
              "input": {"query": QUERY, "k": 5}}], stop_reason="tool_use", index=1),
        scripted_response(
            [
                {"type": "tool_use", "id": "tu_f1", "name": "fetch", "input": {"url": PAGE_URL}},
                {"type": "tool_use", "id": "tu_f2", "name": "fetch", "input": {"url": BLOCKED_URL}},
                {"type": "tool_use", "id": "tu_f3", "name": "fetch", "input": {"url": PAYWALL_URL}},
            ],
            stop_reason="tool_use", index=2),
        scripted_response(
            [{"type": "tool_use", "id": "tu_w", "name": "search_within",
              "input": {"query": "boiling temperature sea level", "k": 3}}],
            stop_reason="tool_use", index=3),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r", "name": "record_premises",
              "input": {"premises": [{"text": PREMISE_TEXT, "span_refs": [span]}]}}],
            stop_reason="tool_use", index=4),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use", index=5),
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=6),
    ]
    monkeypatch.setattr(cli, "adapter_factory", lambda config: FakeAdapter(responses))
    return answer


def test_m7_exit(tmp_path, transport, scripted_adapter, capsys):
    data_dir = tmp_path / "data"
    session_id = "m7-exit-session"

    exit_code = cli.main(
        [
            "ask", QUERY,
            "--session-id", session_id,
            "--adapter", "fake",
            "--model", "fake-model",
            "--cache-mode", "record",
            "--search-provider", "searxng",
            "--searxng-url", SEARX_URL,
            "--contact", "test@example.test",
            "--max-gap-rounds", "0",
            "--data-dir", str(data_dir),
            "--json",
        ]
    )
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "done"
    assert out["claims_total"] == 1

    conn = open_db(data_dir / "parsec.db")
    blobs = BlobStore(data_dir / "blobs")

    # provider response cached under the T11 borrowed-data regime
    row = conn.execute("SELECT * FROM search_cache WHERE provider='searxng'").fetchone()
    assert row is not None and PAGE_URL in row["response_json"]

    # trafilatura path was used (no fallback note) and boilerplate-free spans exist
    doc = conn.execute("SELECT * FROM documents WHERE url=?", (PAGE_URL,)).fetchone()
    assert json.loads(doc["meta_json"]).get("extractor_version") == "2"

    # typed fetch outcomes, recorded as cached outcome documents
    events = list(open_events(conn, session_id))
    fetches = [e.payload for e in events if e.event_type == EventType.FETCH_PERFORMED]
    outcomes = {p["url"]: p["outcome"] for p in fetches}
    assert outcomes["https://example.test/boiling"] == "ok"
    assert outcomes["https://example.test/blocked/secret"] == "blocked_by_robots"
    assert outcomes["https://paywall.test/article"] == "licensed"

    # the licensed/blocked tool results told the model, with the RSL terms
    tool_results = [
        blobs.get_text(e.payload["full_blob"])
        for e in events
        if e.event_type == EventType.TOOL_RESULT and e.payload.get("full_blob")
    ]
    assert any("blocked_by_robots" in t for t in tool_results)
    assert any('"outcome":"licensed"' in t for t in tool_results)
    # example.test's RSL License: directive is surfaced on its blocked outcome
    assert any("blocked_by_robots" in t and "license.xml" in t for t in tool_results)

    # identity-honest UA reached the web
    # (searx + robots + page + paywall calls all carry it via the transport)
    # --- replay: byte-identical, ZERO live calls (search cache + doc cache + outcome docs) ---
    calls_before = transport["calls"]
    outcome = asyncio.run(run_replay(conn, blobs, RealClock(), session_id))
    assert outcome.result.status == "done"
    assert transport["calls"] == calls_before
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match


def open_events(conn, session_id):
    from parsec.store.event_log import EventLog

    return EventLog(conn, RealClock()).read(session_id)
