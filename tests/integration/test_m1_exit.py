"""Loop exit test (M1 criterion, M2 flow): one query → cited answer where
every claim traces claim→premise→span to a cached span, replayable
byte-identically — driven through the real CLI entrypoint with a scripted
adapter and a mock HTTP transport (no network, no keys).
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
from parsec.replay import run_replay
from parsec.retrieval.extract import extract_text
from parsec.retrieval.span_indexer import index_spans
from parsec.store.blobs import BlobStore
from parsec.store.event_log import EventLog

QUERY = "at what temperature does water boil"

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


def page_span_ids(page: bytes) -> list[str]:
    """Compute span IDs exactly as the fetch pipeline will."""
    text, _, _ = extract_text(page, "text/html")
    h = ids.doc_hash(page)
    return [ids.span_id(h, s, e) for s, e in index_spans(text)]


def premise_id(text: str, span_ref: str) -> str:
    """Compute the premise node ID exactly as record_premises will."""
    return ids.node_id(
        "Premise", {"text": text, "span_refs": [span_ref], "claim_class": "stable"}
    )


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
                QUERY: [
                    {"title": "Boiling Point", "url": URL_A, "snippet": "100C at sea level"},
                    {"title": "Altitude Effects", "url": URL_B, "snippet": "lower at altitude"},
                ]
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
    answer = (
        "Here is what the sources say. [narrative]\n"
        f"Water boils at 100 degrees Celsius at sea level. [{p_a}] "
        f"At extreme altitude it boils much cooler, about 70 degrees Celsius on Everest. [{p_b}]"
    )
    responses = [
        scripted_response(
            [{"type": "tool_use", "id": "tu_search", "name": "search_broad", "input": {"query": QUERY, "k": 5}}],
            stop_reason="tool_use",
            index=0,
        ),
        scripted_response(
            [
                {"type": "tool_use", "id": "tu_fetch_a", "name": "fetch", "input": {"url": URL_A}},
                {"type": "tool_use", "id": "tu_fetch_b", "name": "fetch", "input": {"url": URL_B}},
            ],
            stop_reason="tool_use",
            index=1,
        ),
        scripted_response(
            [
                {
                    "type": "tool_use",
                    "id": "tu_rec",
                    "name": "record_premises",
                    "input": {
                        "premises": [
                            {"text": PREMISE_A_TEXT, "span_refs": [span_a]},
                            {"text": PREMISE_B_TEXT, "span_refs": [span_b]},
                        ]
                    },
                }
            ],
            stop_reason="tool_use",
            index=2,
        ),
        scripted_response(
            [{"type": "text", "text": "Recorded two premises on boiling points."}],
            stop_reason="end_turn",
            index=3,
        ),
        # writer phase
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=4),
        # spare for a repair round in negative scenarios: restate the same answer
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=5),
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


def test_loop_exit(tmp_path, transport, fixtures_path, scripted_adapter, capsys):
    data_dir = tmp_path / "data"
    session_id = "loop-exit-session"

    exit_code = run_ask(data_dir, fixtures_path, session_id)
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "done"
    assert out["claims_total"] == 2
    assert out["unresolved"] == []
    assert out["violations"] == []

    conn = open_db(data_dir / "parsec.db")
    blobs = BlobStore(data_dir / "blobs")

    # Full chain: claim -> premise -> span, span verbatim in cached doc.
    claims = conn.execute(
        "SELECT payload_json FROM nodes WHERE session_id=? AND tier=4", (session_id,)
    ).fetchall()
    assert len(claims) == 2
    premise_refs = [r for c in claims for r in json.loads(c["payload_json"])["premise_refs"]]
    assert len(premise_refs) == 2

    span_refs = []
    for pref in premise_refs:
        prow = conn.execute(
            "SELECT payload_json FROM nodes WHERE node_id=? AND session_id=?", (pref, session_id)
        ).fetchone()
        assert prow is not None
        span_refs += json.loads(prow["payload_json"])["span_refs"]
    assert len(set(r.split("#")[0] for r in span_refs)) == 2  # two distinct docs

    for ref in span_refs:
        span = conn.execute("SELECT * FROM spans WHERE span_id=?", (ref,)).fetchone()
        doc = conn.execute("SELECT * FROM documents WHERE doc_hash=?", (span["doc_hash"],)).fetchone()
        assert blobs.exists(doc["raw_blob"]) and blobs.exists(doc["text_blob"])
        text = blobs.get_text(doc["text_blob"])
        assert text[span["char_start"]:span["char_end"]] == span["text"]
        assert conn.execute(
            "SELECT 1 FROM cache_index WHERE doc_hash=?", (span["doc_hash"],)
        ).fetchone() is not None

    edge_types = {
        row["edge_type"]
        for row in conn.execute("SELECT edge_type FROM edges WHERE session_id=?", (session_id,))
    }
    assert edge_types == {"extracts", "aggregates"}

    # --- replay against the frozen corpus: byte-identical, zero HTTP calls ---
    calls_before = transport["calls"]
    outcome = asyncio.run(run_replay(conn, blobs, RealClock(), session_id))
    assert outcome.result.status == "done"
    assert transport["calls"] == calls_before
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match

    event_log = EventLog(conn, RealClock())
    assert event_log.projection(session_id) == event_log.projection(outcome.result.session_id)
