"""M15.9 exit: a search_within result must not depend on documents the
stream could not have seen — including documents the SAME session fetches
later. The recording's sq-1 searches before sq-2 fetches a page that would
outrank sq-1's own; at replay every span already sits in the corpus-wide
table, and the pre-M15.9 ranking diverged at that call (the first real long
Opus run, call 3). Now the session replays byte-identically.
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

QUERY = "boiling point of water at sea level and on everest"
SQ1, SQ2 = "boiling point at sea level", "boiling point on everest"
URL_A, URL_B = "https://example.test/a", "https://example.test/b"
PAGE_A = (
    "<html><body><p>Water boils at 100 degrees Celsius at standard atmospheric pressure at sea "
    "level, the reference value used to calibrate laboratory thermometers worldwide.</p></body></html>"
).encode()
# Page B is the stronger lexical match for sq-1's search query, but sq-2 fetches it later.
PAGE_B = (
    "<html><body><p>Boiling point of water at sea level: the boiling point of water at sea level is "
    "100 degrees Celsius because sea level pressure is one atmosphere; on Everest the boiling point of "
    "water is lower.</p></body></html>"
).encode()
PREMISE_A = "Water boils at 100 degrees Celsius at standard atmospheric pressure at sea level."


def _span(page: bytes) -> str:
    text, _, _ = extract_text(page, "text/html")
    s, e = index_spans(text)[0]
    return ids.span_id(ids.doc_hash(page), s, e)


@pytest.fixture
def transport(monkeypatch):
    pages = {URL_A: PAGE_A, URL_B: PAGE_B}

    def handler(request: httpx.Request) -> httpx.Response:
        page = pages.get(str(request.url).rstrip("/"))
        if page is None:
            return httpx.Response(404, content=b"not found")
        return httpx.Response(200, content=page, headers={"content-type": "text/html"})

    monkeypatch.setattr(cli, "fetch_transport", httpx.MockTransport(handler))


@pytest.fixture
def fixtures_path(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps({
        SQ1: [{"title": "A", "url": URL_A, "snippet": ""}],
        SQ2: [{"title": "B", "url": URL_B, "snippet": ""}],
    }))
    return path


def _script() -> list:
    span_a = _span(PAGE_A)
    p_a = ids.node_id("Premise", {"text": PREMISE_A, "span_refs": [span_a], "claim_class": "stable"})
    answer = f"Reference value. [narrative]\nWater boils at 100 degrees Celsius at sea level. [{p_a}]"
    tool = lambda i, name, inp, idx: scripted_response(  # noqa: E731
        [{"type": "tool_use", "id": i, "name": name, "input": inp}], stop_reason="tool_use", index=idx
    )
    return [
        tool("tu_dec", "submit_subquestions", {"subquestions": [SQ1, SQ2]}, 0),
        # sq-1: search, fetch A, search WITHIN (before B exists), record, submit
        tool("tu_s1", "search_broad", {"query": SQ1, "k": 5}, 1),
        tool("tu_f1", "fetch", {"url": URL_A}, 2),
        tool("tu_w1", "search_within", {"query": "boiling point of water at sea level", "k": 3}, 3),
        tool("tu_r1", "record_premises", {"premises": [{"text": PREMISE_A, "span_refs": [span_a]}]}, 4),
        tool("tu_sub1", "submit_report", {"status": "answered"}, 5),
        # sq-2: fetch the page that would have outranked A, then submit
        tool("tu_s2", "search_broad", {"query": SQ2, "k": 5}, 6),
        tool("tu_f2", "fetch", {"url": URL_B}, 7),
        tool("tu_sub2", "submit_report", {"status": "answered"}, 8),
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=9),
    ]


def test_search_within_replays_after_later_fetches(tmp_path, transport, fixtures_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "adapter_factory", lambda config: FakeAdapter(_script()))
    data_dir = tmp_path / "data"
    code = cli.main([
        "ask", QUERY, "--session-id", "scope-session", "--adapter", "fake", "--model", "fake-model",
        "--cache-mode", "record", "--data-dir", str(data_dir), "--search-fixtures", str(fixtures_path),
        "--max-gap-rounds", "0", "--json",
    ])
    capsys.readouterr()
    assert code == 0

    conn = open_db(data_dir / "parsec.db")
    blobs = BlobStore(data_dir / "blobs")
    event_log = EventLog(conn, RealClock())
    # The live search saw only A: B's span is not in the recorded result.
    results = [
        e for e in event_log.read("scope-session")
        if e.event_type == EventType.TOOL_RESULT and e.stream_id == "sq-1"
    ]
    search_result = json.loads(blobs.get_text(results[2].payload["full_blob"]))
    assert [r["span_id"] for r in search_result["results"]] == [_span(PAGE_A)]

    # Replay: B's spans now sit in the table from the start — and must stay invisible.
    outcome = asyncio.run(run_replay(conn, blobs, RealClock(), "scope-session"))
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match
