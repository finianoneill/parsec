"""M4 exit tests (§11):
1. planted low-quality source → its downstream claims render hedged
   (low tier annotated to the writer, flagged below the stakes threshold,
   marked low-confidence in the appendix);
2. planted relevant-but-ignored source → appears in the
   "consulted but unused" appendix.
Plus: the whole credence/omission pipeline replays byte-identically.
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

QUERY = "at what temperature does water boil according to available sources"
SQ = "boiling point of water"

BLOG_PAGE = (
    "<html><head><title>My Kitchen Blog</title></head><body>"
    "<p>I tested this on my stove last weekend and in my experience water boils at "
    "99 degrees Celsius, at least at my house. Your mileage may vary of course, but "
    "that is what my thermometer consistently showed across several attempts during "
    "the afternoon, and I stand by the measurement even if the textbooks disagree "
    "with me on the exact number for this experiment.</p>"
    "</body></html>"
).encode()

GOOD_PAGE = (
    "<html><head><title>Reference Tables</title></head><body>"
    "<p>Standard reference tables list the boiling point of water as 100 degrees "
    "Celsius at one atmosphere of pressure. These values are maintained by national "
    "metrology institutes and are used for instrument calibration worldwide, forming "
    "the basis of countless engineering and scientific procedures that depend on "
    "accurate thermometric reference points for their validity.</p>"
    "</body></html>"
).encode()

BLOG_URL = "https://myblog.blogspot.com/water"
GOOD_URL = "https://data.reference.example/tables"

PREMISE_TEXT = "Water boils at 99 degrees Celsius."


def first_span(page: bytes) -> str:
    text, _, _ = extract_text(page, "text/html")
    h = ids.doc_hash(page)
    s, e = index_spans(text)[0]
    return ids.span_id(h, s, e)


def premise_id(text: str, span_ref: str) -> str:
    return ids.node_id("Premise", {"text": text, "span_refs": [span_ref], "claim_class": "stable"})


@pytest.fixture
def transport(monkeypatch):
    pages = {BLOG_URL: BLOG_PAGE, GOOD_URL: GOOD_PAGE}
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
                SQ: [
                    {"title": "Kitchen Blog", "url": BLOG_URL, "snippet": "99C on my stove"},
                    {"title": "Reference Tables", "url": GOOD_URL, "snippet": "100C standard"},
                ]
            }
        )
    )
    return path


@pytest.fixture
def scripted_adapter(monkeypatch):
    span_blog = first_span(BLOG_PAGE)
    p_blog = premise_id(PREMISE_TEXT, span_blog)
    answer = (
        "The available evidence is thin. [narrative]\n"
        f"One blog source suggests water boils at 99 degrees Celsius. [{p_blog}]"
    )
    responses = [
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
              "input": {"subquestions": [SQ]}}],
            stop_reason="tool_use", index=0,
        ),
        scripted_response(
            [{"type": "tool_use", "id": "tu_s", "name": "search_broad", "input": {"query": SQ, "k": 5}}],
            stop_reason="tool_use", index=1,
        ),
        # the subagent consults BOTH sources...
        scripted_response(
            [
                {"type": "tool_use", "id": "tu_f1", "name": "fetch", "input": {"url": BLOG_URL}},
                {"type": "tool_use", "id": "tu_f2", "name": "fetch", "input": {"url": GOOD_URL}},
            ],
            stop_reason="tool_use", index=2,
        ),
        # ...but records evidence only from the blog (the planted omission)
        scripted_response(
            [{"type": "tool_use", "id": "tu_r", "name": "record_premises",
              "input": {"premises": [{"text": PREMISE_TEXT, "span_refs": [span_blog]}]}}],
            stop_reason="tool_use", index=3,
        ),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub", "name": "submit_report",
              "input": {"status": "answered"}}],
            stop_reason="tool_use", index=4,
        ),
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=5),
    ]
    monkeypatch.setattr(cli, "adapter_factory", lambda config: FakeAdapter(responses))
    return answer


def test_m4_exit(tmp_path, transport, fixtures_path, scripted_adapter, capsys):
    data_dir = tmp_path / "data"
    session_id = "m4-exit-session"

    exit_code = cli.main(
        [
            "ask", QUERY,
            "--session-id", session_id,
            "--adapter", "fake",
            "--model", "fake-model",
            "--cache-mode", "record",
            "--data-dir", str(data_dir),
            "--search-fixtures", str(fixtures_path),
            "--max-gap-rounds", "0",
            "--json",
        ]
    )
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "done"

    # --- exit test 1: low-quality source -> downstream claim renders hedged ---
    assert len(out["low_confidence"]) == 1
    assert "99 degrees Celsius" in out["low_confidence"][0]
    assert out["low_confidence"][0].startswith("low, single source")
    # the harness-built appendix marks it, tiers only, no raw numbers
    assert "low, single source confidence" in out["answer"]
    assert "0.4" not in out["answer"]

    conn = open_db(data_dir / "parsec.db")
    blobs = BlobStore(data_dir / "blobs")

    # the computed credence is persisted on the claim node
    claim = conn.execute(
        "SELECT * FROM nodes WHERE session_id=? AND tier=4", (session_id,)
    ).fetchone()
    assert claim["credence"] is not None and claim["credence"] < 0.7

    # the writer was TOLD the tier (it never invents hedging)
    event_log = EventLog(conn, RealClock())
    requests = [
        blobs.get_text(ev.payload["request_blob"])
        for ev in event_log.read(session_id)
        if ev.event_type == EventType.LLM_REQUEST
    ]
    writer_body = requests[-1]  # the writer call is the session's last (call_index is per-stream since M11)
    assert "confidence: low, single source" in writer_body

    # --- exit test 2: consulted-but-ignored source appears in the appendix ---
    assert out["unused_sources"] == [GOOD_URL]
    assert "Consulted but unused" in out["answer"]
    assert GOOD_URL in out["answer"]
    omission_events = [
        ev for ev in event_log.read(session_id) if ev.event_type == EventType.OMISSION_DETECTED
    ]
    assert len(omission_events) == 1
    assert omission_events[0].payload["unused_documents"][0]["url"] == GOOD_URL

    # --- the credence/omission pipeline replays byte-identically ---
    calls_before = transport["calls"]
    outcome = asyncio.run(run_replay(conn, blobs, RealClock(), session_id))
    assert transport["calls"] == calls_before
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match


def test_source_tier_override_raises_confidence(tmp_path, transport, fixtures_path, scripted_adapter, capsys, monkeypatch):
    """The tier table is a per-run prior, not a verdict (§2.1): overriding the
    blog domain upward lifts the same claim out of the flagged set."""
    import parsec.cli as cli_mod

    data_dir = tmp_path / "data"
    # No CLI flag for source tiers yet — wrap RunConfig at the cli module
    # boundary so the run is created with the override baked in.
    real_runconfig = cli_mod.RunConfig

    class TieredRunConfig(real_runconfig):
        def __init__(self, **kwargs):
            kwargs.setdefault("source_tiers", {"blogspot.com": 0.9})
            super().__init__(**kwargs)

    monkeypatch.setattr(cli_mod, "RunConfig", TieredRunConfig)
    exit_code = cli_mod.main(
        [
            "ask", QUERY,
            "--session-id", "m4-override-session",
            "--adapter", "fake",
            "--model", "fake-model",
            "--cache-mode", "record",
            "--data-dir", str(data_dir),
            "--search-fixtures", str(fixtures_path),
            "--max-gap-rounds", "0",
            "--json",
        ]
    )
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["low_confidence"] == []
    assert "high, single source confidence" in out["answer"]
