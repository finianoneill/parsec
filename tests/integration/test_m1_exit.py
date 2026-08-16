"""M1 exit test (§11): one query → cited answer where every claim resolves
to a cached span, replayable byte-identically — driven through the real CLI
entrypoint with a scripted adapter and a mock HTTP transport (no network,
no keys).
"""

from __future__ import annotations

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


def page_span_ids(page: bytes) -> list[str]:
    """Compute span IDs exactly as the fetch pipeline will."""
    text, _, _ = extract_text(page, "text/html")
    h = ids.doc_hash(page)
    return [ids.span_id(h, s, e) for s, e in index_spans(text)]


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
    answer = (
        "Here is what the sources say. [narrative]\n"
        f"Water boils at 100 degrees Celsius at standard atmospheric pressure at sea level. [{span_a}] "
        f"At higher altitudes the boiling point decreases, dropping to about 70 degrees Celsius "
        f"on the summit of Mount Everest. [{span_b}]"
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
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=2),
        # One spare response for the negative test's repair round-trip: the fake
        # "model" restates the same answer, so an unfixable violation stays flagged.
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=3),
    ]
    monkeypatch.setattr(cli, "adapter_factory", lambda config: FakeAdapter(responses))
    return answer


def test_m1_exit(tmp_path, transport, fixtures_path, scripted_adapter, capsys):
    data_dir = tmp_path / "data"
    session_id = "m1-exit-session"

    exit_code = cli.main(
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
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "done"
    assert out["claims_total"] == 2
    assert out["unresolved"] == []
    assert out["totals"]["input_tokens"] > 0

    conn = open_db(data_dir / "parsec.db")
    blobs = BlobStore(data_dir / "blobs")

    # Every non-narrative claim resolves: span rows exist, text is the verbatim
    # document slice, docs are cached with raw blobs present.
    claims = conn.execute(
        "SELECT payload_json FROM nodes WHERE session_id=? AND tier=4", (session_id,)
    ).fetchall()
    assert len(claims) == 2
    all_refs = []
    for row in claims:
        payload = json.loads(row["payload_json"])
        assert payload["span_refs"]
        all_refs += payload["span_refs"]
    assert len(set(r.split("#")[0] for r in all_refs)) == 2  # two distinct docs cited

    for ref in all_refs:
        span = conn.execute("SELECT * FROM spans WHERE span_id=?", (ref,)).fetchone()
        assert span is not None
        doc = conn.execute("SELECT * FROM documents WHERE doc_hash=?", (span["doc_hash"],)).fetchone()
        assert doc is not None
        assert blobs.exists(doc["raw_blob"]) and blobs.exists(doc["text_blob"])
        text = blobs.get_text(doc["text_blob"])
        assert text[span["char_start"]:span["char_end"]] == span["text"]
        cached = conn.execute(
            "SELECT 1 FROM cache_index WHERE doc_hash=?", (span["doc_hash"],)
        ).fetchone()
        assert cached is not None

    # extracts edges exist 1:1 with claim->span pairs
    edges = conn.execute(
        "SELECT * FROM edges WHERE session_id=? AND edge_type='extracts'", (session_id,)
    ).fetchall()
    assert len(edges) == 2

    # --- replay against the frozen corpus: byte-identical, zero HTTP calls ---
    calls_before = transport["calls"]
    import asyncio

    outcome = asyncio.run(run_replay(conn, blobs, RealClock(), session_id))
    assert outcome.result.status == "done"
    assert transport["calls"] == calls_before  # frozen corpus honored (transport unused in replay)
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match
    assert outcome.verified

    event_log = EventLog(conn, RealClock())
    assert event_log.projection(session_id) == event_log.projection(outcome.result.session_id)


def test_m1_negative_deleted_span_flagged(tmp_path, transport, fixtures_path, scripted_adapter, capsys):
    data_dir = tmp_path / "data"
    exit_code = cli.main(
        [
            "ask", QUERY,
            "--session-id", "m1-neg-1",
            "--adapter", "fake", "--model", "fake-model",
            "--cache-mode", "record",
            "--data-dir", str(data_dir),
            "--search-fixtures", str(fixtures_path),
            "--json",
        ]
    )
    assert exit_code == 0
    capsys.readouterr()

    # Corrupt the corpus: tamper with one cited span's stored text, then run the
    # identical scripted session again — the citation check must mechanically
    # flag the mismatch. (Tampering, not deletion: record-mode refetch re-runs
    # the indexer, whose INSERT OR IGNORE preserves the corrupted row.)
    conn = open_db(data_dir / "parsec.db")
    victim = page_span_ids(PAGE_A)[0]
    conn.execute("UPDATE spans SET text='TAMPERED EVIDENCE' WHERE span_id=?", (victim,))
    conn.close()

    exit_code = cli.main(
        [
            "ask", QUERY,
            "--session-id", "m1-neg-2",
            "--adapter", "fake", "--model", "fake-model",
            "--cache-mode", "record",
            "--data-dir", str(data_dir),
            "--search-fixtures", str(fixtures_path),
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    # The fake adapter cannot actually repair, so the run ends partial with the
    # violation surfaced — the claim is flagged, not silently accepted.
    assert exit_code == 3
    assert out["status"] == "partial"
    assert any("does not match" in p for p in out["unresolved"])
