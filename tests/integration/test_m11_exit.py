"""M11 exit tests (v2 plan §3):

1. a 3-subagent CONCURRENT run replays byte-identically — per-stream and in
   the recorded join order, even when the replay's natural completion order
   differs from the recording's;
2. killing one subagent mid-wave leaves a blocked-with-reason coverage row
   and a replayable session;
3. wave allowances starve subagents deterministically — per-stream budget
   gates never read interleaving-dependent global totals.
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
from parsec.gateway.fake_adapter import StreamFakeAdapter, scripted_response
from parsec.models.events import EventType
from parsec.replay import run_replay
from parsec.retrieval.extract import extract_text
from parsec.retrieval.span_indexer import index_spans
from parsec.store.blobs import BlobStore
from parsec.store.event_log import EventLog

QUERY = "how tall are the three highest mountains"
SQS = ["height of everest", "height of k2", "height of kangchenjunga"]

SENTENCES = {
    "sq-1": "Mount Everest rises 8849 meters above sea level according to the 2020 joint survey.",
    "sq-2": "K2 rises 8611 meters above sea level according to modern surveys.",
    "sq-3": "Kangchenjunga rises 8586 meters above sea level according to modern surveys.",
}
URLS = {
    "sq-1": "https://peaks.example/everest",
    "sq-2": "https://peaks.example/k2",
    "sq-3": "https://peaks.example/kangchenjunga",
}


def _page(sentence: str) -> bytes:
    return f"<html><body><article><p>{sentence}</p></article></body></html>".encode()


PAGES = {URLS[sq]: _page(SENTENCES[sq]) for sq in URLS}


def _span(sq: str) -> str:
    page = PAGES[URLS[sq]]
    text, _, _ = extract_text(page, "text/html")
    start, end = index_spans(text)[0]
    return ids.span_id(ids.doc_hash(page), start, end)


def _premise(sq: str) -> str:
    return ids.node_id(
        "Premise", {"text": SENTENCES[sq], "span_refs": [_span(sq)], "claim_class": "stable"}
    )


class RecordDelays:
    """Delay chosen streams' calls at RECORD time only, so the recorded join
    order differs from replay's natural completion order — forcing the
    deterministic scheduler to apply the RECORDED order."""

    def __init__(self, inner, delays: dict[str, float]):
        self._inner = inner
        self._delays = delays

    async def complete(self, request):
        from parsec.store.event_log import CURRENT_STREAM

        delay = self._delays.get(CURRENT_STREAM.get())
        if delay:
            await asyncio.sleep(delay)
        return await self._inner.complete(request)


def _subagent_script(sq: str) -> list:
    return [
        scripted_response(
            [{"type": "tool_use", "id": f"tu_f_{sq}", "name": "fetch", "input": {"url": URLS[sq]}}],
            stop_reason="tool_use",
        ),
        scripted_response(
            [{"type": "tool_use", "id": f"tu_r_{sq}", "name": "record_premises",
              "input": {"premises": [{"text": SENTENCES[sq], "span_refs": [_span(sq)]}]}}],
            stop_reason="tool_use",
        ),
        scripted_response(
            [{"type": "tool_use", "id": f"tu_s_{sq}", "name": "submit_report",
              "input": {"status": "answered"}}],
            stop_reason="tool_use",
        ),
    ]


def _decomposer() -> object:
    return scripted_response(
        [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
          "input": {"subquestions": SQS}}],
        stop_reason="tool_use",
    )


def _writer(premise_sqs: list[str], gap_note: str | None = None) -> object:
    lines = ["The measured heights are settled. [narrative]"]
    if gap_note:
        lines.append(f"{gap_note} [narrative]")
    lines += [f"{SENTENCES[sq]} [{_premise(sq)}]" for sq in premise_sqs]
    return scripted_response([{"type": "text", "text": "\n".join(lines)}], stop_reason="end_turn")


@pytest.fixture
def transport(monkeypatch):
    counter = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["calls"] += 1
        page = PAGES.get(str(request.url).rstrip("/"))
        if page is None:
            return httpx.Response(404, content=b"not found")
        return httpx.Response(200, content=page, headers={"content-type": "text/html"})

    monkeypatch.setattr(cli, "fetch_transport", httpx.MockTransport(handler))
    return counter


def _ask(tmp_path, session_id: str, extra: list[str] | None = None) -> tuple[int, dict]:
    data_dir = tmp_path / "data"
    exit_code = cli.main(
        [
            "ask", QUERY, "--session-id", session_id, "--adapter", "fake",
            "--model", "fake-model", "--cache-mode", "record",
            "--data-dir", str(data_dir), "--parallel", "3",
            "--max-gap-rounds", "0", "--json", *(extra or []),
        ]
    )
    return exit_code, data_dir


def _joined_order(event_log: EventLog, session_id: str) -> list[str]:
    return [
        ev.payload["sq_id"]
        for ev in event_log.read(session_id)
        if ev.event_type == EventType.SUBAGENT_JOINED
    ]


def test_exit_1_concurrent_run_replays_byte_identically(tmp_path, transport, monkeypatch, capsys):
    """3 concurrent subagents; sq-1 artificially slow at record time, so the
    recorded join order is NOT dispatch order — and replay (where nothing is
    slow) must still reproduce it from the journal."""
    scripts = {"orchestrator": [_decomposer(), _writer(["sq-1", "sq-2", "sq-3"])]}
    scripts.update({sq: _subagent_script(sq) for sq in ("sq-1", "sq-2", "sq-3")})
    monkeypatch.setattr(
        cli, "adapter_factory",
        lambda config: RecordDelays(StreamFakeAdapter(scripts), {"sq-1": 0.05}),
    )
    exit_code, data_dir = _ask(tmp_path, "m11-parallel")
    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0 and out["status"] == "done"
    for sq in ("sq-1", "sq-2", "sq-3"):
        assert SENTENCES[sq] in out["answer"]

    conn = open_db(data_dir / "parsec.db")
    blobs = BlobStore(data_dir / "blobs")
    event_log = EventLog(conn, RealClock())

    recorded_order = _joined_order(event_log, "m11-parallel")
    assert sorted(recorded_order) == ["sq-1", "sq-2", "sq-3"]
    assert recorded_order[-1] == "sq-1"  # the slow subagent joined last

    calls_before = transport["calls"]
    outcome = asyncio.run(run_replay(conn, blobs, RealClock(), "m11-parallel"))
    assert transport["calls"] == calls_before  # zero live fetches
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match
    # the replayed session applied the RECORDED join order, not its own
    assert _joined_order(event_log, outcome.result.session_id) == recorded_order


def test_exit_2_subagent_killed_mid_wave(tmp_path, transport, monkeypatch, capsys):
    """One subagent's model call dies mid-wave: the wave survives, the
    coverage row resolves as blocked WITH the reason, and the whole session
    — including the journaled failure — replays byte-identically."""
    scripts = {
        "orchestrator": [
            _decomposer(),
            _writer(["sq-1", "sq-3"], gap_note="The K2 measurement could not be researched."),
        ],
        "sq-1": _subagent_script("sq-1"),
        "sq-2": [RuntimeError("boom")],
        "sq-3": _subagent_script("sq-3"),
    }
    monkeypatch.setattr(
        cli, "adapter_factory", lambda config: StreamFakeAdapter(scripts)
    )
    exit_code, data_dir = _ask(tmp_path, "m11-failure")
    out = json.loads(capsys.readouterr().out)
    assert SENTENCES["sq-1"] in out["answer"] and SENTENCES["sq-3"] in out["answer"]

    conn = open_db(data_dir / "parsec.db")
    blobs = BlobStore(data_dir / "blobs")
    event_log = EventLog(conn, RealClock())

    row = conn.execute(
        "SELECT status, reason FROM coverage WHERE session_id=? AND sq_id=?",
        ("m11-failure", "sq-2"),
    ).fetchone()
    assert row["status"] == "blocked"
    assert row["reason"] == "model call failed: RuntimeError: boom"

    failures = [
        ev for ev in event_log.read("m11-failure") if ev.event_type == EventType.LLM_FAILED
    ]
    assert len(failures) == 1 and failures[0].stream_id == "sq-2"
    assert failures[0].payload["kind"] == "RuntimeError"

    outcome = asyncio.run(run_replay(conn, blobs, RealClock(), "m11-failure"))
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match


def test_exit_3_wave_allowance_starves_deterministically(tmp_path, transport, monkeypatch, capsys):
    """Tiny token budget: the wave allowance lets each subagent make exactly
    one call, then its PER-STREAM gate stops it — no gate ever reads a
    sibling's spend, so the outcome is interleaving-proof and replays."""
    scripts = {
        "orchestrator": [
            _decomposer(),
            scripted_response(
                [{"type": "text", "text": "The budget was exhausted before research completed. [narrative]"}],
                stop_reason="end_turn",
            ),
        ],
        # each subagent gets one call's worth of script; the gate stops it there
        "sq-1": _subagent_script("sq-1")[:1],
        "sq-2": _subagent_script("sq-2")[:1],
        "sq-3": _subagent_script("sq-3")[:1],
    }
    monkeypatch.setattr(cli, "adapter_factory", lambda config: StreamFakeAdapter(scripts))
    # decomposer spends 150 tokens; 200-150=50 remaining -> ~16 tokens per
    # subagent -> each stops after its first (150-token) call
    exit_code, data_dir = _ask(tmp_path, "m11-starved", extra=["--max-tokens", "200"])
    capsys.readouterr()

    conn = open_db(data_dir / "parsec.db")
    blobs = BlobStore(data_dir / "blobs")
    rows = conn.execute(
        "SELECT sq_id, status, reason FROM coverage WHERE session_id=? ORDER BY sq_id",
        ("m11-starved",),
    ).fetchall()
    assert [r["status"] for r in rows] == ["blocked"] * 3
    assert all("wave token allowance exhausted" in r["reason"] for r in rows)

    outcome = asyncio.run(run_replay(conn, blobs, RealClock(), "m11-starved"))
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match
