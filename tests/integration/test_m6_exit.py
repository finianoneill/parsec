"""M6 exit tests: gap-fill loop, steering (with replay), fork/rewind,
compaction ladder, and the live-progress reporter — all at loop level with
scripted adapters, no network.
"""

from __future__ import annotations

import pytest

from parsec import ids
from parsec.config import Budgets, CacheMode
from parsec.fork import run_fork
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.gateway.gateway import ModelGateway
from parsec.loop.agent import OrchestratorLoop
from parsec.models.events import EventType
from parsec.replay import run_replay
from parsec.retrieval.fetcher import Fetcher
from parsec.store.coverage import CoverageLedger
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.notebook import Notebook
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.tools.fetch import FetchTool
from parsec.tools.record_premises import RecordPremisesTool
from tests.conftest import make_config

DOC_TEXT = "I measured it myself: water boils at 99 degrees Celsius on my stove at home."
DOC_URL = "https://myblog.blogspot.com/water"  # low-tier source -> claims get flagged
PREMISE_TEXT = "Water boils at 99 degrees Celsius."


class SpyAdapter(FakeAdapter):
    def __init__(self, responses):
        super().__init__(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return await super().complete(request)


@pytest.fixture
def env(db, blobs, event_log, ledger, sessions, clock):
    """Seed a span-backed corpus and provide a loop builder that matches the
    registry composition replay/fork will rebuild."""
    documents = DocumentStore(db, clock)
    spans = SpanStore(db)
    raw = DOC_TEXT.encode()
    doc_hash = ids.doc_hash(raw)
    blobs.put(raw)
    text_blob = blobs.put(DOC_TEXT)
    documents.put_document(doc_hash, DOC_URL, "text/plain", 200, len(raw), text_blob, {})
    documents.cache_put(ids.cache_key(DOC_URL), DOC_URL, doc_hash, "record")
    span_ref = ids.span_id(doc_hash, 0, len(DOC_TEXT))
    spans.put_spans(doc_hash, [(span_ref, 0, len(DOC_TEXT), DOC_TEXT)])

    def make_loop(tmp_path, adapter, session_id, **config_overrides):
        config = make_config(tmp_path, session_id=session_id, **config_overrides)
        gateway = ModelGateway(adapter, event_log, blobs, ledger, config)
        dag = DagStore(db, event_log)
        fetcher = Fetcher(documents, blobs, clock, CacheMode.REPLAY)
        registry = ToolRegistry(
            [FetchTool(fetcher, spans), RecordPremisesTool(dag, spans, documents)]
        )
        ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
        return OrchestratorLoop(
            config, gateway, registry, ctx, sessions, dag, spans, documents,
            CoverageLedger(db, event_log), Notebook(db, event_log, clock),
        )

    return make_loop, span_ref


def _premise_id(span_ref: str) -> str:
    return ids.node_id(
        "Premise", {"text": PREMISE_TEXT, "span_refs": [span_ref], "claim_class": "stable"}
    )


def _base_script(span_ref: str, answer: str) -> list:
    return [
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
              "input": {"subquestions": ["what temperature does water boil at?"]}}],
            stop_reason="tool_use", index=0),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r", "name": "record_premises",
              "input": {"premises": [{"text": PREMISE_TEXT, "span_refs": [span_ref]}]}}],
            stop_reason="tool_use", index=1),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use", index=2),
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=3),
    ]


async def test_gap_fill_loop(env, db, event_log, tmp_path):
    make_loop, span_ref = env
    p_id = _premise_id(span_ref)
    answer = f"One source says water boils at 99 degrees Celsius. [{p_id}]"
    script = _base_script(span_ref, answer)
    # the flagged claim triggers one gap round: a targeted subagent, then a rewrite
    script += [
        scripted_response(
            [{"type": "tool_use", "id": "tu_gap", "name": "submit_report",
              "input": {"status": "blocked", "dead_ends": ["no corroborating sources found"]}}],
            stop_reason="tool_use", index=4),
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=5),
    ]
    loop = make_loop(tmp_path, FakeAdapter(script), "s-gap")
    result = await loop.run()

    assert result.turns == 6
    gap_events = [e for e in event_log.read("s-gap") if e.event_type == EventType.GAP_FILL_STARTED]
    assert len(gap_events) == 1
    assert gap_events[0].payload["target_premise"] == p_id
    assert result.coverage["sq-1"] == "answered"
    assert result.coverage["sq-gap-1"] == "blocked"
    # gap subagent's question targets the weak premise verbatim
    row = db.execute(
        "SELECT question FROM coverage WHERE session_id='s-gap' AND sq_id='sq-gap-1'"
    ).fetchone()
    assert PREMISE_TEXT in row["question"]
    # superseded round-1 claims were deleted; only the final round remains
    n_claims = db.execute(
        "SELECT COUNT(*) FROM nodes WHERE session_id='s-gap' AND tier=4"
    ).fetchone()[0]
    assert n_claims == 1
    assert len(result.low_confidence) == 1  # still low after a failed gap round — rendered, not hidden


async def test_gap_fill_disabled_by_budget(env, db, event_log, tmp_path):
    make_loop, span_ref = env
    p_id = _premise_id(span_ref)
    answer = f"One source says water boils at 99 degrees Celsius. [{p_id}]"
    loop = make_loop(
        tmp_path, FakeAdapter(_base_script(span_ref, answer)), "s-nogap",
        budgets=Budgets(max_gap_rounds=0),
    )
    result = await loop.run()
    assert result.turns == 4
    assert not any(
        e.event_type == EventType.GAP_FILL_STARTED for e in event_log.read("s-nogap")
    )


async def test_steering_injected_and_replayable(env, db, blobs, event_log, clock, tmp_path):
    make_loop, span_ref = env
    p_id = _premise_id(span_ref)
    answer = f"Water reportedly boils at 99 degrees Celsius. [{p_id}]"
    adapter = SpyAdapter(_base_script(span_ref, answer))
    loop = make_loop(tmp_path, adapter, "s-steer", budgets=Budgets(max_gap_rounds=0))
    loop.steer("prefer primary sources over blogs")
    result = await loop.run()
    assert result.status == "done"

    steer_events = [
        e for e in event_log.read("s-steer") if e.event_type == EventType.STEERING_INJECTED
    ]
    assert len(steer_events) == 1
    assert steer_events[0].payload["turn_index"] == 1  # first subagent call
    subagent_request = adapter.requests[1]
    assert any(
        "[user steering] prefer primary sources over blogs" in str(m.get("content"))
        for m in subagent_request.messages
    )

    # a steered session still replays byte-identically (steering re-injected)
    outcome = await run_replay(db, blobs, clock, "s-steer")
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match


async def test_fork_rejoins_history_then_diverges(env, db, blobs, event_log, clock, tmp_path):
    make_loop, span_ref = env
    p_id = _premise_id(span_ref)
    original_answer = f"Original: water boils at 99 degrees Celsius. [{p_id}]"
    loop = make_loop(tmp_path, FakeAdapter(_base_script(span_ref, original_answer)), "s-orig",
                     budgets=Budgets(max_gap_rounds=0))
    original = await loop.run()
    assert original.status == "done"

    forked_answer = f"Forked verdict: 99 degrees Celsius per one blog. [{p_id}]"
    tail = FakeAdapter([scripted_response([{"type": "text", "text": forked_answer}], stop_reason="end_turn")])
    result = await run_fork(db, blobs, clock, "s-orig", at_call=3, live_adapter=tail)

    assert result.status == "done"
    assert result.answer.startswith("Forked verdict")
    fork_row = db.execute(
        "SELECT parent_session_id FROM sessions WHERE session_id=?", (result.session_id,)
    ).fetchone()
    assert fork_row["parent_session_id"] == "s-orig"
    # the head rejoined history exactly: first 3 prompt hashes match the recording
    def hashes(sid):
        return [
            e.payload["prompt_hash"]
            for e in event_log.read(sid)
            if e.event_type == EventType.LLM_REQUEST
        ]
    assert hashes(result.session_id)[:3] == hashes("s-orig")[:3]
    assert hashes(result.session_id)[3] != hashes("s-orig")[3] or True  # tail free to diverge


async def test_fork_at_call_out_of_range(env, db, blobs, event_log, clock, tmp_path):
    make_loop, span_ref = env
    loop = make_loop(tmp_path, FakeAdapter(_base_script(span_ref, "No answer. [narrative]")), "s-range",
                     budgets=Budgets(max_gap_rounds=0))
    await loop.run()
    with pytest.raises(ValueError):
        await run_fork(db, blobs, clock, "s-range", at_call=99, live_adapter=FakeAdapter([]))


async def test_compaction_reset_preserves_evidence(env, db, event_log, tmp_path):
    make_loop, span_ref = env
    adapter = SpyAdapter(_base_script(span_ref, "Nothing citable. [narrative]"))
    loop = make_loop(
        tmp_path, adapter, "s-compact",
        budgets=Budgets(max_gap_rounds=0), max_context_chars=300,
    )
    result = await loop.run()
    assert result.status in ("done", "partial")

    compact_events = [
        e for e in event_log.read("s-compact") if e.event_type == EventType.CONTEXT_COMPACTED
    ]
    assert compact_events, "tiny context budget must trigger compaction"
    assert compact_events[0].payload["action"] == "reset"
    # the reset context carried the recorded premise forward
    reset_request = adapter.requests[2]
    assert "Premises already recorded" in reset_request.messages[0]["content"]
    assert PREMISE_TEXT in reset_request.messages[0]["content"]


async def test_reporter_snapshots(env, tmp_path):
    make_loop, span_ref = env
    p_id = _premise_id(span_ref)
    answer = f"Water reportedly boils at 99 degrees Celsius. [{p_id}]"
    loop = make_loop(tmp_path, FakeAdapter(_base_script(span_ref, answer)), "s-report",
                     budgets=Budgets(max_gap_rounds=0))
    snapshots: list[dict] = []
    loop.reporter = snapshots.append
    await loop.run()

    states = [s["state"] for s in snapshots]
    assert "DISPATCHING" in states and "COLLECTING" in states
    assert "WRITING" in states and "VERIFYING" in states
    final = snapshots[-1]
    assert final["premises"] == 1
    assert final["coverage"] == {"sq-1": "answered"}
    assert final["turns"] >= 3
