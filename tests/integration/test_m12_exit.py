"""M12 exit tests (v2 plan §3):

1. a long subagent run reconstructs its context from the DAG (compaction
   rung 2) and completes with IDENTICAL evidence to the unreconstructed
   run — and the reconstructed session replays byte-identically;
2. brief-gate approval and edits are steering events that replay, and the
   brief's effort estimate is enforced as dispatch caps;
3. KV-cache prefix audit: within every phase, system+tools are byte-stable
   and message lists are append-only except at recorded compaction points.
"""

from __future__ import annotations

import json

import pytest

from parsec import ids
from parsec.config import Budgets, CacheMode
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.gateway.gateway import ModelGateway
from parsec.loop.agent import OrchestratorLoop
from parsec.models.events import EventType
from parsec.replay import run_replay
from parsec.retrieval.embeddings import EmbeddingCache, HashedNgramEmbedder
from parsec.retrieval.fetcher import Fetcher
from parsec.retrieval.span_indexer import index_spans
from parsec.store.coverage import CoverageLedger
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.notebook import Notebook
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.tools.fetch import FetchTool
from parsec.tools.record_premises import RecordPremisesTool
from parsec.tools.search_within import SearchWithinTool
from tests.conftest import make_config

FILLER = (
    " The plant operates continuously through seasonal demand cycles, with"
    " maintenance windows scheduled around reservoir levels and regional load"
    " forecasts, according to the operator's published annual engineering report."
) * 3

SENT_1 = "The Hoover facility generated 4 billion kilowatt hours in 2020."
SENT_2 = "The Glen Canyon facility generated 5 billion kilowatt hours in 2020."
DOC_1 = SENT_1 + FILLER
DOC_2 = SENT_2 + FILLER
URL_1 = "https://energy.example/hoover"
URL_2 = "https://energy.example/glen-canyon"


class SpyAdapter(FakeAdapter):
    def __init__(self, responses):
        super().__init__(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return await super().complete(request)


@pytest.fixture
def env(db, blobs, event_log, ledger, sessions, clock):
    """Two cached documents and a loop builder matching the registry replay
    rebuilds (tool schemas feed prompt hashes)."""
    documents = DocumentStore(db, clock)
    spans = SpanStore(db)
    for url, text in ((URL_1, DOC_1), (URL_2, DOC_2)):
        raw = text.encode()
        doc_hash = ids.doc_hash(raw)
        blobs.put(raw)
        text_blob = blobs.put(text)
        documents.put_document(doc_hash, url, "text/plain", 200, len(raw), text_blob, {})
        documents.cache_put(ids.cache_key(url), url, doc_hash, "record")

    def make_loop(tmp_path, adapter, session_id, **config_overrides):
        config = make_config(tmp_path, session_id=session_id, **config_overrides)
        gateway = ModelGateway(adapter, event_log, blobs, ledger, config)
        dag = DagStore(db, event_log)
        fetcher = Fetcher(documents, blobs, clock, CacheMode.REPLAY)
        registry = ToolRegistry(
            [
                FetchTool(fetcher, spans),
                RecordPremisesTool(dag, spans, documents),
                SearchWithinTool(spans, EmbeddingCache(db, HashedNgramEmbedder())),
            ]
        )
        ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
        return OrchestratorLoop(
            config, gateway, registry, ctx, sessions, dag, spans, documents,
            CoverageLedger(db, event_log), Notebook(db, event_log, clock),
        )

    return make_loop


def _span(text: str) -> str:
    start, end = index_spans(text)[0]
    return ids.span_id(ids.doc_hash(text.encode()), start, end)


def _premise_id(sentence: str, doc_text: str) -> str:
    return ids.node_id(
        "Premise", {"text": sentence, "span_refs": [_span(doc_text)], "claim_class": "stable"}
    )


def _long_script(answer: str) -> list:
    """decompose -> fetch/record x2 (bulky tool results) -> submit -> write."""
    return [
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
              "input": {"subquestions": ["hydropower output of the two dams"]}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_f1", "name": "fetch", "input": {"url": URL_1}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r1", "name": "record_premises",
              "input": {"premises": [{"text": SENT_1, "span_refs": [_span(DOC_1)]}]}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_f2", "name": "fetch", "input": {"url": URL_2}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r2", "name": "record_premises",
              "input": {"premises": [{"text": SENT_2, "span_refs": [_span(DOC_2)]}]}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use"),
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn"),
    ]


def _answer() -> str:
    p1 = _premise_id(SENT_1, DOC_1)
    p2 = _premise_id(SENT_2, DOC_2)
    return f"Both outputs are on record. [narrative]\n{SENT_1} [{p1}] {SENT_2} [{p2}]"


def _actions(event_log, sid: str) -> list[str]:
    return [
        ev.payload["action"]
        for ev in event_log.read(sid)
        if ev.event_type == EventType.CONTEXT_COMPACTED
    ]


def _tier1_ids(db, sid: str) -> set[str]:
    return {
        r["node_id"]
        for r in db.execute("SELECT node_id FROM nodes WHERE session_id=? AND tier=1", (sid,))
    }


async def test_exit_1_reconstruction_preserves_evidence_and_replays(
    env, db, blobs, event_log, clock, tmp_path
):
    # control: roomy context, no compaction
    control = env(tmp_path / "a", FakeAdapter(_long_script(_answer())), "s-control",
                  budgets=Budgets(max_gap_rounds=0))
    result_control = await control.run()
    assert result_control.status == "done"
    assert _actions(event_log, "s-control") == []

    # long run under a tight budget: rung 2 must fire (not the bare reset)
    adapter = SpyAdapter(_long_script(_answer()))
    tight = env(
        tmp_path / "b", adapter, "s-reconstruct",
        max_context_chars=1600, evict_keep_last=1, budgets=Budgets(max_gap_rounds=0),
    )
    result = await tight.run()
    assert result.status == "done"
    actions = _actions(event_log, "s-reconstruct")
    assert "reconstruct" in actions
    assert "reset" not in actions

    # the reconstructed workspace came from the DAG + notebook
    reconstructed = next(
        r for r in adapter.requests
        if len(r.messages) == 1 and "reconstructed from the evidence graph" in str(r.messages[0])
    )
    body = reconstructed.messages[0]["content"]
    assert f"[{_premise_id(SENT_1, DOC_1)}]" in body       # DAG slice, by id
    assert URL_1 in body                                    # with provenance
    assert "## Session notebook" in body and "## Plan" in body

    # identical evidence to the unreconstructed run, and identical claims
    assert _tier1_ids(db, "s-reconstruct") == _tier1_ids(db, "s-control")
    assert result.answer == result_control.answer

    # and the reconstructed session replays byte-identically
    outcome = await run_replay(db, blobs, clock, "s-reconstruct")
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match


def _gated_script() -> list:
    p1 = _premise_id(SENT_1, DOC_1)
    return [
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec1", "name": "submit_subquestions",
              "input": {"scope": "Cover both dams' output.", "effort": "standard",
                        "subquestions": ["hoover output", "glen canyon output"]}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec2", "name": "submit_subquestions",
              "input": {"scope": "Only the Hoover facility's 2020 output.", "effort": "quick",
                        "subquestions": ["hoover output"]}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_f1", "name": "fetch", "input": {"url": URL_1}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r1", "name": "record_premises",
              "input": {"premises": [{"text": SENT_1, "span_refs": [_span(DOC_1)]}]}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use"),
        scripted_response(
            [{"type": "text", "text": f"Settled. [narrative]\n{SENT_1} [{p1}]"}],
            stop_reason="end_turn"),
    ]


async def test_exit_2_brief_gate_edits_and_approval_replay(
    env, db, blobs, event_log, clock, tmp_path
):
    adapter = SpyAdapter(_gated_script())
    loop = env(tmp_path, adapter, "s-gated", brief_gate=True)
    # the user's gate messages, queued as ordinary live steering
    loop.steer("only cover the hoover facility")
    loop.steer("approve")
    result = await loop.run()
    assert result.status == "done"

    events = event_log.read("s-gated")
    briefs = [ev for ev in events if ev.event_type == EventType.RESEARCH_BRIEF]
    proposed = [ev for ev in briefs if ev.payload.get("status") == "proposed"]
    final = [ev for ev in briefs if "limits" in ev.payload]
    assert [ev.payload["effort"] for ev in proposed] == ["standard", "quick"]
    assert len(final) == 1
    # the edit and the approval are recorded steering events, gate-tagged
    gate_msgs = [
        ev.payload
        for ev in events
        if ev.event_type == EventType.STEERING_INJECTED and ev.payload.get("gate") == "brief"
    ]
    assert [g["text"] for g in gate_msgs] == ["only cover the hoover facility", "approve"]
    assert gate_msgs[0]["turn_index"] < gate_msgs[1]["turn_index"]

    # effort "quick" enforced as dispatch caps: one subquestion, gap-fill off
    assert final[0].payload["limits"]["max_subquestions"] == 1
    rows = db.execute(
        "SELECT sq_id FROM coverage WHERE session_id='s-gated'"
    ).fetchall()
    assert [r["sq_id"] for r in rows] == ["sq-1"]
    # the approved brief's scope rides in the subagent's first message
    subagent_first = adapter.requests[2].messages[0]["content"]
    assert subagent_first.startswith("Research brief: Only the Hoover facility's 2020 output.")

    # gated sessions replay byte-identically: approval and edits re-inject
    outcome = await run_replay(db, blobs, clock, "s-gated")
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match


async def test_exit_3_kv_cache_prefix_audit(env, db, blobs, event_log, clock, tmp_path):
    """The §7 audit as a regression test: per stream and per phase (same
    system+tools), request messages are append-only — the only allowed
    prefix break is a recorded compaction."""
    adapter = FakeAdapter(_long_script(_answer()))
    loop = env(tmp_path, adapter, "s-audit", max_context_chars=1600, evict_keep_last=1,
               budgets=Budgets(max_gap_rounds=0))
    await loop.run()

    requests_by_stream: dict[str, list[tuple[int, dict]]] = {}
    compactions_by_stream: dict[str, list[int]] = {}
    for ev in event_log.read("s-audit"):
        if ev.event_type == EventType.LLM_REQUEST:
            body = json.loads(blobs.get_text(ev.payload["request_blob"]))
            requests_by_stream.setdefault(ev.stream_id, []).append((ev.stream_idx, body))
        elif ev.event_type == EventType.CONTEXT_COMPACTED:
            compactions_by_stream.setdefault(ev.stream_id, []).append(ev.stream_idx)

    assert "sq-1" in requests_by_stream and len(requests_by_stream["sq-1"]) >= 4
    audited_pairs = 0
    for stream, requests in requests_by_stream.items():
        requests.sort()
        for (idx_a, a), (idx_b, b) in zip(requests, requests[1:]):
            if (a["system"], a["tools"]) != (b["system"], b["tools"]):
                continue  # phase boundary (decomposer -> writer share the orchestrator stream)
            prefix_stable = b["messages"][: len(a["messages"])] == a["messages"]
            compacted_between = any(
                idx_a < c < idx_b for c in compactions_by_stream.get(stream, [])
            )
            assert prefix_stable or compacted_between, (
                f"stream {stream}: prompt prefix mutated between requests "
                f"{idx_a} and {idx_b} without a recorded compaction"
            )
            audited_pairs += 1
    assert audited_pairs >= 3  # the audit actually exercised consecutive calls
