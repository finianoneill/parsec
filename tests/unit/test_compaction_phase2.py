"""Phase 2: token-aware compaction triggers, reactive overflow recovery,
and writer-prompt degradation."""

from __future__ import annotations

import json

from parsec.config import Budgets
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.loop import compaction, prompts
from parsec.models.events import EventType
from parsec.store.coverage import CoverageLedger
from tests.unit.test_agent_gates import build_loop, decompose_response


class ContextOverflowError(Exception):
    """SDK-shaped: BadRequest whose message marks a too-long prompt."""

    status_code = 400

    def __init__(self):
        super().__init__("prompt is too long: 210012 tokens > 200000 maximum")


class _ScriptAdapter:
    """Ordered outcomes: Exception instances raise, responses return."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        out = self._outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


# -- estimator ---------------------------------------------------------------


def test_estimate_is_chars_based_without_anchor():
    messages = [{"role": "user", "content": "x" * 400}]
    est = compaction.estimate_tokens(messages, static_chars=1000, last_usage_tokens=None)
    assert est == (1000 + compaction.context_chars(messages)) // 4


def test_recorded_usage_floors_the_estimate():
    """The model's own count beats the chars/4 guess when it is higher —
    e.g. token-dense content the heuristic underestimates."""
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "y" * 400},  # appended since the model looked
    ]
    trailing = compaction.trailing_chars(messages)
    assert trailing == compaction.context_chars(messages[2:])
    est = compaction.estimate_tokens(messages, static_chars=0, last_usage_tokens=50_000)
    assert est == 50_000 + trailing // 4  # anchor dominates the tiny char count


def test_static_prefix_counts_system_and_tools():
    assert compaction.static_prefix_chars("sys", []) == 3 + len("[]")


# -- reactive subagent compaction --------------------------------------------


async def test_overflow_compacts_and_retries_the_turn(
    tmp_path, db, blobs, event_log, ledger, sessions, clock
):
    adapter = _ScriptAdapter(
        [
            decompose_response(["what is q?"]),
            ContextOverflowError(),  # subagent turn 1: API rejects
            scripted_response([{"type": "text", "text": "done looking"}], index=1),
            scripted_response([{"type": "text", "text": "No evidence. [narrative]"}], index=2),
        ]
    )
    loop = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter,
        Budgets(max_turns=10), session_id="s-overflow-retry",
    )
    result = await loop.run()
    assert result.status in ("done", "partial")

    compactions = [
        e for e in event_log.read("s-overflow-retry")
        if e.event_type == EventType.CONTEXT_COMPACTED
    ]
    assert len(compactions) == 1
    assert compactions[0].payload["trigger"] == "overflow"
    assert compactions[0].payload["action"] == "evict"  # rung 1 on first overflow
    assert compactions[0].stream_id == "sq-1"
    # the failed attempt is journaled but does not consume a turn
    assert result.turns == 3  # decomposer + subagent + writer


async def test_consecutive_overflows_escalate_then_die(
    tmp_path, db, blobs, event_log, ledger, sessions, clock
):
    """Rungs escalate one per overflow; past rung 3 the subagent dies as any
    other model failure — the run degrades, the wave survives."""
    adapter = _ScriptAdapter(
        [
            decompose_response(["what is q?"]),
            ContextOverflowError(), ContextOverflowError(),
            ContextOverflowError(), ContextOverflowError(),
            scripted_response([{"type": "text", "text": "Nothing. [narrative]"}], index=1),
        ]
    )
    loop = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter,
        Budgets(max_turns=10), session_id="s-overflow-die",
    )
    result = await loop.run()
    assert result.status in ("done", "partial")  # degraded, not dead

    actions = [
        e.payload["action"]
        for e in event_log.read("s-overflow-die")
        if e.event_type == EventType.CONTEXT_COMPACTED
    ]
    assert actions == ["evict", "reconstruct", "reset"]
    rows = CoverageLedger(db, event_log).all("s-overflow-die")
    assert rows[0]["status"] == "blocked"
    assert "model call failed" in rows[0]["reason"]


# -- writer degradation ------------------------------------------------------


async def test_writer_overflow_clips_and_retries(
    tmp_path, db, blobs, event_log, ledger, sessions, clock
):
    adapter = _ScriptAdapter(
        [
            decompose_response(["what is q?"]),
            scripted_response([{"type": "text", "text": "found nothing"}], index=1),
            ContextOverflowError(),  # writer call: API rejects
            scripted_response([{"type": "text", "text": "No evidence. [narrative]"}], index=2),
        ]
    )
    loop = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter,
        Budgets(max_turns=10), session_id="s-writer-clip",
    )
    result = await loop.run()
    assert result.status in ("done", "partial")  # the answer shipped either way

    clips = [
        e for e in event_log.read("s-writer-clip")
        if e.event_type == EventType.CONTEXT_COMPACTED
    ]
    assert len(clips) == 1
    assert clips[0].payload["action"] == "writer_clip"
    assert clips[0].payload["trigger"] == "overflow"
    assert clips[0].payload["level"] == 1
    assert clips[0].stream_id == "orchestrator"


def test_writer_prompt_clips_evidence_text_only():
    long_text = "water boils at one hundred degrees " * 20  # ~700 chars
    row = {
        "node_id": "premise:abc123",
        "payload_json": json.dumps({"text": long_text}),
    }
    full = prompts.writer_user_prompt("q?", [row], [], {"premise:abc123": "https://x.example"}, [])
    clipped = prompts.writer_user_prompt(
        "q?", [row], [], {"premise:abc123": "https://x.example"}, [], max_text_chars=120
    )
    assert long_text in full
    assert long_text not in clipped
    assert "…[clipped to fit context]" in clipped
    assert "[premise:abc123]" in clipped          # id survives
    assert "source: https://x.example" in clipped  # provenance survives
    assert len(clipped) < len(full)


# -- replay ------------------------------------------------------------------


async def test_overflow_compacted_run_replays(
    db, blobs, event_log, ledger, sessions, clock, tmp_path
):
    """The overflow failure, the compaction decision, and the retried call
    all replay byte-identically."""
    from parsec.config import CacheMode
    from parsec.gateway.gateway import ModelGateway
    from parsec.loop.agent import OrchestratorLoop
    from parsec.replay import run_replay
    from parsec.retrieval.embeddings import EmbeddingCache, HashedNgramEmbedder
    from parsec.retrieval.fetcher import Fetcher
    from parsec.store.dag import DagStore
    from parsec.store.documents import DocumentStore
    from parsec.store.notebook import Notebook
    from parsec.store.spans import SpanStore
    from parsec.tools.base import ToolContext, ToolRegistry
    from parsec.tools.fetch import FetchTool
    from parsec.tools.record_premises import RecordPremisesTool
    from parsec.tools.search_within import SearchWithinTool
    from tests.conftest import make_config

    adapter = _ScriptAdapter(
        [
            decompose_response(["what is q?"]),
            ContextOverflowError(),
            scripted_response([{"type": "text", "text": "nothing found"}], index=1),
            scripted_response([{"type": "text", "text": "No evidence. [narrative]"}], index=2),
        ]
    )
    config = make_config(tmp_path, session_id="s-oc-replay", budgets=Budgets(max_turns=10))
    documents = DocumentStore(db, clock)
    spans = SpanStore(db)
    dag = DagStore(db, event_log)
    registry = ToolRegistry(
        [
            FetchTool(Fetcher(documents, blobs, clock, CacheMode.RECORD), spans),
            RecordPremisesTool(dag, spans, documents),
            SearchWithinTool(spans, EmbeddingCache(db, HashedNgramEmbedder())),
        ]
    )
    gateway = ModelGateway(adapter, event_log, blobs, ledger, config)
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    loop = OrchestratorLoop(
        config, gateway, registry, ctx, sessions, dag, spans, documents,
        CoverageLedger(db, event_log), Notebook(db, event_log, clock),
    )
    await loop.run()

    outcome = await run_replay(db, blobs, clock, "s-oc-replay")
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.verified
