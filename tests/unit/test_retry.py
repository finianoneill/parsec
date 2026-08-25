"""Harness-owned model-call retries (Phase 1): classified errors, journaled
LLM_RETRY attempts, deterministic backoff, and replayable attempt sequences."""

from __future__ import annotations

import pytest

from parsec.errors import ModelCallFailed, ModelErrorKind
from parsec.gateway.fake_adapter import scripted_response
from parsec.gateway.gateway import ModelGateway
from parsec.gateway.retry import RetryPolicy, classify_model_error
from parsec.models.events import EventType
from parsec.models.gateway import ModelRequest
from tests.conftest import make_config


class _FlakyAdapter:
    """Scripted outcomes: Exception instances raise, responses return."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)

    async def complete(self, request):
        out = self._outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class RateLimitError(Exception):
    status_code = 429


class OverloadedError(Exception):
    status_code = 529


class BadRequestError(Exception):
    status_code = 400


class APITimeoutError(Exception):
    pass


class AuthenticationError(Exception):
    status_code = 401


# -- classification ----------------------------------------------------------


def test_classify_by_status_and_name():
    assert classify_model_error(RateLimitError("429")) == ModelErrorKind.THROTTLED
    assert classify_model_error(OverloadedError("Overloaded")) == ModelErrorKind.OVERLOADED
    assert classify_model_error(Exception("ThrottlingException: slow down")) == ModelErrorKind.THROTTLED
    assert (
        classify_model_error(BadRequestError("prompt is too long: 210000 tokens > 200000"))
        == ModelErrorKind.CONTEXT_OVERFLOW
    )
    assert classify_model_error(APITimeoutError("timed out")) == ModelErrorKind.TRANSIENT
    assert classify_model_error(AuthenticationError("invalid x-api-key")) == ModelErrorKind.FATAL
    assert classify_model_error(ValueError("boom")) == ModelErrorKind.FATAL


def test_policy_delays_are_pure_and_capped():
    policy = RetryPolicy(max_attempts=6, base_delay_s=2.0, max_delay_s=10.0)
    assert [policy.delay_s(a) for a in range(1, 6)] == [2.0, 4.0, 8.0, 10.0, 10.0]
    assert policy.should_retry(ModelErrorKind.THROTTLED, 5)
    assert not policy.should_retry(ModelErrorKind.THROTTLED, 6)  # attempts exhausted
    assert not policy.should_retry(ModelErrorKind.FATAL, 1)
    assert not policy.should_retry(ModelErrorKind.CONTEXT_OVERFLOW, 1)  # Phase 2's job
    assert not policy.should_retry(None, 1)  # unclassified (pre-taxonomy recording)


# -- gateway retry loop ------------------------------------------------------


def _gateway(adapter, event_log, blobs, ledger, sessions, tmp_path, session_id="s-retry"):
    config = make_config(tmp_path, session_id=session_id)
    sessions.create(config)
    return ModelGateway(adapter, event_log, blobs, ledger, config)


_REQ = ModelRequest(model="fake-model", max_tokens=10, messages=[{"role": "user", "content": "x"}])


async def test_throttle_then_success_journals_retries(
    event_log, blobs, ledger, sessions, tmp_path, clock
):
    adapter = _FlakyAdapter(
        [RateLimitError("429"), RateLimitError("429"), scripted_response([{"type": "text", "text": "hi"}])]
    )
    gw = _gateway(adapter, event_log, blobs, ledger, sessions, tmp_path)
    resp = await gw.complete(_REQ)
    assert resp.text == "hi"
    assert clock.mono == 2.0 + 4.0  # journaled backoff actually waited

    events = event_log.read("s-retry")
    retries = [e for e in events if e.event_type == EventType.LLM_RETRY]
    assert [r.payload["attempt"] for r in retries] == [1, 2]
    assert [r.payload["delay_s"] for r in retries] == [2.0, 4.0]
    assert all(r.payload["error_kind"] == "throttled" for r in retries)
    assert sum(1 for e in events if e.event_type == EventType.LLM_REQUEST) == 1
    assert not [e for e in events if e.event_type == EventType.LLM_FAILED]


async def test_exhaustion_fails_with_error_kind(event_log, blobs, ledger, sessions, tmp_path):
    adapter = _FlakyAdapter([RateLimitError("429")] * 4)
    gw = _gateway(adapter, event_log, blobs, ledger, sessions, tmp_path, session_id="s-exhaust")
    with pytest.raises(ModelCallFailed) as exc:
        await gw.complete(_REQ)
    assert exc.value.error_kind == ModelErrorKind.THROTTLED

    events = event_log.read("s-exhaust")
    assert sum(1 for e in events if e.event_type == EventType.LLM_RETRY) == 3  # max_attempts=4
    failed = next(e for e in events if e.event_type == EventType.LLM_FAILED)
    assert failed.payload["error_kind"] == "throttled"
    assert failed.payload["kind"] == "RateLimitError"


async def test_fatal_never_retries(event_log, blobs, ledger, sessions, tmp_path, clock):
    adapter = _FlakyAdapter([AuthenticationError("invalid x-api-key")])
    gw = _gateway(adapter, event_log, blobs, ledger, sessions, tmp_path, session_id="s-fatal")
    with pytest.raises(ModelCallFailed) as exc:
        await gw.complete(_REQ)
    assert exc.value.error_kind == ModelErrorKind.FATAL
    assert clock.mono == 0.0
    events = event_log.read("s-fatal")
    assert not [e for e in events if e.event_type == EventType.LLM_RETRY]
    assert next(e for e in events if e.event_type == EventType.LLM_FAILED).payload["error_kind"] == "fatal"


async def test_context_overflow_not_retried(event_log, blobs, ledger, sessions, tmp_path):
    adapter = _FlakyAdapter([BadRequestError("prompt is too long: 210000 tokens")])
    gw = _gateway(adapter, event_log, blobs, ledger, sessions, tmp_path, session_id="s-overflow")
    with pytest.raises(ModelCallFailed) as exc:
        await gw.complete(_REQ)
    # Not retryable verbatim — Phase 2 routes this kind into compaction.
    assert exc.value.error_kind == ModelErrorKind.CONTEXT_OVERFLOW
    events = event_log.read("s-overflow")
    assert not [e for e in events if e.event_type == EventType.LLM_RETRY]


# -- replay ------------------------------------------------------------------


async def test_replay_reproduces_retry_sequence(db, blobs, event_log, ledger, sessions, clock, tmp_path):
    """A run that throttled and recovered replays byte-identically: the
    ReplayAdapter serves the recorded failed attempts, the gateway re-journals
    the same LLM_RETRY events, and projections match."""
    from parsec.config import Budgets, CacheMode
    from parsec.loop.agent import OrchestratorLoop
    from parsec.replay import run_replay
    from parsec.retrieval.embeddings import EmbeddingCache, HashedNgramEmbedder
    from parsec.retrieval.fetcher import Fetcher
    from parsec.store.coverage import CoverageLedger
    from parsec.store.dag import DagStore
    from parsec.store.documents import DocumentStore
    from parsec.store.notebook import Notebook
    from parsec.store.spans import SpanStore
    from parsec.tools.base import ToolContext, ToolRegistry
    from parsec.tools.fetch import FetchTool
    from parsec.tools.record_premises import RecordPremisesTool
    from parsec.tools.search_within import SearchWithinTool
    from tests.unit.test_agent_gates import decompose_response

    adapter = _FlakyAdapter(
        [
            RateLimitError("429"),  # decomposer, attempt 1: throttled
            decompose_response(["q?"]),
            RateLimitError("429"),  # subagent, attempt 1: throttled
            scripted_response([{"type": "text", "text": "nothing found"}], index=1),
            scripted_response([{"type": "text", "text": "No evidence. [narrative]"}], index=2),
        ]
    )
    # Registry composition must match what run_replay rebuilds (tool schemas
    # feed prompt hashes).
    config = make_config(tmp_path, session_id="s-retry-replay", budgets=Budgets(max_turns=10))
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
    result = await loop.run()
    assert result.status in ("done", "partial")
    original = event_log.read("s-retry-replay")
    assert sum(1 for e in original if e.event_type == EventType.LLM_RETRY) == 2

    outcome = await run_replay(db, blobs, clock, "s-retry-replay")
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.verified
