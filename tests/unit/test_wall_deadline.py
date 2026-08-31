"""In-call wall-clock deadline: the gateway cuts a live model call at the
remaining wall budget and journals it as a FATAL WallClockExceeded failure,
so one wedged call can no longer overrun max_wall_seconds by the whole
request timeout. Replay and unhooked gateways are unaffected."""

from __future__ import annotations

import asyncio

import pytest

from parsec.errors import ModelCallFailed, ModelErrorKind
from parsec.gateway.fake_adapter import scripted_response
from parsec.gateway.gateway import WALL_CLOCK_KIND, ModelGateway
from parsec.models.events import EventType
from parsec.models.gateway import ModelRequest
from tests.conftest import make_config

_REQ = ModelRequest(model="fake-model", max_tokens=10, messages=[{"role": "user", "content": "x"}])


class _SlowAdapter:
    """Sleeps `delay` real seconds, then answers. Records whether the sleep
    was cancelled (the deadline must not leave the call running)."""

    def __init__(self, delay: float):
        self.delay = delay
        self.calls = 0
        self.cancelled = False

    async def complete(self, request):
        self.calls += 1
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return scripted_response([{"type": "text", "text": "late"}])


class _TimeoutAdapter:
    """An adapter whose OWN timeout fires (the SDK's request timeout)."""

    calls = 0

    async def complete(self, request):
        self.calls += 1
        raise TimeoutError("request timed out")


def _gateway(model_adapter, event_log, blobs, ledger, sessions, tmp_path, sid, **overrides):
    config = make_config(tmp_path, session_id=sid, **overrides)
    sessions.create(config)
    return ModelGateway(model_adapter, event_log, blobs, ledger, config)


def _failures(event_log, sid):
    return [e for e in event_log.read(sid) if e.event_type == EventType.LLM_FAILED]


async def test_inflight_call_is_cut_at_the_wall_budget(event_log, blobs, ledger, sessions, tmp_path):
    adapter = _SlowAdapter(delay=5.0)
    gw = _gateway(adapter, event_log, blobs, ledger, sessions, tmp_path, "s-wall")
    gw.wall_budget = lambda: (299.9, 300.0)  # 0.1s left

    with pytest.raises(ModelCallFailed) as info:
        await gw.complete(_REQ)
    assert info.value.kind == WALL_CLOCK_KIND
    assert info.value.error_kind == ModelErrorKind.FATAL
    assert adapter.calls == 1 and adapter.cancelled

    (failed,) = _failures(event_log, "s-wall")
    assert failed.payload["kind"] == WALL_CLOCK_KIND
    assert failed.payload["error_kind"] == "fatal"
    assert failed.payload["detail"] == "wall-clock budget (300s) exhausted mid-call"
    # FATAL: no retry was journaled or attempted.
    assert not [e for e in event_log.read("s-wall") if e.event_type == EventType.LLM_RETRY]


async def test_exhausted_budget_fails_before_calling(event_log, blobs, ledger, sessions, tmp_path):
    adapter = _SlowAdapter(delay=0.0)
    gw = _gateway(adapter, event_log, blobs, ledger, sessions, tmp_path, "s-spent")
    gw.wall_budget = lambda: (301.0, 300.0)
    with pytest.raises(ModelCallFailed) as info:
        await gw.complete(_REQ)
    assert info.value.kind == WALL_CLOCK_KIND
    assert adapter.calls == 0
    assert len(_failures(event_log, "s-spent")) == 1


async def test_adapter_timeout_stays_transient_and_retryable(
    event_log, blobs, ledger, sessions, tmp_path
):
    """The SDK's request timeout is the adapter's error, not a wall breach:
    it classifies TRANSIENT and goes through the journaled retry loop."""
    adapter = _TimeoutAdapter()
    gw = _gateway(adapter, event_log, blobs, ledger, sessions, tmp_path, "s-sdk")
    gw.wall_budget = lambda: (0.0, 300.0)
    with pytest.raises(ModelCallFailed) as info:
        await gw.complete(_REQ)
    assert info.value.kind == "TimeoutError"
    assert info.value.error_kind == ModelErrorKind.TRANSIENT
    retries = [e for e in event_log.read("s-sdk") if e.event_type == EventType.LLM_RETRY]
    assert len(retries) == 3 and adapter.calls == 4


async def test_no_hook_or_replay_means_no_deadline(event_log, blobs, ledger, sessions, tmp_path):
    fast = _SlowAdapter(delay=0.01)
    gw = _gateway(fast, event_log, blobs, ledger, sessions, tmp_path, "s-nohook")
    assert (await gw.complete(_REQ)).text == "late"  # wall_budget unset

    replayed = _SlowAdapter(delay=0.01)
    gw = _gateway(replayed, event_log, blobs, ledger, sessions, tmp_path, "s-replay", adapter="replay")
    gw.wall_budget = lambda: (301.0, 300.0)  # would fail if honored
    assert (await gw.complete(_REQ)).text == "late"
