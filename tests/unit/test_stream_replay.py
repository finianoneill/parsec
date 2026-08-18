"""Per-stream replay serving and journaled model failures (M11)."""

import pytest

from parsec.errors import ModelCallFailed, ReplayDivergence
from parsec.gateway.fake_adapter import StreamFakeAdapter, scripted_response
from parsec.gateway.gateway import ModelGateway
from parsec.gateway.replay_adapter import ReplayAdapter
from parsec.models.events import EventType
from parsec.models.gateway import ModelRequest
from parsec.store.event_log import stream_scope


def _req(text: str) -> ModelRequest:
    return ModelRequest(model="fake-model", max_tokens=10, messages=[{"role": "user", "content": text}])


@pytest.fixture
async def recorded(event_log, blobs, ledger, sessions, config):
    """Record: 1 orchestrator call, 2 sq-1 calls, and an sq-2 call whose
    adapter raises — all through the real gateway."""
    sessions.create(config)
    gateway = ModelGateway(
        StreamFakeAdapter(
            {
                "orchestrator": [scripted_response([{"type": "text", "text": "plan"}])],
                "sq-1": [
                    scripted_response([{"type": "text", "text": "one"}]),
                    scripted_response([{"type": "text", "text": "two"}]),
                ],
                "sq-2": [RuntimeError("boom")],
            }
        ),
        event_log, blobs, ledger, config,
    )
    await gateway.complete(_req("orchestrator prompt"))
    with stream_scope("sq-1"):
        await gateway.complete(_req("sq1 first"))
        await gateway.complete(_req("sq1 second"))
    with stream_scope("sq-2"):
        with pytest.raises(ModelCallFailed) as exc_info:
            await gateway.complete(_req("sq2 doomed"))
    return exc_info.value


async def test_gateway_journals_failures(recorded, event_log, config):
    failure = recorded
    assert failure.kind == "RuntimeError" and failure.detail == "boom"
    failed = [ev for ev in event_log.read(config.session_id) if ev.event_type == EventType.LLM_FAILED]
    assert len(failed) == 1
    assert failed[0].stream_id == "sq-2"
    assert failed[0].payload == {"call_index": 0, "kind": "RuntimeError", "detail": "boom"}


async def test_replay_serves_per_stream_in_any_arrival_order(recorded, event_log, blobs, config):
    adapter = ReplayAdapter(event_log, blobs, config.session_id)
    assert adapter.recorded_calls == 4  # 1 orchestrator + 2 sq-1 + 1 recorded failure

    # sq-1 first this time — arrival order across streams must not matter
    with stream_scope("sq-1"):
        assert (await adapter.complete(_req("sq1 first"))).text == "one"
        assert (await adapter.complete(_req("sq1 second"))).text == "two"
    assert (await adapter.complete(_req("orchestrator prompt"))).text == "plan"


async def test_recorded_failure_replays_identically(recorded, event_log, blobs, config):
    adapter = ReplayAdapter(event_log, blobs, config.session_id)
    with stream_scope("sq-2"):
        with pytest.raises(ModelCallFailed) as exc_info:
            await adapter.complete(_req("sq2 doomed"))
    assert exc_info.value.kind == "RuntimeError"
    assert exc_info.value.detail == "boom"
    assert str(exc_info.value) == str(recorded)  # byte-identical failure text


async def test_prompt_divergence_names_the_stream(recorded, event_log, blobs, config):
    adapter = ReplayAdapter(event_log, blobs, config.session_id)
    with stream_scope("sq-1"):
        with pytest.raises(ReplayDivergence, match="stream sq-1"):
            await adapter.complete(_req("a different prompt"))


async def test_exhausted_stream_diverges(recorded, event_log, blobs, config):
    adapter = ReplayAdapter(event_log, blobs, config.session_id)
    with stream_scope("sq-9"):
        with pytest.raises(ReplayDivergence, match="exhausted for stream 'sq-9'"):
            await adapter.complete(_req("never recorded"))


async def test_fake_adapter_is_stream_keyed():
    adapter = StreamFakeAdapter({"sq-1": [scripted_response([{"type": "text", "text": "a"}])]})
    with stream_scope("sq-1"):
        assert (await adapter.complete(_req("x"))).text == "a"
        with pytest.raises(IndexError, match="sq-1"):
            await adapter.complete(_req("x"))
