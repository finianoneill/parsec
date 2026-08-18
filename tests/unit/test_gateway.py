import json

import pytest

from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.gateway.gateway import ModelGateway
from parsec.models.events import EventType
from parsec.models.gateway import ModelRequest


@pytest.fixture
def gateway(event_log, blobs, ledger, sessions, config):
    sessions.create(config)
    adapter = FakeAdapter(
        [scripted_response([{"type": "text", "text": "hello"}], input_tokens=100, output_tokens=50)]
    )
    return ModelGateway(adapter, event_log, blobs, ledger, config)


async def test_gateway_records_events_blobs_and_debits(gateway, event_log, blobs, ledger, config):
    req = ModelRequest(model="fake-model", max_tokens=100, messages=[{"role": "user", "content": "hi"}])
    resp = await gateway.complete(req)
    assert resp.text == "hello"

    events = event_log.read(config.session_id)
    types = [e.event_type for e in events]
    assert EventType.LLM_REQUEST in types
    assert EventType.LLM_RESPONSE in types

    req_ev = next(e for e in events if e.event_type == EventType.LLM_REQUEST)
    assert req_ev.payload["prompt_hash"] == req.prompt_hash
    stored = json.loads(blobs.get_text(req_ev.payload["request_blob"]))
    assert stored["messages"] == [{"role": "user", "content": "hi"}]

    resp_ev = next(e for e in events if e.event_type == EventType.LLM_RESPONSE)
    assert blobs.exists(resp_ev.payload["response_blob"])

    totals = ledger.totals(config.session_id)
    assert totals["input_tokens"] == 100
    assert totals["output_tokens"] == 50
    assert "usd" not in totals  # fake-model is free; zero debits are skipped


async def test_call_index_increments_per_stream(gateway, config):
    """M11: call indices are per-stream — replay keys on (stream, index)."""
    from parsec.store.event_log import stream_scope

    gateway.adapter = FakeAdapter(
        [scripted_response([{"type": "text", "text": t}]) for t in ("a", "b", "c")]
    )
    req = ModelRequest(model="fake-model", max_tokens=10, messages=[{"role": "user", "content": "x"}])
    await gateway.complete(req)
    await gateway.complete(req)
    with stream_scope("sq-1"):
        await gateway.complete(req)
    assert gateway.call_indices["orchestrator"] == 2
    assert gateway.call_indices["sq-1"] == 1
