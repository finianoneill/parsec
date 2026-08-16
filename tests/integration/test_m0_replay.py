"""M0 exit test: a recorded gateway session replays byte-identically.

Uses a minimal scripted driver (two model calls with an event-logged tool
step between them) — the full agent loop arrives at M1; this test pins the
record/replay substrate underneath it.
"""

from __future__ import annotations

import json

import pytest

from parsec.canonical import canonical_json
from parsec.errors import ReplayDivergence
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.gateway.gateway import ModelGateway
from parsec.gateway.replay_adapter import ReplayAdapter
from parsec.models.events import EventType
from parsec.models.gateway import ModelRequest
from tests.conftest import make_config


def _scripted_responses():
    return [
        scripted_response(
            [{"type": "tool_use", "id": "tu_1", "name": "echo", "input": {"text": "ping"}}],
            stop_reason="tool_use",
            index=0,
        ),
        scripted_response([{"type": "text", "text": "final answer"}], stop_reason="end_turn", index=1),
    ]


async def _drive(gateway: ModelGateway, event_log, sessions, config):
    """Minimal deterministic loop: call model, log a tool result, call again."""
    sid = config.session_id
    sessions.create(config)
    event_log.append(
        sid,
        "harness",
        EventType.SESSION_STARTED,
        {"session_id": sid, "query": config.query, "config": config.to_json_dict()},
    )
    messages: list[dict] = [{"role": "user", "content": config.query}]
    req = ModelRequest(model=config.model, max_tokens=100, messages=list(messages))
    resp = await gateway.complete(req)
    assert resp.stop_reason == "tool_use"
    tool_use = resp.tool_uses[0]
    event_log.append(sid, "harness", EventType.TOOL_INTENT, {"tool_use_id": tool_use["id"], "tool_name": tool_use["name"], "input": tool_use["input"]})
    tool_output = canonical_json({"echoed": tool_use["input"]["text"]})
    event_log.append(sid, "tool:echo", EventType.TOOL_RESULT, {"tool_use_id": tool_use["id"], "ok": True, "output": tool_output})
    messages.append({"role": "assistant", "content": resp.content})
    messages.append(
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": tool_output}]}
    )
    req2 = ModelRequest(model=config.model, max_tokens=100, messages=list(messages))
    resp2 = await gateway.complete(req2)
    answer_blob = gateway.blobs.put(resp2.text)
    event_log.append(sid, "harness", EventType.ANSWER_EMITTED, {"answer_blob": answer_blob})
    event_log.append(sid, "harness", EventType.SESSION_FINISHED, {"status": "done"})
    return answer_blob


async def test_record_then_replay_byte_identical(db, event_log, blobs, ledger, sessions, tmp_path):
    cfg_a = make_config(tmp_path, session_id="s-orig")
    gw_a = ModelGateway(FakeAdapter(_scripted_responses()), event_log, blobs, ledger, cfg_a)
    answer_a = await _drive(gw_a, event_log, sessions, cfg_a)

    cfg_b = make_config(tmp_path, session_id="s-replay")
    replay_adapter = ReplayAdapter(event_log, blobs, "s-orig")
    gw_b = ModelGateway(replay_adapter, event_log, blobs, ledger, cfg_b)
    answer_b = await _drive(gw_b, event_log, sessions, cfg_b)

    assert event_log.projection("s-orig") == event_log.projection("s-replay")
    assert blobs.get(answer_a) == blobs.get(answer_b)


async def test_divergence_detected(db, event_log, blobs, ledger, sessions, tmp_path):
    cfg_a = make_config(tmp_path, session_id="s-orig2")
    gw_a = ModelGateway(FakeAdapter(_scripted_responses()), event_log, blobs, ledger, cfg_a)
    await _drive(gw_a, event_log, sessions, cfg_a)

    cfg_b = make_config(tmp_path, session_id="s-replay2", query="a DIFFERENT query")
    replay_adapter = ReplayAdapter(event_log, blobs, "s-orig2")
    gw_b = ModelGateway(replay_adapter, event_log, blobs, ledger, cfg_b)
    with pytest.raises(ReplayDivergence) as exc:
        await _drive(gw_b, event_log, sessions, cfg_b)
    assert exc.value.call_index == 0
    assert "DIFFERENT" in exc.value.diff


async def test_replay_exhaustion_detected(db, event_log, blobs, ledger, sessions, tmp_path):
    cfg_a = make_config(tmp_path, session_id="s-orig3")
    gw_a = ModelGateway(FakeAdapter(_scripted_responses()), event_log, blobs, ledger, cfg_a)
    await _drive(gw_a, event_log, sessions, cfg_a)

    cfg_b = make_config(tmp_path, session_id="s-replay3")
    replay_adapter = ReplayAdapter(event_log, blobs, "s-orig3")
    gw_b = ModelGateway(replay_adapter, event_log, blobs, ledger, cfg_b)
    await _drive(gw_b, event_log, sessions, cfg_b)
    req = ModelRequest(model="fake-model", max_tokens=10, messages=[{"role": "user", "content": "extra"}])
    with pytest.raises(ReplayDivergence):
        await gw_b.complete(req)
