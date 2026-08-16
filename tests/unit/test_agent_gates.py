"""Stop-condition gate behavior of the single-agent loop."""

from __future__ import annotations

import pytest

from parsec.config import Budgets
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.gateway.gateway import ModelGateway
from parsec.loop.agent import SingleAgentLoop
from parsec.loop.prompts import FORCED_ANSWER_NUDGE
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from tests.conftest import make_config


def build_loop(tmp_path, db, blobs, event_log, ledger, sessions, clock, responses, budgets):
    config = make_config(tmp_path, session_id="s-gate", budgets=budgets)
    gateway = ModelGateway(FakeAdapter(responses), event_log, blobs, ledger, config)
    registry = ToolRegistry([])
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    documents = DocumentStore(db, clock)
    spans = SpanStore(db)
    dag = DagStore(db, event_log)
    return SingleAgentLoop(config, gateway, registry, ctx, sessions, dag, spans, documents)


async def test_max_turns_forces_partial_answer(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    # max_turns=1 → the very first gate forces the answer nudge; the model's
    # reply (even without citations... it has none needed if narrative) ends the run as partial.
    responses = [
        scripted_response(
            [{"type": "text", "text": "I could not research anything. [narrative]"}],
            stop_reason="end_turn",
        )
    ]
    loop = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, responses, Budgets(max_turns=1)
    )
    result = await loop.run()
    assert result.status == "partial"
    assert result.turns == 1


async def test_token_budget_forces_answer(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    responses = [
        scripted_response(
            [{"type": "tool_use", "id": "t1", "name": "nope", "input": {}}],
            stop_reason="tool_use",
            input_tokens=500_000,  # blows the token cap in one call
        ),
        scripted_response(
            [{"type": "text", "text": "Out of budget. [narrative]"}], stop_reason="end_turn"
        ),
    ]
    loop = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, responses, Budgets(max_turns=10)
    )
    result = await loop.run()
    assert result.status == "partial"
    assert result.turns == 2  # second call was the forced answer


async def test_forced_answer_nudge_injected_once(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    adapter_seen: list[list[dict]] = []

    class SpyAdapter(FakeAdapter):
        async def complete(self, request):
            adapter_seen.append(request.messages)
            return await super().complete(request)

    responses = [
        scripted_response([{"type": "text", "text": "Done. [narrative]"}], stop_reason="end_turn")
    ]
    config = make_config(tmp_path, session_id="s-nudge", budgets=Budgets(max_turns=1))
    gateway = ModelGateway(SpyAdapter(responses), event_log, blobs, ledger, config)
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    loop = SingleAgentLoop(
        config, gateway, ToolRegistry([]), ctx, sessions,
        DagStore(db, event_log), SpanStore(db), DocumentStore(db, clock),
    )
    await loop.run()
    assert adapter_seen[0][-1]["content"] == FORCED_ANSWER_NUDGE
