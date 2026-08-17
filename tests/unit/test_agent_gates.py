"""Stop-condition gates and the writer context firewall of the agent loop."""

from __future__ import annotations

import pytest

from parsec.config import Budgets
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.gateway.gateway import ModelGateway
from parsec.loop.agent import SingleAgentLoop
from parsec.loop.prompts import WRITER_SYSTEM
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from tests.conftest import make_config


def build_loop(tmp_path, db, blobs, event_log, ledger, sessions, clock, responses, budgets, session_id="s-gate"):
    config = make_config(tmp_path, session_id=session_id, budgets=budgets)
    gateway = ModelGateway(FakeAdapter(responses), event_log, blobs, ledger, config)
    registry = ToolRegistry([])
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    documents = DocumentStore(db, clock)
    spans = SpanStore(db)
    dag = DagStore(db, event_log)
    return SingleAgentLoop(config, gateway, registry, ctx, sessions, dag, spans, documents)


async def test_max_turns_skips_research_ends_partial(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    # max_turns=1 → the research gate trips before any research call; the
    # writer still runs (over zero premises) and the run ends partial.
    responses = [
        scripted_response(
            [{"type": "text", "text": "No research was possible. [narrative]"}],
            stop_reason="end_turn",
        )
    ]
    loop = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, responses, Budgets(max_turns=1)
    )
    result = await loop.run()
    assert result.status == "partial"
    assert result.turns == 1
    assert result.claims_total == 0


async def test_token_budget_forces_writer(tmp_path, db, blobs, event_log, ledger, sessions, clock):
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
        tmp_path, db, blobs, event_log, ledger, sessions, clock, responses,
        Budgets(max_turns=10), session_id="s-tokens",
    )
    result = await loop.run()
    assert result.status == "partial"
    assert result.turns == 2  # research call + writer call


async def test_writer_sees_only_query_and_premises(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    """§6.5 firewall: the writer request must not contain the research transcript."""
    seen_requests = []

    class SpyAdapter(FakeAdapter):
        async def complete(self, request):
            seen_requests.append(request)
            return await super().complete(request)

    responses = [
        scripted_response([{"type": "text", "text": "found things"}], stop_reason="end_turn"),
        scripted_response([{"type": "text", "text": "Nothing citable. [narrative]"}], stop_reason="end_turn"),
    ]
    config = make_config(tmp_path, session_id="s-firewall", budgets=Budgets(max_turns=10))
    gateway = ModelGateway(SpyAdapter(responses), event_log, blobs, ledger, config)
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    loop = SingleAgentLoop(
        config, gateway, ToolRegistry([]), ctx, sessions,
        DagStore(db, event_log), SpanStore(db), DocumentStore(db, clock),
    )
    result = await loop.run()
    assert result.status == "done"

    research_req, writer_req = seen_requests
    assert writer_req.tools == []
    assert writer_req.system[0]["text"] == WRITER_SYSTEM
    assert len(writer_req.messages) == 1
    content = writer_req.messages[0]["content"]
    assert content.startswith("Question:")
    assert "Premises:" in content
    assert "found things" not in content  # research transcript is not carried over
