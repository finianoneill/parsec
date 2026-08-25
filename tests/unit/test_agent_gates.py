"""Stop-condition gates, coverage resolution, and the writer context firewall."""

from __future__ import annotations

import pytest

from parsec.config import Budgets
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.gateway.gateway import ModelGateway
from parsec.loop.agent import OrchestratorLoop
from parsec.loop.prompts import DECOMPOSER_SYSTEM, WRITER_SYSTEM
from parsec.store.coverage import CoverageLedger
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.notebook import Notebook
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from tests.conftest import make_config


def build_loop(tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter, budgets, session_id="s-gate"):
    config = make_config(tmp_path, session_id=session_id, budgets=budgets)
    gateway = ModelGateway(adapter, event_log, blobs, ledger, config)
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    return OrchestratorLoop(
        config, gateway, ToolRegistry([]), ctx, sessions,
        DagStore(db, event_log), SpanStore(db), DocumentStore(db, clock),
        CoverageLedger(db, event_log), Notebook(db, event_log, clock),
    )


def decompose_response(questions, index=0):
    return scripted_response(
        [
            {
                "type": "tool_use",
                "id": f"tu_dec_{index}",
                "name": "submit_subquestions",
                "input": {"subquestions": questions},
            }
        ],
        stop_reason="tool_use",
        index=index,
    )


async def test_max_turns_skips_everything_ends_partial(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    # max_turns=1 → planning gate trips (no decomposer call), dispatch gate
    # trips, sq-1 is explicitly blocked, writer still runs → partial.
    adapter = FakeAdapter(
        [scripted_response([{"type": "text", "text": "Nothing gathered. [narrative]"}], stop_reason="end_turn")]
    )
    loop = build_loop(tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter, Budgets(max_turns=1))
    result = await loop.run()
    assert result.status == "partial"
    assert result.turns == 1  # writer only
    assert result.coverage == {"sq-1": "blocked"}
    rows = CoverageLedger(db, event_log).all(loop.config.session_id)
    assert rows[0]["reason"] == "turn budget (1) exhausted before dispatch"


async def test_token_budget_blocks_remaining_subquestions(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    adapter = FakeAdapter(
        [
            # decomposer call blows the token cap
            scripted_response(
                [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
                  "input": {"subquestions": ["part one?", "part two?"]}}],
                stop_reason="tool_use",
                input_tokens=500_000,
            ),
            scripted_response([{"type": "text", "text": "Out of budget. [narrative]"}], stop_reason="end_turn"),
        ]
    )
    loop = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter,
        Budgets(max_turns=10), session_id="s-tokens",
    )
    result = await loop.run()
    assert result.status == "partial"
    assert result.turns == 2  # decomposer + writer
    assert result.coverage == {"sq-1": "blocked", "sq-2": "blocked"}
    rows = CoverageLedger(db, event_log).all("s-tokens")
    assert all("token budget (200000) exhausted before dispatch" == r["reason"] for r in rows)


async def test_invalid_decomposition_falls_back_to_whole_query(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    adapter = FakeAdapter(
        [
            scripted_response([{"type": "text", "text": "here are my thoughts..."}], stop_reason="end_turn"),
            # the structured repair round (Phase 3) also fails to call the tool
            scripted_response([{"type": "text", "text": "still just thoughts"}], stop_reason="end_turn", index=1),
            # subagent for sq-1 ends immediately without a report
            scripted_response([{"type": "text", "text": "cannot research"}], stop_reason="end_turn"),
            scripted_response([{"type": "text", "text": "No evidence found. [narrative]"}], stop_reason="end_turn"),
        ]
    )
    loop = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter,
        Budgets(max_turns=10), session_id="s-fallback",
    )
    result = await loop.run()
    rows = CoverageLedger(db, event_log).all("s-fallback")
    assert len(rows) == 1
    assert rows[0]["question"] == "test query"  # the whole query
    assert rows[0]["status"] == "blocked"
    assert "without calling submit_report" in rows[0]["reason"]


async def test_writer_sees_only_query_evidence_and_gaps(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    """§6.5/T6 firewall: decomposer sees only the query; the writer request
    contains no research transcript."""
    seen_requests = []

    class SpyAdapter(FakeAdapter):
        async def complete(self, request):
            seen_requests.append(request)
            return await super().complete(request)

    adapter = SpyAdapter(
        [
            decompose_response(["what is the boiling point?"]),
            scripted_response([{"type": "text", "text": "found nothing useful"}], stop_reason="end_turn", index=1),
            scripted_response([{"type": "text", "text": "The evidence is insufficient. [narrative]"}], stop_reason="end_turn", index=2),
        ]
    )
    loop = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter,
        Budgets(max_turns=10), session_id="s-firewall",
    )
    result = await loop.run()
    assert result.coverage == {"sq-1": "blocked"}

    decomposer_req, subagent_req, writer_req = seen_requests
    assert decomposer_req.system[0]["text"] == DECOMPOSER_SYSTEM
    assert decomposer_req.messages == [{"role": "user", "content": "test query"}]

    assert writer_req.system[0]["text"] == WRITER_SYSTEM
    assert writer_req.tools == []
    assert len(writer_req.messages) == 1
    content = writer_req.messages[0]["content"]
    assert content.startswith("Question:")
    assert "Unresolved subquestions" in content
    assert "found nothing useful" not in content  # no subagent transcript leaks


async def test_fair_share_turn_split_dispatches_every_subquestion(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    """8-turn budget, 6-turn subagent cap, 2 subquestions: v1 let sq-1 run to
    its full cap and starved sq-2 into "blocked before dispatch"; the fair
    share gives each subquestion its slice of the remaining turns."""
    responses = [decompose_response(["part one?", "part two?"])]
    for i in range(6):  # each subagent burns its slice on unknown tools
        responses.append(
            scripted_response(
                [{"type": "tool_use", "id": f"tu_{i}", "name": "missing_tool", "input": {}}],
                stop_reason="tool_use", index=i + 1,
            )
        )
    responses.append(
        scripted_response([{"type": "text", "text": "Ran dry. [narrative]"}], stop_reason="end_turn", index=9)
    )
    adapter = FakeAdapter(responses)
    loop = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter,
        Budgets(max_turns=8, max_turns_per_subagent=6), session_id="s-fair",
    )
    result = await loop.run()
    rows = CoverageLedger(db, event_log).all("s-fair")
    # both dispatched — neither is "blocked before dispatch"
    assert [r["status"] for r in rows] == ["blocked", "blocked"]
    assert rows[0]["reason"] == "subagent turn cap (3) reached before submit_report"
    # sq-2's slice ends exactly where the global budget does; the global
    # gate is checked first, so its (equally true) reason wins.
    assert rows[1]["reason"] == "turn budget (8) exhausted before submit_report"
    assert result.turns == 8  # decomposer + 3 + 3 + writer


async def test_subagent_turn_cap_marks_partial(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    # Subagent keeps "researching" past its per-subagent cap.
    endless = [decompose_response(["what is q?"])]
    for i in range(3):
        endless.append(
            scripted_response(
                [{"type": "tool_use", "id": f"tu_{i}", "name": "missing_tool", "input": {}}],
                stop_reason="tool_use",
                index=i + 1,
            )
        )
    endless.append(
        scripted_response([{"type": "text", "text": "Ran out. [narrative]"}], stop_reason="end_turn", index=9)
    )
    adapter = FakeAdapter(endless)
    budgets = Budgets(max_turns=20, max_turns_per_subagent=3)
    loop = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter, budgets, session_id="s-cap"
    )
    result = await loop.run()
    rows = CoverageLedger(db, event_log).all("s-cap")
    assert rows[0]["status"] == "blocked"
    assert "cap" in rows[0]["reason"]
