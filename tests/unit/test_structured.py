"""Phase 3: the unified structured-output contract — per-field repair for
the brief, submit_report, and the judge channels."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, ValidationError

from parsec.config import Budgets
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.loop.structured import (
    BriefSubmission,
    StructuredOutcome,
    format_validation_errors,
    parse_prose_json,
    structured_call,
    validate_tool_call,
)
from parsec.models.events import EventType
from tests.unit.test_agent_gates import build_loop, decompose_response


class _Point(BaseModel):
    x: int
    y: int = Field(ge=0)


def _tool_resp(name: str, tool_input: dict, index: int = 0):
    return scripted_response(
        [{"type": "tool_use", "id": f"tu_{index}", "name": name, "input": tool_input}],
        stop_reason="tool_use",
        index=index,
    )


# -- primitives --------------------------------------------------------------


def test_format_validation_errors_names_fields():
    with pytest.raises(ValidationError) as exc:
        _Point.model_validate({"y": -1})
    lines = format_validation_errors(exc.value)
    assert any(line.startswith("x:") for line in lines)
    assert any(line.startswith("y:") for line in lines)


def test_validate_tool_call_paths():
    ok, block, problems = validate_tool_call(_tool_resp("plot", {"x": 1, "y": 2}), "plot", _Point)
    assert ok is not None and block["id"] == "tu_0" and problems == []

    missing, block, problems = validate_tool_call(
        scripted_response([{"type": "text", "text": "hi"}]), "plot", _Point
    )
    assert missing is None and block is None
    assert problems == ["the response did not call plot"]

    invalid, block, problems = validate_tool_call(_tool_resp("plot", {"x": "no"}), "plot", _Point)
    assert invalid is None and block is not None
    assert any(p.startswith("x:") for p in problems)


async def test_structured_call_repairs_with_field_errors_and_forced_tool():
    calls = []

    async def complete(messages, tool_choice):
        calls.append((messages, tool_choice))
        if len(calls) == 1:
            return _tool_resp("plot", {"x": "no"}, index=0)
        return _tool_resp("plot", {"x": 1, "y": 2}, index=1)

    outcome = await structured_call(
        complete, [{"role": "user", "content": "plot it"}], _Point, "plot", "Call plot again."
    )
    assert isinstance(outcome, StructuredOutcome)
    assert outcome.value is not None and outcome.repairs == 1

    repair_messages, forced = calls[1]
    assert forced == {"type": "tool", "name": "plot"}  # final attempt is pinned
    corrective = repair_messages[-1]["content"][0]
    assert corrective["type"] == "tool_result" and corrective["is_error"]
    assert "x:" in corrective["content"]  # the per-field detail reached the model


async def test_structured_call_exhaustion_returns_problems():
    async def complete(messages, tool_choice):
        return scripted_response([{"type": "text", "text": "nope"}])

    outcome = await structured_call(
        complete, [{"role": "user", "content": "plot it"}], _Point, "plot", "Call plot."
    )
    assert outcome.value is None and outcome.repairs == 1
    assert outcome.problems == ["the response did not call plot"]


def test_brief_submission_keeps_old_leniency():
    sub = BriefSubmission.model_validate(
        {"scope": "  s  ", "effort": "heroic", "subquestions": [" what? "], "extra": 1}
    )
    assert sub.scope == "s"
    assert sub.effort == "deep"  # invalid estimate never clamps
    assert sub.subquestions == ["what?"]
    with pytest.raises(ValidationError):
        BriefSubmission.model_validate({"subquestions": ["q?"]})  # too short
    with pytest.raises(ValidationError):
        BriefSubmission.model_validate({"subquestions": []})


def test_parse_prose_json_matches_old_judge_behavior():
    class Reply(BaseModel):
        synthesis_score: float = Field(ge=1, le=5)
        rationale: str = ""

    ok, _ = parse_prose_json('prose then {"synthesis_score": 4} trailing', Reply)
    assert ok is not None and ok.synthesis_score == 4
    bad, problems = parse_prose_json('{"synthesis_score": 9}', Reply)
    assert bad is None and any("synthesis_score" in p for p in problems)
    none, problems = parse_prose_json("not json at all", Reply)
    assert none is None and problems == ["no JSON object found in the reply"]


# -- the brief through the loop ----------------------------------------------


class SpyAdapter(FakeAdapter):
    def __init__(self, responses):
        super().__init__(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return await super().complete(request)


async def test_invalid_brief_is_repaired_not_swallowed(
    tmp_path, db, blobs, event_log, ledger, sessions, clock
):
    adapter = SpyAdapter(
        [
            scripted_response([{"type": "text", "text": "thinking out loud..."}], stop_reason="end_turn"),
            decompose_response(["what is the boiling point?"], index=1),
            scripted_response([{"type": "text", "text": "gave up"}], stop_reason="end_turn", index=2),
            scripted_response([{"type": "text", "text": "No evidence. [narrative]"}], stop_reason="end_turn", index=3),
        ]
    )
    loop = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter,
        Budgets(max_turns=10), session_id="s-brief-repair",
    )
    await loop.run()

    brief_ev = next(
        e for e in event_log.read("s-brief-repair") if e.event_type == EventType.RESEARCH_BRIEF
    )
    # the REPAIRED decomposition won — no silent whole-query fallback
    assert brief_ev.payload["subquestions"] == ["what is the boiling point?"]

    repair_req = adapter.requests[1]
    assert repair_req.tool_choice == {"type": "tool", "name": "submit_subquestions"}
    assert "did not call submit_subquestions" in str(repair_req.messages[-1]["content"])


async def test_submit_report_corrective_names_the_bad_premise(
    tmp_path, db, blobs, event_log, ledger, sessions, clock
):
    bad_submit = scripted_response(
        [{"type": "tool_use", "id": "tu_sub", "name": "submit_report",
          "input": {"status": "answered",
                    "findings": [{"text": "derived fact", "premise_ids": ["premise:deadbeef00000000"],
                                  "edge_type": "deduces"}]}}],
        stop_reason="tool_use", index=1,
    )
    adapter = SpyAdapter(
        [
            decompose_response(["what is q about?"]),
            bad_submit,
            scripted_response([{"type": "text", "text": "giving up"}], stop_reason="end_turn", index=2),
            scripted_response([{"type": "text", "text": "No evidence. [narrative]"}], stop_reason="end_turn", index=3),
        ]
    )
    loop = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter,
        Budgets(max_turns=10), session_id="s-submit-detail",
    )
    await loop.run()

    # the corrective the subagent saw on its next turn names the problem
    followup = adapter.requests[2].messages[-1]["content"][0]
    assert followup["type"] == "tool_result" and followup["is_error"]
    assert "findings[0].premise_ids" in followup["content"]
    assert "premise:deadbeef00000000" in followup["content"]
    assert "record_premises" in followup["content"]


# -- replay ------------------------------------------------------------------


async def test_repaired_brief_run_replays(db, blobs, event_log, ledger, sessions, clock, tmp_path):
    from parsec.config import CacheMode
    from parsec.gateway.gateway import ModelGateway
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
    from tests.conftest import make_config

    adapter = FakeAdapter(
        [
            scripted_response([{"type": "text", "text": "hmm..."}], stop_reason="end_turn"),
            decompose_response(["what is the answer?"], index=1),
            scripted_response([{"type": "text", "text": "nothing found"}], stop_reason="end_turn", index=2),
            scripted_response([{"type": "text", "text": "No evidence. [narrative]"}], stop_reason="end_turn", index=3),
        ]
    )
    config = make_config(tmp_path, session_id="s-repair-replay", budgets=Budgets(max_turns=10))
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

    # the forced-tool_choice repair request replays byte-identically too
    outcome = await run_replay(db, blobs, clock, "s-repair-replay")
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.verified
