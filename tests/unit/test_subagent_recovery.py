"""Turn-cap recovery: the final-turn nudge, and record_premises calls made
in the same response as submit_report landing in the DAG instead of being
silently dropped."""

from __future__ import annotations

from parsec import ids
from parsec.config import Budgets
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.gateway.gateway import ModelGateway
from parsec.loop import prompts
from parsec.loop.agent import OrchestratorLoop
from parsec.store.coverage import CoverageLedger
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.notebook import Notebook
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.tools.record_premises import RecordPremisesTool
from tests.conftest import make_config

DOC_TEXT = "Water boils at 100 degrees Celsius at sea level."


def build_loop(tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter, budgets, session_id):
    """A loop whose registry has a live record_premises tool over a one-span corpus."""
    config = make_config(tmp_path, session_id=session_id, budgets=budgets)
    documents = DocumentStore(db, clock)
    spans = SpanStore(db)
    dag = DagStore(db, event_log)

    raw = DOC_TEXT.encode()
    doc_hash = ids.doc_hash(raw)
    blobs.put(raw)
    text_blob = blobs.put(DOC_TEXT)
    documents.put_document(doc_hash, "https://example.test/w", "text/plain", 200, len(raw), text_blob, {})
    s1 = ids.span_id(doc_hash, 0, len(DOC_TEXT))
    spans.put_spans(doc_hash, [(s1, 0, len(DOC_TEXT), DOC_TEXT)])

    tool = RecordPremisesTool(dag, spans, documents)
    gateway = ModelGateway(adapter, event_log, blobs, ledger, config)
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    loop = OrchestratorLoop(
        config, gateway, ToolRegistry([tool]), ctx, sessions,
        dag, spans, documents, CoverageLedger(db, event_log), Notebook(db, event_log, clock),
    )
    return loop, dag, s1


def decompose_response(questions, index=0):
    return scripted_response(
        [{"type": "tool_use", "id": f"tu_dec_{index}", "name": "submit_subquestions",
          "input": {"subquestions": questions}}],
        stop_reason="tool_use",
        index=index,
    )


async def test_final_turn_nudge_injected_before_cap(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    seen_requests = []

    class SpyAdapter(FakeAdapter):
        async def complete(self, request):
            seen_requests.append(request)
            return await super().complete(request)

    adapter = SpyAdapter(
        [
            decompose_response(["what is q?"]),
            # turn 1: still researching
            scripted_response(
                [{"type": "tool_use", "id": "tu_1", "name": "missing_tool", "input": {}}],
                stop_reason="tool_use", index=1,
            ),
            # turn 2 (the last allowed): gives up
            scripted_response([{"type": "text", "text": "gave up"}], stop_reason="end_turn", index=2),
            scripted_response([{"type": "text", "text": "No evidence. [narrative]"}], stop_reason="end_turn", index=3),
        ]
    )
    loop, dag, s1 = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter,
        Budgets(max_turns=10, max_turns_per_subagent=2), session_id="s-nudge",
    )
    await loop.run()

    first_subagent_req, last_subagent_req = seen_requests[1], seen_requests[2]
    assert not any(
        m.get("content") == prompts.FINAL_TURN_NUDGE for m in first_subagent_req.messages
    )
    assert any(
        m.get("content") == prompts.FINAL_TURN_NUDGE for m in last_subagent_req.messages
    )


async def test_record_premises_beside_submit_report_still_lands(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    # Span ids are content-addressed, so the script can name the span upfront.
    s1 = ids.span_id(ids.doc_hash(DOC_TEXT.encode()), 0, len(DOC_TEXT))
    adapter = FakeAdapter(
        [
            decompose_response(["at what temperature does water boil?"]),
            # one response: record the premise AND submit the report
            scripted_response(
                [
                    {"type": "tool_use", "id": "tu_rec", "name": "record_premises",
                     "input": {"premises": [{"text": DOC_TEXT, "span_refs": [s1]}]}},
                    {"type": "tool_use", "id": "tu_sub", "name": "submit_report",
                     "input": {"status": "answered", "summary": "boiling point found"}},
                ],
                stop_reason="tool_use", index=1,
            ),
            scripted_response(
                [{"type": "text", "text": "The evidence is recorded. [narrative]"}],
                stop_reason="end_turn", index=2,
            ),
        ]
    )
    loop, dag, _ = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter,
        Budgets(max_turns=10), session_id="s-beside",
    )
    result = await loop.run()

    premises = dag.nodes_for_session("s-beside", tier=1)
    assert len(premises) == 1
    assert result.coverage == {"sq-1": "answered"}
