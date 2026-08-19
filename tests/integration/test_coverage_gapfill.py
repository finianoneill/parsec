"""Coverage gap-fill + primary-source audit exit tests (judge-comparison
follow-ups): a run must not stop with subquestions still partial while
budget headroom remains, and a primary source the harness could not read is
a loud coverage failure, never a silent downgrade to secondary evidence.
All at loop level with scripted adapters, no network.
"""

from __future__ import annotations

from parsec import ids
from parsec.config import Budgets, CacheMode, effort_limits
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.gateway.gateway import ModelGateway
from parsec.loop.agent import OrchestratorLoop
from parsec.models.events import EventType
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
import pytest

DOC_TEXT = "The agency guidance states trials must control the Type I error rate."
DOC_URL = "https://example.com/guidance-summary"  # default-tier secondary source
PDF_URL = "https://www.fda.gov/media/78495/download"  # primary tier (.gov)
PDF_NOTE = "unsupported content type: application/pdf"
PREMISE_TEXT = "The agency guidance states trials must control the Type I error rate."


@pytest.fixture
def env(db, blobs, event_log, ledger, sessions, clock):
    """A readable secondary source plus a primary-tier document that yields
    no text (the unminable-PDF shape), both pre-seeded for replay-mode fetch."""
    documents = DocumentStore(db, clock)
    spans = SpanStore(db)

    raw = DOC_TEXT.encode()
    doc_hash = ids.doc_hash(raw)
    blobs.put(raw)
    text_blob = blobs.put(DOC_TEXT)
    documents.put_document(doc_hash, DOC_URL, "text/plain", 200, len(raw), text_blob, {})
    documents.cache_put(ids.cache_key(DOC_URL), DOC_URL, doc_hash, "record")
    span_ref = ids.span_id(doc_hash, 0, len(DOC_TEXT))
    spans.put_spans(doc_hash, [(span_ref, 0, len(DOC_TEXT), DOC_TEXT)])

    pdf_raw = b"%PDF-1.4 fake binary"
    pdf_hash = ids.doc_hash(pdf_raw)
    blobs.put(pdf_raw)
    empty_blob = blobs.put("")
    documents.put_document(
        pdf_hash, PDF_URL, "application/pdf", 200, len(pdf_raw), empty_blob, {"note": PDF_NOTE}
    )
    documents.cache_put(ids.cache_key(PDF_URL), PDF_URL, pdf_hash, "record")

    def make_loop(tmp_path, adapter, session_id, **config_overrides):
        config = make_config(tmp_path, session_id=session_id, **config_overrides)
        gateway = ModelGateway(adapter, event_log, blobs, ledger, config)
        dag = DagStore(db, event_log)
        fetcher = Fetcher(documents, blobs, clock, CacheMode.REPLAY)
        registry = ToolRegistry(
            [
                FetchTool(fetcher, spans),
                RecordPremisesTool(dag, spans, documents),
                SearchWithinTool(spans, EmbeddingCache(db, HashedNgramEmbedder())),
            ]
        )
        ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
        return OrchestratorLoop(
            config, gateway, registry, ctx, sessions, dag, spans, documents,
            CoverageLedger(db, event_log), Notebook(db, event_log, clock),
        )

    return make_loop, span_ref


def _premise_id(span_ref: str) -> str:
    return ids.node_id(
        "Premise", {"text": PREMISE_TEXT, "span_refs": [span_ref], "claim_class": "stable"}
    )


def _research_script(span_ref: str, status: str) -> list:
    """decomposer -> one subagent that records a premise and reports status."""
    return [
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
              "input": {"subquestions": ["summarize the agency guidance"]}}],
            stop_reason="tool_use", index=0),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r", "name": "record_premises",
              "input": {"premises": [{"text": PREMISE_TEXT, "span_refs": [span_ref]}]}}],
            stop_reason="tool_use", index=1),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub", "name": "submit_report",
              "input": {"status": status}}], stop_reason="tool_use", index=2),
    ]


async def test_coverage_gap_fill_retries_partial_subquestion(env, db, blobs, event_log, clock, tmp_path):
    make_loop, span_ref = env
    p_id = _premise_id(span_ref)
    answer = f"Trials must control the Type I error rate. [{p_id}]"
    script = _research_script(span_ref, "partial")
    script += [
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=3),
        # the coverage retry: one subagent answers, then the rewrite
        scripted_response(
            [{"type": "tool_use", "id": "tu_cov", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use", index=4),
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=5),
    ]
    loop = make_loop(tmp_path, FakeAdapter(script), "s-cov",
                     budgets=Budgets(max_gap_rounds=0))
    result = await loop.run()

    assert result.turns == 6
    cov_events = [
        e for e in event_log.read("s-cov")
        if e.event_type == EventType.GAP_FILL_STARTED and e.payload.get("kind") == "coverage"
    ]
    assert len(cov_events) == 1
    assert cov_events[0].payload["target_sq"] == "sq-1"
    assert result.coverage["sq-cov-1"] == "answered"
    # the retry answered, so the original row reads recovered, not still-partial
    assert result.coverage["sq-1"] == "answered"
    row = db.execute(
        "SELECT reason FROM coverage WHERE session_id='s-cov' AND sq_id='sq-1'"
    ).fetchone()
    assert row["reason"] == "recovered by sq-cov-1"
    # the retry prompt names the shortfall
    row = db.execute(
        "SELECT question FROM coverage WHERE session_id='s-cov' AND sq_id='sq-cov-1'"
    ).fetchone()
    assert "PARTIAL" in row["question"]
    # the answer/appendix boundary announces itself
    assert "--- end of answer ---" in result.answer
    assert "not part of the answer" in result.answer

    # the coverage-retried session still replays byte-identically (T4)
    outcome = await run_replay(db, blobs, clock, "s-cov")
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match


async def test_coverage_gap_fill_disabled(env, event_log, tmp_path):
    make_loop, span_ref = env
    p_id = _premise_id(span_ref)
    answer = f"Trials must control the Type I error rate. [{p_id}]"
    script = _research_script(span_ref, "partial")
    script += [scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=3)]
    loop = make_loop(tmp_path, FakeAdapter(script), "s-cov-off",
                     budgets=Budgets(max_gap_rounds=0, max_coverage_gap_rounds=0))
    result = await loop.run()
    assert result.turns == 4
    assert result.coverage["sq-1"] == "partial"
    assert not any(
        e.event_type == EventType.GAP_FILL_STARTED for e in event_log.read("s-cov-off")
    )


async def test_coverage_gap_fill_needs_turn_headroom(env, event_log, tmp_path):
    make_loop, span_ref = env
    p_id = _premise_id(span_ref)
    answer = f"Trials must control the Type I error rate. [{p_id}]"
    script = _research_script(span_ref, "partial")
    script += [scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=3)]
    # 6 turns: after research+writer (4), fewer than MIN_SUBAGENT_TURNS+1
    # remain, so the retry must not dispatch
    loop = make_loop(tmp_path, FakeAdapter(script), "s-cov-tight",
                     budgets=Budgets(max_gap_rounds=0, max_turns=6))
    result = await loop.run()
    assert result.turns == 4
    assert result.coverage["sq-1"] == "partial"
    assert not any(
        e.event_type == EventType.GAP_FILL_STARTED for e in event_log.read("s-cov-tight")
    )


async def test_quick_effort_disables_coverage_gap_fill():
    limits = effort_limits("quick", Budgets())
    assert limits.max_coverage_gap_rounds == 0
    assert effort_limits("deep", Budgets()).max_coverage_gap_rounds == Budgets().max_coverage_gap_rounds


async def test_unreadable_primary_source_demotes_coverage(env, db, event_log, tmp_path):
    make_loop, span_ref = env
    p_id = _premise_id(span_ref)
    answer = f"Trials must control the Type I error rate. [{p_id}]"
    script = [
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
              "input": {"subquestions": ["summarize the agency guidance"]}}],
            stop_reason="tool_use", index=0),
        # the subagent tries the primary PDF (no text), falls back to the
        # secondary page, and reports ANSWERED — the audit must not agree
        scripted_response(
            [{"type": "tool_use", "id": "tu_f1", "name": "fetch", "input": {"url": PDF_URL}}],
            stop_reason="tool_use", index=1),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r", "name": "record_premises",
              "input": {"premises": [{"text": PREMISE_TEXT, "span_refs": [span_ref]}]}}],
            stop_reason="tool_use", index=2),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use", index=3),
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=4),
    ]
    loop = make_loop(tmp_path, FakeAdapter(script), "s-primary",
                     budgets=Budgets(max_gap_rounds=0, max_coverage_gap_rounds=0))
    result = await loop.run()

    assert result.coverage["sq-1"] == "partial"
    row = db.execute(
        "SELECT reason FROM coverage WHERE session_id='s-primary' AND sq_id='sq-1'"
    ).fetchone()
    assert "primary source unreadable" in row["reason"]
    assert PDF_URL in row["reason"]
    # the failure is loud in the appendix, with the extractor's diagnosis
    assert "Primary sources that could not be read" in result.answer
    assert PDF_NOTE in result.answer


async def test_readable_primary_source_not_demoted(env, db, event_log, tmp_path):
    """The audit fires only on unreadable PRIMARY sources: a default-tier
    page that extracted fine must leave an answered row alone."""
    make_loop, span_ref = env
    p_id = _premise_id(span_ref)
    answer = f"Trials must control the Type I error rate. [{p_id}]"
    script = [
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
              "input": {"subquestions": ["summarize the agency guidance"]}}],
            stop_reason="tool_use", index=0),
        scripted_response(
            [{"type": "tool_use", "id": "tu_f1", "name": "fetch", "input": {"url": DOC_URL}}],
            stop_reason="tool_use", index=1),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r", "name": "record_premises",
              "input": {"premises": [{"text": PREMISE_TEXT, "span_refs": [span_ref]}]}}],
            stop_reason="tool_use", index=2),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use", index=3),
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=4),
    ]
    loop = make_loop(tmp_path, FakeAdapter(script), "s-secondary",
                     budgets=Budgets(max_gap_rounds=0, max_coverage_gap_rounds=0))
    result = await loop.run()
    assert result.coverage["sq-1"] == "answered"
    assert "Primary sources that could not be read" not in result.answer
