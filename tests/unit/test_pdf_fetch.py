"""End-to-end for the PDF gap found in session 20260818T235534-376c: primary
sources that are PDFs must become citable spans, and a fetch that yields no
text must say WHY — to the model in the tool result, and to the reader in
the unused-documents appendix."""

from __future__ import annotations

import httpx
import pytest

from parsec.config import CacheMode
from parsec.models.tools import ToolIntent
from parsec.retrieval.fetcher import Fetcher
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.tools.fetch import FetchTool
from parsec.verify.omission import detect_omissions
from tests.unit.test_extract import make_pdf

PDF_TEXT = "Adaptive trial designs may reduce required sample sizes substantially."


def transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".pdf"):
            return httpx.Response(
                200, content=make_pdf(PDF_TEXT), headers={"content-type": "application/pdf"}
            )
        return httpx.Response(
            200, content=b"\x89PNG\r\n\x1a\n", headers={"content-type": "image/png"}
        )

    return httpx.MockTransport(handler)


@pytest.fixture
def setup(db, blobs, event_log, ledger, sessions, config, clock):
    sessions.create(config)
    documents = DocumentStore(db, clock)
    spans = SpanStore(db)
    fetcher = Fetcher(documents, blobs, clock, CacheMode.RECORD, transport=transport())
    registry = ToolRegistry([FetchTool(fetcher, spans)])
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    return registry, ctx


async def test_pdf_fetch_yields_citable_spans(setup):
    registry, ctx = setup
    result = await registry.dispatch(
        ToolIntent(tool_use_id="t1", tool_name="fetch", input={"url": "https://agency.test/guidance.pdf"}),
        ctx,
    )
    assert result.ok
    assert "1 spans" in result.truncated_text
    assert "Adaptive trial designs" in result.truncated_text  # span preview


async def test_unsupported_fetch_tells_the_model_why(setup):
    registry, ctx = setup
    result = await registry.dispatch(
        ToolIntent(tool_use_id="t2", tool_name="fetch", input={"url": "https://agency.test/figure"}),
        ctx,
    )
    assert result.ok
    assert "No indexable text content — unsupported content type: image/png." in result.truncated_text
    assert "Try an HTML version" in result.truncated_text


async def test_unused_documents_carry_extraction_note(setup, db, event_log, config):
    registry, ctx = setup
    await registry.dispatch(
        ToolIntent(tool_use_id="t3", tool_name="fetch", input={"url": "https://agency.test/guidance.pdf"}),
        ctx,
    )
    await registry.dispatch(
        ToolIntent(tool_use_id="t4", tool_name="fetch", input={"url": "https://agency.test/figure"}),
        ctx,
    )
    # no premises recorded: both documents are unused, for different reasons
    DagStore(db, event_log)  # (dag exists; nothing recorded into it)
    report = detect_omissions(db, event_log, config.session_id)
    by_url = {d["url"]: d for d in report.unused_documents}
    assert "note" not in by_url["https://agency.test/guidance.pdf"]  # extractable, just unmined
    assert by_url["https://agency.test/figure"]["note"] == "unsupported content type: image/png"
