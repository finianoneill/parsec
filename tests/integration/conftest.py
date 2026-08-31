"""Integration-level fixtures shared by the M14 (diachronic research)
suites: the mutable fake web, its transport, and one recorded parent
session to diff/refresh/watch against. Corpus constants and scripts live in
m14_corpus.py so test modules can import them without shadowing fixtures."""

from __future__ import annotations

import httpx
import pytest

from parsec.config import Budgets, CacheMode
from parsec.gateway.fake_adapter import FakeAdapter
from parsec.gateway.gateway import ModelGateway
from parsec.loop.agent import OrchestratorLoop
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
from tests.integration.m14_corpus import SENT_A, SENT_B1, URL_A, URL_B, _parent_script

@pytest.fixture
def pages() -> dict[str, str]:
    """Mutable fake web: the price page changes between parent and refresh."""
    return {URL_A: SENT_A, URL_B: SENT_B1}


@pytest.fixture
def transport(pages) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in pages:
            return httpx.Response(
                200, text=pages[url], headers={"content-type": "text/plain"}
            )
        return httpx.Response(404, text="not found")

    return httpx.MockTransport(handler)


@pytest.fixture
async def parent(db, blobs, event_log, ledger, sessions, clock, transport, tmp_path):
    """One completed recorded run: sq-1 answered on stable evidence, sq-2 on
    volatile evidence. Gap-fill disabled to keep the script linear."""
    config = make_config(
        tmp_path, session_id="s-parent", query="mountain height and price",
        respect_robots=False,
        budgets=Budgets(max_gap_rounds=0, max_coverage_gap_rounds=0),
    )
    documents = DocumentStore(db, clock)
    spans = SpanStore(db)
    dag = DagStore(db, event_log)
    fetcher = Fetcher(documents, blobs, clock, CacheMode.RECORD, transport=transport)
    registry = ToolRegistry(
        [
            FetchTool(fetcher, spans),
            RecordPremisesTool(dag, spans, documents),
            SearchWithinTool(spans, EmbeddingCache(db, HashedNgramEmbedder())),
        ]
    )
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    loop = OrchestratorLoop(
        config, ModelGateway(FakeAdapter(_parent_script()), event_log, blobs, ledger, config),
        registry, ctx, sessions, dag, spans, documents,
        CoverageLedger(db, event_log), Notebook(db, event_log, clock),
    )
    result = await loop.run()
    assert result.status == "done"
    return "s-parent"
