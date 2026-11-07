"""Integration-level fixtures. For the CLI-driven M3/M2/M5 exit tests: the
scripted transport, search fixtures, and scripted adapter over the M3 corpus
(m3_corpus.py). For the M14 (diachronic research)
suites: the mutable fake web, its transport, and one recorded parent
session to diff/refresh/watch against. Corpus constants and scripts live in
m14_corpus.py so test modules can import them without shadowing fixtures."""

from __future__ import annotations

import json

import httpx
import pytest

import parsec.cli as cli
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
from tests.integration import m3_corpus
from tests.integration.m14_corpus import SENT_A, SENT_B1, URL_A, URL_B, _parent_script

@pytest.fixture
def m3_transport(monkeypatch):
    """Mock HTTP for the M3 corpus through the CLI's transport seam; returns
    a call counter so replays can prove zero HTTP."""
    pages = {m3_corpus.URL_A: m3_corpus.PAGE_A, m3_corpus.URL_B: m3_corpus.PAGE_B}
    counter = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["calls"] += 1
        page = pages.get(str(request.url).rstrip("/")) or pages.get(str(request.url))
        if page is None:
            return httpx.Response(404, content=b"not found")
        return httpx.Response(200, content=page, headers={"content-type": "text/html"})

    monkeypatch.setattr(cli, "fetch_transport", httpx.MockTransport(handler))
    return counter


@pytest.fixture
def m3_fixtures_path(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps(
            {
                m3_corpus.SQ1: [
                    {"title": "Boiling Point", "url": m3_corpus.URL_A, "snippet": "100C at sea level"}
                ],
                m3_corpus.SQ2: [
                    {"title": "Altitude Effects", "url": m3_corpus.URL_B, "snippet": "lower at altitude"}
                ],
            }
        )
    )
    return path


@pytest.fixture
def m3_scripted_adapter(monkeypatch):
    """Wire the M3 script into the CLI's adapter factory; returns the
    expected answer text."""
    answer, responses = m3_corpus.m3_script()
    monkeypatch.setattr(cli, "adapter_factory", lambda config: FakeAdapter(responses))
    return answer


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
