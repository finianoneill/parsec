"""M14.2 exit tests: `parsec refresh <session>`.

1. A refresh is seeded by the parent's brief and coverage ledger — no
   decomposition model call — with mutability governing the split: an
   answered subquestion whose evidence is all claim_class "stable" carries
   forward (identical content-derived node IDs, no dispatch), a volatile
   one re-researches against the live (changed) web, and the run ends by
   emitting the claim-level diff against the parent.
2. The refreshed session replays byte-identically (the seed re-derives
   from refresh_of in its config), AND the parent still replays after the
   refresh advanced the shared URL cache row (fetch pinning, T4).
3. --all forces every subquestion back to research.
"""

from __future__ import annotations

import json

import httpx
import pytest

from parsec import cli, ids
from parsec.config import Budgets, CacheMode
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.gateway.gateway import ModelGateway
from parsec.loop.agent import OrchestratorLoop
from parsec.models.events import EventType
from parsec.refresh import derive_seed, run_refresh
from parsec.replay import run_replay
from parsec.retrieval.embeddings import EmbeddingCache, HashedNgramEmbedder
from parsec.retrieval.fetcher import Fetcher
from parsec.retrieval.span_indexer import index_spans
from parsec.store.coverage import CoverageLedger
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.notebook import Notebook
from parsec.store.sessions import SessionStore
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.tools.fetch import FetchTool
from parsec.tools.record_premises import RecordPremisesTool
from parsec.tools.search_within import SearchWithinTool
from parsec.verify.diff import diff_sessions
from tests.conftest import make_config

URL_A = "https://geo.example/mountain"
URL_B = "https://market.example/price"
SENT_A = "Olympus Mons is 21.9 kilometers tall."
SENT_B1 = "The listed price is 10 dollars."
SENT_B2 = "The listed price is 12 dollars."


def _span(text: str) -> str:
    start, end = index_spans(text)[0]
    return ids.span_id(ids.doc_hash(text.encode()), start, end)


def _premise_id(sentence: str, claim_class: str = "stable") -> str:
    return ids.node_id(
        "Premise",
        {"text": sentence, "span_refs": [_span(sentence)], "claim_class": claim_class},
    )


def _parent_script() -> list:
    """decompose -> sq-1 (stable evidence) -> sq-2 (volatile evidence) -> write."""
    return [
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
              "input": {"scope": "Cover the mountain's height and the current listed price.",
                        "effort": "standard",
                        "subquestions": ["how tall is the mountain", "what is the listed price"]}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_f1", "name": "fetch", "input": {"url": URL_A}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r1", "name": "record_premises",
              "input": {"premises": [{"text": SENT_A, "span_refs": [_span(SENT_A)],
                                      "claim_class": "stable"}]}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_s1", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_f2", "name": "fetch", "input": {"url": URL_B}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r2", "name": "record_premises",
              "input": {"premises": [{"text": SENT_B1, "span_refs": [_span(SENT_B1)],
                                      "claim_class": "volatile"}]}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_s2", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use"),
        scripted_response(
            [{"type": "text", "text": _answer(SENT_B1)}], stop_reason="end_turn"),
    ]


def _refresh_script(price_sentence: str = SENT_B2, include_sq1: bool = False) -> list:
    """No decomposer call: the brief is seeded. sq-2 re-fetches the (changed)
    price page; sq-1 appears only under --all."""
    script: list = []
    if include_sq1:
        script += [
            scripted_response(
                [{"type": "tool_use", "id": "tu_rf0", "name": "fetch", "input": {"url": URL_A}}],
                stop_reason="tool_use"),
            scripted_response(
                [{"type": "tool_use", "id": "tu_rr0", "name": "record_premises",
                  "input": {"premises": [{"text": SENT_A, "span_refs": [_span(SENT_A)],
                                          "claim_class": "stable"}]}}],
                stop_reason="tool_use"),
            scripted_response(
                [{"type": "tool_use", "id": "tu_rs0", "name": "submit_report",
                  "input": {"status": "answered"}}], stop_reason="tool_use"),
        ]
    script += [
        scripted_response(
            [{"type": "tool_use", "id": "tu_rf1", "name": "fetch", "input": {"url": URL_B}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_rr1", "name": "record_premises",
              "input": {"premises": [{"text": price_sentence,
                                      "span_refs": [_span(price_sentence)],
                                      "claim_class": "volatile"}]}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_rs1", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use"),
        scripted_response(
            [{"type": "text", "text": _answer(price_sentence)}], stop_reason="end_turn"),
    ]
    return script


def _answer(price_sentence: str) -> str:
    p_a = _premise_id(SENT_A)
    p_b = _premise_id(price_sentence, "volatile")
    return f"Height and price are settled. [narrative]\n{SENT_A} [{p_a}]\n{price_sentence} [{p_b}]"


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


async def test_refresh_carries_stable_and_reresearches_volatile(
    parent, db, blobs, event_log, clock, pages, transport
):
    pages[URL_B] = SENT_B2  # the world moved under the volatile subquestion

    result = await run_refresh(
        db, blobs, clock, parent, FakeAdapter(_refresh_script()), fetch_transport=transport
    )
    assert result.status == "done"
    refreshed = result.session_id
    assert refreshed.startswith(f"{parent}-refresh-")

    events = event_log.read(refreshed)

    # The brief was seeded — zero planning calls: every research LLM call is
    # sq-2's, and the orchestrator's only call is the writer.
    briefs = [ev for ev in events if ev.event_type == EventType.RESEARCH_BRIEF]
    assert len(briefs) == 1 and briefs[0].payload["seeded_from"] == parent
    llm_streams = [ev.stream_id for ev in events if ev.event_type == EventType.LLM_REQUEST]
    assert llm_streams == ["sq-2", "sq-2", "sq-2", "orchestrator"]

    # The split is journaled with reasons, and only sq-2 dispatched.
    (seeded,) = [ev for ev in events if ev.event_type == EventType.REFRESH_SEEDED]
    assert seeded.payload["carried"] == [
        {"sq_id": "sq-1", "reason": "stable evidence carried forward"}
    ]
    assert seeded.payload["researched"] == [
        {"sq_id": "sq-2", "reason": "volatile evidence must be re-observed"}
    ]
    started = {ev.payload["sq_id"] for ev in events if ev.event_type == EventType.SUBAGENT_STARTED}
    assert started == {"sq-2"}

    # Carried coverage resolves as answered-with-reason; carried evidence
    # keeps its content-derived identity across sessions.
    coverage = {r["sq_id"]: r for r in CoverageLedger(db, event_log).all(refreshed)}
    assert coverage["sq-1"]["status"] == "answered"
    assert "carried forward" in coverage["sq-1"]["reason"]
    assert coverage["sq-2"]["status"] == "answered"
    tier1 = {
        r["node_id"]
        for r in db.execute("SELECT node_id FROM nodes WHERE session_id=? AND tier=1", (refreshed,))
    }
    assert _premise_id(SENT_A) in tier1
    assert _premise_id(SENT_B2, "volatile") in tier1

    # The refresh ends in a diff against the parent: the stable claim holds
    # by node id, the reworded price claim matches on the fuzzy tier, and the
    # changed page reads as a document delta by hash.
    report = diff_sessions(db, parent, refreshed)
    by_text = {c.text: c for c in report.claims}
    assert by_text[SENT_A].status == "held" and by_text[SENT_A].match == "id"
    assert by_text[SENT_B2].match == "fuzzy" and by_text[SENT_B2].similarity >= 0.8
    docs = {d.url: d.status for d in report.documents}
    assert docs == {URL_B: "changed"}


async def test_refreshed_and_parent_sessions_both_replay(
    parent, db, blobs, event_log, clock, pages, transport
):
    pages[URL_B] = SENT_B2
    result = await run_refresh(
        db, blobs, clock, parent, FakeAdapter(_refresh_script()), fetch_transport=transport
    )
    assert result.status == "done"

    # The refreshed session replays byte-identically: the seed re-derives
    # from refresh_of in its recorded config, carry-forward included.
    outcome = await run_replay(db, blobs, clock, result.session_id)
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match

    # And the PARENT still replays even though the refresh re-fetched URL_B
    # and advanced the shared cache row: replay pins fetches to the doc
    # hashes the parent's own events recorded (T4).
    outcome = await run_replay(db, blobs, clock, parent)
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.answers_match


async def test_refresh_all_reresearches_stable_subquestions_too(
    parent, db, blobs, event_log, clock, pages, transport
):
    pages[URL_B] = SENT_B2
    result = await run_refresh(
        db, blobs, clock, parent,
        FakeAdapter(_refresh_script(include_sq1=True)),
        fetch_transport=transport, refresh_all=True,
    )
    assert result.status == "done"
    events = event_log.read(result.session_id)
    (seeded,) = [ev for ev in events if ev.event_type == EventType.REFRESH_SEEDED]
    assert seeded.payload["carried"] == []
    assert {r["sq_id"]: r["reason"] for r in seeded.payload["researched"]} == {
        "sq-1": "full refresh requested",
        "sq-2": "full refresh requested",
    }
    started = {ev.payload["sq_id"] for ev in events if ev.event_type == EventType.SUBAGENT_STARTED}
    assert started == {"sq-1", "sq-2"}


async def test_derive_seed_is_a_pure_readonly_split(parent, db, clock, tmp_path):
    seed = derive_seed(db, parent)
    assert seed.questions == ("how tall is the mountain", "what is the listed price")
    assert seed.scope.startswith("Cover the mountain")
    assert seed.effort == "standard"
    assert set(seed.carried) == {"sq-1"}
    assert seed.carried["sq-1"].premise_ids == (_premise_id(SENT_A),)
    assert seed.research_reasons == {"sq-2": "volatile evidence must be re-observed"}

    assert derive_seed(db, parent, refresh_all=True).carried == {}

    with pytest.raises(KeyError):
        derive_seed(db, "s-missing")
    # a session row with no recorded plan cannot seed a refresh
    SessionStore(db, clock).create(make_config(tmp_path, session_id="s-bare"))
    with pytest.raises(ValueError):
        derive_seed(db, "s-bare")


async def test_cli_refresh_emits_diff_json(
    parent, db, blobs, clock, pages, transport, tmp_path, capsys
):
    import asyncio

    pages[URL_B] = SENT_B2
    db.close()
    prev = (cli.adapter_factory, cli.fetch_transport)
    cli.adapter_factory = lambda config: FakeAdapter(_refresh_script())
    cli.fetch_transport = transport
    try:
        # A thread, because cli.main calls asyncio.run and this test's own
        # loop is already running.
        code = await asyncio.to_thread(
            cli.main, ["refresh", parent, "--data-dir", str(tmp_path), "--json"]
        )
    finally:
        cli.adapter_factory, cli.fetch_transport = prev
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "done"
    assert payload["session_id"].startswith(f"{parent}-refresh-")
    assert payload["diff"]["counts"]["held"] == len(payload["diff"]["claims"])
    assert code == cli.EXIT_OK  # every claim held -> no material change

    code = await asyncio.to_thread(
        cli.main, ["refresh", "s-nope", "--data-dir", str(tmp_path)]
    )
    assert code == cli.EXIT_USAGE
