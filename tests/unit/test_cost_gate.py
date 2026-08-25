"""Phase 5: the cost gate — the gate primitive's second consumer.

Fires once when spend crosses the configured fraction of the USD cap at a
dispatch boundary; the reply is a recorded, gate-tagged steering event, so
gated runs replay byte-identically."""

from __future__ import annotations

from parsec.config import Budgets
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.gateway.gateway import ModelGateway
from parsec.loop.agent import OrchestratorLoop
from parsec.models.events import EventType
from parsec.store.coverage import CoverageLedger
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.notebook import Notebook
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from tests.conftest import make_config
from tests.unit.test_agent_gates import decompose_response

# Each fake call reports 100 in / 50 out; at 5/25 $/MTok that is $0.00175.
_PRICING = {"fake-model": {"input": 5.0, "output": 25.0}}


def _gated_loop(
    tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter,
    session_id, threshold=0.3, registry=None,
):
    config = make_config(
        tmp_path, session_id=session_id,
        budgets=Budgets(max_usd=0.01, max_turns=10),
        pricing_override=_PRICING,
        cost_gate_threshold=threshold,
    )
    gateway = ModelGateway(adapter, event_log, blobs, ledger, config)
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    return OrchestratorLoop(
        config, gateway, registry or ToolRegistry([]), ctx, sessions,
        DagStore(db, event_log), SpanStore(db), DocumentStore(db, clock),
        CoverageLedger(db, event_log), Notebook(db, event_log, clock),
    )


def _two_sq_script():
    return [
        decompose_response(["part one?", "part two?"]),
        scripted_response([{"type": "text", "text": "nothing on part one"}], index=1),
        scripted_response([{"type": "text", "text": "nothing on part two"}], index=2),
        scripted_response([{"type": "text", "text": "No evidence. [narrative]"}], index=3),
    ]


async def test_approved_cost_gate_continues(
    tmp_path, db, blobs, event_log, ledger, sessions, clock
):
    loop = _gated_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock,
        FakeAdapter(_two_sq_script()), "s-cost-ok",
    )
    # decomposer + sq-1 = $0.0035 >= 0.3 * $0.01, so the gate fires before
    # sq-2 at turn index 2; the reply is scripted the way replay scripts it.
    loop.scripted_gates = {("cost", 2): ["approve"]}
    result = await loop.run()

    assert result.coverage == {"sq-1": "blocked", "sq-2": "blocked"}  # both DISPATCHED (blocked = no report)
    events = event_log.read("s-cost-ok")
    proposals = [e for e in events if e.event_type == EventType.GATE_PROPOSED]
    assert len(proposals) == 1  # fires once, even though spend stays over threshold
    assert proposals[0].payload["gate"] == "cost"
    assert proposals[0].payload["threshold"] == 0.3
    reply = next(
        e for e in events
        if e.event_type == EventType.STEERING_INJECTED and e.payload.get("gate") == "cost"
    )
    assert reply.payload["text"] == "approve" and reply.payload["turn_index"] == 2


async def test_declined_cost_gate_wraps_up(
    tmp_path, db, blobs, event_log, ledger, sessions, clock
):
    # sq-2's subagent response is never consumed: declined before dispatch.
    script = _two_sq_script()
    del script[2]
    loop = _gated_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock,
        FakeAdapter(script), "s-cost-no",
    )
    loop.scripted_gates = {("cost", 2): ["stop, too expensive"]}
    result = await loop.run()

    assert result.status == "partial"
    assert result.answer  # the writer still shipped a best-effort answer
    rows = CoverageLedger(db, event_log).all("s-cost-no")
    assert rows[1]["sq_id"] == "sq-2" and rows[1]["status"] == "blocked"
    assert rows[1]["reason"] == "declined at cost gate before dispatch"


async def test_no_threshold_means_no_gate(
    tmp_path, db, blobs, event_log, ledger, sessions, clock
):
    loop = _gated_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock,
        FakeAdapter(_two_sq_script()), "s-cost-off", threshold=None,
    )
    await loop.run()
    assert not [
        e for e in event_log.read("s-cost-off") if e.event_type == EventType.GATE_PROPOSED
    ]


async def test_live_reply_via_steer_queue(
    tmp_path, db, blobs, event_log, ledger, sessions, clock
):
    """The live path: the reply arrives through loop.steer() while the gate
    is waiting (the CLI's stdin thread), not from a script."""
    loop = _gated_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock,
        FakeAdapter(_two_sq_script()), "s-cost-live",
    )

    orig_gate = loop._cost_gate

    async def gate_with_user():
        # simulate the user typing only once the gate is actually reached
        if not loop._cost_gate_passed and loop.ledger.spent_usd("s-cost-live") >= 0.003:
            loop.steer("approve")
        return await orig_gate()

    loop._cost_gate = gate_with_user
    result = await loop.run()
    assert result.coverage == {"sq-1": "blocked", "sq-2": "blocked"}  # continued past the gate


async def test_declined_cost_gate_replays(
    db, blobs, event_log, ledger, sessions, clock, tmp_path
):
    from parsec.config import CacheMode
    from parsec.replay import run_replay
    from parsec.retrieval.embeddings import EmbeddingCache, HashedNgramEmbedder
    from parsec.retrieval.fetcher import Fetcher
    from parsec.tools.fetch import FetchTool
    from parsec.tools.record_premises import RecordPremisesTool
    from parsec.tools.search_within import SearchWithinTool

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
    script = _two_sq_script()
    del script[2]
    loop = _gated_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock,
        FakeAdapter(script), "s-cost-replay", registry=registry,
    )
    loop.scripted_gates = {("cost", 2): ["no thanks"]}
    await loop.run()

    # replay rebuilds the recorded threshold and re-injects the recorded
    # gate reply through the same (gate, turn) key
    outcome = await run_replay(db, blobs, clock, "s-cost-replay")
    assert outcome.projections_match, outcome.first_divergence
    assert outcome.verified
