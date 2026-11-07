"""Live-run hardening (M15.8), from the first long Opus run that halted on
the wall clock after a sleeping laptop: the writer keeps a wall reserve, a
budget halt returns the draft it had, markdown structure is not a claim,
system sleep is journaled on the call it interrupted, the activity ticker
says "retrying" once the backoff is over, and caffeinate holds off idle
sleep on request. Old recordings keep the behaviour they were recorded with.
"""

from __future__ import annotations

import io

from rich.console import Console

from parsec.activity import ActivityView
from parsec.awake import keep_awake
from parsec.config import Budgets, legacy_flag_overrides
from parsec.errors import ModelCallFailed, ModelErrorKind
from parsec.gateway.fake_adapter import FakeAdapter, StreamFakeAdapter, scripted_response
from parsec.gateway.gateway import WALL_CLOCK_KIND, ModelGateway, wall_clock_detail
from parsec.loop.agent import OrchestratorLoop
from parsec.loop.citations import check_citations, segment_answer
from parsec.models.events import EventType
from parsec.models.gateway import ModelRequest
from parsec.store.coverage import CoverageLedger
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.event_log import _strip_volatile
from parsec.store.notebook import Notebook
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from tests.conftest import make_config
from tests.unit.test_agent_gates import decompose_response

_REQ = ModelRequest(model="fake-model", max_tokens=10, messages=[{"role": "user", "content": "x"}])


def _loop(tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter, budgets, session_id, **cfg):
    config = make_config(tmp_path, session_id=session_id, budgets=budgets, **cfg)
    gateway = ModelGateway(adapter, event_log, blobs, ledger, config)
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    return OrchestratorLoop(
        config, gateway, ToolRegistry([]), ctx, sessions,
        DagStore(db, event_log), SpanStore(db), DocumentStore(db, clock),
        CoverageLedger(db, event_log), Notebook(db, event_log, clock),
    )


# --- 1. a budget halt keeps the draft ---------------------------------------


def _wall_failure() -> ModelCallFailed:
    return ModelCallFailed(WALL_CLOCK_KIND, wall_clock_detail(300), ModelErrorKind.FATAL)


async def test_budget_halt_returns_latest_draft(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    """max_turns=1: no research, the writer runs, its draft cites an unknown
    id, the repair call dies on the wall clock — the draft ships anyway."""
    adapter = StreamFakeAdapter({
        "orchestrator": [
            scripted_response([{"type": "text", "text": "The draft. [premise:deadbeefdeadbeef]"}], stop_reason="end_turn"),
            _wall_failure(),
        ]
    })
    loop = _loop(tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter, Budgets(max_turns=1), "s-draft")
    result = await loop.run()
    assert result.status == "halted_budget"
    assert result.answer.startswith("PARTIAL — budget exhausted (wall_seconds) after a draft was written")
    assert "1 citation problem(s) left unresolved:" in result.answer
    assert "unknown premise/finding id" in result.answer
    assert result.answer.rstrip().endswith("The draft. [premise:deadbeefdeadbeef]")
    assert len(result.unresolved) == 2  # the budget breach + the citation problem
    assert sessions.get("s-draft")["status"] == "halted_budget"


async def test_budget_halt_legacy_flag_dumps_premises(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    adapter = StreamFakeAdapter({
        "orchestrator": [
            scripted_response([{"type": "text", "text": "The draft. [premise:deadbeefdeadbeef]"}], stop_reason="end_turn"),
            _wall_failure(),
        ]
    })
    loop = _loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter, Budgets(max_turns=1),
        "s-legacy", halt_keeps_draft=False,
    )
    result = await loop.run()
    assert result.answer.startswith("PARTIAL — budget exhausted (wall_seconds) before an answer could be written.")
    assert "The draft." not in result.answer


def test_legacy_flag_overrides_only_for_pre_flag_recordings():
    assert legacy_flag_overrides({"query": "q"}) == {
        "halt_keeps_draft": False, "structure_lines_narrative": False,
    }
    assert legacy_flag_overrides({"halt_keeps_draft": True, "structure_lines_narrative": True}) == {}
    assert legacy_flag_overrides({"halt_keeps_draft": False}) == {"structure_lines_narrative": False}


def test_projection_strips_behaviour_flags_and_sleep():
    config = {"halt_keeps_draft": True, "structure_lines_narrative": True, "model": "m"}
    out = _strip_volatile(EventType.SESSION_STARTED, {"session_id": "s", "config": config})
    assert out["config"] == {"model": "m"}
    for et in (EventType.LLM_RETRY, EventType.LLM_FAILED, EventType.LLM_RESPONSE):
        assert "suspended_s" not in _strip_volatile(et, {"kind": "x", "suspended_s": 600.0})
    assert _strip_volatile(EventType.LLM_RETRY, {"kind": "x"}) == {"kind": "x"}


# --- 2. writer wall reserve --------------------------------------------------


async def test_writer_wall_reserve_stops_dispatch(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    """Research used 85% of a 100s wall budget by the time it would dispatch:
    the 20% reserve blocks the subquestions with a stable reason and the
    writer still runs inside the time left."""
    adapter = FakeAdapter([
        decompose_response(["part one?", "part two?"]),
        scripted_response([{"type": "text", "text": "Nothing gathered. [narrative]"}], stop_reason="end_turn"),
    ])
    budgets = Budgets(max_turns=10, max_wall_seconds=100, writer_wall_reserve=0.2)
    loop = _loop(tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter, budgets, "s-reserve")
    orig = loop.gateway.complete

    async def complete_then_age(request):
        resp = await orig(request)
        clock.mono = 85.0  # research ran long
        return resp

    loop.gateway.complete = complete_then_age
    result = await loop.run()
    assert result.status == "partial"
    assert result.turns == 2  # decomposer + writer
    rows = CoverageLedger(db, event_log).all("s-reserve")
    assert [r["status"] for r in rows] == ["blocked", "blocked"]
    assert all(
        r["reason"] == "wall-clock reserve for the writer reached (20% of 100s held back) before dispatch"
        for r in rows
    )


def test_gate_reserve_threshold_and_opt_out(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    for reserve, elapsed, expected in ((0.2, 79.0, False), (0.2, 80.0, True), (0.0, 99.0, False)):
        budgets = Budgets(max_turns=10, max_wall_seconds=100, writer_wall_reserve=reserve)
        loop = _loop(
            tmp_path, db, blobs, event_log, ledger, sessions, clock, FakeAdapter([]), budgets,
            f"s-gate-{reserve}-{elapsed}",
        )
        loop._start_mono = 0.0
        clock.mono = elapsed
        assert loop._gate() is expected, (reserve, elapsed)


# --- 3. markdown structure is not a claim -----------------------------------

_MD = (
    "## 1. ADK 2.0 — the framework\n"
    "---\n"
    "ADK is Apache-2.0 licensed [premise:21fc70a68520ae36].\n"
    "### A heading that cites [premise:21fc70a68520ae36]\n"
    "***\n"
    "Unsupported claim here.\n"
)


def test_structure_lines_skipped_when_flag_on():
    texts = [s.text for s in segment_answer(_MD, structure_lines_narrative=True)]
    assert texts == [
        "ADK is Apache-2.0 licensed.",
        "### A heading that cites",  # a heading that cites stays a claim segment, as written
        "Unsupported claim here.",
    ]


def test_structure_lines_legacy_behaviour_unchanged():
    texts = [s.text for s in segment_answer(_MD)]
    assert "## 1." in texts and "ADK 2.0 — the framework" in texts and "---" in texts


def test_check_citations_honours_structure_flag(db, event_log, sessions, config):
    sessions.create(config)
    dag = DagStore(db, event_log)
    pid = dag.add_node(
        config.session_id, "Premise",
        {"text": "ADK is Apache-2.0 licensed.", "span_refs": ["doc:abcdefabcdef#0-10"], "claim_class": "stable"},
    )
    answer = f"## Summary\n---\nADK is Apache-2.0 licensed. [{pid}]\n"
    assert check_citations(answer, config.session_id, dag, structure_lines_narrative=True).ok
    legacy = check_citations(answer, config.session_id, dag)
    assert [p for p in legacy.problems if "## Summary" in p] and len(legacy.problems) == 2


# --- 4. system sleep is journaled on the call it interrupted ----------------


class _DriftClock:
    """Wall clock and monotonic clock advanced independently: `sleep_for`
    moves only the wall clock, which is what a suspended machine does."""

    def __init__(self):
        self.wall = 1_800_000_000.0
        self.mono = 0.0

    def now_iso(self) -> str:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(self.wall, tz=timezone.utc).isoformat(timespec="milliseconds")

    def monotonic(self) -> float:
        return self.mono

    async def sleep(self, seconds: float) -> None:
        self.wall += seconds
        self.mono += seconds


class _SleepyAdapter:
    """First attempt: the machine sleeps 15 minutes and the socket dies.
    Second attempt: answers normally."""

    def __init__(self, clock: _DriftClock):
        self.clock = clock
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        if self.calls == 1:
            self.clock.wall += 900.0
            raise ConnectionError("Connection error.")
        self.clock.wall += 3.0
        self.clock.mono += 3.0
        return scripted_response([{"type": "text", "text": "ok"}])


async def test_sleep_gap_journaled_on_retry_not_on_clean_calls(db, blobs, tmp_path):
    from parsec.store.event_log import EventLog
    from parsec.store.ledger import Ledger
    from parsec.store.sessions import SessionStore

    clock = _DriftClock()
    event_log, ledger, sessions = EventLog(db, clock), Ledger(db, clock), SessionStore(db, clock)
    config = make_config(tmp_path, session_id="s-sleep")
    sessions.create(config)
    gw = ModelGateway(_SleepyAdapter(clock), event_log, blobs, ledger, config)
    assert (await gw.complete(_REQ)).text == "ok"
    events = {e.event_type: e for e in event_log.read("s-sleep")}
    assert events[EventType.LLM_RETRY].payload["suspended_s"] == 900.0
    assert events[EventType.LLM_RETRY].payload["kind"] == "ConnectionError"
    assert "suspended_s" not in events[EventType.LLM_RESPONSE].payload  # 3s of real time: no gap
    assert b"suspended_s" not in event_log.projection("s-sleep")


# --- 5. the activity ticker says "retrying" once the backoff is over --------


def test_retry_ticker_flips_to_retrying_after_backoff():
    console = Console(file=io.StringIO(), force_terminal=True, width=120)
    with ActivityView(console) as view:
        view.on_event(EventType.STATE_TRANSITION, {"to_state": "WRITING"}, "orchestrator")
        view.on_event(EventType.LLM_REQUEST, {"call_index": 1}, "orchestrator")
        view.on_event(
            EventType.LLM_RETRY,
            {"attempt": 1, "delay_s": 2.0, "error_kind": "transient", "kind": "APIConnectionError",
             "suspended_s": 900.0},
            "orchestrator",
        )
        tail = str(view._tail[-1])
        assert "retry 1 in 2.0s" in tail and "machine slept 15m00s during the call" in tail
        assert "waiting to retry (transient)…" in str(view._render().renderables[0].text)
        view._busy_since -= 10  # 2s backoff elapsed, 8s into the re-attempt
        assert "retrying (attempt 2)… · 8s" in str(view._render().renderables[0].text)
        view.on_event(EventType.LLM_RESPONSE, {"call_index": 1, "suspended_s": 30.0}, "orchestrator")
        assert view._busy_since is None and view._retry_delay is None
        assert "machine slept 30s during the call" in str(view._tail[-1])


# --- 6. keep-awake ------------------------------------------------------------


def test_keep_awake_disabled_or_unavailable(monkeypatch):
    with keep_awake(False) as held:
        assert held is False
    monkeypatch.setattr("parsec.awake.available", lambda: False)
    with keep_awake(True) as held:
        assert held is False


def test_keep_awake_spawns_and_terminates_caffeinate(monkeypatch):
    import parsec.awake as awake

    spawned: list[list[str]] = []

    class FakeProc:
        terminated = False

        def terminate(self):
            FakeProc.terminated = True

        def wait(self, timeout=None):
            return 0

    def fake_popen(argv, **kwargs):
        spawned.append(argv)
        return FakeProc()

    monkeypatch.setattr(awake, "available", lambda: True)
    monkeypatch.setattr(awake.subprocess, "Popen", fake_popen)
    with keep_awake(True) as held:
        assert held is True
    assert spawned[0][:2] == ["caffeinate", "-i"] and spawned[0][2] == "-w"
    assert FakeProc.terminated
