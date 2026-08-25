"""Graceful abort: request_halt() → journaled USER_ABORT, session row
finished as halted_user — never a row stuck 'running' (Phase 0)."""

from __future__ import annotations

from parsec.config import Budgets
from parsec.gateway.fake_adapter import FakeAdapter
from parsec.models.events import EventType
from tests.unit.test_agent_gates import build_loop, decompose_response


async def test_request_halt_finishes_session_as_halted_user(
    tmp_path, db, blobs, event_log, ledger, sessions, clock
):
    adapter = FakeAdapter([decompose_response(["part one?", "part two?"])])
    loop = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, adapter,
        Budgets(max_turns=10), session_id="s-halt",
    )

    # Halt lands after the decomposer call, as a SIGINT between turns would:
    # the next gate (dispatch) raises HaltRequested.
    orig_complete = loop.gateway.complete

    async def complete_then_halt(request):
        resp = await orig_complete(request)
        loop.request_halt()
        return resp

    loop.gateway.complete = complete_then_halt
    result = await loop.run()

    assert result.status == "halted_user"
    assert result.turns == 1  # decomposer only; regression: turns must not land in low_confidence
    assert result.low_confidence == []

    events = event_log.read("s-halt")
    types = [e.event_type for e in events]
    assert EventType.USER_ABORT in types
    assert EventType.SESSION_FINISHED in types

    row = sessions.get("s-halt")
    assert row["status"] == "halted_user"
    assert row["finished_ts"] is not None


async def test_halt_before_any_call_still_journals(
    tmp_path, db, blobs, event_log, ledger, sessions, clock
):
    loop = build_loop(
        tmp_path, db, blobs, event_log, ledger, sessions, clock, FakeAdapter([]),
        Budgets(max_turns=10), session_id="s-halt-early",
    )
    loop.request_halt()
    result = await loop.run()
    assert result.status == "halted_user"
    assert result.turns == 0
    assert sessions.get("s-halt-early")["status"] == "halted_user"
