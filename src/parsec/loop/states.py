"""M1 state machine.

Collapsed from the §3 orchestrator states (multi-agent scale) to a
single-agent shape with a documented mapping, so M3 can adopt the full
names without churn:

    INIT        -> PLANNING
    RESEARCHING -> DISPATCHING + COLLECTING
    ANSWERING   -> WRITING
    CITE_CHECK  -> VERIFYING (stage-1 structural only)
    DONE/HALTED -> DONE/HALTED

Stop-condition gates run before every model call and tool execution, in
§3 priority order. Conditions 3 (coverage ledger) and 4 (saturation) are
M3 concerns — their slots are noted at the gate site in agent.py.
"""

from __future__ import annotations

from enum import StrEnum

from parsec.models.events import EventType
from parsec.store.event_log import EventLog


class AgentState(StrEnum):
    INIT = "INIT"
    RESEARCHING = "RESEARCHING"
    ANSWERING = "ANSWERING"
    CITE_CHECK = "CITE_CHECK"
    DONE = "DONE"
    HALTED = "HALTED"


_ALLOWED: dict[AgentState, set[AgentState]] = {
    AgentState.INIT: {AgentState.RESEARCHING, AgentState.HALTED},
    AgentState.RESEARCHING: {AgentState.RESEARCHING, AgentState.ANSWERING, AgentState.HALTED},
    AgentState.ANSWERING: {AgentState.CITE_CHECK, AgentState.HALTED},
    AgentState.CITE_CHECK: {AgentState.DONE, AgentState.HALTED},
    AgentState.DONE: set(),
    AgentState.HALTED: set(),
}


class StateMachine:
    def __init__(self, event_log: EventLog, session_id: str):
        self.state = AgentState.INIT
        self._event_log = event_log
        self._session_id = session_id

    def transition(self, to: AgentState, reason: str = "") -> None:
        if to not in _ALLOWED[self.state]:
            raise RuntimeError(f"illegal transition {self.state} -> {to}")
        self._event_log.append(
            self._session_id,
            "harness",
            EventType.STATE_TRANSITION,
            {"from_state": self.state.value, "to_state": to.value, "reason": reason},
        )
        self.state = to
