"""Orchestrator state machine — the full §3 names as of M3.

    PLANNING    — decompose the query into subquestions (coverage ledger rows)
    DISPATCHING — pick the next open subquestion, prepare a subagent
    COLLECTING  — subagent tool loop runs; its report is folded into the DAG
    WRITING     — writer composes from premises/findings (coverage gate first)
    VERIFYING   — citation check + stage-1 structural verification
    DONE/HALTED — terminal

DISPATCHING↔COLLECTING cycles once per subquestion. GAP_FILLING and
STEERING arrive at M6. Stop-condition gates run before every model call, in
§3 priority order; slots for coverage-completeness and saturation stops are
marked at the gate site in agent.py.
"""

from __future__ import annotations

from enum import StrEnum

from parsec.models.events import EventType
from parsec.store.event_log import EventLog


class AgentState(StrEnum):
    PLANNING = "PLANNING"
    DISPATCHING = "DISPATCHING"
    COLLECTING = "COLLECTING"
    WRITING = "WRITING"
    VERIFYING = "VERIFYING"
    GAP_FILLING = "GAP_FILLING"
    DONE = "DONE"
    HALTED = "HALTED"


_ALLOWED: dict[AgentState, set[AgentState]] = {
    AgentState.PLANNING: {AgentState.DISPATCHING, AgentState.WRITING, AgentState.HALTED},
    AgentState.DISPATCHING: {AgentState.COLLECTING, AgentState.WRITING, AgentState.HALTED},
    AgentState.COLLECTING: {AgentState.DISPATCHING, AgentState.WRITING, AgentState.HALTED},
    AgentState.WRITING: {AgentState.VERIFYING, AgentState.HALTED},
    AgentState.VERIFYING: {AgentState.DONE, AgentState.GAP_FILLING, AgentState.HALTED},
    AgentState.GAP_FILLING: {AgentState.WRITING, AgentState.HALTED},
    AgentState.DONE: set(),
    AgentState.HALTED: set(),
}


class StateMachine:
    def __init__(self, event_log: EventLog, session_id: str):
        self.state = AgentState.PLANNING
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
