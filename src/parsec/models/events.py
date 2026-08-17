"""Event log types. Every state change in a run is one of these."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class EventType(StrEnum):
    SESSION_STARTED = "session_started"
    STATE_TRANSITION = "state_transition"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_INTENT = "tool_intent"
    TOOL_RESULT = "tool_result"
    FETCH_PERFORMED = "fetch_performed"
    SPAN_INDEXED = "span_indexed"
    NODE_ADDED = "node_added"
    EDGE_ADDED = "edge_added"
    BUDGET_DEBIT = "budget_debit"
    SUBQUESTIONS_PLANNED = "subquestions_planned"
    SUBAGENT_STARTED = "subagent_started"
    SUBAGENT_COMPLETED = "subagent_completed"
    COVERAGE_UPDATED = "coverage_updated"
    NOTEBOOK_APPENDED = "notebook_appended"
    VERIFICATION_COMPLETED = "verification_completed"
    ANSWER_EMITTED = "answer_emitted"
    USER_ABORT = "user_abort"
    ERROR = "error"
    SESSION_FINISHED = "session_finished"


class Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    idx: int
    ts: str
    session_id: str
    actor: str
    event_type: EventType
    payload: dict
    parent_seq: int | None = None
