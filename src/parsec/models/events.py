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
    SUBAGENT_JOINED = "subagent_joined"  # M11: observed completion/merge order
    LLM_FAILED = "llm_failed"            # M11: adapter raised; failure is journaled
    COVERAGE_UPDATED = "coverage_updated"
    NOTEBOOK_APPENDED = "notebook_appended"
    CONTEXT_COMPACTED = "context_compacted"
    STEERING_INJECTED = "steering_injected"
    GAP_FILL_STARTED = "gap_fill_started"
    JUDGE_SCORED = "judge_scored"
    CREDENCE_COMPUTED = "credence_computed"
    OMISSION_DETECTED = "omission_detected"
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
    # M11 per-stream ordering (T8): idx is the volatile arrival ordinal;
    # (stream_id, stream_idx) is the deterministic coordinate replay keys off.
    stream_id: str = "orchestrator"
    stream_idx: int = 0
