"""One structured-output contract (Phase 3).

A pydantic model is the single source of truth for what the model must
produce; validation failures go back to the model as per-field errors
instead of silent fallbacks or generic correctives. Three consumers,
three repair channels:

- the decomposer brief: bounded repair rounds via structured_call(), with
  tool_choice pinned on the final attempt so a prose-only reply cannot
  recur;
- submit_report: per-field problems fed into the subagent's normal
  tool-error feedback (the turn cap bounds repair; the subagent must stay
  free to keep researching, so nothing is forced);
- the judges: prose-JSON validation with one corrective retry (a
  different model family with a deliberately tool-free adapter, so the
  channel is prose by design).

Every repair is an ordinary journaled model call — bounded, recorded, and
replayable like any other turn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from parsec.models.gateway import ModelRequest, ModelResponse

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def format_validation_errors(exc: ValidationError) -> list[str]:
    """Per-field problem lines, the same shape the tool registry feeds back:
    the model can only fix what it can locate."""
    lines = []
    for err in exc.errors():
        loc = " -> ".join(str(p) for p in err["loc"]) or "(root)"
        lines.append(f"{loc}: {err['msg']}")
    return lines


def validate_tool_call(
    resp: ModelResponse, tool_name: str, model_cls: type[BaseModel]
) -> tuple[BaseModel | None, dict | None, list[str]]:
    """Find and validate the named tool call in a response.

    Returns (instance, tool_use_block, problems): instance is None when the
    call is missing or invalid; tool_use_block is the offending block when
    one exists (so the corrective can be a tool_result, not loose prose)."""
    for block in resp.tool_uses:
        if block.get("name") != tool_name:
            continue
        try:
            return model_cls.model_validate(block.get("input", {})), block, []
        except ValidationError as exc:
            return None, block, format_validation_errors(exc)
    return None, None, [f"the response did not call {tool_name}"]


def repair_turn(
    resp: ModelResponse, tool_use: dict | None, problems: list[str], instruction: str
) -> list[dict]:
    """The corrective exchange appended before a repair attempt: the
    assistant's own turn, then the problems — as an is_error tool_result
    when there was a tool call to answer, as plain text otherwise."""
    problem_text = "\n".join(f"- {p}" for p in problems)
    turns: list[dict] = []
    if resp.content:
        turns.append({"role": "assistant", "content": resp.content})
    if tool_use is not None:
        turns.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use["id"],
                        "content": f"invalid {tool_use.get('name')}:\n{problem_text}\n{instruction}",
                        "is_error": True,
                    }
                ],
            }
        )
    else:
        turns.append({"role": "user", "content": f"{problem_text}\n{instruction}"})
    return turns


@dataclass
class StructuredOutcome:
    value: BaseModel | None
    resp: ModelResponse  # the last response (callers may need its transcript)
    problems: list[str]  # empty on success
    repairs: int


async def structured_call(
    complete: Callable[[list[dict], dict | None], Awaitable[ModelResponse]],
    messages: list[dict],
    model_cls: type[BaseModel],
    tool_name: str,
    repair_instruction: str,
    max_repairs: int = 1,
) -> StructuredOutcome:
    """Call, validate, repair: per-field errors go back as a tool error and
    the retry pins tool_choice to the expected tool. `complete` is the
    caller's own gateway closure (it owns turn counting and request
    building), invoked as complete(messages, tool_choice)."""
    resp = await complete(messages, None)
    repairs = 0
    while True:
        value, tool_use, problems = validate_tool_call(resp, tool_name, model_cls)
        if value is not None:
            return StructuredOutcome(value, resp, [], repairs)
        if repairs >= max_repairs:
            return StructuredOutcome(None, resp, problems, repairs)
        repairs += 1
        messages = messages + repair_turn(resp, tool_use, problems, repair_instruction)
        resp = await complete(messages, {"type": "tool", "name": tool_name})


class BriefSubmission(BaseModel):
    """The decomposer's submit_subquestions contract. The wire schema stays
    the hand-written SUBMIT_SUBQUESTIONS_SCHEMA (byte-stable prompts); this
    model VALIDATES what comes back. Lenient where the old parser was
    lenient: unknown keys are ignored and a bad effort coerces to "deep"
    (an invalid estimate must never clamp dispatch)."""

    model_config = ConfigDict(extra="ignore")

    scope: str = ""
    effort: str = "deep"
    subquestions: list[str] = Field(min_length=1)

    @field_validator("scope")
    @classmethod
    def _strip_scope(cls, v: str) -> str:
        return v.strip()

    @field_validator("effort")
    @classmethod
    def _coerce_effort(cls, v: str) -> str:
        return v if v in ("quick", "standard", "deep") else "deep"

    @field_validator("subquestions")
    @classmethod
    def _clean_questions(cls, v: list[str]) -> list[str]:
        cleaned = [q.strip() for q in v]
        for i, q in enumerate(cleaned):
            if len(q) < 3:
                raise ValueError(f"subquestion {i + 1} is too short (need >= 3 characters)")
        return cleaned


# -- prose-JSON structured replies (the judge channel) -----------------------


def parse_prose_json(
    text: str, model_cls: type[BaseModel]
) -> tuple[BaseModel | None, list[str]]:
    """Validate the first JSON object embedded in a prose reply."""
    m = _JSON_RE.search(text)
    if not m:
        return None, ["no JSON object found in the reply"]
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON: {exc}"]
    try:
        return model_cls.model_validate(obj), []
    except ValidationError as exc:
        return None, format_validation_errors(exc)


async def judged_json(
    adapter,
    request: ModelRequest,
    model_cls: type[BaseModel],
    retry_instruction: str,
) -> BaseModel | None:
    """One validated prose-JSON exchange with a single corrective retry.
    Judge channels are advisory by contract: any failure — malformed reply
    after retry, network error — degrades to None, never raises."""
    try:
        resp = await adapter.complete(request)
        value, problems = parse_prose_json(resp.text, model_cls)
        if value is not None:
            return value
        retry = ModelRequest(
            model=request.model,
            max_tokens=request.max_tokens,
            system=request.system,
            tools=request.tools,
            messages=request.messages
            + [
                {"role": "assistant", "content": [{"type": "text", "text": resp.text}]},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was invalid:\n"
                        + "\n".join(f"- {p}" for p in problems)
                        + f"\n{retry_instruction}"
                    ),
                },
            ],
        )
        value, _ = parse_prose_json((await adapter.complete(retry)).text, model_cls)
        return value
    except Exception:
        return None
