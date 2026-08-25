"""Synthesis judge (§11 M5 axis 3): last-run, least trusted, advisory only.

The judge is a DIFFERENT model family than the generator (§6 stage 5 —
never let the pipeline's own model grade its own homework). Its score is a
number in the eval result, never a gate on anything. Judge failures
(bad JSON, network error) degrade to None rather than failing the eval.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from parsec.gateway.base import ModelAdapter
from parsec.loop.structured import judged_json, parse_prose_json
from parsec.models.gateway import ModelRequest

JUDGE_SYSTEM = """You are grading a research report for synthesis quality. You will see the question and the report (citation markers like [premise:...] and a mechanical appendix are part of the harness — ignore their syntax, judge the prose).

Score 1-5 for synthesis quality: does the report directly answer the question, integrate its evidence coherently, acknowledge gaps and conflicts honestly, and hedge in proportion to its stated confidence?

Reply with ONLY a JSON object: {"synthesis_score": <1-5>, "rationale": "<one sentence>"}"""

_RETRY_INSTRUCTION = (
    'Reply with ONLY the JSON object: {"synthesis_score": <1-5>, "rationale": "<one sentence>"}'
)


class SynthesisJudgeReply(BaseModel):
    """Validated instead of regex-scraped (Phase 3); one corrective retry
    before the advisory axis degrades to None."""

    model_config = ConfigDict(extra="ignore")

    synthesis_score: float = Field(ge=1, le=5)
    rationale: str = ""


def judge_request(judge_model: str, query: str, answer: str) -> ModelRequest:
    return ModelRequest(
        model=judge_model,
        max_tokens=500,
        system=[{"type": "text", "text": JUDGE_SYSTEM}],
        messages=[
            {
                "role": "user",
                "content": f"Question:\n{query}\n\nReport:\n{answer}",
            }
        ],
    )


def parse_judge_reply(text: str) -> float | None:
    """Extract a 1-5 score and normalize to [0,1]; None on any malformation."""
    value, _ = parse_prose_json(text, SynthesisJudgeReply)
    return None if value is None else (float(value.synthesis_score) - 1.0) / 4.0


async def judge_synthesis(
    adapter: ModelAdapter, judge_model: str, query: str, answer: str
) -> float | None:
    reply = await judged_json(
        adapter, judge_request(judge_model, query, answer), SynthesisJudgeReply, _RETRY_INSTRUCTION
    )
    return None if reply is None else (float(reply.synthesis_score) - 1.0) / 4.0
