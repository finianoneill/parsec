"""Synthesis judge (§11 M5 axis 3): last-run, least trusted, advisory only.

The judge is a DIFFERENT model family than the generator (§6 stage 5 —
never let the pipeline's own model grade its own homework). Its score is a
number in the eval result, never a gate on anything. Judge failures
(bad JSON, network error) degrade to None rather than failing the eval.
"""

from __future__ import annotations

import json
import re

from parsec.gateway.base import ModelAdapter
from parsec.models.gateway import ModelRequest

JUDGE_SYSTEM = """You are grading a research report for synthesis quality. You will see the question and the report (citation markers like [premise:...] and a mechanical appendix are part of the harness — ignore their syntax, judge the prose).

Score 1-5 for synthesis quality: does the report directly answer the question, integrate its evidence coherently, acknowledge gaps and conflicts honestly, and hedge in proportion to its stated confidence?

Reply with ONLY a JSON object: {"synthesis_score": <1-5>, "rationale": "<one sentence>"}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


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
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        score = obj["synthesis_score"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(score, (int, float)) or not 1 <= score <= 5:
        return None
    return (float(score) - 1.0) / 4.0


async def judge_synthesis(
    adapter: ModelAdapter, judge_model: str, query: str, answer: str
) -> float | None:
    try:
        resp = await adapter.complete(judge_request(judge_model, query, answer))
    except Exception:
        return None  # advisory axis: degrade, never fail the eval
    return parse_judge_reply(resp.text)
