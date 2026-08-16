"""Prompt assembly (§7 cache-aware): the system prompt + tool schemas form
an immutable prefix; conversation is append-only. Nothing mutates the
front of the context mid-session."""

from __future__ import annotations

from parsec.config import RunConfig
from parsec.models.gateway import ModelRequest

SYSTEM_PROMPT = """You are a research agent inside a verification harness. Your job: answer the user's question using ONLY evidence you retrieve with your tools.

Workflow:
1. Use search_broad to find relevant pages.
2. Use fetch on promising URLs. fetch returns citable span IDs of the form [doc:<hash>#<start>-<end>] with text previews.
3. When you have enough evidence, write your final answer.

Citation contract (mechanically enforced — violations are rejected):
- Every factual sentence in your final answer MUST end with one or more citations of the exact form [doc:<hash>#<start>-<end>], using ONLY span IDs returned by fetch in this conversation.
- Purely structural sentences (transitions, headers) must end with the tag [narrative].
- Never cite a span you have not seen returned by fetch. Never invent span IDs.
- If the evidence you found does not answer the question, say so plainly — in sentences tagged [narrative] — rather than citing weakly related spans.

Be concise. Do not pad. Answer directly."""


def build_system() -> list[dict]:
    return [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def build_request(config: RunConfig, tools: list[dict], messages: list[dict]) -> ModelRequest:
    return ModelRequest(
        model=config.model,
        max_tokens=config.max_tokens_per_call,
        system=build_system(),
        tools=tools,
        messages=messages,
    )


FORCED_ANSWER_NUDGE = (
    "Budget limits reached. Stop researching now and write your best final answer from the "
    "evidence already gathered, following the citation contract. If evidence is insufficient, "
    "say so in [narrative] sentences."
)

REPAIR_TEMPLATE = (
    "Your answer violated the citation contract. Problems:\n{problems}\n"
    "Revise the FULL answer. Every factual sentence must end with valid span IDs already "
    "returned by fetch; structural sentences must end with [narrative]. Do not add new claims."
)

TRUNCATED_NUDGE = "Your answer was cut off by the length limit. Restate it more concisely, keeping the citation contract."
