"""Prompt assembly (§7 cache-aware): each phase's system prompt + tool
schemas form an immutable prefix; conversation is append-only. Nothing
mutates the front of the context mid-session.

As of M2 the loop has two phases with separate prefixes:
  research — tool loop; facts enter the DAG only via record_premises
  writer   — no tools; sees ONLY the query + recorded premises (§6.5),
             never raw spans or the research transcript
"""

from __future__ import annotations

import sqlite3

from parsec.config import RunConfig
from parsec.models.gateway import ModelRequest

RESEARCH_SYSTEM = """You are the research phase of a verification harness. Your job: gather evidence for the user's question and record it as premises. You will NOT write the final answer — a separate writer will, using ONLY the premises you record. Anything you do not record is lost.

Workflow:
1. search_broad to find relevant pages.
2. fetch promising URLs. fetch returns span IDs (doc:<hash>#<start>-<end>) with text previews.
3. record_premises for every fact you may need: atomic statements (one subject, one predicate), quantities exactly as the span states them, span_refs listing the supporting span IDs. Rejected premises come back with reasons — fix and re-record.
4. When you have recorded enough premises to answer the question, stop calling tools and reply with a one-line summary of what you found.

Rules:
- Numbers and quoted phrases in a premise must appear exactly in a cited span, or carry a transform_note explaining the derivation.
- Record premises for conflicting evidence too; do not resolve conflicts silently.
- Do not re-issue near-identical searches that returned nothing."""

WRITER_SYSTEM = """You are the writer phase of a verification harness. Write the final answer to the user's question using ONLY the premises provided. You have no tools and no other knowledge for this task: if the premises do not support a statement, you must not make it.

Citation contract (mechanically enforced — violations are rejected):
- Every factual sentence MUST end with one or more citations of the exact form [premise:<id>], using only premise IDs from the provided list.
- Purely structural sentences (transitions, framing) must end with the tag [narrative].
- If the premises cannot answer the question, say so plainly in [narrative] sentences.

Be concise. Answer directly."""


def _system_block(text: str) -> list[dict]:
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def build_research_request(
    config: RunConfig, tools: list[dict], messages: list[dict]
) -> ModelRequest:
    return ModelRequest(
        model=config.model,
        max_tokens=config.max_tokens_per_call,
        system=_system_block(RESEARCH_SYSTEM),
        tools=tools,
        messages=messages,
    )


def build_writer_request(config: RunConfig, messages: list[dict]) -> ModelRequest:
    return ModelRequest(
        model=config.model,
        max_tokens=config.max_tokens_per_call,
        system=_system_block(WRITER_SYSTEM),
        tools=[],
        messages=messages,
    )


def writer_user_prompt(query: str, premises: list[sqlite3.Row], sources: dict[str, str]) -> str:
    """The writer's entire view of the world: query + distilled premises."""
    lines = [f"Question: {query}", "", "Premises:"]
    if not premises:
        lines.append(
            "(none were recorded — state in [narrative] sentences that no supported answer can be given)"
        )
    import json

    for row in premises:
        payload = json.loads(row["payload_json"])
        url = sources.get(row["node_id"], "")
        suffix = f" (source: {url})" if url else ""
        lines.append(f"[{row['node_id']}] {payload['text']}{suffix}")
    return "\n".join(lines)


REPAIR_TEMPLATE = (
    "Your answer violated the citation contract. Problems:\n{problems}\n"
    "Revise the FULL answer. Every factual sentence must end with valid [premise:<id>] citations "
    "from the provided premise list; structural sentences must end with [narrative]. Do not add "
    "new claims."
)

TRUNCATED_NUDGE = "Your answer was cut off by the length limit. Restate it more concisely, keeping the citation contract."
