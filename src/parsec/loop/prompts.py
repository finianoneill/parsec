"""Prompt assembly (§7 cache-aware): each phase's system prompt + tool
schemas form an immutable prefix; conversation is append-only. Nothing
mutates the front of the context mid-session.

Phases as of M3:
  decomposer — orchestrator call; sees ONLY the user query
  subagent   — per-subquestion tool loop; the only consumer of raw
               documents (T6); cannot spawn agents (no dispatch tool exists
               in its registry — the recursion ban is structural, §3)
  writer     — no tools; sees ONLY the query + premises/findings + coverage
               gaps (§6.5), never raw spans or any research transcript
"""

from __future__ import annotations

import json
import sqlite3

from parsec.config import RunConfig
from parsec.models.gateway import ModelRequest
from parsec.models.report import SubagentSubmission

DECOMPOSER_SYSTEM = """You are the planning phase of a research harness. Split the user's question into the smallest set of independent subquestions that together fully cover it. A simple question needs exactly one subquestion. Do not add subquestions the user did not ask about.

Call submit_subquestions exactly once with your decomposition. Do not answer the question."""

SUBAGENT_SYSTEM = """You are a research subagent inside a verification harness, assigned ONE subquestion. Gather evidence for it and record premises. You will not write any answer — a separate writer will, using only recorded premises and findings. Anything you do not record is lost.

Workflow:
1. search_broad to find relevant pages.
2. fetch promising URLs. fetch returns span IDs (doc:<hash>#<start>-<end>) with text previews.
3. record_premises for every fact you may need: atomic statements (one subject, one predicate), quantities exactly as the span states them, span_refs listing the supporting span IDs. Rejected premises come back with reasons — fix and re-record.
4. When done, call submit_report exactly once: status (answered / partial / blocked), findings derived from your recorded premises (each cites premise IDs), conflicts between premises, and dead_ends (queries that yielded nothing, so they are not re-issued).

Rules:
- Numbers and quoted phrases in a premise must appear exactly in a cited span, or carry a transform_note explaining the derivation.
- Record premises for conflicting evidence too; report conflicts upward, never resolve them silently.
- If you cannot find usable sources, submit_report with status "blocked" and your dead_ends. Do not fabricate.
- Work only on your assigned subquestion."""

WRITER_SYSTEM = """You are the writer phase of a verification harness. Write the final answer to the user's question using ONLY the premises and findings provided. You have no tools and no other knowledge for this task: if the premises and findings do not support a statement, you must not make it.

Citation contract (mechanically enforced — violations are rejected):
- Every factual sentence MUST end with one or more citations of the exact form [premise:<id>] or [finding:<id>], using only IDs from the provided list.
- Purely structural sentences (transitions, framing) must end with the tag [narrative].
- If some subquestions are listed as unresolved, acknowledge the gap in [narrative] sentences; do not paper over it.

Hedging register (each premise/finding carries a computed confidence tier — your wording must match it):
- high: state it plainly ("X is Y").
- moderate: hedge ("X is likely Y", "evidence indicates").
- low or single-source: attribute it ("one source suggests", "according to <source>") — never state it as settled fact.

Be concise. Answer directly."""

SUBMIT_SUBQUESTIONS_SCHEMA = {
    "name": "submit_subquestions",
    "description": "Submit the decomposition of the user's question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "subquestions": {
                "type": "array",
                "items": {"type": "string", "minLength": 3},
                "minItems": 1,
            }
        },
        "required": ["subquestions"],
        "additionalProperties": False,
    },
}


def submit_report_schema() -> dict:
    schema = SubagentSubmission.model_json_schema()
    schema.pop("title", None)
    schema["additionalProperties"] = False
    return {
        "name": "submit_report",
        "description": (
            "Submit your final report for the assigned subquestion. Call exactly once, "
            "after recording all premises. findings must cite premise IDs you recorded."
        ),
        "input_schema": schema,
    }


def _system_block(text: str) -> list[dict]:
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def build_decomposer_request(config: RunConfig, query: str) -> ModelRequest:
    return ModelRequest(
        model=config.model,
        max_tokens=config.max_tokens_per_call,
        system=_system_block(DECOMPOSER_SYSTEM),
        tools=[SUBMIT_SUBQUESTIONS_SCHEMA],
        messages=[{"role": "user", "content": query}],
    )


def build_subagent_request(
    config: RunConfig, tools: list[dict], messages: list[dict]
) -> ModelRequest:
    return ModelRequest(
        model=config.model,
        max_tokens=config.max_tokens_per_call,
        system=_system_block(SUBAGENT_SYSTEM),
        tools=tools + [submit_report_schema()],
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


def subagent_user_prompt(sq_id: str, question: str) -> str:
    return f"Subquestion {sq_id}: {question}"


def writer_user_prompt(
    query: str,
    premises: list[sqlite3.Row],
    findings: list[sqlite3.Row],
    sources: dict[str, str],
    unresolved: list[tuple[str, str, str]],  # (sq_id, question, "status: reason")
    confidence: dict[str, str] | None = None,  # node_id -> annotation, e.g. "high" / "low, single source"
) -> str:
    """The writer's entire view of the world: query + distilled evidence + gaps.
    Confidence annotations are computed by the credence model (§6.5) — the
    writer applies the register, it never invents the tier."""
    confidence = confidence or {}
    lines = [f"Question: {query}", "", "Premises:"]
    if not premises:
        lines.append(
            "(none were recorded — state in [narrative] sentences that no supported answer can be given)"
        )
    for row in premises:
        payload = json.loads(row["payload_json"])
        url = sources.get(row["node_id"], "")
        notes = [f"source: {url}"] if url else []
        if row["node_id"] in confidence:
            notes.append(f"confidence: {confidence[row['node_id']]}")
        suffix = f" ({'; '.join(notes)})" if notes else ""
        lines.append(f"[{row['node_id']}] {payload['text']}{suffix}")
    if findings:
        lines += ["", "Findings (derived from premises):"]
        for row in findings:
            payload = json.loads(row["payload_json"])
            deps = ", ".join(payload["premise_ids"])
            conf = f"; confidence: {confidence[row['node_id']]}" if row["node_id"] in confidence else ""
            lines.append(f"[{row['node_id']}] {payload['text']} (from: {deps}{conf})")
    if unresolved:
        lines += ["", "Unresolved subquestions (acknowledge these gaps in [narrative]):"]
        for sq_id, question, why in unresolved:
            lines.append(f"- {sq_id}: {question} — {why}")
    return "\n".join(lines)


REPAIR_TEMPLATE = (
    "Your answer violated the citation contract. Problems:\n{problems}\n"
    "Revise the FULL answer. Every factual sentence must end with valid [premise:<id>] or "
    "[finding:<id>] citations from the provided list; structural sentences must end with "
    "[narrative]. Do not add new claims."
)

TRUNCATED_NUDGE = "Your answer was cut off by the length limit. Restate it more concisely, keeping the citation contract."
