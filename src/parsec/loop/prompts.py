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

from parsec.config import RunConfig, behaviour_level
from parsec.models.gateway import ModelRequest
from parsec.models.report import submission_model

# -- Versioned prompt text ---------------------------------------------------
#
# Every string and schema below is part of a recorded prompt's bytes, and so
# of its hash: editing one in place makes every existing recording diverge on
# replay. They are FROZEN. To change wording or a schema, add a new constant
# and select it by behaviour version in the accessor functions further down
# (decomposer_system, subagent_system, writer_system, repair_template,
# submit_subquestions_schema, submit_report_schema); never edit these.
# tests/unit/test_prompt_golden.py pins their hashes.

DECOMPOSER_SYSTEM = """You are the scoping phase of a research harness. Produce a research brief for the user's question:
- scope: 1-3 sentences stating what the research will and will not cover.
- effort: an honest complexity estimate the harness will enforce as dispatch caps — "quick" (a single simple lookup: one subagent, few calls), "standard" (a question with a couple of facets), "deep" (multi-facet synthesis needing full fan-out).
- subquestions: the smallest set of independent subquestions that together fully cover the scope. A simple question needs exactly one. Do not add subquestions the user did not ask about.

For a survey over a body of documents (regulatory guidance, standards, a literature area), covering the scope requires one subquestion that enumerates the current document landscape from the issuing authority's own index — including recent additions, drafts, and adjacent initiatives — so the synthesis cannot silently miss a document no other subquestion happens to surface. This enumeration is part of what such a question asks, not an addition to it.

Call submit_subquestions exactly once with the brief. Do not answer the question. If the user requests changes to a proposed brief, revise it and call submit_subquestions again."""

SUBAGENT_SYSTEM = """You are a research subagent inside a verification harness, assigned ONE subquestion. Gather evidence for it and record premises. You will not write any answer — a separate writer will, using only recorded premises and findings. Anything you do not record is lost.

Workflow:
1. search_broad to find relevant pages.
2. fetch promising URLs. fetch returns span IDs (doc:<hash>#<start>-<end>) with text previews.
3. record_premises for every fact you may need: atomic statements (one subject, one predicate, under 300 characters), quantities exactly as the span states them, span_refs listing the supporting span IDs. Set claim_class by mutability: "stable" for facts that do not change, "slow" for slowly-changing facts (populations, rankings), "volatile" for prices, versions, and roles. Rejected premises come back with reasons — fix and re-record. Record premises as soon as you have their spans — you have a hard turn cap, and anything unrecorded when it hits is lost. Do not save recording for one final batch.
4. When done, call submit_report exactly once: status (answered / partial / blocked), findings derived from your recorded premises (each cites premise IDs), conflicts between premises, and dead_ends (queries that yielded nothing, so they are not re-issued).

Rules:
- Numbers and quoted phrases in a premise must appear exactly in a cited span, or carry a transform_note explaining the derivation.
- A premise must stand alone: name the specific entity (never a bare "it", "they", or "the study"), state one statement per premise, and give specifics instead of vague terms like "benefits" or "significant" — vague or ambiguous premises are rejected.
- Temporal ordering findings ("X before/after Y") must cite exactly two premises, in the order the events are mentioned, and the dates must appear in the premises or their spans.
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
- moderate: hedge lightly ("X is likely Y", "evidence indicates") — do NOT name or characterize the source in prose; the citation carries provenance, and the appendix carries the tier.
- low: attribute it ("one source suggests", "according to <source>") — never state it as settled fact.

Substance first: the reader came for the answer, not a catalog of your evidence. Never open consecutive sentences with attribution or provenance phrasing ("A secondary summary indicates...", "One source states..."); reserve that framing for the few genuinely low-tier claims. Spend your words on what is known, not on how it came to be known.

Be concise. Answer directly."""

# derivations >= 2: premises are verbatim; derived figures are computes
# findings. Built from the frozen text, replacing exactly the one rule.
_SUBAGENT_TRANSFORM_RULE = (
    "- Numbers and quoted phrases in a premise must appear exactly in a cited span, or carry "
    "a transform_note explaining the derivation.\n"
)
_SUBAGENT_DERIVATION_RULES = (
    "- Numbers and quoted phrases in a premise must appear exactly in a cited span. A premise "
    "states only what its source states.\n"
    "- A figure you derive — a unit conversion, sum, difference, ratio, percent change or "
    "rounding — is never a premise. Submit it in submit_report as a finding with edge_type "
    "\"computes\" and an operation: the operands are numbers taken from premises you recorded "
    "(premise id plus the value as that premise states it), and the harness recomputes the "
    "output. If the arithmetic is wrong the finding comes back with the correct value.\n"
    "- Every number in any other finding must appear in the premises that finding cites.\n"
)
assert SUBAGENT_SYSTEM.count(_SUBAGENT_TRANSFORM_RULE) == 1
SUBAGENT_SYSTEM_DERIVATIONS = SUBAGENT_SYSTEM.replace(
    _SUBAGENT_TRANSFORM_RULE, _SUBAGENT_DERIVATION_RULES
)

# sentence_check >= 1: the citation contract also covers what a sentence SAYS,
# and the writer is told so. Built from the frozen text so the two cannot
# drift apart anywhere else.
_WRITER_CONTRACT_V1_TAIL = (
    "- If some subquestions are listed as unresolved, acknowledge the gap in [narrative] "
    "sentences; do not paper over it.\n"
)
_WRITER_SENTENCE_RULES = (
    "- Every number and every quoted phrase in a sentence must appear exactly in a premise or "
    "finding that the sentence cites. State quantities as the evidence states them: do not "
    "round, convert, or combine figures, and cite every premise a sentence depends on.\n"
    "- [narrative] sentences carry no evidence, so they may contain no numbers and no "
    "quotations. A heading may only use figures that appear in evidence the answer cites.\n"
)
# claim_credence >= 2: a sentence is scored by the weakest thing it cites.
_WRITER_CLAIM_RULES = (
    "- A sentence is only as credible as the weakest premise or finding it cites, and citing "
    "more of them does not strengthen it. Write one fact per sentence: do not join unrelated "
    "facts in one sentence, and cite every premise that sentence depends on.\n"
)
assert WRITER_SYSTEM.count(_WRITER_CONTRACT_V1_TAIL) == 1


def _writer_system(sentence_rules: bool, claim_rules: bool) -> str:
    extra = (_WRITER_SENTENCE_RULES if sentence_rules else "") + (
        _WRITER_CLAIM_RULES if claim_rules else ""
    )
    return WRITER_SYSTEM.replace(_WRITER_CONTRACT_V1_TAIL, _WRITER_CONTRACT_V1_TAIL + extra)


WRITER_SYSTEM_SENTENCE_CHECK = _writer_system(sentence_rules=True, claim_rules=False)

SUBMIT_SUBQUESTIONS_SCHEMA: dict = {
    "name": "submit_subquestions",
    "description": "Submit the research brief: scope, effort estimate, and decomposition.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "description": "1-3 sentences: what the research will and will not cover",
            },
            "effort": {
                "type": "string",
                "enum": ["quick", "standard", "deep"],
                "description": "Complexity estimate; the harness enforces it as dispatch caps",
            },
            "subquestions": {
                "type": "array",
                "items": {"type": "string", "minLength": 3},
                "minItems": 1,
            },
        },
        "required": ["subquestions"],
        "additionalProperties": False,
    },
}


# recency >= 2: the brief says what time each subquestion asks about, and
# the subagent dates what it records. Additions to the frozen text only.
_DECOMPOSER_SUBQUESTIONS_RULE = (
    "Do not add subquestions the user did not ask about.\n"
)
_DECOMPOSER_TARGET_RULE = (
    "- target_times: one entry per subquestion, in the same order — \"now\" when the "
    "subquestion asks about the present state of things, or the period it asks about when it "
    "names one (\"2019\", \"March 2025\", \"Q3 2025\"). Evidence is weighed by how close "
    "its date is to this time, so a question about the past must say so.\n"
)
assert DECOMPOSER_SYSTEM.count(_DECOMPOSER_SUBQUESTIONS_RULE) == 1
DECOMPOSER_SYSTEM_TARGET_TIMES = DECOMPOSER_SYSTEM.replace(
    _DECOMPOSER_SUBQUESTIONS_RULE, _DECOMPOSER_SUBQUESTIONS_RULE + _DECOMPOSER_TARGET_RULE
)
SUBMIT_SUBQUESTIONS_SCHEMA_TARGET_TIMES = {
    **SUBMIT_SUBQUESTIONS_SCHEMA,
    "input_schema": {
        **SUBMIT_SUBQUESTIONS_SCHEMA["input_schema"],
        "properties": {
            **SUBMIT_SUBQUESTIONS_SCHEMA["input_schema"]["properties"],
            "target_times": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "One per subquestion, same order: \"now\", or the period that "
                    "subquestion asks about (e.g. \"2019\", \"March 2025\")"
                ),
            },
        },
    },
}
_SUBAGENT_DATING_RULE = (
    "\n- Your subquestion states the time frame it asks about. When a span says WHEN a fact "
    "held, set as_of on that premise to the date as the span gives it (a year, a month and "
    "year, a quarter or a full date); the harness checks it appears in the span. Leave as_of "
    "out when the span gives no date — never guess one."
)


def decomposer_system(config: RunConfig) -> str:
    if behaviour_level(config.recency) >= 2:
        return DECOMPOSER_SYSTEM_TARGET_TIMES
    return DECOMPOSER_SYSTEM


def subagent_system(config: RunConfig) -> str:
    text = SUBAGENT_SYSTEM
    if behaviour_level(config.derivations) >= 2:
        text = SUBAGENT_SYSTEM_DERIVATIONS
    if behaviour_level(config.recency) >= 2:
        text += _SUBAGENT_DATING_RULE
    return text


def writer_system(config: RunConfig) -> str:
    return _writer_system(
        sentence_rules=behaviour_level(config.sentence_check) >= 1,
        claim_rules=behaviour_level(config.claim_credence) >= 2,
    )


def repair_template(config: RunConfig) -> str:
    if behaviour_level(config.sentence_check) >= 1:
        return REPAIR_TEMPLATE_SENTENCE_CHECK
    return REPAIR_TEMPLATE


def submit_subquestions_schema(config: RunConfig) -> dict:
    if behaviour_level(config.recency) >= 2:
        return SUBMIT_SUBQUESTIONS_SCHEMA_TARGET_TIMES
    return SUBMIT_SUBQUESTIONS_SCHEMA


def submit_report_schema(config: RunConfig) -> dict:
    schema = submission_model(config).model_json_schema()
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


# -- cache_strategy="full" placement (Phase 4) -------------------------------
#
# Three breakpoints per request (limit is 4): system (both strategies),
# the last tool schema (caching the whole tools array), and a ROLLING one
# on the final block of the final message — each request writes cache over
# its whole prompt, and the next append-only request reads that prefix at
# 0.1x. Markers are applied to COPIES at request-build time: the loop's
# stored transcript stays unmarked, and placement is a pure function of
# (config, transcript), so recorded runs rebuild identical bytes (T4).

_EPHEMERAL = {"type": "ephemeral"}
CACHEABLE_BLOCK_TYPES = frozenset({"text", "tool_result", "tool_use", "image", "document"})


def _cache_tools(tools: list[dict]) -> list[dict]:
    if not tools:
        return tools
    return [*tools[:-1], {**tools[-1], "cache_control": _EPHEMERAL}]


def _block_content(content) -> list[dict]:
    """Full mode sends every message in block form: a rolling marker turns a
    string into blocks, and the SAME bytes must appear when that message is
    an unmarked prefix of a later request — so normalize all of them."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content


def _mark_messages(messages: list[dict]) -> list[dict]:
    out = [{**m, "content": _block_content(m["content"])} for m in messages]
    if not out:
        return out
    blocks = out[-1]["content"]
    if blocks and blocks[-1].get("type") in CACHEABLE_BLOCK_TYPES:
        out[-1] = {
            **out[-1],
            "content": [*blocks[:-1], {**blocks[-1], "cache_control": _EPHEMERAL}],
        }
    return out


def apply_cache_strategy(
    config: RunConfig, tools: list[dict], messages: list[dict]
) -> tuple[list[dict], list[dict]]:
    if config.cache_strategy != "full":
        return tools, messages
    return _cache_tools(tools), _mark_messages(messages)


def build_decomposer_request(
    config: RunConfig,
    query: str,
    messages: list[dict] | None = None,
    tool_choice: dict | None = None,
) -> ModelRequest:
    """messages overrides the default single-query turn for brief-gate
    revision rounds (M12) and structured repair rounds (Phase 3) — the
    transcript stays append-only, so the decomposer's system+tools prefix
    keeps its KV-cache hit (§7). tool_choice pins submit_subquestions on a
    repair's final attempt."""
    tools, messages = apply_cache_strategy(
        config,
        [submit_subquestions_schema(config)],
        messages if messages is not None else [{"role": "user", "content": query}],
    )
    return ModelRequest(
        model=config.model,
        max_tokens=config.max_tokens_per_call,
        system=_system_block(decomposer_system(config)),
        tools=tools,
        messages=messages,
        tool_choice=tool_choice,
    )


def build_subagent_request(
    config: RunConfig, tools: list[dict], messages: list[dict]
) -> ModelRequest:
    tools, messages = apply_cache_strategy(
        config, [*tools, submit_report_schema(config)], messages
    )
    return ModelRequest(
        model=config.model,
        max_tokens=config.max_tokens_per_call,
        system=_system_block(subagent_system(config)),
        tools=tools,
        messages=messages,
    )


def build_writer_request(config: RunConfig, messages: list[dict]) -> ModelRequest:
    tools, messages = apply_cache_strategy(config, [], messages)
    return ModelRequest(
        model=config.model,
        max_tokens=config.max_tokens_per_call,
        system=_system_block(writer_system(config)),
        tools=tools,
        messages=messages,
    )


def subagent_user_prompt(
    sq_id: str, question: str, scope: str = "", target_time: str | None = None
) -> str:
    """The brief's scope rides in the FIRST user message only (M12): it never
    mutates mid-session, so the subagent's prompt prefix stays cache-stable.
    target_time (recency >= 2; None for sessions that predate it) adds the
    time frame the subquestion asks about."""
    prompt = f"Subquestion {sq_id}: {question}"
    if target_time is not None:
        frame = (
            "the present — what is true now"
            if target_time == "now"
            else target_time.replace("/", " to ")
        )
        prompt += f"\nTime frame: {frame}"
    if scope:
        return f"Research brief: {scope}\n\n{prompt}"
    return prompt


def writer_user_prompt(
    query: str,
    premises: list[sqlite3.Row],
    findings: list[sqlite3.Row],
    sources: dict[str, str],
    unresolved: list[tuple[str, str, str]],  # (sq_id, question, "status: reason")
    confidence: dict[str, str] | None = None,  # node_id -> annotation, e.g. "high" / "low (single source)"
    max_text_chars: int | None = None,  # Phase 2 degradation: clip evidence TEXT, never ids/sources
) -> str:
    """The writer's entire view of the world: query + distilled evidence + gaps.
    Confidence annotations are computed by the credence model (§6.5) — the
    writer applies the register, it never invents the tier."""
    confidence = confidence or {}

    def clip(text: str) -> str:
        if max_text_chars is None or len(text) <= max_text_chars:
            return text
        return text[:max_text_chars] + "…[clipped to fit context]"

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
        lines.append(f"[{row['node_id']}] {clip(payload['text'])}{suffix}")
    if findings:
        lines += ["", "Findings (derived from premises):"]
        for row in findings:
            payload = json.loads(row["payload_json"])
            deps = ", ".join(payload["premise_ids"])
            conf = f"; confidence: {confidence[row['node_id']]}" if row["node_id"] in confidence else ""
            lines.append(f"[{row['node_id']}] {clip(payload['text'])} (from: {deps}{conf})")
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

REPAIR_TEMPLATE_SENTENCE_CHECK = (
    "Your answer violated the citation contract. Problems:\n{problems}\n"
    "Revise the FULL answer. Every factual sentence must end with valid [premise:<id>] or "
    "[finding:<id>] citations from the provided list; structural sentences must end with "
    "[narrative]. Every number and quoted phrase in a sentence must appear exactly in a premise "
    "or finding that sentence cites: correct the figure to match the evidence, cite the premise "
    "that states it, or remove it. Do not add new claims."
)

TRUNCATED_NUDGE = "Your answer was cut off by the length limit. Restate it more concisely, keeping the citation contract."

FINAL_TURN_NUDGE = (
    "[harness] This is your final turn — the turn cap is reached after this response. "
    "Call record_premises now for any facts not yet recorded, and call submit_report. "
    "Anything unrecorded will be lost."
)
