"""Compaction ladder (§7), full three rungs as of M12.

Rung 1 — EVICT: replace old tool-result contents with explicit markers.
The evidence is not lost: spans and premises live in the store, not the
context (§5 — provenance by construction).
Rung 2 — RECONSTRUCT (M12, WS-F.2): round-based context reconstruction.
The field moved past transcript compaction to workspace reconstruction —
and our evidence DAG IS the external memory, so rung 2 re-renders the
relevant DAG slice (this subagent's premises with span refs and sources)
plus the notebook into a fresh workspace. A deterministic function of the
log — better AND cheaper than the once-planned model-written squeeze,
which is why that never shipped.
Rung 3 — RESET: the minimal fallback when even the reconstructed
workspace exceeds the budget — assignment plus bare premise texts.

Triggering is TOKEN-aware as of Phase 2: the estimate counts the system
prompt and tool schemas (previously invisible to the char trigger) and is
anchored on the previous response's journaled usage — the exact context
the model last consumed is a recorded lower bound, and only content
appended since is estimated at ~4 chars/token. Every decision remains a
pure function of the transcript and recorded data, so compaction replays
byte-identically (T4).
"""

from __future__ import annotations

from parsec.canonical import canonical_json

EVICTION_MARKER = (
    "[evicted to save context: tool result withheld; evidence remains addressable "
    "via recorded span and premise IDs]"
)

# The classic English/JSON heuristic. Deliberately crude: the REACTIVE
# path (context_overflow -> compact -> retry) catches what it misses.
CHARS_PER_TOKEN = 4


def context_chars(messages: list[dict]) -> int:
    return len(canonical_json(messages))


def static_prefix_chars(system_text: str, tools: list[dict]) -> int:
    """Serialized size of the per-phase immutable prefix (system + tool
    schemas) — part of every request, previously uncounted by the trigger."""
    return len(system_text) + len(canonical_json(tools))


def trailing_chars(messages: list[dict]) -> int:
    """Chars of messages appended after the last assistant message — i.e.
    content the model has not seen (and usage has not measured) yet."""
    last_assistant = None
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            last_assistant = i
    if last_assistant is None:
        return context_chars(messages)
    return context_chars(messages[last_assistant + 1 :])


def estimate_tokens(messages: list[dict], static_chars: int, last_usage_tokens: int | None) -> int:
    """Estimated input tokens for the next call on this transcript.

    The char estimate covers everything at ~4 chars/token; when the
    previous response's usage is known (input + output + cache tokens =
    the whole context the model just consumed), it floors the estimate:
    the next call cannot be smaller than that plus what was appended
    since. Both inputs are recorded, so the estimate replays (T4)."""
    chars_est = (static_chars + context_chars(messages)) // CHARS_PER_TOKEN
    if last_usage_tokens is None:
        return chars_est
    return max(chars_est, last_usage_tokens + trailing_chars(messages) // CHARS_PER_TOKEN)


def evict_tool_results(messages: list[dict], keep_last: int) -> tuple[list[dict], int]:
    """Rung 1: replace tool-result contents with markers, oldest first,
    preserving the last `keep_last` tool-result messages. Returns the new
    message list and how many results were evicted."""
    tool_msg_indices = [
        i
        for i, m in enumerate(messages)
        if m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any(b.get("type") == "tool_result" for b in m["content"])
    ]
    to_evict = tool_msg_indices[: max(0, len(tool_msg_indices) - keep_last)]
    if not to_evict:
        return messages, 0
    evicted = 0
    out = []
    for i, m in enumerate(messages):
        if i in to_evict:
            blocks = []
            for b in m["content"]:
                if b.get("type") == "tool_result" and b.get("content") != EVICTION_MARKER:
                    blocks.append({**b, "content": EVICTION_MARKER})
                    evicted += 1
                else:
                    blocks.append(b)
            out.append({**m, "content": blocks})
        else:
            out.append(m)
    return out, evicted


def reconstruct_context(subagent_prompt: str, workspace_md: str) -> list[dict]:
    """Rung 2: fresh context = the assignment + a workspace re-rendered from
    the evidence DAG and notebook (built by the orchestrator, which owns the
    stores). One user message, so the phase's cache prefix stays stable."""
    return [{"role": "user", "content": f"{subagent_prompt}\n\n{workspace_md}"}]


def render_workspace(
    premise_lines: list[str], notebook_md: str
) -> str:
    """The rung-2 workspace body: what is already saved (do not re-record),
    with provenance, plus the session notebook."""
    lines = [
        "(Your context was compacted. This workspace was reconstructed from the "
        "evidence graph — everything below is already saved in the store; do NOT "
        "re-record it. Continue researching what is still missing, or call "
        "submit_report.)",
        "",
        "## Premises you have recorded",
    ]
    lines += premise_lines if premise_lines else ["(none yet)"]
    if notebook_md:
        lines += ["", "## Session notebook", notebook_md]
    return "\n".join(lines)


def reset_context(
    subagent_prompt: str, recorded_premise_texts: list[str]
) -> list[dict]:
    """Rung 3: controlled restart — the new context is the assignment plus
    everything durable the subagent has produced so far."""
    lines = [
        subagent_prompt,
        "",
        "(Your context was reset to stay within budget. Evidence you already "
        "recorded is safe in the store and listed below — do not re-record it. "
        "Continue researching what is still missing, or call submit_report.)",
    ]
    if recorded_premise_texts:
        lines.append("")
        lines.append("Premises already recorded:")
        lines += [f"- {t}" for t in recorded_premise_texts]
    return [{"role": "user", "content": "\n".join(lines)}]
