"""Compaction ladder (§7), v1 rungs for subagent contexts.

Rung 1 — EVICT: replace old tool-result contents with explicit markers.
The evidence is not lost: spans and premises live in the store, not the
context (§5 — provenance by construction).
Rung 3 — RESET: fresh context seeded from the recorded evidence, a
controlled restart rather than a truncation.
Rung 2 (model-written squeeze into the notebook) is deferred: it costs
model calls; the notebook already receives distilled evidence at
submit_report time.

Every decision here is a pure function of the transcript's character
counts, so compaction replays byte-identically (T4).
"""

from __future__ import annotations

from parsec.canonical import canonical_json

EVICTION_MARKER = (
    "[evicted to save context: tool result withheld; evidence remains addressable "
    "via recorded span and premise IDs]"
)


def context_chars(messages: list[dict]) -> int:
    return len(canonical_json(messages))


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
