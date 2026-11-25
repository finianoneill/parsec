"""Tool layer (T1): registry, schema validation, execution, truncation.

The model emits intents; only this layer executes. Validation failures are
returned to the model as error tool results — they never crash the loop.
Full tool output always lands in the blob store; only a truncated copy
enters model context, with an explicit marker when truncation occurred.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from parsec.canonical import canonical_json, sha256_hex
from parsec.config import Clock, RunConfig
from parsec.models.events import EventType
from parsec.models.tools import ToolIntent, ToolResult
from parsec.store.blobs import BlobStore
from parsec.store.event_log import EventLog
from parsec.store.ledger import Ledger


@dataclass
class ToolContext:
    conn: sqlite3.Connection
    blobs: BlobStore
    event_log: EventLog
    ledger: Ledger
    config: RunConfig
    clock: Clock
    # recency >= 2: the time each subquestion asks about, by stream (sq id) —
    # "now" or a stored interval. The loop fills it in at dispatch; a tool
    # reads its own stream's entry. Empty for sessions that predate it.
    stream_targets: dict[str, str] = field(default_factory=dict)


class Tool(Protocol):
    # Read-only members: a settable `input_model: type[BaseModel]` is
    # invariant, so no tool declaring its own input class would conform.
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def input_model(self) -> type[BaseModel]: ...
    @property
    def max_context_chars(self) -> int: ...

    # `input` is the tool's own input_model instance; a Protocol parameter
    # typed BaseModel would reject every concrete tool (parameters are
    # contravariant), so it is Any here and validated by the registry.
    async def run(self, input: Any, ctx: ToolContext) -> tuple[dict, str]:
        """Return (full_result_dict, context_text). Truncation applied by the registry."""
        ...


def tool_spec(tool: Tool, config: RunConfig) -> tuple[str, type[BaseModel]]:
    """(description, input model) a tool presents under this session's
    behaviour versions. A tool whose wording or schema differs by version
    defines `spec_for(config)` returning that pair; the rest present their
    class attributes. Export and dispatch both come through here, so the
    schema the model was shown is always the one its call is validated by."""
    spec_for = getattr(tool, "spec_for", None)
    if spec_for is not None:
        description, input_model = spec_for(config)
        return description, input_model
    return tool.description, tool.input_model


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in sorted(tools, key=lambda t: t.name)}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def export_schemas(self, config: RunConfig) -> list[dict]:
        """Anthropic tool schemas, deterministic order (prefix stability, §7),
        as the session's behaviour versions define them — the schemas are
        prompt bytes, so a recording must get back the ones it was sent."""
        out = []
        for tool in self._tools.values():
            description, input_model = tool_spec(tool, config)
            schema = input_model.model_json_schema()
            schema.pop("title", None)
            schema["additionalProperties"] = False
            out.append({"name": tool.name, "description": description, "input_schema": schema})
        return out

    async def dispatch(self, intent: ToolIntent, ctx: ToolContext) -> ToolResult:
        sid = ctx.config.session_id
        intent_seq = ctx.event_log.append(
            sid,
            "model",
            EventType.TOOL_INTENT,
            {"tool_use_id": intent.tool_use_id, "tool_name": intent.tool_name, "input": intent.input},
        )

        tool = self._tools.get(intent.tool_name)
        if tool is None:
            result = ToolResult(
                tool_use_id=intent.tool_use_id,
                tool_name=intent.tool_name,
                ok=False,
                error=f"unknown tool: {intent.tool_name!r}; available: {', '.join(self.names)}",
            )
            return self._record(result, ctx, intent_seq)

        try:
            validated = tool_spec(tool, ctx.config)[1].model_validate(intent.input)
        except ValidationError as exc:
            result = ToolResult(
                tool_use_id=intent.tool_use_id,
                tool_name=intent.tool_name,
                ok=False,
                error=f"invalid input for {tool.name}: {exc.error_count()} error(s): {exc}",
            )
            return self._record(result, ctx, intent_seq)

        start = ctx.clock.monotonic()
        try:
            full, context_text = await tool.run(validated, ctx)
        except Exception as exc:  # tool failure surfaces to the model, not the loop
            result = ToolResult(
                tool_use_id=intent.tool_use_id,
                tool_name=intent.tool_name,
                ok=False,
                error=f"{tool.name} failed: {type(exc).__name__}: {exc}",
            )
            return self._record(result, ctx, intent_seq)
        finally:
            wall_ms = (ctx.clock.monotonic() - start) * 1000
            ctx.ledger.debit(sid, "wall_ms", wall_ms, f"tool:{intent.tool_name}", ref_seq=intent_seq)

        full_blob = ctx.blobs.put(canonical_json(full))
        truncated = self._truncate(context_text, tool.max_context_chars, full_blob)
        result = ToolResult(
            tool_use_id=intent.tool_use_id,
            tool_name=intent.tool_name,
            ok=True,
            full_blob=full_blob,
            truncated_text=truncated,
        )
        return self._record(result, ctx, intent_seq)

    @staticmethod
    def _truncate(text: str, limit: int, full_blob: str) -> str:
        if len(text) <= limit:
            return text
        omitted = len(text) - limit
        return (
            text[:limit]
            + f"\n…[truncated: full result {full_blob}, {omitted} chars omitted; spans remain addressable]"
        )

    @staticmethod
    def _record(result: ToolResult, ctx: ToolContext, intent_seq: int) -> ToolResult:
        ctx.event_log.append(
            ctx.config.session_id,
            f"tool:{result.tool_name}",
            EventType.TOOL_RESULT,
            {
                "tool_use_id": result.tool_use_id,
                "ok": result.ok,
                "error": result.error,
                "full_blob": result.full_blob,
                "truncated_hash": sha256_hex(result.truncated_text),
            },
            parent_seq=intent_seq,
        )
        return result
