"""Tool layer (T1): registry, schema validation, execution, truncation.

The model emits intents; only this layer executes. Validation failures are
returned to the model as error tool results — they never crash the loop.
Full tool output always lands in the blob store; only a truncated copy
enters model context, with an explicit marker when truncation occurred.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Protocol

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


class Tool(Protocol):
    name: str
    description: str
    input_model: type[BaseModel]
    max_context_chars: int

    async def run(self, input: BaseModel, ctx: ToolContext) -> tuple[dict, str]:
        """Return (full_result_dict, context_text). Truncation applied by the registry."""
        ...


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in sorted(tools, key=lambda t: t.name)}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def export_schemas(self) -> list[dict]:
        """Anthropic tool schemas, deterministic order (prefix stability, §7)."""
        out = []
        for tool in self._tools.values():
            schema = tool.input_model.model_json_schema()
            schema.pop("title", None)
            schema["additionalProperties"] = False
            out.append(
                {"name": tool.name, "description": tool.description, "input_schema": schema}
            )
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
            validated = tool.input_model.model_validate(intent.input)
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
