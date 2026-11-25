import json

import pytest
from pydantic import BaseModel, ConfigDict, Field

from parsec.models.events import EventType
from parsec.models.tools import ToolIntent
from parsec.tools.base import ToolContext, ToolRegistry


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)


class EchoTool:
    name = "echo"
    description = "Echo text back"
    input_model = EchoInput
    max_context_chars = 50

    async def run(self, input: EchoInput, ctx: ToolContext) -> tuple[dict, str]:
        return {"echoed": input.text}, input.text


class BoomTool:
    name = "boom"
    description = "Always fails"
    input_model = EchoInput
    max_context_chars = 50

    async def run(self, input, ctx):
        raise RuntimeError("kaput")


@pytest.fixture
def ctx(db, blobs, event_log, ledger, sessions, config, clock):
    sessions.create(config)
    return ToolContext(db, blobs, event_log, ledger, config, clock)


@pytest.fixture
def registry():
    return ToolRegistry([EchoTool(), BoomTool()])


def test_schema_export_deterministic_sorted(registry, config):
    schemas = registry.export_schemas(config)
    assert [s["name"] for s in schemas] == ["boom", "echo"]
    assert schemas[1]["input_schema"]["additionalProperties"] is False
    assert registry.export_schemas(config) == schemas


class ShoutInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    volume: int


class VersionedEchoTool(EchoTool):
    """A tool whose contract changed at sentence_check 1: the schema the model
    is shown and the validation of its call must both follow the session."""

    def spec_for(self, config):
        if config.sentence_check == 1:
            return "Echo text back, loudly", ShoutInput
        return self.description, self.input_model

    async def run(self, input, ctx) -> tuple[dict, str]:
        return {"echoed": input.text}, input.text


def test_tool_spec_follows_the_sessions_behaviour_version(config):
    registry = ToolRegistry([VersionedEchoTool()])
    legacy = registry.export_schemas(config)[0]
    current = registry.export_schemas(config.model_copy(update={"sentence_check": 1}))[0]
    assert legacy["description"] == "Echo text back"
    assert "volume" not in legacy["input_schema"]["properties"]
    assert current["description"] == "Echo text back, loudly"
    assert "volume" in current["input_schema"]["properties"]


async def test_dispatch_validates_with_the_schema_the_session_was_shown(ctx, config):
    registry = ToolRegistry([VersionedEchoTool()])
    intent = ToolIntent(tool_use_id="tu_1", tool_name="echo", input={"text": "hi"})
    # legacy session: the legacy input model accepts the call
    assert (await registry.dispatch(intent, ctx)).ok
    # a session on the newer contract rejects the same call (volume missing)
    newer = ToolContext(
        ctx.conn, ctx.blobs, ctx.event_log, ctx.ledger,
        config.model_copy(update={"sentence_check": 1}), ctx.clock,
    )
    result = await registry.dispatch(intent, newer)
    assert not result.ok and "volume" in result.error


async def test_dispatch_success_records_events(registry, ctx, event_log, config):
    intent = ToolIntent(tool_use_id="t1", tool_name="echo", input={"text": "hi"})
    result = await registry.dispatch(intent, ctx)
    assert result.ok and result.truncated_text == "hi"
    types = [e.event_type for e in event_log.read(config.session_id)]
    assert EventType.TOOL_INTENT in types and EventType.TOOL_RESULT in types


async def test_invalid_input_is_error_result_not_exception(registry, ctx):
    intent = ToolIntent(tool_use_id="t2", tool_name="echo", input={"wrong": 1})
    result = await registry.dispatch(intent, ctx)
    assert not result.ok
    assert "invalid input" in result.error


async def test_unknown_tool(registry, ctx):
    intent = ToolIntent(tool_use_id="t3", tool_name="nope", input={})
    result = await registry.dispatch(intent, ctx)
    assert not result.ok and "unknown tool" in result.error


async def test_tool_exception_surfaced_as_error(registry, ctx):
    intent = ToolIntent(tool_use_id="t4", tool_name="boom", input={"text": "x"})
    result = await registry.dispatch(intent, ctx)
    assert not result.ok and "kaput" in result.error


async def test_truncation_marker_and_full_blob(registry, ctx, blobs):
    long_text = "z" * 200
    intent = ToolIntent(tool_use_id="t5", tool_name="echo", input={"text": long_text})
    result = await registry.dispatch(intent, ctx)
    assert result.ok
    assert "truncated" in result.truncated_text
    assert len(result.truncated_text) < 250
    full = json.loads(blobs.get_text(result.full_blob))
    assert full["echoed"] == long_text


async def test_wall_ms_debited(registry, ctx, ledger, config):
    intent = ToolIntent(tool_use_id="t6", tool_name="echo", input={"text": "hi"})
    await registry.dispatch(intent, ctx)
    by_actor = ledger.totals_by_actor(config.session_id)
    assert ("tool:echo", "wall_ms") in by_actor
