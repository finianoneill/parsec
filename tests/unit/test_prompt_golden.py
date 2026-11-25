"""Golden hashes for everything that is prompt bytes under LEGACY behaviour.

Replay asserts each request's prompt hash against the recording, so a changed
system prompt, nudge, or tool schema makes every existing recording diverge.
These hashes were computed at 366f1f6 — the commit the remediation's frozen
recordings come from — and must never change.

If this test fails you have done one of two things:
  - edited frozen prompt text or a frozen wire model in place. Revert it; add
    a new constant / model class and select it by behaviour version instead
    (see the accessors in parsec.loop.prompts and tool_spec in
    parsec.tools.base);
  - upgraded a dependency (pydantic above all) that changed how JSON schemas
    are generated. Old recordings no longer replay under that version — pin
    it back, or freeze the legacy schemas as literal dicts first.

Do not update the hashes to make the test pass.
"""

from __future__ import annotations

import pytest

from parsec.canonical import hash_obj, sha256_hex
from parsec.config import LEGACY_BEHAVIOUR, RunConfig
from parsec.loop import prompts
from parsec.tools.base import ToolRegistry
from parsec.tools.fetch import FetchTool
from parsec.tools.record_premises import RecordPremisesTool
from parsec.tools.search_broad import SearchBroadTool
from parsec.tools.search_within import SearchWithinTool

GOLDEN = {
    "decomposer_system": "afd47531f7068481",
    "subagent_system": "658c651ff5710b2f",
    "writer_system": "163e64b139a2c07d",
    "repair_template": "e47fa0e160d1af7b",
    "TRUNCATED_NUDGE": "09bd74cad4b5ff0e",
    "FINAL_TURN_NUDGE": "b98d0579a1c0e7e5",
    "submit_subquestions": "e6e497ccce195b4b",
    "submit_report": "36261ee22cc153af",
    "tool:fetch": "9ca6d9c60a91bc43",
    "tool:record_premises": "d205b2cf1748c1de",
    "tool:search_broad": "77b5492688f3dc8d",
    "tool:search_within": "0fde9a7a021cd3b9",
    # Whole requests: catches structural drift the pieces cannot (block
    # layout, cache_control placement, tool ordering).
    "req:decomposer:system": "2c2465d548d20c82",
    "req:subagent:system": "f2044bfc641bed4d",
    "req:writer:system": "43c3857a30f9a72b",
    "req:decomposer:full": "2e3bf8a84ae6cd9b",
    "req:subagent:full": "903d274502633f73",
    "req:writer:full": "20e10531c6f194ab",
}


def _legacy_config(cache_strategy: str = "system") -> RunConfig:
    return RunConfig(
        session_id="s", query="q", model="m", cache_strategy=cache_strategy, **LEGACY_BEHAVIOUR
    )


def _tool_schemas(config: RunConfig) -> list[dict]:
    # Schema export reads only each tool's declared contract, never its
    # collaborators, so the tools need no stores behind them here.
    tools = [
        cls.__new__(cls)
        for cls in (FetchTool, RecordPremisesTool, SearchWithinTool, SearchBroadTool)
    ]
    return ToolRegistry(tools).export_schemas(config)


def _actual() -> dict[str, str]:
    config = _legacy_config()
    out = {
        "decomposer_system": sha256_hex(prompts.decomposer_system(config)),
        "subagent_system": sha256_hex(prompts.subagent_system(config)),
        "writer_system": sha256_hex(prompts.writer_system(config)),
        "repair_template": sha256_hex(prompts.repair_template(config)),
        "TRUNCATED_NUDGE": sha256_hex(prompts.TRUNCATED_NUDGE),
        "FINAL_TURN_NUDGE": sha256_hex(prompts.FINAL_TURN_NUDGE),
        "submit_subquestions": hash_obj(prompts.submit_subquestions_schema(config)),
        "submit_report": hash_obj(prompts.submit_report_schema(config)),
    }
    for schema in _tool_schemas(config):
        out[f"tool:{schema['name']}"] = hash_obj(schema)
    for strategy in ("system", "full"):
        cfg = _legacy_config(strategy)
        messages = [{"role": "user", "content": "hello"}]
        requests = {
            "decomposer": prompts.build_decomposer_request(cfg, "q"),
            "subagent": prompts.build_subagent_request(cfg, _tool_schemas(cfg), list(messages)),
            "writer": prompts.build_writer_request(cfg, list(messages)),
        }
        for phase, request in requests.items():
            out[f"req:{phase}:{strategy}"] = request.prompt_hash
    return {key: value[:16] for key, value in out.items()}


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_legacy_prompt_bytes_are_frozen(key):
    assert _actual()[key] == GOLDEN[key], (
        f"{key} changed under legacy behaviour — existing recordings will no longer "
        "replay. See this module's docstring; do not update the hash."
    )


def test_every_prompt_piece_is_pinned():
    assert set(_actual()) == set(GOLDEN)
