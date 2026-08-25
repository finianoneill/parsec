"""Phase 4: cache_strategy breakpoint placement.

"full" adds two breakpoints to the legacy system-block one: the last tool
schema (caching the whole tools array) and a rolling marker on the final
block of the final message. Placement is applied to copies at build time —
the loop's transcript stays unmarked — and is a pure function of
(config, transcript), so recorded runs rebuild identical bytes."""

from __future__ import annotations

import copy

from parsec.loop import prompts
from tests.conftest import make_config


def _strip_cache(obj):
    if isinstance(obj, dict):
        return {k: _strip_cache(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_cache(x) for x in obj]
    return obj


def _markers(obj) -> int:
    if isinstance(obj, dict):
        return int("cache_control" in obj) + sum(_markers(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_markers(x) for x in obj)
    return 0


_TOOLS = [
    {"name": "alpha", "description": "a", "input_schema": {"type": "object"}},
    {"name": "beta", "description": "b", "input_schema": {"type": "object"}},
]


def test_system_strategy_is_the_legacy_wire_shape(tmp_path):
    config = make_config(tmp_path, cache_strategy="system")
    req = prompts.build_subagent_request(
        config, list(_TOOLS), [{"role": "user", "content": "go"}]
    )
    assert _markers(req.tools) == 0
    assert _markers(req.messages) == 0
    assert _markers(req.system) == 1  # the one pre-Phase-4 breakpoint
    assert req.messages == [{"role": "user", "content": "go"}]  # not even blockified


def test_full_strategy_marks_tools_and_rolling_message(tmp_path):
    config = make_config(tmp_path, cache_strategy="full")
    messages = [{"role": "user", "content": "go"}]
    snapshot = copy.deepcopy(messages)
    req = prompts.build_subagent_request(config, list(_TOOLS), messages)

    assert "cache_control" in req.tools[-1]  # last tool caches the whole array
    assert all("cache_control" not in t for t in req.tools[:-1])
    last_block = req.messages[-1]["content"][-1]
    assert last_block == {"type": "text", "text": "go", "cache_control": {"type": "ephemeral"}}
    # 4-breakpoint API limit: system + tools + rolling = 3
    assert _markers(req.system) + _markers(req.tools) + _markers(req.messages) == 3

    assert messages == snapshot  # the loop's transcript is never mutated
    # module-constant schema not mutated either (decomposer path)
    prompts.build_decomposer_request(config, "q")
    assert "cache_control" not in prompts.SUBMIT_SUBQUESTIONS_SCHEMA


def test_rolling_marker_moves_but_prefix_is_stable(tmp_path):
    config = make_config(tmp_path, cache_strategy="full")
    turn1 = [{"role": "user", "content": "go"}]
    turn2 = turn1 + [
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "r"}]},
    ]
    req1 = prompts.build_subagent_request(config, list(_TOOLS), turn1)
    req2 = prompts.build_subagent_request(config, list(_TOOLS), turn2)

    # exactly one rolling marker each, on the final block of the final message
    assert _markers(req1.messages) == 1 and _markers(req2.messages) == 1
    assert "cache_control" in req2.messages[-1]["content"][-1]
    # markers stripped, request 2 extends request 1 byte-for-byte
    s1, s2 = _strip_cache(req1.messages), _strip_cache(req2.messages)
    assert s2[: len(s1)] == s1
    # and the static prefix is byte-identical between turns
    assert (req1.system, req1.tools) == (req2.system, req2.tools)


def test_writer_and_empty_tools(tmp_path):
    config = make_config(tmp_path, cache_strategy="full")
    req = prompts.build_writer_request(config, [{"role": "user", "content": "write"}])
    assert req.tools == []  # empty tools stay empty, no crash
    assert _markers(req.messages) == 1


def test_unmarkable_final_block_is_skipped(tmp_path):
    config = make_config(tmp_path, cache_strategy="full")
    messages = [{"role": "user", "content": [{"type": "mystery", "data": "?"}]}]
    req = prompts.build_writer_request(config, messages)
    assert _markers(req.messages) == 0  # unknown block type: no marker, no crash
