"""Golden hashes for everything that is prompt bytes, per RELEASED behaviour.

Replay asserts each request's prompt hash against the recording, so a changed
system prompt, nudge, or tool schema makes every existing recording diverge.
One set of hashes per release:

  0.1.0  LEGACY_BEHAVIOUR — computed at 366f1f6, the commit the remediation's
         first frozen recordings come from
  0.2.0  sentence_check 1, claim_credence 2, derivations 2, recency 2

Neither set may ever change. A new release adds a set; it does not edit one.

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

from parsec import __version__
from parsec.canonical import hash_obj, sha256_hex
from parsec.config import CURRENT_BEHAVIOUR, LEGACY_BEHAVIOUR, RunConfig
from parsec.loop import prompts
from parsec.tools.base import ToolRegistry
from parsec.tools.fetch import FetchTool
from parsec.tools.record_premises import RecordPremisesTool
from parsec.tools.search_broad import SearchBroadTool
from parsec.tools.search_within import SearchWithinTool

V0_2_0 = {"sentence_check": 1, "claim_credence": 2, "derivations": 2, "recency": 2}
# Prompts at recency >= 2 carry the session's reference date; pin one.
AS_OF = "2026-11-25T12:00:00.000+00:00"

GOLDEN_V0_2_0: dict[str, str] = {
    "FINAL_TURN_NUDGE": "b98d0579a1c0e7e5",
    "TRUNCATED_NUDGE": "09bd74cad4b5ff0e",
    "decomposer_system": "c040c4ee84f4e973",
    "repair_template": "73b4348767811b56",
    "req:decomposer:full": "b8a1fa5a084f6839",
    "req:decomposer:system": "5ec416a4d4d8e570",
    "req:subagent:full": "a2884c9614033190",
    "req:subagent:system": "0651d11fa8160b53",
    "req:writer:full": "1348dc5e2e3d0649",
    "req:writer:system": "2d17ee51ec247043",
    "subagent_system": "bc597ff611acfaa5",
    "submit_report": "12d88f46b09b99cc",
    "submit_subquestions": "8ffd64bbb2415888",
    "tool:fetch": "9ca6d9c60a91bc43",
    "tool:record_premises": "0ab8a313710c45c1",
    "tool:search_broad": "77b5492688f3dc8d",
    "tool:search_within": "0fde9a7a021cd3b9",
    "writer_system": "d1c594d98e697d11",
}

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


def _config(behaviour: dict, cache_strategy: str = "system") -> RunConfig:
    return RunConfig(
        session_id="s", query="q", model="m", cache_strategy=cache_strategy,
        as_of_ts=AS_OF, **behaviour,
    )


def _tool_schemas(config: RunConfig) -> list[dict]:
    # Schema export reads only each tool's declared contract, never its
    # collaborators, so the tools need no stores behind them here.
    tools = [
        cls.__new__(cls)
        for cls in (FetchTool, RecordPremisesTool, SearchWithinTool, SearchBroadTool)
    ]
    return ToolRegistry(tools).export_schemas(config)


def _actual(behaviour: dict) -> dict[str, str]:
    config = _config(behaviour)
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
        cfg = _config(behaviour, strategy)
        messages = [{"role": "user", "content": "hello"}]
        requests = {
            "decomposer": prompts.build_decomposer_request(cfg, "q"),
            "subagent": prompts.build_subagent_request(cfg, _tool_schemas(cfg), list(messages)),
            "writer": prompts.build_writer_request(cfg, list(messages)),
        }
        for phase, request in requests.items():
            out[f"req:{phase}:{strategy}"] = request.prompt_hash
    return {key: value[:16] for key, value in out.items()}


RELEASES = {"0.1.0": (LEGACY_BEHAVIOUR, GOLDEN), "0.2.0": (V0_2_0, GOLDEN_V0_2_0)}


@pytest.mark.parametrize(
    ("release", "key"), [(r, k) for r, (_, golden) in RELEASES.items() for k in sorted(golden)]
)
def test_released_prompt_bytes_are_frozen(release, key):
    behaviour, golden = RELEASES[release]
    assert _actual(behaviour)[key] == golden[key], (
        f"{key} changed under the {release} behaviour versions — recordings made by that "
        "release will no longer replay. See this module's docstring; do not update the hash."
    )


@pytest.mark.parametrize("release", sorted(RELEASES))
def test_every_prompt_piece_is_pinned(release):
    behaviour, golden = RELEASES[release]
    assert set(_actual(behaviour)) == set(golden)


NEWEST_RELEASE = "0.2.0"


def unpinned_behaviour(current: dict, version: str, pinned: dict) -> list[str]:
    """Why `current` is not what the newest pinned release records, as a list
    of problems (empty when it is). `pinned` is that release's behaviour over
    every key, with the legacy value for keys the release never set.

    Between releases the package version is a development version
    ("0.2.1.dev0") and a behaviour may record "<n>-dev": that is work merged
    since the newest release, and it carries no replay guarantee. Everything
    else must still be exactly what that release pinned. A released version
    may record no -dev value at all, so the release that turns a -dev into
    an integer has to pin that integer's prompt bytes first."""
    in_development = {key for key, value in current.items() if str(value).endswith("-dev")}
    problems = []
    if in_development and ".dev" not in version:
        problems.append(
            f"version {version} is a release but records -dev behaviour for "
            f"{', '.join(sorted(in_development))}: fine between releases, never in one"
        )
    for key, value in current.items():
        if key not in in_development and value != pinned.get(key):
            problems.append(
                f"{key} is {value!r}, which no pinned release records "
                f"(the newest pins {pinned.get(key)!r}): add a golden set for it"
            )
    return problems


def test_the_newest_pinned_release_is_what_new_sessions_record():
    """If CURRENT_BEHAVIOUR moves on, its bytes need a pinned set of their
    own before the release that ships them."""
    newest, _ = RELEASES[NEWEST_RELEASE]
    assert unpinned_behaviour(CURRENT_BEHAVIOUR, __version__, {**LEGACY_BEHAVIOUR, **newest}) == []


def test_the_pin_check_allows_dev_versions_only_between_releases():
    pinned = {**LEGACY_BEHAVIOUR, **V0_2_0}
    in_progress = {**pinned, "claim_credence": "3-dev"}
    assert unpinned_behaviour(pinned, "0.2.0", pinned) == []
    assert unpinned_behaviour(in_progress, "0.2.1.dev0", pinned) == []
    # a release may not ship a -dev behaviour
    assert len(unpinned_behaviour(in_progress, "0.2.1", pinned)) == 1
    # an integer no release pinned is caught, in development or not
    unpinned = {**pinned, "claim_credence": 3}
    assert len(unpinned_behaviour(unpinned, "0.2.1.dev0", pinned)) == 1
    assert len(unpinned_behaviour(unpinned, "0.2.1", pinned)) == 1
    # so is a behaviour key the release never had, once it stops being -dev
    assert unpinned_behaviour({**pinned, "run_status": "1-dev"}, "0.2.1.dev0", pinned) == []
    assert len(unpinned_behaviour({**pinned, "run_status": 1}, "0.2.1.dev0", pinned)) == 1
