"""Run configuration. Frozen into sessions.config_json at session start;
replay reads it back so a replayed run is configured identically."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from parsec import __version__ as _PARSEC_VERSION
from parsec.canonical import sha256_hex

DEFAULT_MODEL = "claude-opus-5"


class CacheMode(StrEnum):
    RECORD = "record"
    REPLAY = "replay"
    LIVE_PREFER_CACHE = "live-prefer-cache"


class Budgets(BaseModel):
    """Hard caps. Defaults are deliberately embarrassingly low (§10.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_usd: float = 0.50
    max_total_tokens: int = 200_000
    max_wall_seconds: int = 300
    max_turns: int = 12          # global model-call cap (orchestrator + subagents + writer)
    max_turns_per_subagent: int = 6
    max_subquestions: int = 4
    # M11 (T8): concurrent subagents per wave, capped at 5 per §3 of the v1
    # brief. 1 = sequential — the permanent config fallback, and what
    # `parsec fork --at-call` requires (global call order is only meaningful
    # in sequential recordings). Parallelism applies to RESEARCH only; the
    # writer is always single (WS-E.3).
    parallel_subagents: int = Field(default=1, ge=1, le=5)
    max_gap_rounds: int = 1      # §3 gap-filling: bounded rewrite rounds targeting weak evidence
    # Coverage gap-fill: a run must not stop with subquestions still partial
    # while a substantial share of its budget sits unspent — partial coverage
    # plus headroom is unfinished work, not a result. Bounded like weak-
    # evidence gap-fill; 0 disables.
    max_coverage_gap_rounds: int = 2
    # Fraction of BOTH the usd and token budgets that must remain for a
    # coverage retry to dispatch (clock-free: wall time never participates,
    # so the decision replays byte-identically).
    coverage_gap_headroom: float = Field(default=0.25, ge=0.0, le=1.0)


class ModelRetry(BaseModel):
    """Harness-owned retry policy for model calls (throttle/overload/
    transient failures). Every retry is journaled as an LLM_RETRY event
    with a delay that is a pure function of the attempt number, so retried
    runs replay byte-identically. max_attempts=1 disables retries.
    Defaults are sized to the (deliberately low) default wall budget."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_attempts: int = Field(default=4, ge=1)
    base_delay_s: float = Field(default=2.0, gt=0)
    max_delay_s: float = Field(default=30.0, gt=0)


class EffortLimits(BaseModel):
    """Effective dispatch caps for one run, derived from the decomposer's
    effort estimate (M12, WS-F.4): spend scales with query complexity, and
    the estimate is enforced by the harness gates — never by the prompt.
    Effort can only clamp BELOW the configured budgets, never raise them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_subquestions: int
    max_turns_per_subagent: int
    max_gap_rounds: int
    max_coverage_gap_rounds: int


# The least a dispatched subagent can usefully do: search, fetch/record,
# submit. Planning more subquestions than can each get this many turns dooms
# the tail of the plan to "blocked before dispatch".
MIN_SUBAGENT_TURNS = 3


def effort_limits(effort: str, budgets: "Budgets") -> EffortLimits:
    """quick: one subagent, few calls, no gap-fill. standard: small fan-out.
    deep (and anything unrecognized — the v1-compatible default): the full
    configured caps. Every tier is additionally clamped to the number of
    subquestions the TURN budget can actually dispatch (decomposer and
    writer take a turn each) — the caps must agree with each other."""
    turn_fit = max(1, (budgets.max_turns - 2) // MIN_SUBAGENT_TURNS)
    if effort == "quick":
        return EffortLimits(
            max_subquestions=1,
            max_turns_per_subagent=min(3, budgets.max_turns_per_subagent),
            max_gap_rounds=0,
            max_coverage_gap_rounds=0,
        )
    if effort == "standard":
        return EffortLimits(
            max_subquestions=min(2, budgets.max_subquestions, turn_fit),
            max_turns_per_subagent=budgets.max_turns_per_subagent,
            max_gap_rounds=budgets.max_gap_rounds,
            max_coverage_gap_rounds=budgets.max_coverage_gap_rounds,
        )
    return EffortLimits(
        max_subquestions=min(budgets.max_subquestions, turn_fit),
        max_turns_per_subagent=budgets.max_turns_per_subagent,
        max_gap_rounds=budgets.max_gap_rounds,
        max_coverage_gap_rounds=budgets.max_coverage_gap_rounds,
    )


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    query: str
    model: str = DEFAULT_MODEL
    max_tokens_per_call: int = 8192
    cache_mode: CacheMode = CacheMode.LIVE_PREFER_CACHE
    # Prompt-cache breakpoint placement (Phase 4). "system": the pre-Phase-4
    # wire shape — one ephemeral breakpoint on the system block. "full":
    # additionally cache the tool-schema array (breakpoint on the last tool)
    # and roll a breakpoint onto the final block of the final message, so
    # each request writes cache over its whole prompt and the next
    # append-only request reads it (multi-turn subagent prefixes re-bill at
    # 0.1x instead of full price). The FIELD default stays "system" because
    # cache_control is part of the prompt bytes and hashes: a recorded
    # config missing this key must rebuild pre-Phase-4 requests exactly.
    # The CLI records "full" for new runs.
    cache_strategy: Literal["system", "full"] = "system"
    adapter: Literal["anthropic", "bedrock", "fake", "replay"] = "anthropic"
    # Bedrock (Mantle client): auth is the AWS credential chain — aws_profile
    # pins the ~/.aws/credentials profile (e.g. the one okta-awscli writes).
    aws_region: str | None = None
    aws_profile: str | None = None
    budgets: Budgets = Field(default_factory=Budgets)
    data_dir: Path = Path("data")
    search_fixtures: Path | None = None
    search_k_default: int = 5
    # Live search providers (T11): provider responses are BORROWED data —
    # cached TTL-bounded per provider policy, unlike the permanent
    # self-fetch archive. API keys come from env (BRAVE_API_KEY, SERPER_API_KEY).
    search_provider: Literal["fixture", "searxng", "brave", "serper"] = "fixture"
    searxng_url: str | None = None
    provider_cache_ttl_s: int = 7 * 24 * 3600
    # M12 brief gate (WS-F.1): pause after the research brief is proposed and
    # wait for a steering message — "approve" dispatches, anything else is an
    # edit fed back to the decomposer. Approval and edits are recorded
    # steering events, so a gated session still replays byte-identically.
    brief_gate: bool = False
    # Cost gate (Phase 5): pause once for approval when spend crosses this
    # fraction of budgets.max_usd at a dispatch boundary. None = off.
    # Approval continues; any other reply stops dispatching and the writer
    # wraps up what was gathered. The reply is a recorded, gate-tagged
    # steering event, so gated sessions replay byte-identically.
    cost_gate_threshold: float | None = Field(default=None, gt=0, le=1)
    # Politeness 2.0: robots respected by purpose; contact appended to the UA.
    respect_robots: bool = True
    robots_ttl_s: int = 24 * 3600
    contact: str | None = None
    # Per-request HTTP timeout for live model calls, in seconds. A tripped
    # timeout surfaces through the gateway as a journaled TRANSIENT failure
    # and a visible retry — bounded and diagnosable, never a silently wedged
    # call. Default 180s: a full 8192-token response at Opus speeds takes
    # ~2 minutes, so anything past three is a stall, not a long answer (the
    # SDK's own default is 10 minutes per attempt, which in practice let one
    # stalled call eat a whole wall-clock budget). None = the SDK default.
    request_timeout_s: float | None = Field(default=180.0, gt=0)
    # The Anthropic SDK's transparent retry count for live model calls.
    # Default 0: the harness owns retries (model_retry below) and journals
    # each one — SDK-internal retries are invisible to the journal and were
    # indistinguishable from a hang. Escape hatch only. Stripped from
    # projections like adapter above — a transport detail no recorded
    # trajectory depends on.
    model_max_retries: int = Field(default=0, ge=0)
    # Harness retry policy (T1): applied by the gateway, journaled per
    # attempt. Stripped from projections — its effects ARE the journaled
    # LLM_RETRY events, so comparing the knob too would only break replay
    # of pre-taxonomy recordings.
    model_retry: ModelRetry = Field(default_factory=ModelRetry)
    # M14.2 refresh provenance: the recorded session this run was seeded
    # from, and whether stable-evidence carry-forward was disabled. The seed
    # (brief + carry/re-research split) is a pure function of that immutable
    # parent recording plus refresh_all, so replay re-derives it identically
    # from these two fields (T4) — no seed data is duplicated into config.
    refresh_of: str | None = None
    refresh_all: bool = False
    # Provenance stamp: the parsec version that recorded this session.
    # Replay/fork warn on mismatch — recordings only replay byte-identically
    # against the code that produced them, so skew is the first thing to
    # suspect on divergence. Stripped from projections.
    parsec_version: str | None = Field(default_factory=lambda: _PARSEC_VERSION)
    # Pricing table pinned at session start so replay reproduces recorded costs.
    pricing_override: dict[str, dict[str, float]] | None = None
    # Credence model (§2.1, T3; 2.0 at M10). source_tiers merges over the
    # built-in domain table; stakes_threshold is the stage-3 flagging floor
    # for report claims; volatile_penalty is the flat mutability floor for
    # volatile claims. Age decay is clock-free: evidence age is measured
    # against the newest evidence timestamp in the corpus, both recorded, so
    # replay stays byte-identical.
    source_tiers: dict[str, float] | None = None
    stakes_threshold: float = 0.7
    volatile_penalty: float = 0.85
    volatile_half_life_days: float = 30.0
    slow_half_life_days: float = 365.0
    # Truth-discovery source reliability (M10): adjusts tier priors ±cap from
    # agreement patterns in the session's own graph. Opt-in (plan risk §4.3:
    # small-corpus truth discovery can self-reinforce) — provenance-stamped.
    learned_reliability: bool = False
    # Fitted calibration (`parsec calibrate` output), frozen into the session
    # so range-backed tier rendering ("high (72–96%)") replays identically.
    calibration: dict | None = None
    # Grounded-NLI premise support tier (M9, T9): "lexical" is the always-on
    # deterministic checker, "hhem" escalates to HHEM-2.1-Open (needs the
    # `nli` extra), "none" disables. Advisory only — it never gates.
    nli_checker: Literal["lexical", "hhem", "none"] = "lexical"
    # Compaction ladder (§7): applied proactively when the token estimate
    # (system + tools + transcript, floored by the previous response's
    # journaled usage) crosses max_context_tokens, and reactively when the
    # API rejects a call as context_overflow (escalate one rung, retry).
    # Decisions are a pure function of the transcript and recorded data, so
    # compaction replays deterministically. Rung 1 evicts old tool results
    # down to markers; rung 2 reconstructs the workspace from the DAG +
    # notebook; rung 3 resets to the assignment + recorded premise texts.
    # The writer phase degrades separately: evidence lines are clipped
    # (300 then 120 chars) instead of laddered — IDs and the citation
    # universe are never dropped.
    max_context_tokens: int = 18_000
    evict_keep_last: int = 2
    # DEPRECATED (Phase 2): superseded by max_context_tokens (the char
    # trigger ignored system + tool schemas and could not react to a real
    # overflow). Unread; retained so pre-Phase-2 session configs still load.
    max_context_chars: int = 60_000

    def to_json_dict(self) -> dict:
        return self.model_dump(mode="json")


def make_session_id(query: str, now_utc: str) -> str:
    stamp = now_utc.replace(":", "").replace("-", "")[:15]  # YYYYMMDDTHHMMSS
    return f"{stamp}-{sha256_hex(query)[:4]}"


class Clock(Protocol):
    def now_iso(self) -> str: ...
    def monotonic(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


class RealClock:
    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        import asyncio

        await asyncio.sleep(seconds)
