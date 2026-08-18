"""Run configuration. Frozen into sessions.config_json at session start;
replay reads it back so a replayed run is configured identically."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

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


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    query: str
    model: str = DEFAULT_MODEL
    max_tokens_per_call: int = 8192
    cache_mode: CacheMode = CacheMode.LIVE_PREFER_CACHE
    adapter: Literal["anthropic", "fake", "replay"] = "anthropic"
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
    # Politeness 2.0: robots respected by purpose; contact appended to the UA.
    respect_robots: bool = True
    robots_ttl_s: int = 24 * 3600
    contact: str | None = None
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
    # Compaction ladder (§7), applied to subagent contexts. Decisions are a
    # pure function of the transcript (char counts), so compaction replays
    # deterministically. Rung 1 evicts old tool results down to markers;
    # rung 3 resets the context seeded from recorded evidence. Rung 2
    # (model-written squeeze) is deferred — it would spend model calls.
    max_context_chars: int = 60_000
    evict_keep_last: int = 2

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
