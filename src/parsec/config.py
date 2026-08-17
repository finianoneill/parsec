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
    # Credence model (§2.1, T3). source_tiers merges over the built-in domain
    # table; stakes_threshold is the stage-3 flagging floor for report claims;
    # volatile_penalty is the flat recency proxy for volatile claims (real
    # time-decay is deferred until calibration data exists — a clock read here
    # would break byte-identical replay).
    source_tiers: dict[str, float] | None = None
    stakes_threshold: float = 0.7
    volatile_penalty: float = 0.85
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
