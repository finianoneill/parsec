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
    # Pricing table pinned at session start so replay reproduces recorded costs.
    pricing_override: dict[str, dict[str, float]] | None = None

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
