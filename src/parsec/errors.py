"""Parsec error hierarchy."""

from __future__ import annotations


class ParsecError(Exception):
    """Base class for all parsec errors."""


class BudgetExceeded(ParsecError):
    """A hard budget cap (usd, tokens, wall-clock) was breached."""

    def __init__(self, category: str, spent: float, cap: float):
        self.category = category
        self.spent = spent
        self.cap = cap
        super().__init__(f"budget exceeded: {category} spent={spent} cap={cap}")


class ReplayDivergence(ParsecError):
    """A replayed run issued a request that differs from the recording."""

    def __init__(self, call_index: int, recorded_hash: str, live_hash: str, diff: str = ""):
        self.call_index = call_index
        self.recorded_hash = recorded_hash
        self.live_hash = live_hash
        self.diff = diff
        msg = (
            f"replay divergence at call {call_index}: "
            f"recorded prompt_hash={recorded_hash[:12]} live={live_hash[:12]}"
        )
        if diff:
            msg += f"\n{diff}"
        super().__init__(msg)


class ModelCallFailed(ParsecError):
    """An adapter raised mid-call. The gateway journals the failure as an
    LLM_FAILED event and raises this typed wrapper, so a replayed run can
    reproduce the SAME failure at the same per-stream call — a subagent
    dying mid-wave leaves a replayable stream (M11 failure semantics)."""

    def __init__(self, kind: str, detail: str):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}")


class CacheMiss(ParsecError):
    """Replay-mode fetch requested a URL absent from the frozen corpus."""

    def __init__(self, url: str):
        self.url = url
        super().__init__(f"cache miss in replay mode: {url}")


class ToolValidationError(ParsecError):
    """Tool intent failed schema validation (surfaced to the model, not raised through the loop)."""


class HaltRequested(ParsecError):
    """User abort (SIGINT) or explicit halt."""
