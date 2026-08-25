"""Harness-owned model-call retries (T1).

The SDK's transparent retries are disabled (model_max_retries defaults to
0); the gateway owns the retry loop instead, so every attempt is journaled
as an LLM_RETRY event — the activity view can tell a retry from a hang,
and replay reproduces the exact attempt sequence from the recording.
"""

from __future__ import annotations

from dataclasses import dataclass

from parsec.errors import ModelErrorKind

RETRYABLE = frozenset(
    {ModelErrorKind.THROTTLED, ModelErrorKind.OVERLOADED, ModelErrorKind.TRANSIENT}
)

# Substrings the API uses for a too-long prompt (matched case-insensitively).
# Ugly, but message text is what the provider gives for this class of error.
_OVERFLOW_MARKERS = ("prompt is too long", "input is too long", "context window exceeded")


def classify_model_error(exc: Exception) -> ModelErrorKind:
    """Map an adapter/SDK exception onto the taxonomy. Duck-typed on the
    Anthropic SDK's exception shape (status_code attribute, class name,
    message substrings) — both live adapters use that SDK, and no SDK
    import is needed here. Unknown errors classify FATAL: when in doubt,
    don't spend budget retrying."""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    msg = str(exc).lower()
    if status == 429 or name == "RateLimitError" or "rate limit" in msg or "throttl" in msg:
        return ModelErrorKind.THROTTLED
    if status == 529 or "overloaded" in msg:
        return ModelErrorKind.OVERLOADED
    if any(marker in msg for marker in _OVERFLOW_MARKERS):
        return ModelErrorKind.CONTEXT_OVERFLOW
    if status is not None and 500 <= status < 600:
        return ModelErrorKind.TRANSIENT
    if "Timeout" in name or "Connection" in name:
        return ModelErrorKind.TRANSIENT
    return ModelErrorKind.FATAL


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff. Deliberately jitter-free: the delay is
    a pure function of the attempt number, so the journaled LLM_RETRY
    payload replays byte-identically."""

    max_attempts: int = 4
    base_delay_s: float = 2.0
    max_delay_s: float = 30.0

    def should_retry(self, error_kind: ModelErrorKind | None, attempt: int) -> bool:
        """attempt is the 1-based number of the attempt that just failed."""
        return error_kind in RETRYABLE and attempt < self.max_attempts

    def delay_s(self, attempt: int) -> float:
        return min(self.base_delay_s * (2 ** (attempt - 1)), self.max_delay_s)
