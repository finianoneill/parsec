"""Static USD-per-million-token pricing.

Pinned into RunConfig at session start (pricing_override) so a replayed
run reproduces recorded costs even if this table changes later.
"""

from __future__ import annotations

from parsec.models.gateway import Cost, Usage

# $/MTok: input, output; cache read = 0.1x input, cache write (5m) = 1.25x input.
PRICES: dict[str, dict[str, float]] = {
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    # deterministic test adapter — free
    "fake-model": {"input": 0.0, "output": 0.0},
}

CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25


def resolve_prices(model: str, override: dict[str, dict[str, float]] | None = None) -> dict[str, float]:
    table = dict(PRICES)
    if override:
        table.update(override)
    if model in table:
        return table[model]
    # Unknown model: price as the most expensive known tier so budgets stay conservative.
    return table["claude-opus-5"]


def compute_cost(model: str, usage: Usage, override: dict[str, dict[str, float]] | None = None) -> Cost:
    p = resolve_prices(model, override)
    per = 1_000_000
    breakdown = {
        "input": usage.input_tokens * p["input"] / per,
        "output": usage.output_tokens * p["output"] / per,
        "cache_read": usage.cache_read_input_tokens * p["input"] * CACHE_READ_MULT / per,
        "cache_write": usage.cache_creation_input_tokens * p["input"] * CACHE_WRITE_MULT / per,
    }
    return Cost(usd=sum(breakdown.values()), breakdown=breakdown)
