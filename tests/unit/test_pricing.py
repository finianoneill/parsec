from parsec.gateway.pricing import compute_cost, resolve_prices
from parsec.models.gateway import Usage


def test_opus_cost_hand_computed():
    usage = Usage(input_tokens=1000, output_tokens=500)
    cost = compute_cost("claude-opus-5", usage)
    # 1000/1M * $5 + 500/1M * $25 = 0.005 + 0.0125
    assert abs(cost.usd - 0.0175) < 1e-9


def test_cache_read_and_write_multipliers():
    usage = Usage(cache_read_input_tokens=1_000_000, cache_creation_input_tokens=1_000_000)
    cost = compute_cost("claude-opus-5", usage)
    assert abs(cost.breakdown["cache_read"] - 0.50) < 1e-9   # 0.1 * $5
    assert abs(cost.breakdown["cache_write"] - 6.25) < 1e-9  # 1.25 * $5


def test_unknown_model_priced_conservatively():
    assert resolve_prices("mystery-model") == resolve_prices("claude-opus-5")


def test_override_pins_prices():
    override = {"claude-opus-5": {"input": 1.0, "output": 1.0}}
    cost = compute_cost("claude-opus-5", Usage(input_tokens=1_000_000), override)
    assert abs(cost.usd - 1.0) < 1e-9


def test_fake_model_free():
    assert compute_cost("fake-model", Usage(input_tokens=999, output_tokens=999)).usd == 0.0
