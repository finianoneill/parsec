"""Platt calibration, Brier/smooth-ECE metrics, risk-coverage (M10, WS-D.5)."""

import pytest

from parsec.verify.calibration import (
    MIN_LABELS,
    PlattScaling,
    brier,
    calibration_report,
    fit_platt,
    risk_coverage,
    smooth_ece,
    tier_ranges,
)


def overconfident_pairs(per_bucket: int = 15) -> list[tuple[float, int]]:
    """Deterministic synthetic labels for a systematically overconfident
    heuristic: it says p, the true outcome rate is p**2."""
    pairs = []
    for i in range(20):
        p = 0.05 + 0.9 * i / 19
        ones = round((p**2) * per_bucket)
        pairs += [(p, 1)] * ones + [(p, 0)] * (per_bucket - ones)
    return pairs


def test_fit_needs_minimum_labels():
    with pytest.raises(ValueError, match=str(MIN_LABELS)):
        fit_platt([(0.5, 1)] * (MIN_LABELS - 1))


def test_platt_improves_brier_and_ece_on_miscalibrated_data():
    pairs = overconfident_pairs()
    assert len(pairs) >= 200
    scaling = fit_platt(pairs)
    assert brier(pairs, scaling) < brier(pairs)
    assert smooth_ece(pairs, scaling) < smooth_ece(pairs)


def test_apply_is_monotonic_and_bounded():
    scaling = fit_platt(overconfident_pairs())
    outputs = [scaling.apply(p / 20) for p in range(1, 20)]
    assert all(0.0 < q < 1.0 for q in outputs)
    assert outputs == sorted(outputs)


def test_payload_round_trip():
    scaling = fit_platt(overconfident_pairs())
    clone = PlattScaling.from_payload(scaling.to_payload())
    assert abs(clone.apply(0.7) - scaling.apply(0.7)) < 1e-6
    assert clone.n == scaling.n


def test_risk_coverage_curve_shape():
    curve = risk_coverage(overconfident_pairs())
    assert len(curve) == 10
    assert curve[-1]["coverage"] == 1.0
    # asserting only the most-confident claims first is never riskier overall
    assert curve[0]["risk"] <= curve[-1]["risk"]


def test_tier_ranges_are_ordered_percent_bands():
    ranges = tier_ranges(fit_platt(overconfident_pairs()))
    assert set(ranges) == {"low", "moderate", "high"}
    for lo, hi in ranges.values():
        assert 0 <= lo <= hi <= 100
    assert ranges["low"][0] <= ranges["moderate"][0] <= ranges["high"][0]


def test_calibration_report_payload():
    payload = calibration_report(overconfident_pairs())
    assert payload["method"] == "platt" and payload["n"] >= 200
    assert payload["brier_calibrated"] < payload["brier_raw"]
    assert not payload["underpowered"]
    assert len(payload["risk_coverage"]) == 10
    assert set(payload["tier_ranges"]) == {"low", "moderate", "high"}


def test_underpowered_flag():
    payload = calibration_report(overconfident_pairs(per_bucket=2))  # n=40 < 200
    assert payload["underpowered"]
