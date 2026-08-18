"""`parsec calibrate` (WS-D.5): turn heuristic credences into measured ones.

The v1 brief's promise (§10.3: "a 0.87 looks like calibrated probability; at
v1 it's a heuristic ordinal"), now concrete: every rendered credence is
logged (CREDENCE_COMPUTED events, persisted node credences, eval label
harvests); this module fits Platt scaling over (credence, outcome) pairs and
reports Brier score, kernel-smoothed ECE, and a risk-coverage curve.

Platt over isotonic is deliberate: isotonic regression needs ~1000+ points
before it stops overfitting (Niculescu-Mizil & Caruana); our label budgets
start in the hundreds. The fit is a fixed-iteration Newton solve — pure
Python, deterministic, no dependencies.

After calibration, tiers render with quantified RANGES on expansion
("high (72–96%)", never "0.68"): ranges don't damage trust, unquantified
hedges do (van der Bles). The fitted parameters travel inside RunConfig
(frozen at session start), so range-backed rendering replays byte-identically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from parsec.verify.credence import HIGH, MODERATE

Pair = tuple[float, int]  # (heuristic credence in (0,1), outcome 0/1)

MIN_LABELS = 20          # hard floor to fit at all
RECOMMENDED_LABELS = 200  # below this, report loudly that the fit is weak

_EPS = 1e-6
_MAX_Z = 8.0


def _clamp(p: float) -> float:
    return min(1.0 - _EPS, max(_EPS, p))


def _logit(p: float) -> float:
    z = math.log(_clamp(p) / (1.0 - _clamp(p)))
    return min(_MAX_Z, max(-_MAX_Z, z))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass(frozen=True)
class PlattScaling:
    a: float
    b: float
    n: int

    def apply(self, credence: float) -> float:
        return _sigmoid(self.a * _logit(credence) + self.b)

    def to_payload(self) -> dict:
        return {"method": "platt", "a": round(self.a, 6), "b": round(self.b, 6), "n": self.n}

    @classmethod
    def from_payload(cls, payload: dict) -> "PlattScaling":
        return cls(a=float(payload["a"]), b=float(payload["b"]), n=int(payload["n"]))


def fit_platt(pairs: list[Pair], iterations: int = 50) -> PlattScaling:
    """Newton-Raphson logistic fit of outcome ~ sigmoid(a*logit(p) + b), with
    Platt's target smoothing so extreme labels don't force infinite weights.
    Fixed iteration count and pure arithmetic: deterministic."""
    if len(pairs) < MIN_LABELS:
        raise ValueError(f"calibration needs >= {MIN_LABELS} labeled claims, got {len(pairs)}")
    n_pos = sum(label for _, label in pairs)
    n_neg = len(pairs) - n_pos
    t_pos = (n_pos + 1.0) / (n_pos + 2.0)
    t_neg = 1.0 / (n_neg + 2.0)
    data = [(_logit(p), t_pos if label else t_neg) for p, label in pairs]

    a, b = 1.0, 0.0
    for _ in range(iterations):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for z, t in data:
            q = _sigmoid(a * z + b)
            w = max(q * (1.0 - q), 1e-12)
            g_a += (q - t) * z
            g_b += q - t
            h_aa += w * z * z
            h_ab += w * z
            h_bb += w
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        step_a = (h_bb * g_a - h_ab * g_b) / det
        step_b = (h_aa * g_b - h_ab * g_a) / det
        a, b = a - step_a, b - step_b
        if abs(step_a) < 1e-10 and abs(step_b) < 1e-10:
            break
    return PlattScaling(a=a, b=b, n=len(pairs))


def brier(pairs: list[Pair], scaling: PlattScaling | None = None) -> float:
    total = 0.0
    for p, label in pairs:
        q = scaling.apply(p) if scaling else p
        total += (q - label) ** 2
    return total / len(pairs)


def smooth_ece(
    pairs: list[Pair], scaling: PlattScaling | None = None, bandwidth: float = 0.1
) -> float:
    """Kernel-smoothed expected calibration error: at each prediction, compare
    it to a Gaussian-weighted local outcome rate; average the gaps weighted by
    local density. Smooth in the inputs — no bin-boundary artifacts."""
    preds = [(scaling.apply(p) if scaling else p, label) for p, label in pairs]
    total_gap = 0.0
    for q, _ in preds:
        wsum = rate = 0.0
        for q2, label2 in preds:
            w = math.exp(-((q - q2) ** 2) / (2.0 * bandwidth**2))
            wsum += w
            rate += w * label2
        total_gap += abs(q - rate / wsum)
    return total_gap / len(preds)


def risk_coverage(
    pairs: list[Pair], scaling: PlattScaling | None = None
) -> list[dict[str, float]]:
    """Assert the most-confident claims first: at each decile of coverage,
    the error rate among the claims asserted so far. What a user pays in
    risk for demanding more coverage."""
    scored = sorted(
        ((scaling.apply(p) if scaling else p, label) for p, label in pairs),
        key=lambda x: (-x[0], x[1]),
    )
    curve = []
    n = len(scored)
    for decile in range(1, 11):
        k = max(1, round(n * decile / 10))
        wrong = sum(1 - label for _, label in scored[:k])
        curve.append({"coverage": round(k / n, 3), "risk": round(wrong / k, 4)})
    return curve


def tier_ranges(scaling: PlattScaling) -> dict[str, tuple[int, int]]:
    """The user-facing payoff: each tier's heuristic band mapped through the
    fitted calibration into a quantified probability range (percent)."""
    bands = {"low": (0.05, MODERATE), "moderate": (MODERATE, HIGH), "high": (HIGH, 0.99)}
    out = {}
    for tier, (lo, hi) in bands.items():
        a, b = scaling.apply(lo), scaling.apply(hi)
        out[tier] = (int(round(min(a, b) * 100)), int(round(max(a, b) * 100)))
    return out


def calibration_report(pairs: list[Pair]) -> dict:
    """Fit + before/after metrics, ready to persist as calibration.json and
    to freeze into RunConfig.calibration."""
    scaling = fit_platt(pairs)
    payload = scaling.to_payload()
    payload["brier_raw"] = round(brier(pairs), 6)
    payload["brier_calibrated"] = round(brier(pairs, scaling), 6)
    payload["smooth_ece_raw"] = round(smooth_ece(pairs), 6)
    payload["smooth_ece_calibrated"] = round(smooth_ece(pairs, scaling), 6)
    payload["risk_coverage"] = risk_coverage(pairs, scaling)
    payload["tier_ranges"] = {t: list(r) for t, r in tier_ranges(scaling).items()}
    payload["underpowered"] = len(pairs) < RECOMMENDED_LABELS
    return payload
