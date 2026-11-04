"""Regression comparison: two eval-result files, identical corpora, did the
harness get worse? This is what makes any change measurable.

M8 statistics: identical frozen corpora make this a paired-differences
setting (per Anthropic's "Adding Error Bars to Evals") — per-axis verdicts
come from the mean paired per-case delta with a 95% CI, not from comparing
two independent means. Three states per axis:

  regressed    — mean delta < -epsilon AND the CI excludes zero (n>=2)
  improved     — mean delta > +epsilon AND the CI excludes zero (n>=2)
  inconclusive — everything else (including single-case comparisons within
                 epsilon and CIs straddling zero)

Per-case deltas are still reported with an epsilon flag for triage; gating
should use the axis verdicts.

M15: the axes are discovered from each file's metric manifest (the union
of both sides; pre-M15 files fall back to the original five), and every
axis carries its tier. Only MECHANICAL axes can flip `ok` — an advisory
axis (the synthesis judge) reports its verdict and nothing more, which is
what "the judge never gates" has always meant (§6, T9). Loss histograms
(M15.2) are diffed per pattern alongside, so a change reads as "retrieval
losses 3 → 0, extraction losses 1 → 4" and not only as a score moving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from parsec.evals.losses import CATEGORY, LossPattern, histogram_from_results

LEGACY_AXES = ("citation_faithfulness", "coverage", "nugget_recall", "claim_support", "synthesis")
LEGACY_TIERS = {axis: ("advisory" if axis == "synthesis" else "mechanical") for axis in LEGACY_AXES}
Z95 = 1.96


@dataclass
class AxisDelta:
    case_id: str
    axis: str
    before: float | None
    after: float | None
    delta: float | None
    regressed: bool


@dataclass
class AxisVerdict:
    axis: str
    verdict: str  # improved | regressed | inconclusive
    n_cases: int
    mean_delta: float | None
    ci95: float | None  # half-width; None when n < 2
    tier: str = "mechanical"

    def to_payload(self) -> dict:
        return {
            "axis": self.axis,
            "verdict": self.verdict,
            "n_cases": self.n_cases,
            "mean_delta": self.mean_delta,
            "ci95": self.ci95,
            "tier": self.tier,
        }


@dataclass
class LossDelta:
    pattern: str
    category: str
    before: int
    after: int

    @property
    def delta(self) -> int:
        return self.after - self.before

    def to_payload(self) -> dict:
        return {
            "pattern": self.pattern,
            "category": self.category,
            "before": self.before,
            "after": self.after,
            "delta": self.delta,
        }


@dataclass
class Comparison:
    deltas: list[AxisDelta] = field(default_factory=list)
    verdicts: list[AxisVerdict] = field(default_factory=list)
    loss_deltas: list[LossDelta] = field(default_factory=list)
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)

    @property
    def regressions(self) -> list[AxisDelta]:
        return [d for d in self.deltas if d.regressed]

    @property
    def regressed_axes(self) -> list[AxisVerdict]:
        """Mechanical axes that regressed — the gating set."""
        return [v for v in self.verdicts if v.verdict == "regressed" and v.tier == "mechanical"]

    @property
    def advisory_regressions(self) -> list[AxisVerdict]:
        """Advisory axes that regressed: reported, never gating."""
        return [v for v in self.verdicts if v.verdict == "regressed" and v.tier != "mechanical"]

    @property
    def ok(self) -> bool:
        return not self.regressed_axes

    def to_payload(self) -> dict:
        return {
            "ok": self.ok,
            "verdicts": [v.to_payload() for v in self.verdicts],
            "advisory_regressions": [v.axis for v in self.advisory_regressions],
            "regressions": [d.__dict__ for d in self.regressions],
            "deltas": [d.__dict__ for d in self.deltas],
            "loss_deltas": [d.to_payload() for d in self.loss_deltas],
            "only_in_a": self.only_in_a,
            "only_in_b": self.only_in_b,
        }


def _axis_verdict(axis: str, deltas: list[float], epsilon: float, tier: str) -> AxisVerdict:
    n = len(deltas)
    if n == 0:
        return AxisVerdict(axis, "inconclusive", 0, None, None, tier)
    mean = sum(deltas) / n
    if n == 1:
        # no variance estimate: fall back to the practical-significance floor
        verdict = "regressed" if mean < -epsilon else "improved" if mean > epsilon else "inconclusive"
        return AxisVerdict(axis, verdict, 1, round(mean, 6), None, tier)
    variance = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    ci95 = Z95 * math.sqrt(variance / n)
    if mean < -epsilon and mean + ci95 < 0:
        verdict = "regressed"
    elif mean > epsilon and mean - ci95 > 0:
        verdict = "improved"
    else:
        verdict = "inconclusive"
    return AxisVerdict(axis, verdict, n, round(mean, 6), round(ci95, 6), tier)


def discover_axes(a: dict, b: dict) -> tuple[list[str], dict[str, str]]:
    """(axes in manifest order, axis -> tier) from both files' metric
    manifests; a file without one (pre-M15) contributes the legacy five."""
    axes: list[str] = []
    tiers: dict[str, str] = {}
    for payload in (a, b):
        entries = payload.get("metrics") or [
            {"name": axis, "tier": LEGACY_TIERS[axis]} for axis in LEGACY_AXES
        ]
        for entry in entries:
            if entry["name"] not in tiers:
                axes.append(entry["name"])
            tiers[entry["name"]] = entry.get("tier", "mechanical")
    return axes, tiers


def compare_runs(a: dict, b: dict, epsilon: float = 0.05) -> Comparison:
    """a/b are eval-result payloads as written by `parsec eval run`."""
    axes, tiers = discover_axes(a, b)
    a_by_case = {r["case_id"]: r for r in a["results"]}
    b_by_case = {r["case_id"]: r for r in b["results"]}
    comparison = Comparison(
        only_in_a=sorted(set(a_by_case) - set(b_by_case)),
        only_in_b=sorted(set(b_by_case) - set(a_by_case)),
    )
    paired: dict[str, list[float]] = {axis: [] for axis in axes}
    for case_id in sorted(set(a_by_case) & set(b_by_case)):
        sa, sb = a_by_case[case_id]["scores"], b_by_case[case_id]["scores"]
        for axis in axes:
            before, after = sa.get(axis), sb.get(axis)
            if before is None and after is None:
                continue
            delta = None if (before is None or after is None) else round(after - before, 6)
            if delta is not None:
                paired[axis].append(delta)
            comparison.deltas.append(
                AxisDelta(case_id, axis, before, after, delta, delta is not None and delta < -epsilon)
            )
    for axis in axes:
        if paired[axis] or any(d.axis == axis for d in comparison.deltas):
            comparison.verdicts.append(_axis_verdict(axis, paired[axis], epsilon, tiers[axis]))

    shared = set(a_by_case) & set(b_by_case)
    before = histogram_from_results([r for r in a["results"] if r["case_id"] in shared])
    after = histogram_from_results([r for r in b["results"] if r["case_id"] in shared])
    for pattern in LossPattern:
        if pattern.value in before or pattern.value in after:
            comparison.loss_deltas.append(
                LossDelta(
                    pattern.value, CATEGORY[pattern],
                    before.get(pattern.value, 0), after.get(pattern.value, 0),
                )
            )
    return comparison
