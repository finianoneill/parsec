"""Regression comparison: two eval-result files, identical corpora, did the
harness get worse? (§11 M5 — this is what makes any change measurable.)

An axis regresses when it drops by more than epsilon on a case where both
runs produced a score. New/removed cases and newly-None scores are
reported but are not regressions by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

AXES = ("citation_faithfulness", "coverage", "synthesis")


@dataclass
class AxisDelta:
    case_id: str
    axis: str
    before: float | None
    after: float | None
    delta: float | None
    regressed: bool


@dataclass
class Comparison:
    deltas: list[AxisDelta] = field(default_factory=list)
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)

    @property
    def regressions(self) -> list[AxisDelta]:
        return [d for d in self.deltas if d.regressed]

    @property
    def ok(self) -> bool:
        return not self.regressions

    def to_payload(self) -> dict:
        return {
            "ok": self.ok,
            "regressions": [d.__dict__ for d in self.regressions],
            "deltas": [d.__dict__ for d in self.deltas],
            "only_in_a": self.only_in_a,
            "only_in_b": self.only_in_b,
        }


def compare_runs(a: dict, b: dict, epsilon: float = 0.05) -> Comparison:
    """a/b are eval-result payloads as written by `parsec eval run`."""
    a_by_case = {r["case_id"]: r for r in a["results"]}
    b_by_case = {r["case_id"]: r for r in b["results"]}
    comparison = Comparison(
        only_in_a=sorted(set(a_by_case) - set(b_by_case)),
        only_in_b=sorted(set(b_by_case) - set(a_by_case)),
    )
    for case_id in sorted(set(a_by_case) & set(b_by_case)):
        sa, sb = a_by_case[case_id]["scores"], b_by_case[case_id]["scores"]
        for axis in AXES:
            before, after = sa.get(axis), sb.get(axis)
            if before is None and after is None:
                continue
            delta = None if (before is None or after is None) else round(after - before, 6)
            regressed = delta is not None and delta < -epsilon
            comparison.deltas.append(
                AxisDelta(case_id, axis, before, after, delta, regressed)
            )
    return comparison
