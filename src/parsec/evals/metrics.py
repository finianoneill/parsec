"""The metric seam (M15.1): one contract for every eval axis.

Before M15 the five axes were a tuple spelled out in three places (scoring,
runner aggregate, regression compare) and adding one meant editing all
three. Now a metric is any object with a `name`, a `tier`, and an async
`score(ctx)`; built-ins register by name, the runner records a manifest of
which metrics scored a run, and `eval compare` discovers its axes from that
manifest instead of a hardcoded list.

The tier is the T9 discipline made structural rather than conventional:

  mechanical — no model, deterministic over the frozen world. Gates.
  advisory   — a judge model's opinion. Reported with its verdict and never
               flips a comparison to "regressed" (the judge is a different
               model family and still gates nothing; §6 stage 5).

The borrowed idea is Google's agent-evaluation custom-metric registry
(`CodeExecutionMetric` / `LLMMetric`, registered by name and selected per
run); the parsec shape keeps the mechanical/advisory split in the type.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Iterable, Literal, Protocol

from parsec.evals.case import EvalCase
from parsec.evals.judge import judge_synthesis
from parsec.evals.scoring import (
    score_citation_faithfulness,
    score_coverage,
    score_nuggets,
)
from parsec.evals.support import MechanicalSupportChecker, SupportChecker, score_claim_support
from parsec.gateway.base import ModelAdapter
from parsec.store.blobs import BlobStore
from parsec.store.event_log import EventLog
from parsec.store.ledger import Ledger

Tier = Literal["mechanical", "advisory"]


@dataclass
class MetricContext:
    """Everything a metric may read. All of it is the recorded session plus
    the case's gold — a metric that needs more is a metric that cannot be
    recomputed from the recording, which is the line we do not cross."""

    conn: sqlite3.Connection
    blobs: BlobStore
    event_log: EventLog
    ledger: Ledger
    session_id: str
    case: EvalCase
    status: str = "done"
    answer: str | None = None
    support_checker: SupportChecker = field(default_factory=MechanicalSupportChecker)


@dataclass
class MetricResult:
    name: str
    tier: Tier
    value: float | None  # None = not applicable (no gold, no claims, no judge)
    details: dict = field(default_factory=dict)


class Metric(Protocol):
    name: str
    tier: Tier

    async def score(self, ctx: MetricContext) -> MetricResult: ...


class Scores:
    """A run's metric results as one flat namespace: `scores.nugget_recall`
    is the value, `scores.nugget_misses` the detail, `scores.tiers` the
    manifest. `to_payload()` is the flat dict `eval compare` reads, so a
    results file written before M15 and one written after look the same
    except for the manifest."""

    def __init__(self, results: Iterable[MetricResult]):
        self.results: list[MetricResult] = list(results)
        self.values: dict[str, float | None] = {r.name: r.value for r in self.results}
        self.tiers: dict[str, Tier] = {r.name: r.tier for r in self.results}
        self.details: dict = {}
        for r in self.results:
            self.details.update(r.details)

    def __getattr__(self, name: str):
        # Only reached when normal attribute lookup fails: values first, then
        # details, so `scores.coverage` and `scores.must_find_misses` both work.
        values = self.__dict__.get("values", {})
        if name in values:
            return values[name]
        details = self.__dict__.get("details", {})
        if name in details:
            return details[name]
        raise AttributeError(name)

    def get(self, name: str) -> float | None:
        return self.values.get(name)

    def to_payload(self) -> dict:
        return {**self.values, **self.details}

    @classmethod
    def mean(cls, runs: list["Scores"]) -> "Scores":
        """Per-metric mean across runs (None-aware); details from the first
        run, which is what the pre-M15 aggregate did too."""
        first = runs[0]
        merged = []
        for r in first.results:
            vals = [s.values.get(r.name) for s in runs if s.values.get(r.name) is not None]
            value = round(sum(vals) / len(vals), 6) if vals else None
            merged.append(MetricResult(r.name, r.tier, value, dict(r.details)))
        return cls(merged)


# --- registry ---------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def register(cls):
    """Class decorator: make a zero-argument metric selectable by name
    (`parsec eval run --metrics a,b`). Metrics that need configuration (the
    judge) are constructed explicitly by `default_metrics`."""
    _REGISTRY[cls.name] = cls
    return cls


def registered_metrics() -> dict[str, type]:
    return dict(_REGISTRY)


def mechanical_metrics(names: Iterable[str] | None = None) -> list[Metric]:
    if names is None:
        return [cls() for cls in _REGISTRY.values()]
    out = []
    for name in names:
        if name not in _REGISTRY:
            raise ValueError(
                f"unknown metric {name!r}; registered: {', '.join(sorted(_REGISTRY))}"
            )
        out.append(_REGISTRY[name]())
    return out


def default_metrics(
    judge_adapter: ModelAdapter | None = None,
    judge_model: str = "",
    names: Iterable[str] | None = None,
) -> list[Metric]:
    """The standard axis set: every registered mechanical metric (or the
    named subset) plus the advisory synthesis judge — always present in the
    manifest, scoring None when no judge is wired, so results files keep a
    stable shape with or without a judge."""
    return [*mechanical_metrics(names), SynthesisJudge(judge_adapter, judge_model)]


async def run_metrics(metrics: list[Metric], ctx: MetricContext) -> Scores:
    return Scores([await m.score(ctx) for m in metrics])


def manifest(metrics: Iterable[Metric]) -> list[dict]:
    return [{"name": m.name, "tier": m.tier} for m in metrics]


# --- built-ins --------------------------------------------------------------


@register
class CitationFaithfulness:
    """Fraction of non-narrative claims untouched by structural-verification
    violations. No model."""

    name = "citation_faithfulness"
    tier: Tier = "mechanical"

    async def score(self, ctx: MetricContext) -> MetricResult:
        value, total, faithful = score_citation_faithfulness(ctx.conn, ctx.blobs, ctx.session_id)
        return MetricResult(
            self.name, self.tier, value, {"claims_total": total, "claims_faithful": faithful}
        )


@register
class Coverage:
    """Fraction of gold must-find items present in the claim texts."""

    name = "coverage"
    tier: Tier = "mechanical"

    async def score(self, ctx: MetricContext) -> MetricResult:
        value, hits, misses = score_coverage(ctx.conn, ctx.session_id, ctx.case.must_find)
        return MetricResult(
            self.name, self.tier, value, {"must_find_hits": hits, "must_find_misses": misses}
        )


@register
class NuggetRecall:
    """Weighted binary rubric recall with contradiction patterns."""

    name = "nugget_recall"
    tier: Tier = "mechanical"

    async def score(self, ctx: MetricContext) -> MetricResult:
        value, hits, misses, contra = score_nuggets(ctx.conn, ctx.session_id, ctx.case.nuggets)
        return MetricResult(
            self.name,
            self.tier,
            value,
            {"nugget_hits": hits, "nugget_misses": misses, "nugget_contradictions": contra},
        )


@register
class ClaimSupport:
    """Every claim graded against the verbatim spans behind it, from the
    frozen cache, by the configured SupportChecker (exact-match floor)."""

    name = "claim_support"
    tier: Tier = "mechanical"

    async def score(self, ctx: MetricContext) -> MetricResult:
        value, details = score_claim_support(ctx.conn, ctx.session_id, ctx.support_checker)
        grades: dict[str, int] = {}
        for d in details:
            grades[d.grade] = grades.get(d.grade, 0) + 1
        return MetricResult(self.name, self.tier, value, {"claim_support_grades": grades})


class SynthesisJudge:
    """Advisory: a different model family scores the prose 1-5, normalized.
    Never a gate; None when no judge is wired or the run produced no answer."""

    name = "synthesis"
    tier: Tier = "advisory"

    def __init__(self, adapter: ModelAdapter | None = None, model: str = ""):
        self.adapter = adapter
        self.model = model

    async def score(self, ctx: MetricContext) -> MetricResult:
        if self.adapter is None or not ctx.answer:
            return MetricResult(self.name, self.tier, None)
        value = await judge_synthesis(self.adapter, self.model, ctx.case.query, ctx.answer)
        return MetricResult(self.name, self.tier, value)
