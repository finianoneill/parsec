"""Subagent contract (§4). Defined and schema-tested at M0; consumed at M3.

The load-bearing rule: a finding with zero premise_ids is rejected at
schema level — it cannot enter the DAG.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PremiseDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Atomicity guard (§10.1): one subject, one predicate — enforced softly via
    # a hard length cap; specificity itself is an eval metric, not a schema rule.
    text: str = Field(min_length=1, max_length=300)
    span_refs: list[str] = Field(min_length=1)
    # FreshQA-style mutability classes (M10): stable never decays, slow
    # decays gently with evidence age, volatile decays fast and can be
    # superseded by newer contradicting evidence.
    claim_class: Literal["stable", "slow", "volatile"] = "stable"
    # Exempts the premise from the exact-number containment check; stored on
    # the extracts edge so the derivation stays auditable (§6 stage 1).
    transform_note: str | None = None


class FindingDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    premise_ids: list[str] = Field(min_length=1)
    edge_type: Literal["deduces", "induces", "temporal"]


class Conflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: str
    b: str
    note: str = ""


class SubagentReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subquestion_id: str
    status: Literal["answered", "partial", "blocked"]
    premises: list[PremiseDraft] = Field(default_factory=list)
    findings: list[FindingDraft] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    dead_ends: list[str] = Field(default_factory=list)
    tokens_spent: int = 0


class SubagentSubmission(BaseModel):
    """What the subagent's submit_report tool accepts (§4, adapted).

    Deviation from the spec's inline-premises shape, deliberately stronger:
    premises enter the DAG at record time via the validated record_premises
    tool (self-pruning happens there), so the submission references recorded
    premise node IDs instead of restating premise text. tokens_spent is
    computed by the harness from the ledger, never self-reported.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "partial", "blocked"]
    findings: list[FindingDraft] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    dead_ends: list[str] = Field(default_factory=list)
    summary: str = Field(default="", max_length=500)
