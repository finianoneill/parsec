"""Subagent contract (§4). Defined and schema-tested at M0; consumed at M3.

The load-bearing rule: a finding with zero premise_ids is rejected at
schema level — it cannot enter the DAG.

These models are wire contracts: their JSON schemas are sent to the model as
tool schemas, so their fields, descriptions and constraints are prompt bytes.
They are FROZEN for the behaviour versions that already use them — a changed
contract is a new model class, selected by version (submission_model below,
ToolRegistry's tool_spec for tool inputs), never an edit to these.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from parsec.config import RunConfig, behaviour_level

# Atomicity guard (§10.1): one subject, one predicate — enforced softly via a
# hard length cap. The record_premises tool checks it per premise so one
# overlong text cannot fail a whole batch at schema level.
PREMISE_MAX_CHARS = 300


class PremiseDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Specificity itself is an eval metric, not a schema rule.
    text: str = Field(min_length=1, max_length=PREMISE_MAX_CHARS)
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


# -- derivations >= 2 ---------------------------------------------------------
#
# New contract classes, not edits to the ones above: premises lose
# transform_note, findings gain `computes` and its operation. Their
# docstrings and Field descriptions are sent to the model as part of the
# tool schema, so they are written for the model, not for maintainers.


class PremiseDraftV2(BaseModel):
    """One atomic premise, stated exactly as its spans state it."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=PREMISE_MAX_CHARS)
    span_refs: list[str] = Field(min_length=1)
    claim_class: Literal["stable", "slow", "volatile"] = "stable"


class PremiseDraftV3(PremiseDraftV2):
    """One atomic premise, stated exactly as its spans state it."""

    # recency >= 2 (docstring above is sent to the model; this is not).
    as_of: str | None = Field(
        default=None,
        description=(
            "When the fact held, IF the cited span states that date: a year ('2019'), a month "
            "and year ('March 2025'), a quarter ('Q3 2025') or a full date ('2025-03-01'). "
            "The harness checks the date appears in the span. Omit it when the span gives no date."
        ),
    )


class Operand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    premise_id: str = Field(description="A premise you recorded that states this number")
    value: str = Field(description="The number exactly as that premise states it, e.g. '21.9'")


class Operation(BaseModel):
    """A derivation the harness recomputes with exact decimal arithmetic."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["convert", "sum", "diff", "ratio", "percent_change", "round", "other"] = Field(
        description=(
            "convert: one operand between unit_from and unit_to. sum: two or more operands. "
            "diff: first minus second. ratio: first divided by second. percent_change: from "
            "the first operand to the second. round: one operand to `precision`. other: "
            "anything else, described in `description` — recorded as UNVERIFIED and never "
            "treated as evidence, so prefer a recomputable op."
        )
    )
    operands: list[Operand] = Field(min_length=1)
    unit_from: str | None = Field(default=None, description="convert only, e.g. 'km'")
    unit_to: str | None = Field(default=None, description="convert only, e.g. 'mi'")
    precision: str = Field(
        default="1", description="A power of ten the output is rounded to: '1', '0.1', '0.01'"
    )
    output: str = Field(description="The result as your finding states it, e.g. '13.6'")
    description: str | None = Field(default=None, description="op 'other' only: what was done")


class FindingDraftV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    premise_ids: list[str] = Field(min_length=1)
    edge_type: Literal["deduces", "induces", "temporal", "computes"]
    operation: Operation | None = Field(
        default=None, description="Required when edge_type is 'computes'; omit otherwise"
    )


class SubagentSubmissionV2(BaseModel):
    """Your final report for the assigned subquestion."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "partial", "blocked"]
    findings: list[FindingDraftV2] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    dead_ends: list[str] = Field(default_factory=list)
    summary: str = Field(default="", max_length=500)


def submission_model(
    config: RunConfig,
) -> type[SubagentSubmission] | type[SubagentSubmissionV2]:
    """The submit_report contract for this session's behaviour versions. The
    schema shown to the subagent and the validation of what it sends back
    both come from here, so they cannot drift apart."""
    if behaviour_level(config.derivations) >= 2:
        return SubagentSubmissionV2
    return SubagentSubmission
