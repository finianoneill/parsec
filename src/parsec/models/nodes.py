"""Evidence DAG node payloads (§2.1) and edge types.

Tier layout: 0 SourceSpan, 1 Premise, 2 Finding, 3 Synthesis, 4 ReportClaim.
At M1 only tiers 0 and 4 are written (claims cite spans directly via
`extracts` edges — a documented collapsed chain until M2/M3 insert 1–3).

The schema-level rules from §4 live here: a Finding with zero premises and
a non-narrative ReportClaim with zero span refs cannot be constructed.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EdgeType(StrEnum):
    EXTRACTS = "extracts"
    DEDUCES = "deduces"
    INDUCES = "induces"
    TEMPORAL = "temporal"
    AGGREGATES = "aggregates"
    CONTRADICTS = "contradicts"


class _Node(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceSpanNode(_Node):
    TIER: ClassVar[int] = 0
    span_id: str
    doc_hash: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text: str
    url: str
    fetched_ts: str


class PremiseNode(_Node):
    TIER: ClassVar[int] = 1
    text: str
    span_refs: list[str] = Field(min_length=1)
    claim_class: Literal["stable", "slow", "volatile"]


class FindingNode(_Node):
    TIER: ClassVar[int] = 2
    text: str
    premise_ids: list[str] = Field(min_length=1)
    edge_type: Literal["deduces", "induces", "temporal"]


class SynthesisNode(_Node):
    TIER: ClassVar[int] = 3
    text: str
    child_ids: list[str] = Field(min_length=1)
    conflict_notes: list[str] = Field(default_factory=list)


class ReportClaimNode(_Node):
    TIER: ClassVar[int] = 4
    text: str
    # Premise and/or Finding node IDs; the claim→(premise|finding)→…→span
    # path is what stage-1 verification walks.
    refs: list[str] = Field(default_factory=list)
    narrative: bool = False

    @model_validator(mode="after")
    def _non_narrative_needs_refs(self) -> "ReportClaimNode":
        if not self.narrative and not self.refs:
            raise ValueError("non-narrative ReportClaim requires at least one premise/finding ref")
        return self


NODE_TIERS: dict[str, int] = {
    "SourceSpan": 0,
    "Premise": 1,
    "Finding": 2,
    "Synthesis": 3,
    "ReportClaim": 4,
}
