"""Citation checking over premise references (M2).

The writer cites [premise:<id>] per sentence. This module deterministically
segments the answer, resolves every premise ref against the session's DAG
(shallow check — the deep claim→premise→span walk is the verification
engine's job, verify/structural.py), and writes tier-4 ReportClaim nodes
with `aggregates` edges to their premises.

Sentence segmentation is a regex splitter — imperfect but deterministic and
replay-stable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from parsec.ids import PREMISE_REF_RE
from parsec.store.dag import DagStore

NARRATIVE_TAG = "[narrative]"

# Split after sentence punctuation (or a closing citation bracket) + whitespace,
# unless another citation/tag follows — citations trail their sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\]])\s+(?!\[)")


@dataclass
class Segment:
    text: str
    refs: list[str]  # premise node IDs
    narrative: bool


@dataclass
class CitationCheck:
    segments: list[Segment] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def claim_segments(self) -> list[Segment]:
        return [s for s in self.segments if not s.narrative]


def segment_answer(answer: str) -> list[Segment]:
    segments: list[Segment] = []
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        for raw in _SENTENCE_SPLIT_RE.split(line):
            raw = raw.strip()
            if not raw:
                continue
            refs = [m.group(0) for m in PREMISE_REF_RE.finditer(raw)]
            narrative = NARRATIVE_TAG in raw
            clean = PREMISE_REF_RE.sub("", raw).replace(NARRATIVE_TAG, "")
            clean = re.sub(r"\[\s*\]", "", clean)
            clean = " ".join(clean.split())
            if not clean:
                continue
            segments.append(Segment(text=clean, refs=refs, narrative=narrative and not refs))
    return segments


def check_citations(answer: str, session_id: str, dag: DagStore) -> CitationCheck:
    result = CitationCheck(segments=segment_answer(answer))
    known = {
        row["node_id"]
        for row in dag.nodes_for_session(session_id, tier=1)
    }
    for seg in result.segments:
        if seg.narrative:
            continue
        if not seg.refs:
            result.problems.append(f"uncited sentence: {seg.text[:120]!r}")
            continue
        for ref in seg.refs:
            if ref not in known:
                result.problems.append(
                    f"unknown premise id (not recorded this session): {ref}"
                )
    return result


def write_claims(session_id: str, check: CitationCheck, dag: DagStore) -> int:
    """Write ReportClaim nodes + aggregates edges to premises. Returns claim count.

    (M3 will insert Synthesis between claims and premises for multi-agent
    merges; single-agent claims aggregate their premises directly.)
    """
    written = 0
    for seg in check.claim_segments:
        claim_id = dag.add_node(
            session_id,
            "ReportClaim",
            {"text": seg.text, "premise_refs": seg.refs, "narrative": False},
        )
        for ref in seg.refs:
            dag.add_edge(session_id, claim_id, ref, "aggregates")
        written += 1
    return written
