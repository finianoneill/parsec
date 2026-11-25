"""Citation checking over premise/finding references.

The writer cites [premise:<id>] or [finding:<id>] per sentence. This module
deterministically segments the answer, resolves every ref against the
session's DAG (shallow check — the deep claim→…→span walk is the
verification engine's job, verify/structural.py), and writes tier-4
ReportClaim nodes with `aggregates` edges to their evidence.

Sentence segmentation is a regex splitter — imperfect but deterministic and
replay-stable.

With `sentence_check` on (behaviour version sentence_check >= 1) the check
goes past "does the cited id exist": every number and quotation in a
sentence must appear in the text of what that sentence cites, a [narrative]
sentence may carry neither, and a heading may only use figures that appear
in evidence cited somewhere in the answer. Sentences that still fail after
the repair round are marked inline by mark_unverified(). With it off, this
module behaves exactly as it did for recordings that predate the check.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from parsec.ids import NODE_REF_RE
from parsec.store.dag import DagStore
from parsec.verify.containment import check_sentence, has_checkable_content

NARRATIVE_TAG = "[narrative]"
# Appended to a sentence that failed the citation check and could not be
# repaired: the reader sees which sentence the harness could not defend.
UNVERIFIED_TAG = "[unverified]"

# Split after sentence punctuation + whitespace unless a citation/tag follows
# (citations trail their sentence), and after a closing citation bracket only
# where a new sentence appears to start — a mid-sentence citation must not
# split its sentence into two claims.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?!\[)"
    r"|(?<=\])\s+(?=[\"'“(]?[A-Z0-9#*])"
)


# Markdown structure the writer emits around its claims: ATX headings and
# horizontal rules. With structure_lines_narrative these are skipped whole,
# never sentence-split ("## 1. ADK 2.0" used to become the claims "## 1."
# and "ADK 2.0") and never reported as uncited sentences (M15.8).
_STRUCTURE_LINE_RE = re.compile(r"^(?:#{1,6}\s|(?:-{3,}|\*{3,}|_{3,})\s*$)")


@dataclass
class Segment:
    text: str
    refs: list[str]  # premise/finding node IDs
    narrative: bool
    # A markdown heading or rule: reader-visible, but never a claim. Only
    # present in the segment list when the caller asked to keep structure.
    structure: bool = False
    # Where the raw piece sits in the answer: line index (as splitlines()
    # numbers them) and the character range within that line.
    line: int = -1
    start: int = 0
    end: int = 0


@dataclass
class CitationCheck:
    segments: list[Segment] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    # segment index -> that segment's problems (the same strings as above)
    failing: dict[int, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def claim_segments(self) -> list[Segment]:
        return [s for s in self.segments if not s.narrative]

    def fail(self, index: int, problem: str) -> None:
        self.problems.append(problem)
        self.failing.setdefault(index, []).append(problem)


def _clean(raw: str, strip_unverified: bool = False) -> str:
    clean = NODE_REF_RE.sub("", raw).replace(NARRATIVE_TAG, "")
    if strip_unverified:  # only text the sentence check itself can have added
        clean = clean.replace(UNVERIFIED_TAG, "")
    clean = re.sub(r"\[\s*\]", "", clean)
    # Removing a tag leaves a gap before its trailing punctuation
    # ("realm ." / "meaning ,") — close it.
    clean = re.sub(r"\s+([.,;:!?])", r"\1", clean)
    return " ".join(clean.split())


def segment_answer(
    answer: str, structure_lines_narrative: bool = False, keep_structure: bool = False
) -> list[Segment]:
    """Split an answer into sentence segments. keep_structure additionally
    returns heading/rule lines as `structure` segments (they are otherwise
    dropped when structure_lines_narrative is set)."""
    segments: list[Segment] = []
    for line_no, original in enumerate(answer.splitlines()):
        lead = len(original) - len(original.lstrip())
        line = original.strip()
        if not line:
            continue
        if (
            structure_lines_narrative
            and _STRUCTURE_LINE_RE.match(line)
            and not NODE_REF_RE.search(line)
        ):
            # structure, not a claim; a heading that cites is still checked
            if keep_structure and _clean(line, True).strip("#-*_ "):
                segments.append(
                    Segment(
                        text=_clean(line, True), refs=[], narrative=True, structure=True,
                        line=line_no, start=lead, end=lead + len(line),
                    )
                )
            continue
        separators = [(m.start(), m.end()) for m in _SENTENCE_SPLIT_RE.finditer(line)]
        for piece_start, piece_end in _piece_ranges(separators, len(line)):
            piece = line[piece_start:piece_end]
            raw = piece.strip()
            if not raw:
                continue
            refs = [m.group(0) for m in NODE_REF_RE.finditer(raw)]
            narrative = NARRATIVE_TAG in raw
            clean = _clean(raw, strip_unverified=keep_structure)
            if not clean:
                continue
            start = lead + piece_start + (len(piece) - len(piece.lstrip()))
            segments.append(
                Segment(
                    text=clean, refs=refs, narrative=narrative and not refs,
                    line=line_no, start=start, end=start + len(raw),
                )
            )
    return segments


def _piece_ranges(separators: list[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    """The ranges between separator matches — what re.split would return,
    with offsets."""
    ranges: list[tuple[int, int]] = []
    pos = 0
    for sep_start, sep_end in separators:
        ranges.append((pos, sep_start))
        pos = sep_end
    ranges.append((pos, length))
    return ranges


def check_citations(
    answer: str,
    session_id: str,
    dag: DagStore,
    structure_lines_narrative: bool = False,
    sentence_check: bool = False,
) -> CitationCheck:
    result = CitationCheck(
        segments=segment_answer(answer, structure_lines_narrative, keep_structure=sentence_check)
    )
    known: dict[str, str] = {}
    for tier in (1, 2):
        for row in dag.nodes_for_session(session_id, tier=tier):
            known[row["node_id"]] = json.loads(row["payload_json"]).get("text", "")
    # Evidence cited anywhere in the answer: what a heading may draw on.
    cited_anywhere = [
        known[ref] for seg in result.segments for ref in seg.refs if ref in known
    ]
    for index, seg in enumerate(result.segments):
        if seg.structure:
            for problem in check_sentence(seg.text, cited_anywhere):
                result.fail(index, f"heading {seg.text[:120]!r}: {problem} anywhere in the answer")
            continue
        if seg.narrative:
            if sentence_check and has_checkable_content(seg.text):
                result.fail(
                    index,
                    f"narrative sentence carries a number or quotation and must cite "
                    f"its evidence instead: {seg.text[:120]!r}",
                )
            continue
        if not seg.refs:
            result.fail(index, f"uncited sentence: {seg.text[:120]!r}")
            continue
        for ref in seg.refs:
            if ref not in known:
                result.fail(index, f"unknown premise/finding id (not recorded this session): {ref}")
        if sentence_check:
            cited = [known[ref] for ref in seg.refs if ref in known]
            if cited:
                for problem in check_sentence(seg.text, cited):
                    result.fail(
                        index,
                        f"sentence {seg.text[:120]!r}: {problem} ({', '.join(seg.refs)})",
                    )
    return result


def mark_unverified(answer: str, check: CitationCheck) -> str:
    """The answer with UNVERIFIED_TAG appended to every failing sentence (and
    heading). Everything else is left byte for byte as the writer wrote it."""
    if not check.failing:
        return answer
    lines = answer.splitlines(keepends=True)
    by_line: dict[int, list[int]] = {}
    for index in check.failing:
        seg = check.segments[index]
        if seg.line >= 0:
            by_line.setdefault(seg.line, []).append(seg.end)
    for line_no, ends in by_line.items():
        line = lines[line_no]
        for end in sorted(set(ends), reverse=True):  # right to left: offsets stay valid
            line = f"{line[:end]} {UNVERIFIED_TAG}{line[end:]}"
        lines[line_no] = line
    return "".join(lines)


def write_claims(
    session_id: str, check: CitationCheck, dag: DagStore, sentence_check: bool = False
) -> int:
    """Write ReportClaim nodes + aggregates edges to their premises/findings.
    Returns claim count. (Synthesis-tier merges across subagents arrive with
    the gap-fill work; claims aggregate their evidence nodes directly.)

    With sentence_check, narrative sentences are written too (as narrative
    nodes, no edges) so `parsec verify` can re-check them from the graph
    alone, and a sentence that failed the citation check carries
    `unverified: true`. Headings are never written: they are not claims.
    """
    written = 0
    for index, seg in enumerate(check.segments):
        if seg.structure:
            continue
        if seg.narrative:
            if sentence_check:
                payload: dict = {"text": seg.text, "refs": [], "narrative": True}
                if index in check.failing:
                    payload["unverified"] = True
                dag.add_node(session_id, "ReportClaim", payload)
            continue
        payload = {"text": seg.text, "refs": seg.refs, "narrative": False}
        if sentence_check and index in check.failing:
            payload["unverified"] = True
        claim_id = dag.add_node(session_id, "ReportClaim", payload)
        for ref in seg.refs:
            dag.add_edge(session_id, claim_id, ref, "aggregates")
        written += 1
    return written
