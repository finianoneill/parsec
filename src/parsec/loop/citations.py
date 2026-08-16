"""Citation checking: the M1 slice of structural verification (§6 stage 1).

Deterministically segments the answer into claim sentences, resolves every
[doc:<hash>#<start>-<end>] reference against the span store (span row exists,
its document is cached, its text is the verbatim slice), and writes
ReportClaim + SourceSpan nodes with `extracts` edges.

Sentence segmentation is a regex splitter — imperfect but deterministic and
replay-stable; flagged for upgrade at M2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from parsec.ids import SPAN_ID_RE, parse_span_id
from parsec.store.blobs import BlobStore
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.spans import SpanStore

NARRATIVE_TAG = "[narrative]"

# Split after sentence punctuation (or a closing citation bracket) + whitespace,
# unless another citation/tag follows — citations trail their sentence:
# "X is true. [doc:...#0-10] Y is false. [doc:...#5-9]" -> two segments.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\]])\s+(?!\[)")


@dataclass
class Segment:
    text: str
    refs: list[str]
    narrative: bool


@dataclass
class CitationCheck:
    segments: list[Segment] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)  # human/model-readable violations

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
            refs = [m.group(0) for m in SPAN_ID_RE.finditer(raw)]
            narrative = NARRATIVE_TAG in raw
            clean = SPAN_ID_RE.sub("", raw).replace(NARRATIVE_TAG, "")
            clean = re.sub(r"\[\s*\]", "", clean)
            clean = " ".join(clean.split())
            if not clean:
                continue
            segments.append(Segment(text=clean, refs=refs, narrative=narrative and not refs))
    return segments


def check_citations(
    answer: str,
    spans: SpanStore,
    documents: DocumentStore,
    blobs: BlobStore,
) -> CitationCheck:
    result = CitationCheck(segments=segment_answer(answer))
    for seg in result.segments:
        if seg.narrative:
            continue
        if not seg.refs:
            result.problems.append(f"uncited sentence: {seg.text[:120]!r}")
            continue
        for ref in seg.refs:
            problem = _resolve_ref(ref, spans, documents, blobs)
            if problem:
                result.problems.append(problem)
    return result


def _resolve_ref(ref: str, spans: SpanStore, documents: DocumentStore, blobs: BlobStore) -> str | None:
    parsed = parse_span_id(ref)
    if parsed is None:
        return f"malformed span id: {ref}"
    row = spans.get(ref)
    if row is None:
        return f"unknown span id (never returned by fetch): {ref}"
    doc = documents.get_document(row["doc_hash"])
    if doc is None:
        return f"span {ref} references an unknown document"
    if not blobs.exists(doc["raw_blob"]) or not blobs.exists(doc["text_blob"]):
        return f"span {ref}: document content missing from blob store"
    text = blobs.get_text(doc["text_blob"])
    if text[row["char_start"] : row["char_end"]] != row["text"]:
        return f"span {ref}: stored text does not match document slice"
    return None


def write_claims(
    session_id: str,
    check: CitationCheck,
    dag: DagStore,
    spans: SpanStore,
    documents: DocumentStore,
) -> int:
    """Write ReportClaim + SourceSpan nodes and extracts edges. Returns claim count."""
    written = 0
    span_node_ids: dict[str, str] = {}
    for seg in check.claim_segments:
        claim_id = dag.add_node(
            session_id,
            "ReportClaim",
            {"text": seg.text, "span_refs": seg.refs, "narrative": False},
        )
        for ref in seg.refs:
            if ref not in span_node_ids:
                row = spans.get(ref)
                doc = documents.get_document(row["doc_hash"])
                span_node_ids[ref] = dag.add_node(
                    session_id,
                    "SourceSpan",
                    {
                        "span_id": ref,
                        "doc_hash": row["doc_hash"],
                        "char_start": row["char_start"],
                        "char_end": row["char_end"],
                        "text": row["text"],
                        "url": doc["url"],
                        "fetched_ts": doc["fetched_ts"],
                    },
                )
            dag.add_edge(session_id, claim_id, span_node_ids[ref], "extracts")
        written += 1
    return written
