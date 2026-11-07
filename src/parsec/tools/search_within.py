"""search_within: hybrid search over the spans this stream can see.

Every deep-research architecture re-queries gathered evidence; ours does it
over citable, replay-deterministic spans. BM25 and vector similarity
(cached deterministic embeddings, exact brute-force cosine) are fused with
reciprocal-rank fusion; ties break lexicographically by span_id.

Scope (M15.9). The spans table is corpus-wide and append-only: at replay it
already holds everything the session — and every later session in the data
dir — fetched, so a search ranked over "all spans" returned different
results on replay than it did live (the first real long run diverged at its
fourth model call on exactly this). A stream now searches only documents it
could have seen at that moment: its own fetches, plus the fetches of
streams that COMPLETED before it STARTED. In sequential mode that is "the
session so far"; under parallel waves (M11) siblings stay invisible to each
other until the next wave, which is the only interleaving-independent
reading (T8). SUBAGENT_STARTED is journaled before a subagent's first await,
so the set is a pure function of the recording.

BM25 is computed in-process over that set rather than by SQLite FTS5: FTS5's
rank uses corpus-wide term statistics, so even a filtered query would have
ranked differently as the corpus grew. Query terms are iterated in sorted
order so the floating-point sums are bit-identical across processes.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from parsec.models.events import EventType
from parsec.retrieval.embeddings import EmbeddingCache, cosine
from parsec.store.event_log import CURRENT_STREAM, EventLog
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext

RRF_K = 60
CANDIDATES = 25
BM25_K1 = 1.2
BM25_B = 0.75
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_SQL_CHUNK = 500


class SearchWithinInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="What to look for in already-fetched documents")
    k: int = Field(default=5, ge=1, le=10, description="Number of spans to return")


def visible_doc_hashes(event_log: EventLog, session_id: str, stream_id: str) -> set[str]:
    """Documents a stream may search: its own fetches plus those of streams
    that completed before it started. The orchestrator stream (no STARTED
    event of its own) sees every completed stream's fetches."""
    events = event_log.read(session_id)
    my_start: int | None = None
    for ev in events:
        if ev.event_type == EventType.SUBAGENT_STARTED and ev.stream_id == stream_id:
            my_start = ev.idx
            break
    completed = {
        ev.payload.get("sq_id", ev.stream_id)
        for ev in events
        if ev.event_type == EventType.SUBAGENT_COMPLETED and (my_start is None or ev.idx < my_start)
    }
    return {
        ev.payload["doc_hash"]
        for ev in events
        if ev.event_type == EventType.FETCH_PERFORMED
        and (ev.stream_id == stream_id or ev.stream_id in completed)
    }


def bm25_rank(query: str, rows: list[tuple[str, str]]) -> list[str]:
    """Top candidates by Okapi BM25 over (span_id, text) rows; deterministic
    for a given row set (sorted query terms, span_id tie-break)."""
    terms = sorted(set(_TOKEN_RE.findall(query.lower())))
    if not terms or not rows:
        return []
    docs = [(span_id, _TOKEN_RE.findall(text.lower())) for span_id, text in rows]
    n_docs = len(docs)
    avgdl = sum(len(toks) for _, toks in docs) / n_docs
    df = Counter()
    for _, toks in docs:
        for term in set(toks):
            df[term] += 1
    scored: list[tuple[float, str]] = []
    for span_id, toks in docs:
        tf = Counter(toks)
        dl = len(toks)
        score = 0.0
        for term in terms:
            f = tf.get(term)
            if not f:
                continue
            n = df[term]
            idf = math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5))
            score += idf * f * (BM25_K1 + 1.0) / (f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl))
        if score > 0.0:
            scored.append((score, span_id))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [span_id for _, span_id in scored[:CANDIDATES]]


class SearchWithinTool:
    name = "search_within"
    description = (
        "Search the text of documents ALREADY fetched in this session for relevant spans. "
        "Returns citable span IDs with previews. Use this to re-check gathered evidence "
        "before searching the web again."
    )
    input_model = SearchWithinInput
    max_context_chars = 4000

    def __init__(self, spans: SpanStore, embeddings: EmbeddingCache):
        self.spans = spans
        self.embeddings = embeddings

    async def run(self, input: SearchWithinInput, ctx: ToolContext) -> tuple[dict, str]:
        rows = self._visible_spans(ctx)
        bm25_ranked = bm25_rank(input.query, rows)
        vector_ranked = self._vector(input.query, rows)

        scores: dict[str, float] = {}
        for ranked in (bm25_ranked, vector_ranked):
            for rank, span_id in enumerate(ranked):
                scores[span_id] = scores.get(span_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        fused = sorted(scores, key=lambda s: (-scores[s], s))[: input.k]

        full = {
            "query": input.query,
            "results": [
                {"span_id": s, "rrf_score": round(scores[s], 6)} for s in fused
            ],
        }
        if not fused:
            return full, "no matching spans in the fetched corpus — fetch more sources first"
        text_by_id = dict(rows)
        lines = []
        for span_id in fused:
            preview = " ".join(text_by_id.get(span_id, "")[:200].split())
            lines.append(f"[{span_id}] {preview}")
        return full, "\n".join(lines)

    def _visible_spans(self, ctx: ToolContext) -> list[tuple[str, str]]:
        docs = sorted(visible_doc_hashes(ctx.event_log, ctx.config.session_id, CURRENT_STREAM.get()))
        rows: list[tuple[str, str]] = []
        for i in range(0, len(docs), _SQL_CHUNK):
            chunk = docs[i : i + _SQL_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            rows += [
                (r["span_id"], r["text"])
                for r in ctx.conn.execute(
                    f"SELECT span_id, text FROM spans WHERE doc_hash IN ({placeholders})", chunk
                )
            ]
        rows.sort()
        return rows

    def _vector(self, query: str, rows: list[tuple[str, str]]) -> list[str]:
        if not rows:
            return []
        qvec = self.embeddings.embed([query])[0]
        span_vecs = self.embeddings.embed([text for _, text in rows])
        scored = sorted(
            zip((span_id for span_id, _ in rows), span_vecs),
            key=lambda pair: (-cosine(qvec, pair[1]), pair[0]),
        )
        return [span_id for span_id, _ in scored[:CANDIDATES]]
