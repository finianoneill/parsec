"""search_within: hybrid search over the already-fetched corpus.

Every deep-research architecture re-queries gathered evidence; ours does it
over citable, replay-deterministic spans. BM25 (SQLite FTS5) and vector
similarity (cached embeddings, exact brute-force cosine) are fused with
reciprocal-rank fusion; ties break lexicographically by span_id so results
are fully deterministic.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from parsec.retrieval.embeddings import EmbeddingCache, cosine
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext

RRF_K = 60
CANDIDATES = 25
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


class SearchWithinInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="What to look for in already-fetched documents")
    k: int = Field(default=5, ge=1, le=10, description="Number of spans to return")


class SearchWithinTool:
    name = "search_within"
    description = (
        "Search the text of documents ALREADY fetched this session (and earlier sessions in "
        "this corpus) for relevant spans. Returns citable span IDs with previews. Use this to "
        "re-check gathered evidence before searching the web again."
    )
    input_model = SearchWithinInput
    max_context_chars = 4000

    def __init__(self, spans: SpanStore, embeddings: EmbeddingCache):
        self.spans = spans
        self.embeddings = embeddings

    async def run(self, input: SearchWithinInput, ctx: ToolContext) -> tuple[dict, str]:
        bm25_ranked = self._bm25(ctx, input.query)
        vector_ranked = self._vector(ctx, input.query)

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
        lines = []
        for span_id in fused:
            row = self.spans.get(span_id)
            preview = " ".join(row["text"][:200].split()) if row else ""
            lines.append(f"[{span_id}] {preview}")
        return full, "\n".join(lines)

    def _bm25(self, ctx: ToolContext, query: str) -> list[str]:
        tokens = _TOKEN_RE.findall(query.lower())
        if not tokens:
            return []
        match = " OR ".join(f'"{t}"' for t in tokens)
        try:
            rows = ctx.conn.execute(
                "SELECT span_id FROM spans_fts WHERE spans_fts MATCH ? ORDER BY rank, span_id LIMIT ?",
                (match, CANDIDATES),
            ).fetchall()
        except Exception:
            return []  # FTS unavailable or query unparsable -> vector-only
        return [r["span_id"] for r in rows]

    def _vector(self, ctx: ToolContext, query: str) -> list[str]:
        rows = ctx.conn.execute("SELECT span_id, text FROM spans ORDER BY span_id").fetchall()
        if not rows:
            return []
        qvec = self.embeddings.embed([query])[0]
        span_vecs = self.embeddings.embed([r["text"] for r in rows])
        scored = sorted(
            zip((r["span_id"] for r in rows), span_vecs),
            key=lambda pair: (-cosine(qvec, pair[1]), pair[0]),
        )
        return [span_id for span_id, _ in scored[:CANDIDATES]]
