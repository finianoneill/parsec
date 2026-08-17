"""Embedding seam for corpus-internal vector search.

Embedding must be a pure, cacheable function of (model, text) or vector
search breaks replay determinism (T4). Vectors are cached content-addressed
by (model_id, text_hash) in the embeddings table.

The default provider is a zero-dependency hashed character-n-gram TF
embedder: deterministic across platforms, CPU-only, adequate for
small-corpus semantic recall alongside BM25. A neural provider (e.g.
nomic-embed / EmbeddingGemma via ONNX int8) can implement the same
protocol later without touching callers.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
from typing import Protocol

from parsec.canonical import canonical_json, sha256_hex


class EmbeddingProvider(Protocol):
    model_id: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashedNgramEmbedder:
    """Character 3-gram TF vectors hashed into a fixed dimension, L2-normed.
    Pure Python, deterministic across platforms."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.model_id = f"hashed-ngram-3g-{dim}-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        normalized = " ".join(text.lower().split())
        for i in range(max(0, len(normalized) - 2)):
            gram = normalized[i : i + 3]
            slot = int.from_bytes(hashlib.md5(gram.encode()).digest()[:4], "big") % self.dim
            vec[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [round(v / norm, 8) for v in vec]
        return vec


class EmbeddingCache:
    def __init__(self, conn: sqlite3.Connection, provider: EmbeddingProvider):
        self.conn = conn
        self.provider = provider

    def embed(self, texts: list[str]) -> list[list[float]]:
        import json

        hashes = [sha256_hex(t) for t in texts]
        cached: dict[str, list[float]] = {}
        for h in hashes:
            row = self.conn.execute(
                "SELECT vector_json FROM embeddings WHERE model_id=? AND text_hash=?",
                (self.provider.model_id, h),
            ).fetchone()
            if row is not None:
                cached[h] = json.loads(row["vector_json"])
        missing = [(t, h) for t, h in zip(texts, hashes) if h not in cached]
        if missing:
            vectors = self.provider.embed([t for t, _ in missing])
            for (_, h), vec in zip(missing, vectors):
                self.conn.execute(
                    "INSERT OR IGNORE INTO embeddings (model_id, text_hash, vector_json)"
                    " VALUES (?,?,?)",
                    (self.provider.model_id, h, canonical_json(vec)),
                )
                cached[h] = vec
        return [cached[h] for h in hashes]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # inputs are L2-normalized
