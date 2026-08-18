"""Embedding-based syndication clustering (v2 plan WS-D.1, thesis T10).

v1 counted corroboration per URL domain — the weakest link in the credence
model: hashing/domain clustering measurably fails on syndicated news, so
twelve outlets republishing one wire story looked like twelve independent
confirmations. Independence must be judged by CONTENT, not domain.

Spans are clustered with union-find under two merge rules:

  shared domain   — same outlet is never independent of itself (the v1 rule,
                    kept as a floor), and
  near-dup content — embedding cosine >= SYNDICATION_COSINE marks syndicated
                     or lightly-edited copies as one cluster.

The embedder is the deterministic hashed-n-gram provider already used by
`search_within` (pure Python, CPU-only, cacheable) — clustering is a pure
function of the span texts, so credence stays replayable (T4). A MinHash
prefilter (the literature's speed trick) is deliberately skipped: per-premise
span counts are tiny and exactness beats speed here.
"""

from __future__ import annotations

from typing import Callable

from parsec.retrieval.embeddings import cosine

# Near-dup threshold on character-3-gram cosine: syndicated copies with light
# edits stay well above this; independent write-ups of the same fact fall
# well below it.
SYNDICATION_COSINE = 0.9

Embed = Callable[[list[str]], list[list[float]]]


def cluster_spans(spans: list[dict], embed: Embed) -> list[list[int]]:
    """Cluster spans ({"domain": ..., "text": ...}) into independence groups.

    Returns clusters as lists of indices into `spans`, ordered by each
    cluster's smallest index — fully deterministic."""
    n = len(spans)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    by_domain: dict[str, int] = {}
    for i, span in enumerate(spans):
        domain = span["domain"]
        if domain in by_domain:
            union(by_domain[domain], i)
        else:
            by_domain[domain] = i

    vectors = embed([span["text"] for span in spans])
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) != find(j) and cosine(vectors[i], vectors[j]) >= SYNDICATION_COSINE:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return [clusters[root] for root in sorted(clusters)]
