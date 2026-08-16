"""Search provider seam.

`search_broad` is stubbed at M1: a fixture-backed provider behind the same
protocol a real web-search provider will implement later. Fixture misses
return a deterministic empty result (never an error) so the model can
rephrase and replay stays stable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

from parsec.models.tools import SearchHit

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_query(query: str) -> str:
    return " ".join(_WORD_RE.findall(query.lower()))


class SearchProvider(Protocol):
    async def search(self, query: str, k: int) -> list[SearchHit]: ...


class FixtureSearchProvider:
    """Loads a JSON file mapping normalized query -> list of hit dicts.

    Lookup: exact normalized match first; else best token-overlap (Jaccard)
    above 0.5 with deterministic tie-break by fixture key order; else [].
    """

    def __init__(self, path: Path | str):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self._fixtures: dict[str, list[SearchHit]] = {}
        for key, hits in raw.items():
            self._fixtures[normalize_query(key)] = [
                SearchHit(rank=i + 1, **hit) if "rank" not in hit else SearchHit(**hit)
                for i, hit in enumerate(hits)
            ]

    async def search(self, query: str, k: int) -> list[SearchHit]:
        norm = normalize_query(query)
        if norm in self._fixtures:
            return self._fixtures[norm][:k]
        q_tokens = set(norm.split())
        if not q_tokens:
            return []
        best_key, best_score = None, 0.5
        for key in self._fixtures:  # dict preserves insertion order → deterministic tie-break
            k_tokens = set(key.split())
            union = q_tokens | k_tokens
            score = len(q_tokens & k_tokens) / len(union) if union else 0.0
            if score > best_score:
                best_key, best_score = key, score
        if best_key is None:
            return []
        return self._fixtures[best_key][:k]
