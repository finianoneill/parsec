"""Live search providers behind the SearchProvider protocol.

Adapters are deliberately thin and disposable (vendor churn is the norm).
Every live provider is wrapped in CachedSearchProvider, which enforces the
T11 split: provider responses are borrowed data, cached TTL-bounded —
never the permanent archive that self-fetched documents get. In replay
cache mode the wrapper is cache-only, so recorded sessions replay with
zero provider calls; a miss raises CacheMiss (surfaced to the model as a
tool error, and to a replayed run as the divergence it is).

API keys are checked lazily (at search time, not construction) so replay
runs of live sessions never need credentials.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import httpx

from parsec.canonical import canonical_json
from parsec.config import CacheMode, Clock, RunConfig
from parsec.errors import CacheMiss
from parsec.models.tools import SearchHit
from parsec.retrieval.search_provider import FixtureSearchProvider, SearchProvider, normalize_query

SEARCH_TIMEOUT_S = 20.0


class SearxngProvider:
    name = "searxng"

    def __init__(self, base_url: str, transport: httpx.AsyncBaseTransport | None = None):
        self._base_url = base_url.rstrip("/")
        self._transport = transport

    async def search(self, query: str, k: int) -> list[SearchHit]:
        async with httpx.AsyncClient(transport=self._transport, timeout=SEARCH_TIMEOUT_S) as client:
            resp = await client.get(
                f"{self._base_url}/search", params={"q": query, "format": "json"}
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            SearchHit(
                title=r.get("title", ""),
                url=r["url"],
                snippet=r.get("content", "") or "",
                rank=i + 1,
            )
            for i, r in enumerate(data.get("results", [])[:k])
        ]


class BraveProvider:
    name = "brave"

    def __init__(self, api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self._api_key = api_key
        self._transport = transport

    def _key(self) -> str:
        import os

        key = self._api_key or os.environ.get("BRAVE_API_KEY", "")
        if not key:
            raise ValueError("Brave provider requires BRAVE_API_KEY")
        return key

    async def search(self, query: str, k: int) -> list[SearchHit]:
        async with httpx.AsyncClient(transport=self._transport, timeout=SEARCH_TIMEOUT_S) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": k},
                headers={"X-Subscription-Token": self._key(), "Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        results = (data.get("web") or {}).get("results", [])
        return [
            SearchHit(
                title=r.get("title", ""),
                url=r["url"],
                snippet=r.get("description", "") or "",
                rank=i + 1,
            )
            for i, r in enumerate(results[:k])
        ]


class SerperProvider:
    name = "serper"

    def __init__(self, api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self._api_key = api_key
        self._transport = transport

    def _key(self) -> str:
        import os

        key = self._api_key or os.environ.get("SERPER_API_KEY", "")
        if not key:
            raise ValueError("Serper provider requires SERPER_API_KEY")
        return key

    async def search(self, query: str, k: int) -> list[SearchHit]:
        async with httpx.AsyncClient(transport=self._transport, timeout=SEARCH_TIMEOUT_S) as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": k},
                headers={"X-API-KEY": self._key()},
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            SearchHit(
                title=r.get("title", ""),
                url=r["link"],
                snippet=r.get("snippet", "") or "",
                rank=i + 1,
            )
            for i, r in enumerate(data.get("organic", [])[:k])
        ]


class CachedSearchProvider:
    """TTL-bounded provider cache (T11) with the fetch cache's mode semantics:
    record = always live + write-through; replay = cache only (frozen);
    live-prefer-cache = fresh-cache hit wins, stale/miss goes live."""

    def __init__(
        self,
        inner: SearchProvider,
        conn: sqlite3.Connection,
        clock: Clock,
        mode: CacheMode,
        ttl_s: int,
    ):
        self.inner = inner
        self.name = getattr(inner, "name", "live")
        self.conn = conn
        self.clock = clock
        self.mode = mode
        self.ttl_s = ttl_s

    async def search(self, query: str, k: int) -> list[SearchHit]:
        query_norm = normalize_query(query)
        row = self.conn.execute(
            "SELECT fetched_ts, response_json FROM search_cache WHERE provider=? AND query_norm=?",
            (self.name, query_norm),
        ).fetchone()

        if self.mode == CacheMode.REPLAY:
            if row is None:
                raise CacheMiss(f"search:{self.name}:{query_norm}")
            return self._hits(row)[:k]

        if self.mode == CacheMode.LIVE_PREFER_CACHE and row is not None and self._fresh(row):
            return self._hits(row)[:k]

        hits = await self.inner.search(query, max(k, 10))  # cache a fuller page than requested
        self.conn.execute(
            "INSERT OR REPLACE INTO search_cache (provider, query_norm, fetched_ts, response_json)"
            " VALUES (?,?,?,?)",
            (
                self.name,
                query_norm,
                self.clock.now_iso(),
                canonical_json([h.model_dump() for h in hits]),
            ),
        )
        return hits[:k]

    def _hits(self, row) -> list[SearchHit]:
        return [SearchHit(**h) for h in json.loads(row["response_json"])]

    def _fresh(self, row) -> bool:
        try:
            fetched = datetime.fromisoformat(row["fetched_ts"])
            now = datetime.fromisoformat(self.clock.now_iso())
        except ValueError:
            return False
        return (now - fetched).total_seconds() < self.ttl_s


def build_search_provider(
    config: RunConfig,
    conn: sqlite3.Connection,
    clock: Clock,
    transport: httpx.AsyncBaseTransport | None = None,
) -> SearchProvider | None:
    """Config-driven provider construction, shared by ask/replay/fork/evals."""
    if config.search_provider == "fixture":
        if config.search_fixtures is None:
            return None
        return FixtureSearchProvider(config.search_fixtures)
    if config.search_provider == "searxng":
        if not config.searxng_url:
            raise ValueError("search_provider=searxng requires searxng_url")
        inner: SearchProvider = SearxngProvider(config.searxng_url, transport)
    elif config.search_provider == "brave":
        inner = BraveProvider(transport=transport)
    elif config.search_provider == "serper":
        inner = SerperProvider(transport=transport)
    else:
        raise ValueError(f"unknown search provider: {config.search_provider}")
    return CachedSearchProvider(inner, conn, clock, config.cache_mode, config.provider_cache_ttl_s)
