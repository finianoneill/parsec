import json

import httpx
import pytest

from parsec.config import CacheMode
from parsec.errors import CacheMiss
from parsec.retrieval.providers import (
    BraveProvider,
    CachedSearchProvider,
    SearxngProvider,
    SerperProvider,
    build_search_provider,
)


def transport_returning(payload: dict, captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["calls"] = captured.get("calls", 0) + 1
        if request.method == "POST":
            captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


async def test_searxng_parsing():
    captured: dict = {}
    payload = {"results": [
        {"title": "A", "url": "https://a.example/1", "content": "snippet a"},
        {"title": "B", "url": "https://b.example/2"},
        {"title": "C", "url": "https://c.example/3", "content": "snippet c"},
    ]}
    provider = SearxngProvider("http://searx.local", transport_returning(payload, captured))
    hits = await provider.search("boiling point", k=2)
    assert "q=boiling+point" in captured["url"] and "format=json" in captured["url"]
    assert [h.url for h in hits] == ["https://a.example/1", "https://b.example/2"]
    assert hits[0].rank == 1 and hits[0].snippet == "snippet a"
    assert hits[1].snippet == ""


async def test_brave_parsing_and_auth():
    captured: dict = {}
    payload = {"web": {"results": [{"title": "A", "url": "https://a.example", "description": "d"}]}}
    provider = BraveProvider(api_key="brave-key", transport=transport_returning(payload, captured))
    hits = await provider.search("q", k=5)
    assert captured["headers"]["x-subscription-token"] == "brave-key"
    assert hits[0].snippet == "d"


async def test_brave_key_checked_lazily(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    provider = BraveProvider()  # construction must not require the key (replay runs)
    with pytest.raises(ValueError):
        await provider.search("q", 5)


async def test_serper_parsing():
    captured: dict = {}
    payload = {"organic": [{"title": "A", "link": "https://a.example", "snippet": "s"}]}
    provider = SerperProvider(api_key="serper-key", transport=transport_returning(payload, captured))
    hits = await provider.search("some query", k=3)
    assert captured["headers"]["x-api-key"] == "serper-key"
    assert captured["body"] == {"q": "some query", "num": 3}
    assert hits[0].url == "https://a.example"


@pytest.fixture
def searx(db):
    def make(mode, captured, clock):
        payload = {"results": [{"title": "A", "url": "https://a.example/1", "content": "s"}]}
        inner = SearxngProvider("http://searx.local", transport_returning(payload, captured))
        return CachedSearchProvider(inner, db, clock, mode, ttl_s=3600)

    return make


async def test_cache_record_write_through(searx, db, clock):
    captured: dict = {}
    provider = searx(CacheMode.RECORD, captured, clock)
    await provider.search("boiling point", 5)
    await provider.search("Boiling  Point!", 5)  # same normalized query, record mode refetches
    assert captured["calls"] == 2
    rows = db.execute("SELECT * FROM search_cache").fetchall()
    assert len(rows) == 1 and rows[0]["provider"] == "searxng"


async def test_cache_replay_is_cache_only(searx, db, clock):
    captured: dict = {}
    await searx(CacheMode.RECORD, captured, clock).search("known query", 5)
    replay = searx(CacheMode.REPLAY, captured, clock)
    hits = await replay.search("known query", 5)
    assert hits[0].url == "https://a.example/1"
    assert captured["calls"] == 1  # no live call in replay
    with pytest.raises(CacheMiss):
        await replay.search("never recorded", 5)


async def test_cache_ttl_governs_live_prefer_cache(searx, db, clock):
    captured: dict = {}
    lpc = searx(CacheMode.LIVE_PREFER_CACHE, captured, clock)
    await lpc.search("q", 5)
    await lpc.search("q", 5)  # fresh -> cache hit
    assert captured["calls"] == 1
    # stale the row: TTL is 3600s, backdate it
    db.execute("UPDATE search_cache SET fetched_ts='2020-01-01T00:00:00.000+00:00'")
    await lpc.search("q", 5)
    assert captured["calls"] == 2  # stale -> live refetch


def test_factory_shapes(db, clock, tmp_path, config):
    assert build_search_provider(config.model_copy(update={"search_fixtures": None}), db, clock) is None
    with pytest.raises(ValueError):
        build_search_provider(
            config.model_copy(update={"search_provider": "searxng", "searxng_url": None}),
            db, clock,
        )
    provider = build_search_provider(
        config.model_copy(update={"search_provider": "searxng", "searxng_url": "http://x"}),
        db, clock,
    )
    assert isinstance(provider, CachedSearchProvider)