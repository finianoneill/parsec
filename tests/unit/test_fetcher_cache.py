import httpx
import pytest

from parsec.config import CacheMode
from parsec.errors import CacheMiss
from parsec.retrieval.fetcher import Fetcher, canonicalize_url
from parsec.store.documents import DocumentStore

PAGE = b"<html><head><title>T</title></head><body><p>Cached content here.</p></body></html>"


def make_transport(counter: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        counter["calls"] = counter.get("calls", 0) + 1
        return httpx.Response(200, content=PAGE, headers={"content-type": "text/html"})

    return httpx.MockTransport(handler)


@pytest.fixture
def documents(db, clock):
    return DocumentStore(db, clock)


def test_canonicalize_url():
    assert canonicalize_url("HTTPS://Example.COM:443/path?b=2&a=1&utm_source=x#frag") == (
        "https://example.com/path?a=1&b=2"
    )
    assert canonicalize_url("http://example.com") == "http://example.com/"
    assert canonicalize_url("https://example.com/p?fbclid=zz") == "https://example.com/p"


async def test_record_writes_through(documents, blobs, clock):
    counter = {}
    f = Fetcher(documents, blobs, clock, CacheMode.RECORD, transport=make_transport(counter))
    doc = await f.fetch("https://example.test/page")
    assert not doc.from_cache
    assert "Cached content here." in doc.text
    assert doc.title == "T"
    assert counter["calls"] == 1
    assert documents.get_document(doc.doc_hash) is not None


async def test_live_prefer_cache_hits_cache(documents, blobs, clock):
    counter = {}
    rec = Fetcher(documents, blobs, clock, CacheMode.RECORD, transport=make_transport(counter))
    await rec.fetch("https://example.test/page")

    lpc = Fetcher(documents, blobs, clock, CacheMode.LIVE_PREFER_CACHE, transport=make_transport(counter))
    doc = await lpc.fetch("https://example.test/page")
    assert doc.from_cache
    assert counter["calls"] == 1  # no second network call
    assert "Cached content here." in doc.text


async def test_record_mode_always_refetches(documents, blobs, clock):
    counter = {}
    f = Fetcher(documents, blobs, clock, CacheMode.RECORD, transport=make_transport(counter))
    await f.fetch("https://example.test/page")
    await f.fetch("https://example.test/other")
    assert counter["calls"] == 2


async def test_replay_mode_cache_only(documents, blobs, clock):
    counter = {}
    rec = Fetcher(documents, blobs, clock, CacheMode.RECORD, transport=make_transport(counter))
    await rec.fetch("https://example.test/page")

    rep = Fetcher(documents, blobs, clock, CacheMode.REPLAY, transport=make_transport(counter))
    doc = await rep.fetch("https://example.test/page")
    assert doc.from_cache
    with pytest.raises(CacheMiss):
        await rep.fetch("https://example.test/never-fetched")
    assert counter["calls"] == 1


async def test_politeness_interval(documents, blobs, clock):
    counter = {}
    f = Fetcher(documents, blobs, clock, CacheMode.RECORD, transport=make_transport(counter))
    await f.fetch("https://example.test/a")
    before = clock.mono
    await f.fetch("https://example.test/b")
    assert clock.mono > before  # slept via injected clock
