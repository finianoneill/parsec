import json

import pytest

from parsec.retrieval.search_provider import FixtureSearchProvider, normalize_query


@pytest.fixture
def provider(tmp_path):
    fixtures = {
        "python packaging history": [
            {"title": "A", "url": "https://example.test/a", "snippet": "about packaging"},
            {"title": "B", "url": "https://example.test/b", "snippet": "more"},
        ],
        "rust borrow checker": [
            {"title": "C", "url": "https://example.test/c"},
        ],
    }
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(fixtures))
    return FixtureSearchProvider(path)


def test_normalize_query():
    assert normalize_query("  Python, Packaging — History!  ") == "python packaging history"


async def test_exact_match(provider):
    hits = await provider.search("Python Packaging History", k=5)
    assert [h.url for h in hits] == ["https://example.test/a", "https://example.test/b"]
    assert hits[0].rank == 1


async def test_k_limits(provider):
    hits = await provider.search("python packaging history", k=1)
    assert len(hits) == 1


async def test_fuzzy_match(provider):
    hits = await provider.search("history of python packaging", k=5)
    assert hits and hits[0].url == "https://example.test/a"


async def test_miss_returns_empty(provider):
    assert await provider.search("completely unrelated topic", k=5) == []


async def test_deterministic(provider):
    a = await provider.search("python packaging", k=5)
    b = await provider.search("python packaging", k=5)
    assert a == b
