import pytest

from parsec import ids
from parsec.models.tools import ToolIntent
from parsec.retrieval.embeddings import EmbeddingCache, HashedNgramEmbedder, cosine
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.tools.search_within import SearchWithinTool


def test_hashed_embedder_deterministic_and_normalized():
    emb = HashedNgramEmbedder()
    a1, a2 = emb.embed(["water boils at 100 degrees"] * 2)
    assert a1 == a2
    assert abs(cosine(a1, a1) - 1.0) < 1e-6
    b = emb.embed(["completely unrelated text about turtles"])[0]
    assert cosine(a1, b) < cosine(a1, a1)


def test_embedding_cache_computes_once(db):
    class CountingEmbedder(HashedNgramEmbedder):
        calls = 0

        def embed(self, texts):
            CountingEmbedder.calls += len(texts)
            return super().embed(texts)

    cache = EmbeddingCache(db, CountingEmbedder())
    cache.embed(["hello world"])
    cache.embed(["hello world"])
    assert CountingEmbedder.calls == 1
    assert db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 1


@pytest.fixture
def corpus(db, blobs, event_log, ledger, sessions, config, clock):
    sessions.create(config)
    spans = SpanStore(db)
    from parsec.store.documents import DocumentStore

    documents = DocumentStore(db, clock)
    texts = [
        "Water boils at 100 degrees Celsius at standard atmospheric pressure at sea level.",
        "On the summit of Mount Everest water boils at about 70 degrees Celsius.",
        "The Amazon rainforest hosts an extraordinary diversity of tree species and wildlife.",
    ]
    span_ids = []
    for i, text in enumerate(texts):
        raw = text.encode()
        doc_hash = ids.doc_hash(raw)
        blobs.put(raw)
        text_blob = blobs.put(text)
        documents.put_document(doc_hash, f"https://example.test/{i}", "text/plain", 200, len(raw), text_blob, {})
        sid_ = ids.span_id(doc_hash, 0, len(text))
        spans.put_spans(doc_hash, [(sid_, 0, len(text), text)])
        span_ids.append(sid_)
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    tool = SearchWithinTool(spans, EmbeddingCache(db, HashedNgramEmbedder()))
    return ToolRegistry([tool]), ctx, span_ids


async def test_hybrid_search_finds_relevant_span(corpus):
    registry, ctx, span_ids = corpus
    intent = ToolIntent(
        tool_use_id="t1", tool_name="search_within",
        input={"query": "boiling temperature at sea level", "k": 2},
    )
    result = await registry.dispatch(intent, ctx)
    assert result.ok
    assert span_ids[0] in result.truncated_text  # sea-level span ranks in top 2
    assert span_ids[2] not in result.truncated_text  # rainforest span does not


async def test_search_within_deterministic(corpus):
    registry, ctx, _ = corpus
    intent = ToolIntent(tool_use_id="t2", tool_name="search_within", input={"query": "water boils"})
    a = await registry.dispatch(intent, ctx)
    intent2 = ToolIntent(tool_use_id="t3", tool_name="search_within", input={"query": "water boils"})
    b = await registry.dispatch(intent2, ctx)
    assert a.truncated_text == b.truncated_text


async def test_empty_corpus(db, blobs, event_log, ledger, sessions, config, clock):
    sessions.create(config)
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    tool = SearchWithinTool(SpanStore(db), EmbeddingCache(db, HashedNgramEmbedder()))
    registry = ToolRegistry([tool])
    intent = ToolIntent(tool_use_id="t1", tool_name="search_within", input={"query": "anything"})
    result = await registry.dispatch(intent, ctx)
    assert result.ok and "no matching spans" in result.truncated_text
