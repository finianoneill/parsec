import pytest

from parsec import ids
from parsec.models.tools import ToolIntent
from parsec.retrieval.embeddings import EmbeddingCache, HashedNgramEmbedder, cosine
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.store.event_log import stream_scope
from parsec.tools.search_within import SearchWithinTool, bm25_rank, visible_doc_hashes


def journal_fetch(event_log, session_id, doc_hash, url):
    from parsec.models.events import EventType

    event_log.append(
        session_id, "tool:fetch", EventType.FETCH_PERFORMED,
        {"cache_key": "k", "doc_hash": doc_hash, "url": url, "status_code": 200,
         "outcome": "ok", "from_cache": False, "mode": "record"},
    )


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
        # Fetched by this stream: the visibility rule (M15.9) keys off the
        # journaled fetch, exactly as FetchTool records it.
        journal_fetch(event_log, config.session_id, doc_hash, f"https://example.test/{i}")
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


def _add_doc(db, blobs, clock, spans, text: str, url: str) -> tuple[str, str]:
    from parsec.store.documents import DocumentStore

    raw = text.encode()
    doc_hash = ids.doc_hash(raw)
    blobs.put(raw)
    DocumentStore(db, clock).put_document(doc_hash, url, "text/plain", 200, len(raw), blobs.put(text), {})
    sid_ = ids.span_id(doc_hash, 0, len(text))
    spans.put_spans(doc_hash, [(sid_, 0, len(text), text)])
    return doc_hash, sid_


async def test_search_scoped_to_what_the_stream_could_see(db, blobs, event_log, ledger, sessions, config, clock):
    """M15.9: sq-2 sees its own fetch and sq-1's (sq-1 completed before sq-2
    started); it does not see a concurrent sibling's fetch, nor a document
    in the corpus that this session never fetched; the orchestrator sees
    every completed stream."""
    from parsec.models.events import EventType

    sessions.create(config)
    sid = config.session_id
    spans = SpanStore(db)
    text = "Water boils at 100 degrees Celsius at sea level; the boiling point is pressure dependent."
    h_old, s_old = _add_doc(db, blobs, clock, spans, text + " (older session)", "https://old.example/x")
    h1, s1 = _add_doc(db, blobs, clock, spans, text + " (sq-1)", "https://a.example/1")
    h2, s2 = _add_doc(db, blobs, clock, spans, text + " (sq-2)", "https://a.example/2")
    h3, s3 = _add_doc(db, blobs, clock, spans, text + " (sq-3 sibling)", "https://a.example/3")

    with stream_scope("sq-1"):
        event_log.append(sid, "subagent:sq-1", EventType.SUBAGENT_STARTED, {"sq_id": "sq-1"})
        journal_fetch(event_log, sid, h1, "https://a.example/1")
        event_log.append(sid, "subagent:sq-1", EventType.SUBAGENT_COMPLETED,
                         {"sq_id": "sq-1", "status": "answered", "premises": [], "findings": [], "dead_ends": []})
    with stream_scope("sq-2"):
        event_log.append(sid, "subagent:sq-2", EventType.SUBAGENT_STARTED, {"sq_id": "sq-2"})
    with stream_scope("sq-3"):
        event_log.append(sid, "subagent:sq-3", EventType.SUBAGENT_STARTED, {"sq_id": "sq-3"})
        journal_fetch(event_log, sid, h3, "https://a.example/3")
        event_log.append(sid, "subagent:sq-3", EventType.SUBAGENT_COMPLETED,
                         {"sq_id": "sq-3", "status": "answered", "premises": [], "findings": [], "dead_ends": []})
    with stream_scope("sq-2"):
        journal_fetch(event_log, sid, h2, "https://a.example/2")

    assert visible_doc_hashes(event_log, sid, "sq-2") == {h1, h2}
    assert visible_doc_hashes(event_log, sid, "sq-3") == {h1, h3}
    assert visible_doc_hashes(event_log, sid, "orchestrator") == {h1, h3}
    assert h_old not in visible_doc_hashes(event_log, sid, "sq-2")

    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    registry = ToolRegistry([SearchWithinTool(spans, EmbeddingCache(db, HashedNgramEmbedder()))])
    with stream_scope("sq-2"):
        result = await registry.dispatch(
            ToolIntent(tool_use_id="t", tool_name="search_within", input={"query": "boiling point sea level", "k": 10}),
            ctx,
        )
    assert s1 in result.truncated_text and s2 in result.truncated_text
    assert s3 not in result.truncated_text and s_old not in result.truncated_text


def test_bm25_rank_is_deterministic_and_relevant():
    rows = [
        ("doc:a#0-1", "The Amazon rainforest hosts many tree species."),
        ("doc:b#0-1", "Water boils at 100 degrees Celsius at sea level."),
        ("doc:c#0-1", "Sea level pressure is one atmosphere; water boils at 100 degrees there."),
    ]
    ranked = bm25_rank("water boils sea level", rows)
    assert ranked[0] in ("doc:b#0-1", "doc:c#0-1") and "doc:a#0-1" not in ranked
    assert ranked == bm25_rank("level sea boils water", rows)  # term order irrelevant
    assert bm25_rank("turtles", rows) == [] and bm25_rank("", rows) == []
