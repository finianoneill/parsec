import pytest

from parsec import ids
from parsec.loop.citations import check_citations, segment_answer, write_claims
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.spans import SpanStore

DOC_TEXT = "Water boils at 100 degrees Celsius at sea level.\n\nThe boiling point drops at altitude."


@pytest.fixture
def stores(db, blobs, event_log, sessions, config, clock):
    sessions.create(config)
    documents = DocumentStore(db, clock)
    spans = SpanStore(db)
    dag = DagStore(db, event_log)

    raw = DOC_TEXT.encode()
    doc_hash = ids.doc_hash(raw)
    blobs.put(raw)
    text_blob = blobs.put(DOC_TEXT)
    documents.put_document(doc_hash, "https://example.test/water", "text/plain", 200, len(raw), text_blob, {})
    documents.cache_put(ids.cache_key("https://example.test/water"), "https://example.test/water", doc_hash, "record")
    s1 = ids.span_id(doc_hash, 0, 48)
    s2 = ids.span_id(doc_hash, 50, 89)
    spans.put_spans(doc_hash, [(s1, 0, 48, DOC_TEXT[0:48]), (s2, 50, 89, DOC_TEXT[50:89])])
    return documents, spans, dag, s1, s2


def test_segment_answer_splits_and_classifies():
    segs = segment_answer(
        "Here is what I found. [narrative]\n"
        "Water boils at 100C. [doc:abcdefabcdef#0-48] "
        "It drops at altitude. [doc:abcdefabcdef#50-89]"
    )
    assert len(segs) == 3
    assert segs[0].narrative and not segs[0].refs
    assert segs[1].refs == ["doc:abcdefabcdef#0-48"]
    assert segs[2].refs == ["doc:abcdefabcdef#50-89]".rstrip("]")]
    assert segs[1].text == "Water boils at 100C."


def test_check_passes_with_valid_refs(stores, blobs):
    documents, spans, dag, s1, s2 = stores
    answer = f"Summary follows. [narrative]\nWater boils at 100C at sea level. [{s1}]"
    check = check_citations(answer, spans, documents, blobs)
    assert check.ok
    assert len(check.claim_segments) == 1


def test_uncited_sentence_flagged(stores, blobs):
    documents, spans, dag, s1, s2 = stores
    check = check_citations("Water boils at 100C.", spans, documents, blobs)
    assert not check.ok
    assert "uncited" in check.problems[0]


def test_unknown_span_flagged(stores, blobs):
    documents, spans, dag, s1, s2 = stores
    check = check_citations("A claim. [doc:000000000000#0-10]", spans, documents, blobs)
    assert not check.ok
    assert "unknown span" in check.problems[0]


def test_corrupted_span_text_flagged(stores, blobs, db):
    documents, spans, dag, s1, s2 = stores
    db.execute("UPDATE spans SET text='TAMPERED' WHERE span_id=?", (s1,))
    check = check_citations(f"Water boils. [{s1}]", spans, documents, blobs)
    assert not check.ok
    assert "does not match" in check.problems[0]


def test_write_claims_builds_dag(stores, blobs, config, db):
    documents, spans, dag, s1, s2 = stores
    answer = f"Intro. [narrative]\nBoiling at sea level is 100C. [{s1}] Altitude lowers it. [{s2}]"
    check = check_citations(answer, spans, documents, blobs)
    assert check.ok
    n = write_claims(config.session_id, check, dag, spans, documents)
    assert n == 2
    claims = dag.nodes_for_session(config.session_id, tier=4)
    source_spans = dag.nodes_for_session(config.session_id, tier=0)
    edges = dag.edges_for_session(config.session_id)
    assert len(claims) == 2 and len(source_spans) == 2 and len(edges) == 2
    assert all(e["edge_type"] == "extracts" for e in edges)
