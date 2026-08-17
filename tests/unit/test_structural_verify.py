import pytest

from parsec import ids
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.spans import SpanStore
from parsec.verify.structural import verify_session

DOC_TEXT = "Water boils at 100 degrees Celsius at sea level.\n\nEverest boiling is about 70 degrees."


@pytest.fixture
def graph(db, blobs, event_log, sessions, config, clock):
    """A well-formed claim -> premise -> span chain plus the corpus behind it."""
    sessions.create(config)
    sid = config.session_id
    documents = DocumentStore(db, clock)
    spans = SpanStore(db)
    dag = DagStore(db, event_log)

    raw = DOC_TEXT.encode()
    doc_hash = ids.doc_hash(raw)
    blobs.put(raw)
    text_blob = blobs.put(DOC_TEXT)
    documents.put_document(doc_hash, "https://example.test/w", "text/plain", 200, len(raw), text_blob, {})
    span_ref = ids.span_id(doc_hash, 0, 48)
    spans.put_spans(doc_hash, [(span_ref, 0, 48, DOC_TEXT[0:48])])

    span_node = dag.add_node(
        sid,
        "SourceSpan",
        {
            "span_id": span_ref,
            "doc_hash": doc_hash,
            "char_start": 0,
            "char_end": 48,
            "text": DOC_TEXT[0:48],
            "url": "https://example.test/w",
            "fetched_ts": clock.now_iso(),
        },
    )
    premise = dag.add_node(
        sid,
        "Premise",
        {"text": "Water boils at 100 degrees Celsius.", "span_refs": [span_ref], "claim_class": "stable"},
    )
    dag.add_edge(sid, premise, span_node, "extracts")
    claim = dag.add_node(
        sid,
        "ReportClaim",
        {"text": "Water boils at 100 degrees Celsius.", "premise_refs": [premise], "narrative": False},
    )
    dag.add_edge(sid, claim, premise, "aggregates")
    return sid, dag, span_node, premise, claim, span_ref, doc_hash


def test_clean_graph_passes(graph, db, blobs):
    sid = graph[0]
    report = verify_session(db, blobs, sid)
    assert report.ok, [v.detail for v in report.violations]
    assert report.checked_claims == 1
    assert report.checked_premises == 1
    assert report.checked_spans == 1


def test_claim_without_path_flagged(graph, db, blobs):
    sid, dag = graph[0], graph[1]
    dag.add_node(sid, "ReportClaim", {"text": "An orphan claim.", "premise_refs": ["premise:deadbeefdeadbeef"], "narrative": False})
    report = verify_session(db, blobs, sid)
    assert any(v.check == "claim-path" for v in report.violations)


def test_premise_without_extracts_flagged(graph, db, blobs):
    sid, dag = graph[0], graph[1]
    dag.add_node(sid, "Premise", {"text": "Floating premise.", "span_refs": ["doc:000000000000#0-1"], "claim_class": "stable"})
    report = verify_session(db, blobs, sid)
    assert any(v.check == "tier-integrity" and "extracts" in v.detail for v in report.violations)


def test_tampered_span_row_flagged(graph, db, blobs):
    sid, span_ref = graph[0], graph[5]
    db.execute("UPDATE spans SET text='TAMPERED' WHERE span_id=?", (span_ref,))
    report = verify_session(db, blobs, sid)
    assert any(v.check == "corpus-integrity" and "diverges" in v.detail for v in report.violations)


def test_span_offsets_tampered_flagged(graph, db, blobs):
    sid, span_ref = graph[0], graph[5]
    # shift offsets so the row no longer matches the document slice
    db.execute("UPDATE spans SET char_start=5, char_end=53 WHERE span_id=?", (span_ref,))
    report = verify_session(db, blobs, sid)
    assert any(
        v.check == "corpus-integrity" and "verbatim slice" in v.detail for v in report.violations
    )


def test_deleted_span_row_flagged(graph, db, blobs):
    sid, span_ref = graph[0], graph[5]
    db.execute("DELETE FROM spans WHERE span_id=?", (span_ref,))
    report = verify_session(db, blobs, sid)
    assert any(v.check == "corpus-integrity" and "missing" in v.detail for v in report.violations)


def test_containment_recheck_flags_bad_premise(graph, db, blobs, event_log, config):
    sid, dag, span_node = graph[0], graph[1], graph[2]
    bad = dag.add_node(
        sid,
        "Premise",
        {"text": "Water boils at 90 degrees.", "span_refs": [graph[5]], "claim_class": "stable"},
    )
    dag.add_edge(sid, bad, span_node, "extracts")
    report = verify_session(db, blobs, sid)
    assert any(v.check == "containment" and "'90'" in v.detail for v in report.violations)


def test_illegal_edge_type_combo_flagged(graph, db, blobs):
    sid, dag, span_node, premise, claim = graph[:5]
    dag.add_edge(sid, claim, span_node, "extracts")  # claims may not extract directly (M2+)
    report = verify_session(db, blobs, sid)
    assert any(
        v.check == "tier-integrity" and "may not connect" in v.detail for v in report.violations
    )


def test_cycle_flagged(graph, db, blobs):
    sid, dag, _, premise, claim = graph[:5]
    dag.add_edge(sid, premise, claim, "contradicts")  # premise -> claim closes a cycle
    report = verify_session(db, blobs, sid)
    assert any(v.check == "acyclic" for v in report.violations)
