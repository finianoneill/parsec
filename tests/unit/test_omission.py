import pytest

from parsec.models.events import EventType
from parsec.store.dag import DagStore
from parsec.verify.omission import detect_omissions


@pytest.fixture
def setup(db, event_log, sessions, config):
    sessions.create(config)
    dag = DagStore(db, event_log)
    sid = config.session_id

    def fetched(doc_hash, url):
        event_log.append(
            sid, "tool:fetch", EventType.FETCH_PERFORMED,
            {"cache_key": "k" + doc_hash[:8], "doc_hash": doc_hash, "url": url,
             "status_code": 200, "from_cache": False, "mode": "record"},
        )

    return dag, sid, fetched


def test_unused_document_and_uncited_premise_surface(setup, db, event_log):
    dag, sid, fetched = setup
    used_hash, unused_hash = "a" * 64, "b" * 64
    fetched(used_hash, "https://used.example/page")
    fetched(unused_hash, "https://ignored.example/page")

    span = dag.add_node(
        sid, "SourceSpan",
        {"span_id": "doc:aaaaaaaaaaaa#0-10", "doc_hash": used_hash, "char_start": 0,
         "char_end": 10, "text": "t", "url": "https://used.example/page", "fetched_ts": "t"},
    )
    cited = dag.add_node(sid, "Premise", {"text": "Cited fact.", "span_refs": ["doc:aaaaaaaaaaaa#0-10"], "claim_class": "stable"})
    dag.add_edge(sid, cited, span, "extracts")
    uncited = dag.add_node(sid, "Premise", {"text": "Orphan fact.", "span_refs": ["doc:aaaaaaaaaaaa#0-10"], "claim_class": "stable"})
    dag.add_edge(sid, uncited, span, "extracts")
    claim = dag.add_node(sid, "ReportClaim", {"text": "Cited fact.", "refs": [cited], "narrative": False})
    dag.add_edge(sid, claim, cited, "aggregates")

    report = detect_omissions(db, event_log, sid)
    assert report.unused_documents == [{"url": "https://ignored.example/page", "doc_hash": unused_hash}]
    assert [p["node_id"] for p in report.uncited_premises] == [uncited]


def test_all_used_is_empty(setup, db, event_log):
    dag, sid, fetched = setup
    used_hash = "a" * 64
    fetched(used_hash, "https://used.example/page")
    span = dag.add_node(
        sid, "SourceSpan",
        {"span_id": "doc:aaaaaaaaaaaa#0-10", "doc_hash": used_hash, "char_start": 0,
         "char_end": 10, "text": "t", "url": "https://used.example/page", "fetched_ts": "t"},
    )
    p = dag.add_node(sid, "Premise", {"text": "Fact.", "span_refs": ["doc:aaaaaaaaaaaa#0-10"], "claim_class": "stable"})
    dag.add_edge(sid, p, span, "extracts")
    claim = dag.add_node(sid, "ReportClaim", {"text": "Fact.", "refs": [p], "narrative": False})
    dag.add_edge(sid, claim, p, "aggregates")
    report = detect_omissions(db, event_log, sid)
    assert report.empty


def test_duplicate_fetches_reported_once(setup, db, event_log):
    dag, sid, fetched = setup
    h = "c" * 64
    fetched(h, "https://ignored.example/page")
    fetched(h, "https://ignored.example/page")
    report = detect_omissions(db, event_log, sid)
    assert len(report.unused_documents) == 1
