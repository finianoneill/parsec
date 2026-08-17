import pytest

from parsec import ids
from parsec.evals.judge import parse_judge_reply
from parsec.evals.regression import compare_runs
from parsec.evals.scoring import score_citation_faithfulness, score_coverage, score_session
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.spans import SpanStore

DOC_TEXT = "Water boils at 100 degrees Celsius at sea level."


@pytest.fixture
def graph(db, blobs, event_log, sessions, config, clock):
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
    span_ref = ids.span_id(doc_hash, 0, len(DOC_TEXT))
    spans.put_spans(doc_hash, [(span_ref, 0, len(DOC_TEXT), DOC_TEXT)])
    span_node = dag.add_node(
        sid, "SourceSpan",
        {"span_id": span_ref, "doc_hash": doc_hash, "char_start": 0, "char_end": len(DOC_TEXT),
         "text": DOC_TEXT, "url": "https://example.test/w", "fetched_ts": clock.now_iso()},
    )
    premise = dag.add_node(
        sid, "Premise",
        {"text": "Water boils at 100 degrees Celsius.", "span_refs": [span_ref], "claim_class": "stable"},
    )
    dag.add_edge(sid, premise, span_node, "extracts")
    claim = dag.add_node(
        sid, "ReportClaim",
        {"text": "Water boils at 100 degrees Celsius.", "refs": [premise], "narrative": False},
    )
    dag.add_edge(sid, claim, premise, "aggregates")
    return sid, dag, span_ref


def test_citation_faithfulness_clean(graph, db, blobs):
    sid = graph[0]
    score, total, faithful = score_citation_faithfulness(db, blobs, sid)
    assert (score, total, faithful) == (1.0, 1, 1)


def test_citation_faithfulness_drops_on_corruption(graph, db, blobs):
    sid, _, span_ref = graph
    db.execute("UPDATE spans SET text='TAMPERED' WHERE span_id=?", (span_ref,))
    score, total, faithful = score_citation_faithfulness(db, blobs, sid)
    assert (score, total, faithful) == (0.0, 1, 0)


def test_citation_faithfulness_none_without_claims(db, blobs, sessions, config):
    sessions.create(config)
    score, total, faithful = score_citation_faithfulness(db, blobs, config.session_id)
    assert score is None and total == 0


def test_coverage_substring_and_regex(graph, db):
    sid = graph[0]
    score, hits, misses = score_coverage(
        db, sid, ["100 DEGREES celsius", "re:boils at \\d+", "mount everest"]
    )
    assert abs(score - 2 / 3) < 1e-9
    assert misses == ["mount everest"]


def test_coverage_none_without_gold(graph, db):
    assert score_coverage(db, graph[0], []) == (None, [], [])


def test_score_session_bundle(graph, db, blobs):
    scores = score_session(db, blobs, graph[0], ["100 degrees"], synthesis=0.75)
    assert scores.citation_faithfulness == 1.0
    assert scores.coverage == 1.0
    assert scores.synthesis == 0.75


def test_judge_parse():
    assert parse_judge_reply('{"synthesis_score": 4, "rationale": "solid"}') == 0.75
    assert parse_judge_reply('prose then {"synthesis_score": 1} trailing') == 0.0
    assert parse_judge_reply('{"synthesis_score": 9}') is None
    assert parse_judge_reply("no json at all") is None
    assert parse_judge_reply('{"wrong_key": 3}') is None


def _results(label, **case_scores):
    return {
        "label": label,
        "results": [
            {"case_id": cid, "session_id": f"eval-{cid}", "status": "done",
             "scores": {"citation_faithfulness": s[0], "coverage": s[1], "synthesis": s[2]},
             "turns": 5, "error": None}
            for cid, s in case_scores.items()
        ],
    }


def test_compare_detects_regression():
    a = _results("a", case1=(1.0, 0.66, 0.75))
    b = _results("b", case1=(1.0, 0.33, 0.75))
    comparison = compare_runs(a, b, epsilon=0.05)
    assert not comparison.ok
    assert [d.axis for d in comparison.regressions] == ["coverage"]


def test_compare_tolerates_epsilon_and_none(a=None):
    a = _results("a", case1=(1.0, 0.5, None))
    b = _results("b", case1=(0.97, 0.5, 0.75))  # -0.03 within epsilon; None->score not a regression
    comparison = compare_runs(a, b, epsilon=0.05)
    assert comparison.ok


def test_compare_case_sets():
    a = _results("a", case1=(1.0, 1.0, None), gone=(1.0, 1.0, None))
    b = _results("b", case1=(1.0, 1.0, None), new=(1.0, 1.0, None))
    comparison = compare_runs(a, b)
    assert comparison.only_in_a == ["gone"]
    assert comparison.only_in_b == ["new"]
    assert comparison.ok
