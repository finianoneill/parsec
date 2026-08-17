import pytest

from parsec.loop.citations import check_citations, segment_answer, write_claims
from parsec.store.dag import DagStore


@pytest.fixture
def dag_with_premises(db, event_log, sessions, config):
    sessions.create(config)
    dag = DagStore(db, event_log)
    p1 = dag.add_node(
        config.session_id,
        "Premise",
        {"text": "Water boils at 100 degrees Celsius.", "span_refs": ["doc:abcdefabcdef#0-48"], "claim_class": "stable"},
    )
    p2 = dag.add_node(
        config.session_id,
        "Premise",
        {"text": "Everest boiling is about 70 degrees.", "span_refs": ["doc:abcdefabcdef#50-90"], "claim_class": "stable"},
    )
    return dag, p1, p2


def test_segment_answer_splits_and_classifies(dag_with_premises):
    dag, p1, p2 = dag_with_premises
    segs = segment_answer(
        "Here is what I found. [narrative]\n"
        f"Water boils at 100C. [{p1}] "
        f"It drops at altitude. [{p2}]"
    )
    assert len(segs) == 3
    assert segs[0].narrative and not segs[0].refs
    assert segs[1].refs == [p1]
    assert segs[2].refs == [p2]
    assert segs[1].text == "Water boils at 100C."


def test_check_passes_with_valid_refs(dag_with_premises, config):
    dag, p1, p2 = dag_with_premises
    answer = f"Summary follows. [narrative]\nWater boils at 100C at sea level. [{p1}]"
    check = check_citations(answer, config.session_id, dag)
    assert check.ok
    assert len(check.claim_segments) == 1


def test_uncited_sentence_flagged(dag_with_premises, config):
    dag, p1, p2 = dag_with_premises
    check = check_citations("Water boils at 100C.", config.session_id, dag)
    assert not check.ok
    assert "uncited" in check.problems[0]


def test_unknown_premise_flagged(dag_with_premises, config):
    dag, p1, p2 = dag_with_premises
    check = check_citations("A claim. [premise:0000000000000000]", config.session_id, dag)
    assert not check.ok
    assert "unknown premise" in check.problems[0]


def test_premise_from_other_session_rejected(dag_with_premises, config, db, event_log, sessions, tmp_path):
    from tests.conftest import make_config

    dag, p1, p2 = dag_with_premises
    other = make_config(tmp_path, session_id="s-other")
    sessions.create(other)
    p_other = dag.add_node(
        "s-other",
        "Premise",
        {"text": "A different session's fact 42.", "span_refs": ["doc:abcdefabcdef#0-10"], "claim_class": "stable"},
    )
    check = check_citations(f"A claim. [{p_other}]", config.session_id, dag)
    assert not check.ok


def test_write_claims_builds_dag(dag_with_premises, config):
    dag, p1, p2 = dag_with_premises
    answer = f"Intro. [narrative]\nBoiling at sea level is 100C. [{p1}] Altitude lowers it. [{p2}]"
    check = check_citations(answer, config.session_id, dag)
    assert check.ok
    n = write_claims(config.session_id, check, dag)
    assert n == 2
    claims = dag.nodes_for_session(config.session_id, tier=4)
    edges = dag.edges_for_session(config.session_id)
    assert len(claims) == 2
    assert len(edges) == 2
    assert all(e["edge_type"] == "aggregates" for e in edges)
    assert {e["dst_node_id"] for e in edges} == {p1, p2}
