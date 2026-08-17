import pytest

from parsec.evals.support import (
    GroundedSupportChecker,
    MechanicalSupportChecker,
    make_support_checker,
    score_claim_support,
)
from parsec.store.dag import DagStore

SPAN_TEXT = (
    "Water boils at 100 degrees Celsius at standard atmospheric pressure, the "
    "reference value used to calibrate thermometers worldwide."
)


def test_full_support():
    grade = MechanicalSupportChecker().grade(
        "Water boils at 100 degrees Celsius at standard pressure.", [SPAN_TEXT]
    )
    assert grade == "full"


def test_number_mismatch_is_none():
    grade = MechanicalSupportChecker().grade("Water boils at 90 degrees Celsius.", [SPAN_TEXT])
    assert grade == "none"


def test_quote_must_appear():
    checker = MechanicalSupportChecker()
    assert checker.grade('The source calls it the "reference value".', [SPAN_TEXT]) == "full"
    assert checker.grade('The source calls it the "gold standard".', [SPAN_TEXT]) == "none"


def test_low_overlap_partial_and_none():
    checker = MechanicalSupportChecker()
    assert checker.grade(
        "Thermometers worldwide depend on fixed calibration references from water.", [SPAN_TEXT]
    ) in ("full", "partial")
    assert checker.grade("The Amazon rainforest hosts many turtles.", [SPAN_TEXT]) == "none"


def test_no_evidence_is_none():
    assert MechanicalSupportChecker().grade("Anything at all.", []) == "none"


def test_grounded_checker_keeps_exact_match_floor():
    checker = GroundedSupportChecker()
    # number mismatch is a hard fail regardless of prose overlap
    assert checker.grade("Water boils at 90 degrees Celsius.", [SPAN_TEXT]) == "none"


def test_grounded_checker_grades_via_verdicts():
    checker = GroundedSupportChecker()
    assert checker.grade("Water boils at 100 degrees Celsius at standard pressure.", [SPAN_TEXT]) == "full"
    # paraphrased but unsupported: no numbers/quotes for the floor to catch,
    # the grounded tier grades it none
    assert checker.grade("Acme's profits doubled.", [SPAN_TEXT]) == "none"
    # partial content overlap -> partial
    assert checker.grade(
        "Water pressure calibrates worldwide shipping lanes.", [SPAN_TEXT]
    ) == "partial"
    assert checker.grade("Anything.", []) == "none"


def test_make_support_checker():
    assert isinstance(make_support_checker("mechanical"), MechanicalSupportChecker)
    assert isinstance(make_support_checker("grounded"), GroundedSupportChecker)
    with pytest.raises(ValueError):
        make_support_checker("bogus")


@pytest.fixture
def graph(db, event_log, sessions, config):
    sessions.create(config)
    sid = config.session_id
    dag = DagStore(db, event_log)
    span = dag.add_node(
        sid, "SourceSpan",
        {"span_id": "doc:aaaaaaaaaaaa#0-10", "doc_hash": "a" * 64, "char_start": 0,
         "char_end": 10, "text": SPAN_TEXT, "url": "https://x.example", "fetched_ts": "t"},
    )
    premise = dag.add_node(
        sid, "Premise",
        {"text": "Water boils at 100 degrees Celsius.", "span_refs": ["doc:aaaaaaaaaaaa#0-10"], "claim_class": "stable"},
    )
    dag.add_edge(sid, premise, span, "extracts")
    supported = dag.add_node(
        sid, "ReportClaim",
        {"text": "Water boils at 100 degrees Celsius at standard atmospheric pressure.", "refs": [premise], "narrative": False},
    )
    dag.add_edge(sid, supported, premise, "aggregates")
    overreach = dag.add_node(
        sid, "ReportClaim",
        {"text": "Water boils at 90 degrees Celsius in most kitchens.", "refs": [premise], "narrative": False},
    )
    dag.add_edge(sid, overreach, premise, "aggregates")
    return sid


def test_score_claim_support_walks_chain(graph, db):
    score, details = score_claim_support(db, graph, MechanicalSupportChecker())
    grades = {d.claim_text[:20]: d.grade for d in details}
    assert grades["Water boils at 100 d"] == "full"
    assert grades["Water boils at 90 de"] == "none"  # cites evidence that does NOT say 90
    assert score == 0.5


def test_score_none_without_claims(db, sessions, config):
    sessions.create(config)
    score, details = score_claim_support(db, config.session_id, MechanicalSupportChecker())
    assert score is None and details == []
