import pytest

from parsec.evals.case import Nugget
from parsec.evals.regression import compare_runs
from parsec.evals.scoring import score_nuggets
from parsec.store.dag import DagStore


@pytest.fixture
def claims(db, event_log, sessions, config):
    sessions.create(config)
    dag = DagStore(db, event_log)

    def write(texts):
        for t in texts:
            dag.add_node(
                config.session_id, "ReportClaim",
                {"text": t, "refs": ["premise:aaaaaaaaaaaaaaaa"], "narrative": False},
            )

    return write, config.session_id


NUGGETS = [
    Nugget(
        text="sea-level boiling point is 100C",
        weight="vital",
        patterns=["100 degrees Celsius"],
        contradiction_patterns=[r"re:9\d degrees celsius"],
    ),
    Nugget(
        text="everest boiling ~70C",
        weight="okay",
        patterns=[r"re:everest.*70 degrees"],
    ),
]


def test_nugget_full_recall(claims, db):
    write, sid = claims
    write([
        "Water boils at 100 degrees Celsius at sea level.",
        "On Mount Everest water boils at about 70 degrees Celsius.",
    ])
    recall, hits, misses, contra = score_nuggets(db, sid, NUGGETS)
    assert recall == 1.0 and misses == [] and contra == []


def test_nugget_weighted_partial_recall(claims, db):
    write, sid = claims
    write(["Water boils at 100 degrees Celsius at sea level."])
    recall, hits, misses, contra = score_nuggets(db, sid, NUGGETS)
    assert abs(recall - 1.0 / 1.5) < 1e-9  # vital hit, okay missed
    assert misses == ["everest boiling ~70C"]


def test_nugget_contradiction_detected(claims, db):
    write, sid = claims
    write(["My tests show water boils at 90 degrees Celsius."])
    recall, hits, misses, contra = score_nuggets(db, sid, NUGGETS)
    assert contra == ["sea-level boiling point is 100C"]
    assert recall == 0.0


def test_nugget_none_without_gold(claims, db):
    _, sid = claims
    assert score_nuggets(db, sid, []) == (None, [], [], [])


def _results(label, cases: dict[str, dict]):
    return {
        "label": label,
        "results": [
            {"case_id": cid, "session_id": f"eval-{cid}", "status": "done",
             "scores": scores, "turns": 5, "error": None}
            for cid, scores in cases.items()
        ],
    }


def _scores(**kw):
    base = {"citation_faithfulness": 1.0, "coverage": None, "nugget_recall": 1.0,
            "claim_support": 1.0, "synthesis": None}
    base.update(kw)
    return base


def test_verdict_regressed_with_ci_over_multiple_cases():
    a = _results("a", {f"c{i}": _scores() for i in range(3)})
    b = _results("b", {f"c{i}": _scores(nugget_recall=0.5) for i in range(3)})
    comparison = compare_runs(a, b, epsilon=0.05)
    verdict = next(v for v in comparison.verdicts if v.axis == "nugget_recall")
    assert verdict.verdict == "regressed"
    assert verdict.n_cases == 3
    assert verdict.mean_delta == -0.5
    assert verdict.ci95 == 0.0  # identical deltas -> zero variance
    assert not comparison.ok


def test_verdict_inconclusive_when_mixed():
    a = _results("a", {"c1": _scores(nugget_recall=1.0), "c2": _scores(nugget_recall=0.2)})
    b = _results("b", {"c1": _scores(nugget_recall=0.2), "c2": _scores(nugget_recall=1.0)})
    comparison = compare_runs(a, b, epsilon=0.05)
    verdict = next(v for v in comparison.verdicts if v.axis == "nugget_recall")
    assert verdict.verdict == "inconclusive"  # mean 0, wide CI
    assert comparison.ok


def test_verdict_improved():
    a = _results("a", {f"c{i}": _scores(claim_support=0.4) for i in range(2)})
    b = _results("b", {f"c{i}": _scores(claim_support=0.9) for i in range(2)})
    comparison = compare_runs(a, b)
    verdict = next(v for v in comparison.verdicts if v.axis == "claim_support")
    assert verdict.verdict == "improved"
    assert comparison.ok


def test_single_case_epsilon_fallback():
    a = _results("a", {"c1": _scores(nugget_recall=1.0)})
    b = _results("b", {"c1": _scores(nugget_recall=0.97)})
    comparison = compare_runs(a, b, epsilon=0.05)
    verdict = next(v for v in comparison.verdicts if v.axis == "nugget_recall")
    assert verdict.verdict == "inconclusive"  # within epsilon, no CI possible
    b_bad = _results("b", {"c1": _scores(nugget_recall=0.5)})
    comparison = compare_runs(a, b_bad, epsilon=0.05)
    verdict = next(v for v in comparison.verdicts if v.axis == "nugget_recall")
    assert verdict.verdict == "regressed" and verdict.ci95 is None
