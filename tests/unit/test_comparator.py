"""The regression comparator as a gate: it must read each metric in the
right direction and refuse to pass a comparison that did not measure the
same things on both sides."""

from __future__ import annotations

import json

import parsec.cli as cli
from parsec.evals.metrics import default_metrics, manifest
from parsec.evals.regression import compare_runs

HIGHER = {"name": "nugget_recall", "tier": "mechanical", "direction": "higher"}
BRIER = {"name": "brier", "tier": "mechanical", "direction": "lower"}


def _run(label: str, scores: dict[str, dict], metrics: list[dict] | None) -> dict:
    payload: dict = {
        "label": label,
        "results": [
            {"case_id": case_id, "session_id": f"eval-{case_id}", "status": "done",
             "scores": dict(values), "turns": 1, "error": None}
            for case_id, values in scores.items()
        ],
    }
    if metrics is not None:
        payload["metrics"] = metrics
    return payload


def _cases(axis: str, value: float, n: int = 3) -> dict[str, dict]:
    return {f"c{i}": {axis: value} for i in range(n)}


def test_built_in_metrics_declare_their_direction():
    entries = manifest(default_metrics(None, "judge-model"))
    assert entries and all(entry["direction"] == "higher" for entry in entries)


def test_lower_is_better_improvement_reads_as_improved():
    """Brier 0.40 -> 0.10 is the model getting better, not worse."""
    comparison = compare_runs(
        _run("a", _cases("brier", 0.40), [BRIER]), _run("b", _cases("brier", 0.10), [BRIER])
    )
    verdict = comparison.verdicts[0]
    assert verdict.verdict == "improved" and verdict.direction == "lower"
    assert verdict.mean_delta == -0.3  # the reported delta stays raw
    assert comparison.ok and comparison.regressions == []


def test_lower_is_better_regression_is_caught():
    comparison = compare_runs(
        _run("a", _cases("brier", 0.10), [BRIER]), _run("b", _cases("brier", 0.40), [BRIER])
    )
    assert comparison.verdicts[0].verdict == "regressed"
    assert not comparison.ok
    assert all(delta.regressed for delta in comparison.deltas)


def test_single_case_uses_direction_too():
    improved = compare_runs(
        _run("a", {"c": {"brier": 0.4}}, [BRIER]), _run("b", {"c": {"brier": 0.1}}, [BRIER])
    )
    worse = compare_runs(
        _run("a", {"c": {"brier": 0.1}}, [BRIER]), _run("b", {"c": {"brier": 0.4}}, [BRIER])
    )
    assert improved.verdicts[0].verdict == "improved" and improved.ok
    assert worse.verdicts[0].verdict == "regressed" and not worse.ok


def test_higher_is_better_is_unchanged_and_the_default():
    for metrics in ([HIGHER], [{"name": "nugget_recall", "tier": "mechanical"}]):
        dropped = compare_runs(
            _run("a", _cases("nugget_recall", 0.9), metrics),
            _run("b", _cases("nugget_recall", 0.5), metrics),
        )
        assert dropped.verdicts[0].verdict == "regressed" and not dropped.ok


def test_a_missing_case_fails_the_comparison_in_either_direction():
    full = _run("a", _cases("nugget_recall", 1.0, n=3), [HIGHER])
    short = _run("b", _cases("nugget_recall", 1.0, n=2), [HIGHER])
    dropped, added = compare_runs(full, short), compare_runs(short, full)
    assert dropped.only_in_a == ["c2"] and not dropped.ok
    assert added.only_in_b == ["c2"] and not added.ok
    assert dropped.regressed_axes == []  # nothing regressed: it is a coverage failure
    assert "c2" in dropped.coverage_problems[0]


def test_a_missing_metric_fails_the_comparison():
    both = _run("a", {"c": {"nugget_recall": 1.0, "brier": 0.1}}, [HIGHER, BRIER])
    one = _run("b", {"c": {"nugget_recall": 1.0}}, [HIGHER])
    comparison = compare_runs(both, one)
    assert comparison.metrics_only_in_a == ["brier"]
    assert not comparison.ok


def test_a_mechanical_score_on_one_side_only_fails_the_comparison():
    a = _run("a", {"c": {"nugget_recall": 0.8}}, [HIGHER])
    b = _run("b", {"c": {"nugget_recall": None}}, [HIGHER])
    comparison = compare_runs(a, b)
    assert comparison.unpaired_scores == [("c", "nugget_recall")]
    assert not comparison.ok


def test_an_advisory_score_on_one_side_only_never_gates():
    """A judge wired into one run and not the other is a configuration
    difference on an axis that may not gate anyway."""
    judged = [HIGHER, {"name": "synthesis", "tier": "advisory", "direction": "higher"}]
    a = _run("a", {"c": {"nugget_recall": 1.0, "synthesis": None}}, judged)
    b = _run("b", {"c": {"nugget_recall": 1.0, "synthesis": 0.7}}, judged)
    assert compare_runs(a, b).ok


def test_runs_that_disagree_on_direction_are_not_comparable():
    flipped = {**BRIER, "direction": "higher"}
    comparison = compare_runs(
        _run("a", _cases("brier", 0.2), [BRIER]), _run("b", _cases("brier", 0.2), [flipped])
    )
    assert comparison.direction_conflicts == ["brier"]
    assert not comparison.ok


def test_identical_runs_compare_ok():
    run = _run("a", _cases("nugget_recall", 0.7), [HIGHER])
    comparison = compare_runs(run, run)
    assert comparison.ok and comparison.coverage_problems == []


def test_cli_compare_exits_nonzero_and_says_why_when_a_case_is_missing(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(_run("a", _cases("nugget_recall", 1.0, n=2), [HIGHER])))
    b.write_text(json.dumps(_run("b", _cases("nugget_recall", 1.0, n=1), [HIGHER])))

    assert cli.main(["eval", "compare", str(a), str(b)]) == cli.EXIT_PARTIAL
    out = capsys.readouterr().out
    assert "not comparable" in out and "c1" in out
    assert "no significant regressions" not in out

    assert cli.main(["eval", "compare", str(a), str(b), "--json"]) == cli.EXIT_PARTIAL
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["coverage_problems"]
    assert cli.main(["eval", "compare", str(a), str(a)]) == cli.EXIT_OK
