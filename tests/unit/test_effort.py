"""Effort-scaled dispatch (M12, WS-F.4): the decomposer's estimate becomes
harness-enforced caps — clamping down, never raising configured budgets."""

from parsec.config import Budgets, effort_limits


def test_quick_clamps_hard():
    limits = effort_limits("quick", Budgets())
    assert limits.max_subquestions == 1
    assert limits.max_turns_per_subagent == 3
    assert limits.max_gap_rounds == 0


def test_standard_is_a_small_fanout():
    b = Budgets()
    limits = effort_limits("standard", b)
    assert limits.max_subquestions == 2
    assert limits.max_turns_per_subagent == b.max_turns_per_subagent
    assert limits.max_gap_rounds == b.max_gap_rounds


def test_deep_is_the_full_configured_caps():
    b = Budgets()
    limits = effort_limits("deep", b)
    assert limits.max_subquestions == b.max_subquestions
    assert limits.max_turns_per_subagent == b.max_turns_per_subagent
    assert limits.max_gap_rounds == b.max_gap_rounds


def test_unknown_effort_never_clamps():
    b = Budgets()
    assert effort_limits("??", b) == effort_limits("deep", b)  # v1-compatible default


def test_effort_never_raises_configured_caps():
    tight = Budgets(max_turns_per_subagent=2, max_subquestions=1, max_gap_rounds=0)
    for effort in ("quick", "standard", "deep"):
        limits = effort_limits(effort, tight)
        assert limits.max_subquestions <= 1
        assert limits.max_turns_per_subagent <= 2
        assert limits.max_gap_rounds == 0
