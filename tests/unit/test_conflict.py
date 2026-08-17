"""Dual-perspective conflict retrieval (M9, WS-C.5)."""

from parsec.verify.conflict import dual_perspective_question, negation_query

CLAIM = "Water boils at 100 degrees Celsius at sea level."


def test_negation_query_targets_counter_evidence():
    q = negation_query(CLAIM)
    assert "water" in q and "boils" in q and "100" in q
    assert "disputed" in q and "contradicting evidence" in q


def test_negation_query_caps_and_dedupes_terms():
    q = negation_query("cats cats cats " + " ".join(f"word{i}" for i in range(20)))
    terms = q.split()
    assert terms.count("cats") == 1
    assert len(terms) <= 8 + 4  # capped content terms + the counter-evidence suffix


def test_dual_perspective_question_carries_both_sides():
    question = dual_perspective_question(CLAIM)
    assert CLAIM in question
    assert negation_query(CLAIM) in question
    assert "do not resolve" in question
