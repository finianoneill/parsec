"""Intake's "is this the same proposition?" test, legacy and order-aware."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from parsec.verify.containment import extract_numbers
from parsec.verify.propositions import (
    dup_token_list,
    dup_tokens,
    near_duplicate,
    ordered_similarity,
)


def _entry(text: str) -> dict:
    return {"text": text, "tokens": dup_tokens(text), "numbers": set(extract_numbers(text))}


def _same(a: str, b: str, order_aware: bool) -> bool:
    return near_duplicate(a, dup_tokens(a), _entry(b), order_aware)


ALDER = "Alder acquired Birch in the merger."
BIRCH = "Birch acquired Alder in the merger."


def test_reversed_subject_and_object_are_different_propositions():
    assert not _same(ALDER, BIRCH, order_aware=True)
    assert not _same(BIRCH, ALDER, order_aware=True)


def test_the_legacy_matcher_still_merges_them_for_old_recordings():
    assert _same(ALDER, BIRCH, order_aware=False)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Water boils at 100 degrees Celsius at sea level.",
         "Water boils at 100 degrees Celsius at sea level."),
        ("Water boils at 100 degrees Celsius at sea level.",
         "water boils at 100 degrees celsius, at sea level"),
        ("The bridge across the river opened to traffic in 1932 after six years of work.",
         "The bridge across the river opened to traffic in 1932 after six years of building work."),
    ],
)
def test_true_restatements_still_merge(a, b):
    assert _same(a, b, order_aware=True) and _same(a, b, order_aware=False)


def test_a_different_quantity_is_never_a_duplicate():
    a = "Water boils at 100 degrees Celsius at sea level."
    b = "Water boils at 90 degrees Celsius at sea level."
    assert not _same(a, b, order_aware=True) and not _same(a, b, order_aware=False)


def test_a_moved_clause_is_split_which_is_the_safe_error():
    """Order-aware matching cannot tell a moved clause from a swapped
    argument, so it keeps both as separate premises. That under-counts
    corroboration (and flags the claim for gap-fill); it never merges two
    different facts."""
    a = "Water boils at 100 degrees Celsius at sea level."
    b = "At sea level water boils at 100 degrees Celsius."
    assert _same(a, b, order_aware=False)
    assert not _same(a, b, order_aware=True)


def test_ordered_similarity_values():
    tokens = dup_token_list
    assert ordered_similarity(tokens(ALDER), tokens(ALDER)) == 1.0
    assert ordered_similarity(tokens(ALDER), tokens(BIRCH)) == pytest.approx(4 / 6)
    assert ordered_similarity([], []) == 1.0
    assert ordered_similarity(["a"], []) == 0.0


words = st.lists(st.sampled_from(["alder", "birch", "acquired", "in", "the", "merger", "x"]), max_size=8)


@given(words, words)
def test_ordered_similarity_is_symmetric_and_bounded(a, b):
    score = ordered_similarity(a, b)
    assert 0.0 <= score <= 1.0
    assert score == ordered_similarity(b, a)
    assert (score == 1.0) == (a == b)
