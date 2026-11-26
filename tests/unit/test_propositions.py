"""Intake's "is this the same proposition?" test, at each claim_credence level."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from parsec.verify.containment import extract_numbers
from parsec.verify.propositions import (
    FUNCTION_WORDS,
    content_tokens,
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


# -- level 3: identical content words -----------------------------------------


def _same3(a: str, b: str) -> bool:
    return near_duplicate(a, dup_tokens(a), _entry(b), level=3)


def test_level_defaults_to_what_order_aware_selects():
    """Callers written before `level` existed keep their behaviour."""
    for a, b in [(ALDER, BIRCH), (ALDER, ALDER)]:
        assert near_duplicate(a, dup_tokens(a), _entry(b)) == near_duplicate(
            a, dup_tokens(a), _entry(b), level=1
        )
        assert near_duplicate(a, dup_tokens(a), _entry(b), order_aware=True) == near_duplicate(
            a, dup_tokens(a), _entry(b), level=2
        )


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("The vaccine is safe for children under five",
         "The vaccine is not safe for children under five"),
        ("The vaccine is safe", "The vaccine is not safe"),
        ("Revenue at the company rose 12 percent in fiscal 2025 compared with 2024",
         "Revenue at the company fell 12 percent in fiscal 2025 compared with 2024"),
        ("The drug was approved by the FDA in March 2024 for adults with severe asthma",
         "The drug was approved by the EMA in March 2024 for adults with severe asthma"),
        ("The study found the treatment was effective in patients over 65 years old",
         "The study found the treatment was ineffective in patients over 65 years old"),
        ("Alder acquired Birch in the 2019 merger that was approved by regulators in Europe",
         "Birch acquired Alder in the 2019 merger that was approved by regulators in Europe"),
        ("Inflation will stay above the target through the end of the year 2025",
         "Inflation will stay below the target through the end of the year 2025"),
        ("The bank said that inflation may fall before the end of the year 2025",
         "The bank said that inflation will fall before the end of the year 2025"),
        ("The board approved the merger and the dividend for shareholders this year",
         "The board approved the merger or the dividend for shareholders this year"),
    ],
)
def test_level_3_never_merges_texts_that_differ_in_a_content_word(a, b):
    assert _same(a, b, order_aware=True)  # the level-2 matcher merges every one of these
    assert not _same3(a, b) and not _same3(b, a)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Water boils at 100 degrees Celsius at sea level.",
         "water boils at 100 degrees celsius, at sea level"),
        ("The solar system has eight planets.", "Solar System has eight planets"),
        ("The bridge is green.", "A bridge is green."),
        ("The bridge", "bridge"),  # an overlap pre-filter would split this one
    ],
)
def test_level_3_merges_case_punctuation_and_article_variants(a, b):
    assert _same3(a, b) and _same3(b, a)


def test_level_3_still_requires_identical_numbers():
    assert not _same3("Water boils at 100 degrees.", "Water boils at 90 degrees.")
    # the same digits grouped differently are different tokens: split, the safe error
    assert not _same3("The perigee is 356,400 km.", "The perigee is 356400 km.")


def test_level_3_splits_a_true_paraphrase_which_is_the_safe_error():
    a = "The bridge across the river opened to traffic in 1932 after six years of work."
    b = "The bridge across the river opened to traffic in 1932 after six years of building work."
    assert _same(a, b, order_aware=True)
    assert not _same3(a, b)


def test_a_text_of_only_articles_matches_nothing():
    assert not _same3("the", "a")


def test_negations_are_content_words():
    negations = {"not", "no", "never", "none", "nor", "neither", "without", "cannot", "t"}
    assert not negations & FUNCTION_WORDS
    assert content_tokens("The vaccine isn't safe") == ["vaccine", "isn", "t", "safe"]
    assert not _same3("The vaccine is safe", "The vaccine isn't safe")


_content_words = st.text(alphabet="bcdfghjklmnpqrstvwxyz", min_size=2, max_size=8).filter(
    lambda word: word not in FUNCTION_WORDS
)
_sentences = st.lists(_content_words, min_size=1, max_size=12)


@given(words=_sentences, extra=_content_words, data=st.data())
def test_level_3_no_edit_to_a_content_word_ever_matches(words, extra, data):
    """Insert, substitute, delete or swap any content word: never the same
    proposition, in either direction, at any sentence length."""
    original = " ".join(words)
    position = data.draw(st.integers(min_value=0, max_value=len(words) - 1))
    edits = [[*words[:position], extra, *words[position:]]]  # insert (a negation is one)
    if extra != words[position]:
        edits.append([*words[:position], extra, *words[position + 1:]])  # substitute
    if len(words) > 1:
        edits.append(words[:position] + words[position + 1:])  # delete
        other = data.draw(st.integers(min_value=0, max_value=len(words) - 1))
        if words[other] != words[position]:
            swapped = list(words)
            swapped[position], swapped[other] = swapped[other], swapped[position]
            edits.append(swapped)
    for edit in edits:
        changed = " ".join(edit)
        assert not _same3(original, changed) and not _same3(changed, original)


@given(words=_sentences, data=st.data())
def test_level_3_articles_and_case_never_split(words, data):
    original = " ".join(words)
    position = data.draw(st.integers(min_value=0, max_value=len(words)))
    article = data.draw(st.sampled_from(sorted(FUNCTION_WORDS)))
    reworded = " ".join([*words[:position], article, *words[position:]]).upper() + "."
    assert _same3(original, reworded) and _same3(reworded, original)
