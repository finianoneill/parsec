"""Claimify-style premise lints (M9, WS-C.3): the ambiguity-refusal gate."""

from parsec.verify.lints import lint_premise


def test_vague_generic_referent_is_refused_with_reason():
    """The v1 brief's canonical rot case (§10.1)."""
    reasons = lint_premise("The study showed benefits.")
    assert any("ambiguous referent" in r for r in reasons)
    assert any("vague term" in r for r in reasons)


def test_pronoun_subject_is_refused():
    reasons = lint_premise("It doubled last year.")
    assert any("ambiguous subject" in r for r in reasons)


def test_generic_referent_with_named_entity_passes():
    assert lint_premise("The study by Smith et al. found a 12% reduction.") == []


def test_vague_term_with_quantity_passes():
    # "about 70" carries a number; the premise is checkable
    assert lint_premise("On the summit of Mount Everest water boils at about 70 degrees Celsius.") == []


def test_vague_term_with_verbatim_quote_passes():
    assert lint_premise('The trial registry entry reports "no significant difference".') == []


def test_multi_sentence_premise_is_refused():
    reasons = lint_premise("Water boils at 100 degrees. Ice melts at 0 degrees.")
    assert any("multiple sentences" in r for r in reasons)


def test_abbreviations_are_not_sentence_boundaries():
    assert lint_premise("The U.S. Government budgeted 5 billion dollars in 2024.") == []


def test_specific_premise_passes():
    assert lint_premise("Water boils at 100 degrees Celsius at sea level.") == []
