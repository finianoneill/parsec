"""Grounded premise-support tier (M9, WS-C.1/C.2): the lexical checker's
verdicts, its span-level unsupported-term flags, and the config factory."""

import pytest

from parsec.verify.nli import (
    HHEMChecker,
    LexicalGroundedChecker,
    make_grounded_checker,
)

SPAN = (
    "Water boils at 100 degrees Celsius at standard atmospheric pressure, the "
    "reference value used to calibrate thermometers worldwide."
)
REPORT_SPAN = (
    "The 2019 annual report describes steady quarterly revenue growth at Acme Corporation."
)


def test_restated_premise_is_supported():
    v = LexicalGroundedChecker().check("Water boils at 100 degrees Celsius.", [SPAN])
    assert v.verdict == "supported"
    assert v.score == 1.0
    assert v.unsupported_terms == ()
    assert not v.flagged


def test_paraphrased_but_unsupported_is_caught_with_term_flags():
    """The M9 exit scenario: no numbers, no quotes — exact-match containment
    has nothing to bite on — but the evidence plainly does not say this."""
    v = LexicalGroundedChecker().check("Acme's profits doubled.", [REPORT_SPAN])
    assert v.verdict == "unsupported"
    assert v.flagged
    # span-level detail: exactly the overreaching terms, stemmed
    assert "profit" in v.unsupported_terms and "doubl" in v.unsupported_terms
    assert "acme" not in v.unsupported_terms


def test_partial_overlap_is_uncertain():
    v = LexicalGroundedChecker().check(
        "Acme Corporation's profits doubled in 2019.", [REPORT_SPAN]
    )
    assert v.verdict == "uncertain"
    assert v.flagged


def test_negation_mismatch_is_contradicted():
    v = LexicalGroundedChecker().check(
        "The vaccine is not effective.", ["Trials showed the vaccine is effective."]
    )
    assert v.verdict == "contradicted"


def test_best_single_span_decides_but_terms_use_the_union():
    v = LexicalGroundedChecker().check(
        "Water boils at 100 degrees Celsius.",
        ["Completely unrelated text about turtles.", SPAN],
    )
    assert v.verdict == "supported"
    assert v.unsupported_terms == ()


def test_no_evidence_is_unsupported():
    v = LexicalGroundedChecker().check("Anything at all.", [])
    assert v.verdict == "unsupported" and v.score == 0.0


def test_no_content_terms_is_uncertain():
    assert LexicalGroundedChecker().check("It is that.", [SPAN]).verdict == "uncertain"


def test_verdicts_are_deterministic():
    a = LexicalGroundedChecker().check("Acme's profits doubled.", [REPORT_SPAN])
    b = LexicalGroundedChecker().check("Acme's profits doubled.", [REPORT_SPAN])
    assert a == b


def test_factory():
    assert isinstance(make_grounded_checker("lexical"), LexicalGroundedChecker)
    assert isinstance(make_grounded_checker("hhem"), HHEMChecker)
    assert make_grounded_checker("none") is None
    with pytest.raises(ValueError):
        make_grounded_checker("bogus")


def test_hhem_without_extra_raises_actionable_error():
    checker = HHEMChecker()
    try:
        import transformers  # noqa: F401

        pytest.skip("nli extra installed; the missing-dependency path is untestable here")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match="nli"):
        checker.check("A claim.", ["Some evidence."])
