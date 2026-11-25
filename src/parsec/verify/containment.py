"""Mechanical containment checks (§6 stage 1, exact-match slice).

String-level validation of a premise against its supporting spans:
numbers must appear exactly (modulo thousands separators) in at least one
span, quoted phrases must appear verbatim. No model, no judgment — these
run constantly and must be ~free (§8). Prose entailment lives one tier up
in parsec.verify.nli (M9, advisory per T9) — this exact-match check stays
the floor. Per the spec, a premise carrying an explicit transformation
note is exempt from the number check, and the note is stored on the
extracts edge for auditability.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from parsec.ids import SPAN_ID_RE

# 100 · 3.14 · 1,600 · 2026
_NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?!\w)")
_QUOTE_RE = re.compile(r"\"([^\"]{3,300})\"|“([^”]{3,300})”")


def _normalize_number(token: str) -> str:
    return token.replace(",", "")


def extract_numbers(text: str) -> list[str]:
    # Span-id offsets ("doc:ab12#140-312") are addresses, not asserted numbers.
    text = SPAN_ID_RE.sub(" ", text)
    return [_normalize_number(m.group(1)) for m in _NUMBER_RE.finditer(text)]


def extract_quotes(text: str) -> list[str]:
    return [" ".join((m.group(1) or m.group(2)).split()) for m in _QUOTE_RE.finditer(text)]


def check_containment(
    premise_text: str,
    span_texts: list[str],
    transform_note: str | None = None,
) -> list[str]:
    """Return a list of violations (empty = passes).

    Every number in the premise must appear in at least one supporting span
    (exact match after separator normalization) unless a transformation note
    is supplied. Every quoted phrase must appear verbatim (whitespace-
    normalized) in at least one span.
    """
    problems: list[str] = []
    span_numbers = {n for t in span_texts for n in extract_numbers(t)}
    normalized_spans = [" ".join(t.split()) for t in span_texts]

    if transform_note is None:
        for num in extract_numbers(premise_text):
            if num not in span_numbers:
                problems.append(
                    f"number {num!r} does not appear in any cited span "
                    "(add a transform_note if it was derived, e.g. unit conversion)"
                )
    for quote in extract_quotes(premise_text):
        if not any(quote in t for t in normalized_spans):
            problems.append(f"quoted phrase {quote!r} not found verbatim in any cited span")
    return problems


def check_premise_verbatim(premise_text: str, span_texts: list[str]) -> list[str]:
    """derivations >= 2: a premise states only what its spans state. The same
    exact-match test as check_containment with no exemption — and wording
    that sends a derived figure where it now belongs."""
    return [
        problem.replace(
            "(add a transform_note if it was derived, e.g. unit conversion)",
            "(a premise states only what its span states; submit a derived figure "
            "as a computes finding in submit_report)",
        )
        for problem in check_containment(premise_text, span_texts, None)
    ]


# -- sentence support (WS0) ---------------------------------------------------
#
# The premise check above is an exact string match and stays exactly as it
# is: recordings replay through it. The sentence check is newer and compares
# numbers by VALUE, so formatting never decides the verdict — "3,260" is
# "3260" and "3.260" is "3.26", while "32.6" is never "3.26".

# The sentence check has its own extractors. The premise extractors above
# are frozen (recordings replay through them) and have three blind spots that
# matter once numbers are compared by value:
#   - "3,260.5" was read as 3260, so "3,260.9" matched "3,260.5";
#   - a sign was dropped, so "-5" matched "5";
#   - quotations under three characters were skipped, so "no" was unchecked.
_SENTENCE_NUMBER_RE = re.compile(
    # A leading minus counts as a sign only where it cannot be a hyphen or a
    # range dash: not after a word character, a digit, a dot or a closing
    # bracket ("COVID-19", "10-20", "sq-2" keep their unsigned numbers).
    r"(?<![\w.])(?:(?<![\w.)\]])([-\u2212])(?=\d))?"
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?!\w)"
)
_SENTENCE_QUOTE_RE = re.compile(r"\"([^\"]{1,300})\"|“([^”]{1,300})”")

# A list marker or heading rule at the start of a line is layout, not a claim
# ("1. First point", "- item", "## Title").
_LEADING_MARKUP_RE = re.compile(r"^\s*(?:#{1,6}\s+|[-*+]\s+|\d{1,3}[.)]\s+)+")
# Subquestion ids ("sq-2", "sq-gap-1") are addresses, like span ids.
_SUBQUESTION_ID_RE = re.compile(r"\bsq-(?:gap-|cov-)?\d+\b")


def canonical_number(token: str) -> str:
    """A number token as its value: separators and trailing zeros removed,
    a unicode minus read as a minus."""
    plain = token.replace(",", "").replace("\u2212", "-")
    try:
        value = Decimal(plain)
    except InvalidOperation:
        return plain
    if value == 0:
        return "0"  # no "-0"
    return format(value.normalize(), "f")


def _number_tokens(text: str) -> list[str]:
    """Number tokens as written (sign included), span ids removed."""
    text = SPAN_ID_RE.sub(" ", text)
    return [(m.group(1) or "") + m.group(2) for m in _SENTENCE_NUMBER_RE.finditer(text)]


def _asserted_number_tokens(sentence: str) -> list[str]:
    """Tokens a reader would take the sentence to be asserting: layout
    markers and subquestion ids are not among them."""
    return _number_tokens(_SUBQUESTION_ID_RE.sub(" ", _LEADING_MARKUP_RE.sub("", sentence)))


def sentence_quotes(text: str) -> list[str]:
    """Quoted phrases of any length, whitespace-normalized."""
    quotes = (" ".join((m.group(1) or m.group(2)).split()) for m in _SENTENCE_QUOTE_RE.finditer(text))
    return [q for q in quotes if q]


def sentence_numbers(text: str) -> list[str]:
    """The numbers a sentence asserts, by value, in order of appearance."""
    return [canonical_number(n) for n in _asserted_number_tokens(text)]


def unsupported_numbers(text: str, evidence_texts: list[str]) -> list[str]:
    """Numbers in `text` (as written) whose value appears in no evidence text."""
    supported = {canonical_number(n) for t in evidence_texts for n in _number_tokens(t)}
    return [n for n in _asserted_number_tokens(text) if canonical_number(n) not in supported]


def unsupported_quotes(text: str, evidence_texts: list[str]) -> list[str]:
    """Quoted phrases in `text` that appear verbatim (whitespace-normalized)
    in no evidence text."""
    normalized = [" ".join(t.split()) for t in evidence_texts]
    return [q for q in sentence_quotes(text) if not any(q in t for t in normalized)]


def check_sentence(sentence: str, cited_texts: list[str]) -> list[str]:
    """Problems with one report sentence against the texts of what it cites
    (empty = supported). This is the one check behind the writer's citation
    check, `parsec verify` and eval grading, so the three cannot disagree.

    It catches wrong numbers, numeric dates and misquotes. It does not catch
    a wrong direction ("rose" for "fell") or a swapped entity."""
    problems = [
        f"number {num!r} does not appear in the cited evidence"
        for num in unsupported_numbers(sentence, cited_texts)
    ]
    problems += [
        f"quoted phrase {quote!r} is not found verbatim in the cited evidence"
        for quote in unsupported_quotes(sentence, cited_texts)
    ]
    return problems


def has_checkable_content(sentence: str) -> bool:
    """True when a sentence carries a number or a quotation — content that
    needs a citation to be checked against."""
    return bool(sentence_numbers(sentence) or sentence_quotes(sentence))
