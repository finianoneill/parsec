"""Mechanical containment checks (§6 stage 1, exact-match slice).

String-level validation of a premise against its supporting spans:
numbers must appear exactly (modulo thousands separators) in at least one
span, quoted phrases must appear verbatim. No model, no judgment — these
run constantly and must be ~free (§8). Prose entailment (NLI-lite) is a
later milestone; per the spec, a premise carrying an explicit
transformation note is exempt from the number check, and the note is
stored on the extracts edge for auditability.
"""

from __future__ import annotations

import re

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
