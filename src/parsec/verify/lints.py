"""Claimify-style premise quality lints (v2 plan WS-C.3; §10.1 of the v1
brief — the anti-"DAG theater" defense).

The failure mode: extraction produces vague premises ("the study showed
benefits") that trivially pass containment while laundering meaning. The
fix is an explicit ambiguity-refusal gate — a premise that cannot be
decontextualized is REJECTED at record time with the reason, not recorded
vaguely.

Lints (all mechanical, no model):

- decontextualization — does the claim carry its referents? A premise whose
  subject is a bare pronoun, or that leans on a generic definite phrase
  ("the study", "the company") while naming no specific entity, cannot be
  verified in isolation.
- vagueness — degree words ("benefits", "significant", "several") with no
  number and no verbatim quote anywhere in the premise carry no checkable
  content; the subagent must state what the span actually says.
- granularity — one statement per premise ("molecular, not atomic"): a
  multi-sentence premise smuggles several claims behind one containment
  check.

`record_premises` applies these as a hard gate on NEW premises;
verification re-lints recorded premises as advisories only, so sessions
recorded before this milestone stay verifiable.
"""

from __future__ import annotations

import re

_PRONOUN_SUBJECT_RE = re.compile(
    r"^(it|they|he|she|this|that|these|those|such)\b", re.IGNORECASE
)
_GENERIC_REFERENT_RE = re.compile(
    r"\b(?:the|this|that|these|those)\s+"
    r"(stud(?:y|ies)|report|paper|article|survey|analysis|trial|experiment|"
    r"research(?:ers)?|authors?|scientists?|compan(?:y|ies)|organization|"
    r"agency|government|team|group)\b",
    re.IGNORECASE,
)
# Degree/quantity words that carry no checkable content on their own.
_VAGUE_TERMS = frozenset(
    "benefit benefits benefited beneficial improved improves improvement "
    "improvements significant significantly substantial substantially "
    "many several some numerous various recently soon".split()
)
_WORD_RE = re.compile(r"[a-z]+", re.IGNORECASE)
# Sentence boundary: terminator + whitespace + a capital, excluding
# single-letter abbreviations ("U.S. Government" must not match).
_MULTI_SENTENCE_RE = re.compile(r"(?<![A-Z])[.!?]\s+[A-Z0-9]")


def _names_specific_entity(text: str) -> bool:
    """A capitalized token past the first word, an all-caps token, or a
    quoted title counts as a carried referent."""
    if '"' in text or "“" in text:
        return True
    tokens = text.split()
    for token in tokens[1:]:
        stripped = token.strip("\"'().,;:!?“”")
        if len(stripped) > 1 and stripped[0].isupper():
            return True
    return False


def lint_premise(text: str) -> list[str]:
    """Return refusal reasons (empty = passes). Reasons are written for the
    model that must fix and re-record the premise."""
    reasons: list[str] = []
    stripped = text.strip()

    if _MULTI_SENTENCE_RE.search(stripped):
        reasons.append(
            "premise contains multiple sentences — record one atomic statement per premise"
        )

    if _PRONOUN_SUBJECT_RE.match(stripped):
        reasons.append(
            f"ambiguous subject {stripped.split()[0]!r}: a premise must carry its referent — "
            "name the specific entity so the statement stands alone"
        )
    else:
        m = _GENERIC_REFERENT_RE.search(stripped)
        if m and not _names_specific_entity(stripped):
            reasons.append(
                f"ambiguous referent {m.group(0)!r} with no named entity: say WHICH "
                f"{m.group(1).lower()} so the premise can be verified in isolation"
            )

    has_number = any(ch.isdigit() for ch in stripped)
    has_quote = '"' in stripped or "“" in stripped
    if not has_number and not has_quote:
        words = {w.lower() for w in _WORD_RE.findall(stripped)}
        vague = sorted(words & _VAGUE_TERMS)
        if vague:
            reasons.append(
                f"vague term(s) {', '.join(repr(v) for v in vague)} with no quantity or "
                "verbatim quote: state the specific figure or quote what the span actually says"
            )
    return reasons
