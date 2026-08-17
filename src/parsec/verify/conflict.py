"""Dual-perspective retrieval for conflict discovery (v2 plan WS-C.5).

The cheap mechanical trick from the literature: do not hope conflicts
surface by accident — when a claim needs corroborating or refuting, also
search for its NEGATION. Hits feed `contradicts` edges (via the subagent's
conflicts contract) instead of being silently absent.

Used by the gap-fill loop: the one targeted subagent dispatched at a weak
premise is instructed to search both perspectives and to record conflicting
evidence as premises + conflict reports, never to resolve the disagreement
itself (§4).
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from had has have in is it its of on or "
    "that the this to was were will with".split()
)
_MAX_TERMS = 8


def negation_query(claim_text: str) -> str:
    """A search query targeting counter-evidence for the claim: its content
    terms (mention order, deduped) plus counter-evidence keywords."""
    seen: list[str] = []
    for word in _WORD_RE.findall(claim_text):
        lower = word.lower()
        if lower in _STOPWORDS or lower in seen:
            continue
        seen.append(lower)
        if len(seen) >= _MAX_TERMS:
            break
    return " ".join(seen + ["wrong", "disputed", "contradicting evidence"])


def dual_perspective_question(claim_text: str) -> str:
    """The gap-fill subquestion: both perspectives, conflicts reported upward."""
    return (
        f'Find independent sources that confirm or refute: "{claim_text}". '
        f'Search BOTH perspectives: the claim as stated, and its negation '
        f'(for example: "{negation_query(claim_text)}"). If you find evidence '
        "against the claim, record it as premises and report the conflicts — "
        "do not resolve the disagreement yourself."
    )
