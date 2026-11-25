"""When are two premise texts the same proposition?

Intake asks this for every new premise: a near-duplicate of a recorded
premise is not recorded again — its spans attach to the existing one, so
corroboration accrues to a single node instead of fragmenting across
paraphrases. Getting it wrong in the merging direction is the dangerous
error: two different facts become one premise, and one of them inherits the
other's evidence.

Two matchers, selected by the session's `claim_credence` behaviour version:

  level 1 (legacy) — token-SET overlap plus identical numbers. Blind to
      word order: "Alder acquired Birch in the merger" and "Birch acquired
      Alder in the merger" share every token and were merged, though they
      say opposite things.
  level 2 — the same, AND the token SEQUENCES must be close (normalized
      edit distance). A swapped subject and object, or a moved clause, now
      reads as a different sentence.

Level 2 errs toward splitting: a true paraphrase with its clauses reordered
is kept as a separate premise. That under-counts corroboration and so
under-credits the claim, which flags it for gap-fill — the safe direction.
"""

from __future__ import annotations

import re

from parsec.verify.containment import extract_numbers

_DUP_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Token-set Jaccard: paraphrases of one sentence clear 0.8; a genuinely
# different subject does not.
DUP_JACCARD = 0.8
# Ordered similarity (1 - token edit distance / longer length). One changed
# word in a ten-word sentence scores 0.9; a subject/object swap in a short
# sentence scores about 0.67.
DUP_ORDERED = 0.8


def dup_token_list(text: str) -> list[str]:
    return _DUP_TOKEN_RE.findall(text.lower())


def dup_tokens(text: str) -> frozenset[str]:
    return frozenset(dup_token_list(text))


def ordered_similarity(a: list[str], b: list[str]) -> float:
    """1.0 for identical token sequences, falling with the number of token
    insertions, deletions and substitutions needed to turn one into the other."""
    if not a and not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,                        # delete
                    current[j - 1] + 1,                     # insert
                    previous[j - 1] + (token_a != token_b), # substitute
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b))


def near_duplicate(
    text_a: str, tokens_a: frozenset[str], entry_b: dict, order_aware: bool = False
) -> bool:
    """Is `text_a` the same proposition as the recorded premise `entry_b`
    ({"text", "tokens", "numbers"})? Quantities must match EXACTLY either
    way: "90 degrees" rephrasing "100 degrees" is a conflict to record,
    never a duplicate."""
    a, b = tokens_a, entry_b["tokens"]
    if not a or not b:
        return False
    if len(a & b) / len(a | b) < DUP_JACCARD:
        return False
    if set(extract_numbers(text_a)) != entry_b["numbers"]:
        return False
    if order_aware:
        return (
            ordered_similarity(dup_token_list(text_a), dup_token_list(entry_b["text"]))
            >= DUP_ORDERED
        )
    return True
