"""When are two premise texts the same proposition?

Intake asks this for every new premise: a near-duplicate of a recorded
premise is not recorded again — its spans attach to the existing one, so
corroboration accrues to a single node instead of fragmenting across
paraphrases. Getting it wrong in the merging direction is the dangerous
error: two different facts become one premise, and one of them inherits the
other's evidence.

Three matchers, selected by the session's `claim_credence` behaviour version:

  level 1 (legacy) — token-SET overlap plus identical numbers. Blind to
      word order: "Alder acquired Birch in the merger" and "Birch acquired
      Alder in the merger" share every token and were merged, though they
      say opposite things.
  level 2 — the same, AND the token SEQUENCES must be close (normalized
      edit distance). This separates a swapped subject and object in a
      short sentence, and nothing else: both tests are similarity
      thresholds, so one changed word in a sentence long enough still
      clears them. "is safe" merged with "is not safe", "rose 12 percent"
      with "fell 12 percent", "approved by the FDA" with "approved by the
      EMA", and the Alder/Birch swap again once the sentence ran past a
      dozen words.
  level 3 — no threshold. Two texts are the same proposition only if they
      state the same numbers and their CONTENT words are identical, in the
      same order. The only words that are not content are the articles
      (FUNCTION_WORDS below). Case and punctuation never counted.

Level 3 errs toward splitting, by design: a true paraphrase ("opened in
1932" / "was opened in 1932") is kept as a separate premise. That
under-counts corroboration and so under-credits the claim, which flags it
for gap-fill — the safe direction. Words are not stemmed for the same
reason: "approve" and "approved" differ in tense, and tense can be the
whole difference between two facts.
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

# Level 3: the words whose presence or absence changes nothing a premise
# says. Articles only, and that is deliberate. Every other "function word"
# can be the whole difference between two facts: a negation (not, no,
# never, without, and the "t" that tokenizing "isn't" leaves behind), a
# preposition (above/below, before/after, to/from), a conjunction (and/or),
# an auxiliary or modal (is/was, may/will), a quantifier (all/some), a
# pronoun (he/she). Widen this set only with evidence that a specific word
# never distinguishes two facts — the cost of leaving a word out is an
# under-credited claim; the cost of putting a wrong one in is a contradiction
# recorded as corroboration.
FUNCTION_WORDS = frozenset({"a", "an", "the"})


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


def content_tokens(text: str) -> list[str]:
    """The words that carry what `text` says, in order: every token except
    the articles."""
    return [token for token in dup_token_list(text) if token not in FUNCTION_WORDS]


def near_duplicate(
    text_a: str,
    tokens_a: frozenset[str],
    entry_b: dict,
    order_aware: bool = False,
    level: int | None = None,
) -> bool:
    """Is `text_a` the same proposition as the recorded premise `entry_b`
    ({"text", "tokens", "numbers"})? Quantities must match EXACTLY at every
    level: "90 degrees" rephrasing "100 degrees" is a conflict to record,
    never a duplicate.

    `level` is the session's claim_credence level (see the module
    docstring). Without it, `order_aware` selects level 2 over level 1."""
    if level is None:
        level = 2 if order_aware else 1
    a, b = tokens_a, entry_b["tokens"]
    if not a or not b:
        return False
    if level >= 3:
        # No overlap pre-filter here: Jaccard counts the articles, so it
        # would split "The bridge" from "bridge" before the content words
        # were ever compared.
        content = content_tokens(text_a)
        return (
            bool(content)
            and content == content_tokens(entry_b["text"])
            and set(extract_numbers(text_a)) == entry_b["numbers"]
        )
    if len(a & b) / len(a | b) < DUP_JACCARD:
        return False
    if set(extract_numbers(text_a)) != entry_b["numbers"]:
        return False
    if level >= 2:
        return (
            ordered_similarity(dup_token_list(text_a), dup_token_list(entry_b["text"]))
            >= DUP_ORDERED
        )
    return True
