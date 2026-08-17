"""Grounded premise-support checking (v2 plan WS-C.1/C.2, thesis T9).

The 2026 audits' recurring finding: mechanical citation checks overstate
faithfulness by 20-50 points — a premise can pass exact number/quote
containment while its cited span says something else entirely. This module
is the tier between exact-match containment and the judge: does the
evidence actually SUPPORT the premise?

Two-tier local stack, per the plan:

- `LexicalGroundedChecker` — the always-on tier. Model-free and fully
  deterministic (byte-identical replay needs no journaling): stemmed
  content-word coverage of the premise by its cited spans, plus a negation-
  parity check against the best-supporting span. It also reports WHICH
  premise terms no span supports — the LettuceDetect-style span-level
  unsupported-content flag, mapped onto our span-addressed evidence.
- `HHEMChecker` — opt-in escalation to Vectara's HHEM-2.1-Open grounded
  factual-consistency model (110M, Apache). Needs the `nli` extra
  (`uv sync --extra nli`); selected via `RunConfig.nli_checker = "hhem"`.

T9 discipline: verdicts are ADVISORY. They warn the subagent at
record_premises time and land as recorded advisories in verification
stage 2 — they never sole-gate a premise. Gating waits for benchmark
evidence (LLM-AggreFact / Qualifire); exact-match stays the floor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

Verdict = Literal["supported", "uncertain", "unsupported", "contradicted"]

# Coverage thresholds for the lexical tier (fraction of premise content
# terms found in the evidence). Deliberately lenient: this tier exists to
# catch premises whose content the evidence plainly does not carry, not to
# adjudicate paraphrase — that is the escalation tier's job.
SUPPORTED_MIN = 0.7
UNCERTAIN_MIN = 0.4

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from had has have in is it its of on or "
    "that the this to was were will with".split()
)
# Negators are handled by the parity check, never counted as content.
_NEGATORS = frozenset({"not", "no", "never", "none", "cannot", "nor", "without"})


def _stem(word: str) -> str:
    """Light suffix stripping so morphology ("doubled"/"doubling") matches.
    Applied identically to both sides — consistency matters, not linguistics."""
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def content_terms(text: str) -> set[str]:
    return {
        _stem(w)
        for w in _WORD_RE.findall(text.lower())
        if w not in _STOPWORDS
        and w not in _NEGATORS
        and (len(w) > 1 or w.isdigit())  # "Acme's" -> acme + s; drop the noise "s"
    }


def _has_negation(text: str) -> bool:
    words = set(_WORD_RE.findall(text.lower()))
    return bool(words & _NEGATORS) or "n't" in text.lower()


@dataclass(frozen=True)
class SupportVerdict:
    verdict: Verdict
    score: float  # best single-span support score in [0, 1]
    unsupported_terms: tuple[str, ...]  # premise terms found in NO span
    checker: str

    @property
    def flagged(self) -> bool:
        return self.verdict != "supported"

    def describe(self) -> str:
        detail = f"{self.verdict} by {self.checker} (score {self.score:.2f})"
        if self.unsupported_terms:
            detail += f"; unsupported terms: {', '.join(self.unsupported_terms)}"
        return detail


class GroundedChecker(Protocol):
    name: str

    def check(self, premise_text: str, evidence_texts: list[str]) -> SupportVerdict: ...


class LexicalGroundedChecker:
    """Deterministic always-on tier. Verdict from the BEST single evidence
    text (a premise needs one span that carries it, not all of them);
    unsupported_terms from the union (a term any span carries is covered)."""

    name = "lexical-nli-v1"

    def check(self, premise_text: str, evidence_texts: list[str]) -> SupportVerdict:
        terms = content_terms(premise_text)
        if not terms:
            return SupportVerdict("uncertain", 0.0, (), self.name)
        if not evidence_texts:
            return SupportVerdict("unsupported", 0.0, tuple(sorted(terms)), self.name)

        best_score, best_text = 0.0, evidence_texts[0]
        union: set[str] = set()
        for text in evidence_texts:
            ev_terms = content_terms(text)
            union |= ev_terms
            score = len(terms & ev_terms) / len(terms)
            if score > best_score:
                best_score, best_text = score, text
        unsupported = tuple(sorted(terms - union))

        if best_score >= SUPPORTED_MIN:
            if _has_negation(premise_text) != _has_negation(best_text):
                return SupportVerdict("contradicted", best_score, unsupported, self.name)
            return SupportVerdict("supported", best_score, unsupported, self.name)
        if best_score >= UNCERTAIN_MIN:
            return SupportVerdict("uncertain", best_score, unsupported, self.name)
        return SupportVerdict("unsupported", best_score, unsupported, self.name)


class HHEMChecker:
    """Escalation tier: HHEM-2.1-Open grounded consistency scores.

    Deterministic for a pinned model revision on CPU, so replay holds as
    long as the environment is unchanged — same contract as any other
    version-stamped pure function in this codebase. Term-level detail still
    comes from the lexical tier (HHEM emits one score per pair)."""

    name = "hhem-2.1-open"

    def __init__(self, supported_min: float = 0.5, uncertain_min: float = 0.3):
        self.supported_min = supported_min
        self.uncertain_min = uncertain_min
        self._model = None
        self._lexical = LexicalGroundedChecker()

    def _load(self):
        if self._model is None:
            try:
                from transformers import AutoModelForSequenceClassification
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise RuntimeError(
                    "nli_checker='hhem' needs the optional nli dependencies: "
                    "uv sync --extra nli  (or pip install 'parsec[nli]')"
                ) from exc
            self._model = AutoModelForSequenceClassification.from_pretrained(
                "vectara/hallucination_evaluation_model", trust_remote_code=True
            )
        return self._model

    def check(self, premise_text: str, evidence_texts: list[str]) -> SupportVerdict:
        if not evidence_texts:
            return SupportVerdict(
                "unsupported", 0.0,
                tuple(sorted(content_terms(premise_text))), self.name,
            )
        model = self._load()
        scores = model.predict([(ev, premise_text) for ev in evidence_texts])
        best = max(float(s) for s in scores)
        unsupported = self._lexical.check(premise_text, evidence_texts).unsupported_terms
        if best >= self.supported_min:
            return SupportVerdict("supported", best, unsupported, self.name)
        if best >= self.uncertain_min:
            return SupportVerdict("uncertain", best, unsupported, self.name)
        return SupportVerdict("unsupported", best, unsupported, self.name)


def make_grounded_checker(name: str) -> GroundedChecker | None:
    """Config seam (RunConfig.nli_checker). "none" disables the tier."""
    if name == "none":
        return None
    if name == "lexical":
        return LexicalGroundedChecker()
    if name == "hhem":
        return HHEMChecker()
    raise ValueError(f"unknown nli_checker {name!r}; expected lexical | hhem | none")
