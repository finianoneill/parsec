"""Dates for recency (WS2): when an observation was made, and what time a
question is asking about.

Three different times were being conflated in the fetch timestamp:

  publication time  when a document says it was published (document meta)
  observation time  when a FACT held, per premise per span — a paragraph
                    can compare 2019 with 2026, so this cannot live on the
                    span; it lives on the premise's edge to that span
  target time       the time the subquestion asks about: "now", or a
                    period it names ("in 2019")

Recency decay is then the gap between an observation and the target — not
"how long ago did we download this". A 2019 figure is strong evidence for
"what was it in 2019?" and weak evidence for "what is it now?".

Everything here is pure and clock-free. "Now" is never read from a clock:
it is the session's recorded `as_of_ts` (T4).
"""

from __future__ import annotations

import calendar
import re
from datetime import date

from parsec.verify.temporal import extract_date_intervals

Interval = tuple[date, date]

NOW = "now"
DATE_SOURCE_AS_OF = "as_of"          # stated by the premise, found in its span
DATE_SOURCE_PUBLISHED = "published"  # the document's publication date
DATE_SOURCE_UNKNOWN = "unknown"

_QUARTER_RE = re.compile(r"\bQ([1-4])\s+(1[5-9]\d{2}|20\d{2})\b", re.IGNORECASE)
_STORED_RE = re.compile(r"(\d{4}-\d{2}-\d{2})/(\d{4}-\d{2}-\d{2})")


def _real_date(year: int, month: int, day: int) -> date:
    """temporal.py marks "end of month" as day 31; clamp to the calendar."""
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def text_intervals(text: str) -> list[Interval]:
    """Every date expression in the text as a [first day, last day] interval:
    ISO dates, "March 2025", "5 March 2025", "Q3 2025", bare years."""
    intervals: list[Interval] = []

    def quarter(m: re.Match) -> str:
        q, year = int(m.group(1)), int(m.group(2))
        intervals.append((date(year, 3 * q - 2, 1), _real_date(year, 3 * q, 31)))
        return " " * len(m.group(0))  # consumed: its year is not a second date

    remaining = _QUARTER_RE.sub(quarter, text)
    for lo, hi in extract_date_intervals(remaining):
        try:
            # A day-level date must be a real one ("2024-02-30" is not);
            # only a month's or year's END is clamped to the calendar.
            first = date(*lo)
            last = first if lo == hi else _real_date(*hi)
        except ValueError:
            continue
        intervals.append((first, last))
    return intervals


def format_interval(interval: Interval) -> str:
    return f"{interval[0].isoformat()}/{interval[1].isoformat()}"


def parse_interval(stored: object) -> Interval | None:
    """Inverse of format_interval (the form stored on edges and premises).
    Anything that is not a well-formed interval string — including a stored
    value of the wrong JSON type — is None, never an exception: a damaged
    edge must be reportable, not a crash."""
    if not isinstance(stored, str):
        return None
    m = _STORED_RE.fullmatch(stored.strip())
    if not m:
        return None
    try:
        lo, hi = date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))
    except ValueError:
        return None
    return (lo, hi) if lo <= hi else None


# The explicit interval SHAPE, whether or not its dates are real.
_INTERVAL_SHAPE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\s*/\s*\d{4}-\d{2}-\d{2}")


def parse_period(text: object, allow_interval: bool = True) -> Interval | None:
    """One date expression as an interval, or None if the text does not hold
    exactly one.

    Text written as an explicit "from/to" interval is parsed strictly: a bad
    endpoint ("2024-02-30/2024-03-01") is None, not whichever half happens
    to be a real date. allow_interval=False refuses that form altogether —
    for dates a model supplies, where an interval is a way to claim an
    arbitrarily wide span ("1900-01-01/2099-12-31") that any date satisfies."""
    if not isinstance(text, str) or not text.strip():
        return None
    if _INTERVAL_SHAPE_RE.fullmatch(text.strip()):
        return parse_interval(text.strip().replace(" ", "")) if allow_interval else None
    found = text_intervals(text)
    return found[0] if len(found) == 1 else None


def normalize_target(text: str | None) -> str | None:
    """A subquestion's target time in stored form: "now", or an interval.
    None when the text is neither (the caller reports it)."""
    if text is None or text.strip().lower() in ("", NOW, "present", "today", "current"):
        return NOW
    period = parse_period(text)
    return format_interval(period) if period is not None else None


def check_date_containment(
    as_of: object, span_text: str, allow_interval: bool = False
) -> tuple[Interval | None, str | None]:
    """(interval, problem). An `as_of` is accepted only if the span states
    that date at least as precisely: "2019" is supported by a span saying
    "March 2019"; "March 2019" is not supported by a span saying only "2019".
    Same floor as numbers and quotes: the model may not date a fact with a
    date the source does not give.

    A date from a model is one calendar expression — a year, a month, a
    quarter or a day. allow_interval=True is for re-checking a date the
    harness itself stored, which is in interval form."""
    period = parse_period(as_of, allow_interval=allow_interval)
    if period is None:
        return None, (
            f"as_of {as_of!r} is not a date the harness can read; use a year (2019), a month "
            "and year (March 2025), a quarter (Q3 2025) or a full date (2025-03-01)"
        )
    lo, hi = period
    if any(lo <= s_lo and s_hi <= hi for s_lo, s_hi in text_intervals(span_text)):
        return period, None
    return None, (
        f"as_of {as_of!r} does not appear in the cited span; give the date as the span "
        "states it, or leave as_of out"
    )


def published_interval(published: str | None) -> Interval | None:
    """A document's publication date ("2019-06-01...") as a one-day interval."""
    try:
        day = date.fromisoformat((published or "")[:10])
    except ValueError:
        return None
    return (day, day)


def target_interval(target_time: str | None, as_of_ts: str | None) -> Interval | None:
    """The time a premise is evidence FOR: the session's recorded "now", or
    the period its subquestion named."""
    if target_time and target_time != NOW:
        return parse_interval(target_time)
    return published_interval(as_of_ts)


def gap_days(observed: Interval, target: Interval) -> int:
    """Days between an observation and the target time; 0 when they overlap.
    Symmetric: a 2026 reading is as far from "in 2019" as the reverse."""
    if observed[1] < target[0]:
        return (target[0] - observed[1]).days
    if target[1] < observed[0]:
        return (observed[0] - target[1]).days
    return 0


def describe_interval(interval: Interval) -> str:
    """An interval the way a person would say it: "2019", a single day, or
    "from to"."""
    lo, hi = interval
    if lo == hi:
        return lo.isoformat()
    if (lo.month, lo.day, hi.month, hi.day) == (1, 1, 12, 31) and lo.year == hi.year:
        return str(lo.year)
    return f"{lo.isoformat()} to {hi.isoformat()}"


def stated_dates(edge_payloads: list[dict]) -> list[Interval]:
    """The observation dates a premise GAVE and its spans state (as_of), in
    order, without repeats. Publication dates are excluded: they date the
    document, and nothing checked them against the text."""
    seen: list[Interval] = []
    for payload in edge_payloads:
        if payload.get("date_source") != DATE_SOURCE_AS_OF:
            continue
        interval = parse_interval(payload.get("observed_at"))
        if interval is not None and interval not in seen:
            seen.append(interval)
    return sorted(seen)


def stated_years_text(edge_payloads: list[dict]) -> str:
    """The years of a premise's stated dates, as text the sentence check can
    read as evidence: a sentence may say "in 2019" of a premise whose span
    was checked to say so, even when the premise's own wording omits it."""
    years = sorted({str(d.year) for iv in stated_dates(edge_payloads) for d in iv})
    return " ".join(years)
