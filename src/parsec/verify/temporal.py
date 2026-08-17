"""Mechanical temporal validator (v2 plan WS-C.4; §2.1's deferred
`temporal` edge check, now landed).

ChronoFact-lite, no model and no clock: normalize time expressions found in
premise texts (falling back to their cited spans) into [earliest, latest]
date intervals, then check ordering findings by interval constraint. Only
findings that are mechanically decidable are judged; everything else is
surfaced as an advisory rather than guessed at.

Convention (documented in the subagent contract): a temporal ordering
finding cites its two premises in the order the events are mentioned in the
finding text — "A before B" means premise_ids[0] is claimed to precede
premise_ids[1]. The check is conservative Allen-style: "A before B" is
violated only when A's EARLIEST possible date is after B's LATEST possible
date (definitely out of order), so granularity mismatches ("2020" vs
"2020-03-05") never false-positive.
"""

from __future__ import annotations

import re

# (year, month, day) — granularity encoded by interval endpoints.
DateTuple = tuple[int, int, int]
Interval = tuple[DateTuple, DateTuple]

_MONTHS = {
    name: i + 1
    for i, name in enumerate(
        "january february march april may june july august september october november december".split()
    )
}
_MONTH_PATTERN = "|".join(_MONTHS)
_YEAR = r"(1[5-9]\d{2}|20\d{2})"

_ISO_RE = re.compile(rf"\b{_YEAR}-(\d{{2}})(?:-(\d{{2}}))?\b")
# "March 5, 2020" / "March 2020"
_MDY_RE = re.compile(rf"\b({_MONTH_PATTERN})\s+(?:(\d{{1,2}}),?\s+)?{_YEAR}\b", re.IGNORECASE)
# "5 March 2020"
_DMY_RE = re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_PATTERN})\s+{_YEAR}\b", re.IGNORECASE)
_BARE_YEAR_RE = re.compile(rf"\b{_YEAR}\b")

_BEFORE_RE = re.compile(r"\b(before|prior to|precedes?|preceded|predates?|predated|earlier than)\b", re.IGNORECASE)
_AFTER_RE = re.compile(r"\b(after|later than|follows|followed|postdates?|postdated|subsequent to)\b", re.IGNORECASE)


def extract_date_intervals(text: str) -> list[Interval]:
    """All time expressions in the text as [earliest, latest] intervals.
    More specific patterns run first and blank out their matches so a bare
    year inside "March 2020" is not double-counted."""
    intervals: list[Interval] = []
    working = text

    def consume(pattern: re.Pattern, build) -> None:
        nonlocal working

        def replace(m: re.Match) -> str:
            intervals.append(build(m))
            return " " * len(m.group(0))

        working = pattern.sub(replace, working)

    consume(_ISO_RE, lambda m: _iso_interval(m))
    consume(_DMY_RE, lambda m: _month_interval(m.group(2), m.group(1), m.group(3)))
    consume(_MDY_RE, lambda m: _month_interval(m.group(1), m.group(2), m.group(3)))
    consume(
        _BARE_YEAR_RE,
        lambda m: ((int(m.group(1)), 1, 1), (int(m.group(1)), 12, 31)),
    )
    return intervals


def _iso_interval(m: re.Match) -> Interval:
    year, month = int(m.group(1)), int(m.group(2))
    if m.group(3):
        day = int(m.group(3))
        return (year, month, day), (year, month, day)
    return (year, month, 1), (year, month, 31)


def _month_interval(month_name: str, day: str | None, year: str) -> Interval:
    y, mo = int(year), _MONTHS[month_name.lower()]
    if day:
        d = int(day)
        return (y, mo, d), (y, mo, d)
    return (y, mo, 1), (y, mo, 31)


def envelope(intervals: list[Interval]) -> Interval | None:
    """The [earliest, latest] envelope over every expression found."""
    if not intervals:
        return None
    return min(lo for lo, _ in intervals), max(hi for _, hi in intervals)


def _fmt(d: DateTuple) -> str:
    return f"{d[0]:04d}-{d[1]:02d}-{d[2]:02d}"


def _premise_interval(pid: str, nodes: dict, out_edges: dict) -> Interval | None:
    """Dates from the premise text itself, else from its cited spans."""
    node = nodes.get(pid)
    if node is None:
        return None
    found = extract_date_intervals(node["payload"].get("text", ""))
    if not found:
        for e in out_edges.get(pid, []):
            if e["edge_type"] == "extracts" and e["dst_node_id"] in nodes:
                found += extract_date_intervals(nodes[e["dst_node_id"]]["payload"].get("text", ""))
    return envelope(found)


def check_temporal_findings(
    nodes: dict[str, dict], out_edges: dict[str, list[dict]]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(violations, advisories) as (finding_id, detail) pairs.

    A violation means the ordering the finding asserts is definitely
    contradicted by the evidence dates. An advisory means the finding could
    not be mechanically decided (no ordering keyword, wrong premise count,
    or missing timestamps) — surfaced, never guessed."""
    violations: list[tuple[str, str]] = []
    advisories: list[tuple[str, str]] = []

    for nid in sorted(nodes):
        node = nodes[nid]
        if node["type"] != "Finding" or node["payload"].get("edge_type") != "temporal":
            continue
        text = node["payload"].get("text", "")
        direction = _direction(text)
        if direction is None:
            advisories.append(
                (nid, "temporal finding has no recognizable ordering keyword; not mechanically checkable")
            )
            continue
        premise_ids = node["payload"].get("premise_ids", [])
        if len(premise_ids) != 2:
            advisories.append(
                (nid, f"temporal ordering check needs exactly 2 premises, finding cites {len(premise_ids)}")
            )
            continue
        intervals = [_premise_interval(pid, nodes, out_edges) for pid in premise_ids]
        missing = [pid for pid, iv in zip(premise_ids, intervals) if iv is None]
        if missing:
            advisories.append(
                (nid, f"no timestamps found on premise(s) {', '.join(missing)}; ordering not checkable")
            )
            continue
        (a_lo, a_hi), (b_lo, b_hi) = intervals
        a_id, b_id = premise_ids
        if direction == "before" and a_lo > b_hi:
            violations.append(
                (
                    nid,
                    f"finding claims {a_id} precedes {b_id}, but evidence dates say "
                    f"{a_id} is {_fmt(a_lo)}..{_fmt(a_hi)} and {b_id} is {_fmt(b_lo)}..{_fmt(b_hi)}",
                )
            )
        elif direction == "after" and a_hi < b_lo:
            violations.append(
                (
                    nid,
                    f"finding claims {a_id} follows {b_id}, but evidence dates say "
                    f"{a_id} is {_fmt(a_lo)}..{_fmt(a_hi)} and {b_id} is {_fmt(b_lo)}..{_fmt(b_hi)}",
                )
            )
    return violations, advisories


def _direction(text: str) -> str | None:
    """First ordering keyword by position decides the claimed direction."""
    before = _BEFORE_RE.search(text)
    after = _AFTER_RE.search(text)
    if before and (not after or before.start() < after.start()):
        return "before"
    if after:
        return "after"
    return None
