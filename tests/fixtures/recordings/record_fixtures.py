"""Recorder for the frozen recordings in this directory.

    python tests/fixtures/recordings/record_fixtures.py <out_dir>

Runs four scripted, keyless, offline sessions and leaves each one's working
directory under <out_dir>/<name>/ (a `data/` dir plus the search fixtures the
run read). `parsec eval export-case` then turns each `data/` into the text
corpus committed here.

READ THIS BEFORE RE-RUNNING IT. The committed recordings were made by running
this script against the source tree at commit 366f1f6 — the last commit
before the remediation work — and they exist to prove that every later
release still replays what that code recorded. Re-recording them with newer
code would turn that proof into a same-build round trip, which shows nothing.
The script is kept so the recordings can be audited (this is exactly what
the model "said" and what the fake web served), not so they can be
regenerated. A new scenario gets a new recording; these four are frozen.

It therefore uses only APIs that exist at 366f1f6.

Scenarios
  demo      the built-in offline demo, unchanged
  full      sequential run through most of the loop: a rejected premise, a
            transform_note premise, a near-duplicate that corroborates,
            findings (deduces + temporal), a recorded conflict, an invalid
            submit_report that is corrected, an uncited sentence that forces
            a writer repair round, and a low-credence claim that triggers a
            gap-fill subagent and a rewrite
  parallel  three concurrent subagents whose completion order differs from
            their dispatch order
  refresh   a parent session with stable and volatile evidence, then
            `parsec refresh` after the volatile page changed (two sessions)
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

import httpx

import parsec.cli as cli
from parsec import ids
from parsec.gateway.fake_adapter import StreamFakeAdapter, scripted_response
from parsec.retrieval.extract import extract_text
from parsec.retrieval.span_indexer import index_spans

# -- the fake web -------------------------------------------------------------

URL_GOV = "https://geo.example.gov/olympus"
URL_BLOG = "https://marsfan.blogspot.com/olympus"
URL_WIRE = "https://news.example.org/mars"
URL_SURVEY = "https://survey.example.gov/mars/olympus"
URL_PRICE = "https://market.example/price"

PAGE_GOV = (
    "Olympus Mons is 21.9 kilometers tall. Olympus Mons is about 600 kilometers wide. "
    "Olympus Mons was discovered by Mariner 9 in 1971."
)
PAGE_BLOG = (
    "Olympus Mons is 25 kilometers tall according to older surveys. "
    "The Viking orbiters mapped Olympus Mons in 1976."
)
PAGE_WIRE = "Olympus Mons is about 600 kilometers wide, roughly the size of Arizona."
PAGE_SURVEY = "Laser altimetry confirms that Olympus Mons is 21.9 kilometers tall."
PRICE_BEFORE = "The listed price is 10 dollars."
PRICE_AFTER = "The listed price is 12 dollars."

P_HEIGHT = "Olympus Mons is 21.9 kilometers tall."
P_HEIGHT_BLOG = "Olympus Mons is 25 kilometers tall according to older surveys."
P_MILES = "Olympus Mons is about 13.6 miles tall."
P_WIDTH = "Olympus Mons is about 600 kilometers wide."
P_DISCOVERED = "Olympus Mons was discovered by Mariner 9 in 1971."
P_MAPPED = "The Viking orbiters mapped Olympus Mons in 1976."
F_ORDER = "Mariner 9 discovered Olympus Mons before the Viking orbiters mapped it."
F_TALLEST = "Olympus Mons is 21.9 kilometers tall, which is about 13.6 miles."


def span_of(page: str) -> str:
    """The page's (single) span id, computed exactly as the fetch pipeline will."""
    raw = page.encode()
    text, _, _ = extract_text(raw, "text/plain")
    start, end = index_spans(text)[0]
    return ids.span_id(ids.doc_hash(raw), start, end)


def premise_id(text: str, span_refs: list[str], claim_class: str = "stable") -> str:
    return ids.node_id(
        "Premise", {"text": text, "span_refs": span_refs, "claim_class": claim_class}
    )


def finding_id(text: str, premise_ids: list[str], edge_type: str) -> str:
    return ids.node_id(
        "Finding", {"text": text, "premise_ids": premise_ids, "edge_type": edge_type}
    )


def transport(pages: dict[str, str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in pages:
            return httpx.Response(200, text=pages[url], headers={"content-type": "text/plain"})
        return httpx.Response(404, text="not found")  # includes every robots.txt

    return httpx.MockTransport(handler)


_n = 0


def call(*blocks: dict) -> object:
    """One scripted model turn that calls tools."""
    global _n
    _n += 1
    return scripted_response(list(blocks), stop_reason="tool_use", index=_n)


def say(text: str) -> object:
    global _n
    _n += 1
    return scripted_response([{"type": "text", "text": text}], stop_reason="end_turn", index=_n)


def tool(name: str, **tool_input: object) -> dict:
    global _n
    _n += 1
    return {"type": "tool_use", "id": f"tu_{_n}", "name": name, "input": tool_input}


class Delayed:
    """Delay some streams' calls so parallel completion order is not dispatch order."""

    def __init__(self, inner: StreamFakeAdapter, delays: dict[str, float]):
        self._inner, self._delays = inner, delays

    async def complete(self, request):  # type: ignore[no-untyped-def]
        from parsec.store.event_log import CURRENT_STREAM

        delay = self._delays.get(CURRENT_STREAM.get())
        if delay:
            await asyncio.sleep(delay)
        return await self._inner.complete(request)


def run_cli(argv: list[str], adapter: object, pages: dict[str, str]) -> int:
    previous = (cli.adapter_factory, cli.fetch_transport)
    cli.adapter_factory = lambda config: adapter
    cli.fetch_transport = transport(pages)
    try:
        return cli.main(argv)
    finally:
        cli.adapter_factory, cli.fetch_transport = previous


def ask(query: str, session_id: str, adapter: object, pages: dict[str, str], *extra: str) -> int:
    # Relative paths on purpose: the fixture-search provider re-reads
    # `search_fixtures` on replay, so the recorded path must resolve wherever
    # the recording is later replayed from.
    return run_cli(
        [
            "ask", query,
            "--session-id", session_id,
            "--adapter", "fake",
            "--model", "fake-model",
            "--cache-mode", "record",
            "--data-dir", "data",
            "--search-provider", "fixture",
            "--search-fixtures", "queries.json",
            "--max-turns", "40",
            "--json",
            *extra,
        ],
        adapter,
        pages,
    )


def write_queries(fixtures: dict[str, list[dict]]) -> None:
    Path("queries.json").write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")


def hit(title: str, url: str) -> dict:
    return {"title": title, "url": url, "snippet": title}


# -- scenarios ----------------------------------------------------------------


def record_demo() -> list[str]:
    assert cli.main(["demo", "--data-dir", "data"]) == 0
    conn = sqlite3.connect("data/parsec.db")
    try:
        return [conn.execute("SELECT session_id FROM sessions").fetchone()[0]]
    finally:
        conn.close()


def record_full() -> list[str]:
    sq1, sq2 = "how tall is olympus mons", "how wide is olympus mons"
    write_queries(
        {
            sq1: [hit("Olympus Mons facts", URL_GOV), hit("Mars fan blog", URL_BLOG)],
            sq2: [hit("Mars wire story", URL_WIRE)],
            "olympus mons height survey": [hit("Altimetry survey", URL_SURVEY)],
        }
    )
    gov, blog, wire, survey = (span_of(p) for p in (PAGE_GOV, PAGE_BLOG, PAGE_WIRE, PAGE_SURVEY))
    p_height = premise_id(P_HEIGHT, [gov])
    p_height_blog = premise_id(P_HEIGHT_BLOG, [blog])
    p_miles = premise_id(P_MILES, [gov])
    p_discovered = premise_id(P_DISCOVERED, [gov])
    p_mapped = premise_id(P_MAPPED, [blog])
    p_width = premise_id(P_WIDTH, [wire])
    f_order = finding_id(F_ORDER, [p_discovered, p_mapped], "temporal")
    f_tallest = finding_id(F_TALLEST, [p_height, p_miles], "deduces")

    body = (
        "## Olympus Mons\n"
        f"Olympus Mons is 21.9 kilometers tall. [{p_height}]\n"
        f"That is about 13.6 miles. [{f_tallest}]\n"
        f"It is likely about 600 kilometers wide. [{p_width}]\n"
        f"One source suggests it is 25 kilometers tall. [{p_height_blog}]\n"
        f"Mariner 9 discovered it before the Viking orbiters mapped it. [{f_order}]\n"
    )
    closing = "The sources disagree on its height. [narrative]"
    draft_uncited = body + "It is the largest volcano on Mars.\n" + closing
    draft_cited = body + closing

    scripts = {
        "orchestrator": [
            call(tool("submit_subquestions", scope="Height and width of Olympus Mons.",
                      effort="deep", subquestions=[sq1, sq2])),
            say(draft_uncited),   # one uncited sentence -> repair round
            say(draft_cited),     # repaired; the 25 km claim is flagged -> gap-fill
            say(draft_cited),     # rewrite after the gap-fill subagent
        ],
        "sq-1": [
            call(tool("search_broad", query=sq1, k=5)),
            call(tool("fetch", url=URL_GOV), tool("fetch", url=URL_BLOG)),
            call(tool("record_premises", premises=[
                {"text": P_HEIGHT, "span_refs": [gov]},
                {"text": P_HEIGHT_BLOG, "span_refs": [blog]},
                # rejected: 13.6 is not in the span and carries no note
                {"text": P_MILES, "span_refs": [gov]},
                {"text": P_DISCOVERED, "span_refs": [gov]},
                {"text": P_MAPPED, "span_refs": [blog]},
            ])),
            call(tool("record_premises", premises=[
                {"text": P_MILES, "span_refs": [gov],
                 "transform_note": "21.9 km x 0.621371 = 13.6 mi"},
            ])),
            # invalid: cites a premise id this subagent never recorded
            call(tool("submit_report", status="answered", findings=[
                {"text": F_ORDER, "premise_ids": ["premise:0000000000000000"],
                 "edge_type": "temporal"},
            ])),
            call(tool("submit_report", status="answered",
                      findings=[
                          {"text": F_ORDER, "premise_ids": [p_discovered, p_mapped],
                           "edge_type": "temporal"},
                          {"text": F_TALLEST, "premise_ids": [p_height, p_miles],
                           "edge_type": "deduces"},
                      ],
                      conflicts=[{"a": p_height, "b": p_height_blog,
                                  "note": "21.9 km vs 25 km"}],
                      dead_ends=["olympus mons height in feet"],
                      summary="Two sources disagree on the height.")),
        ],
        "sq-2": [
            call(tool("search_broad", query=sq2, k=5)),
            call(tool("fetch", url=URL_WIRE)),
            call(tool("search_within", query="kilometers wide", k=3)),
            call(tool("record_premises", premises=[{"text": P_WIDTH, "span_refs": [wire]}])),
            # near-duplicate: same fact, the .gov span attaches as corroboration
            call(tool("record_premises", premises=[{"text": P_WIDTH, "span_refs": [gov]}])),
            call(tool("submit_report", status="answered", summary="Width corroborated.")),
        ],
        "sq-gap-1": [
            call(tool("search_broad", query="olympus mons height survey", k=5)),
            call(tool("fetch", url=URL_SURVEY)),
            # near-duplicate of the .gov height premise: a second independent span
            call(tool("record_premises", premises=[{"text": P_HEIGHT, "span_refs": [survey]}])),
            call(tool("submit_report", status="answered",
                      summary="An independent survey confirms 21.9 km.")),
        ],
    }
    pages = {URL_GOV: PAGE_GOV, URL_BLOG: PAGE_BLOG, URL_WIRE: PAGE_WIRE, URL_SURVEY: PAGE_SURVEY}
    code = ask(
        "how big is olympus mons", "frozen-full", StreamFakeAdapter(scripts), pages,
        "--max-gap-rounds", "1", "--max-coverage-gap-rounds", "0",
        "--max-turns-per-subagent", "8",
    )
    assert code in (0, 2), code
    return ["frozen-full"]


def record_parallel() -> list[str]:
    questions = {
        "sq-1": ("how tall is olympus mons", URL_SURVEY, PAGE_SURVEY, P_HEIGHT),
        "sq-2": ("how wide is olympus mons", URL_WIRE, PAGE_WIRE, P_WIDTH),
        "sq-3": ("when was olympus mons mapped", URL_BLOG, PAGE_BLOG, P_MAPPED),
    }
    write_queries({q: [hit(q, url)] for q, url, _, _ in questions.values()})
    cited = {sq: premise_id(text, [span_of(page)]) for sq, (_, _, page, text) in questions.items()}
    answer = (
        f"Olympus Mons is 21.9 kilometers tall. [{cited['sq-1']}]\n"
        f"It is likely about 600 kilometers wide. [{cited['sq-2']}]\n"
        f"The Viking orbiters reportedly mapped it in 1976. [{cited['sq-3']}]"
    )
    scripts: dict[str, list] = {
        "orchestrator": [
            call(tool("submit_subquestions", effort="deep",
                      subquestions=[q for q, _, _, _ in questions.values()])),
            say(answer),
        ]
    }
    for sq, (query, url, page, text) in questions.items():
        scripts[sq] = [
            call(tool("search_broad", query=query, k=5)),
            call(tool("fetch", url=url)),
            call(tool("record_premises", premises=[{"text": text, "span_refs": [span_of(page)]}])),
            call(tool("submit_report", status="answered")),
        ]
    # sq-1 is slowest and sq-3 fastest: the recorded join order is 3, 2, 1.
    adapter = Delayed(StreamFakeAdapter(scripts), {"sq-1": 0.15, "sq-2": 0.07})
    pages = {url: page for _, url, page, _ in questions.values()}
    code = ask(
        "olympus mons height, width and mapping", "frozen-parallel", adapter, pages,
        "--parallel", "3", "--max-gap-rounds", "0", "--max-coverage-gap-rounds", "0",
    )
    assert code in (0, 2), code
    return ["frozen-parallel"]


def record_refresh() -> list[str]:
    write_queries({})
    gov, before, after = span_of(PAGE_GOV), span_of(PRICE_BEFORE), span_of(PRICE_AFTER)
    p_height = premise_id(P_HEIGHT, [gov])
    p_before = premise_id(PRICE_BEFORE, [before], "volatile")
    p_after = premise_id(PRICE_AFTER, [after], "volatile")

    def answer(price_premise: str, price: str) -> str:
        return (
            f"Olympus Mons is 21.9 kilometers tall. [{p_height}]\n"
            f"One listing gives the price as {price} dollars. [{price_premise}]"
        )

    parent = {
        "orchestrator": [
            call(tool("submit_subquestions",
                      scope="The mountain's height and the current listed price.",
                      effort="standard",
                      subquestions=["how tall is the mountain", "what is the listed price"])),
            say(answer(p_before, "10")),
        ],
        "sq-1": [
            call(tool("fetch", url=URL_GOV)),
            call(tool("record_premises", premises=[
                {"text": P_HEIGHT, "span_refs": [gov], "claim_class": "stable"}])),
            call(tool("submit_report", status="answered")),
        ],
        "sq-2": [
            call(tool("fetch", url=URL_PRICE)),
            call(tool("record_premises", premises=[
                {"text": PRICE_BEFORE, "span_refs": [before], "claim_class": "volatile"}])),
            call(tool("submit_report", status="answered")),
        ],
    }
    pages = {URL_GOV: PAGE_GOV, URL_PRICE: PRICE_BEFORE}
    code = ask(
        "mountain height and price", "frozen-refresh-parent", StreamFakeAdapter(parent), pages,
        "--max-gap-rounds", "0", "--max-coverage-gap-rounds", "0",
    )
    assert code in (0, 2), code

    # The world moves: the price page changes, the mountain does not. The
    # refresh carries sq-1 forward and re-researches only sq-2.
    pages[URL_PRICE] = PRICE_AFTER
    refresh = {
        "orchestrator": [say(answer(p_after, "12"))],
        "sq-2": [
            call(tool("fetch", url=URL_PRICE)),
            call(tool("record_premises", premises=[
                {"text": PRICE_AFTER, "span_refs": [after], "claim_class": "volatile"}])),
            call(tool("submit_report", status="answered")),
        ],
    }
    code = run_cli(
        ["refresh", "frozen-refresh-parent", "--data-dir", "data", "--json"],
        StreamFakeAdapter(refresh), pages,
    )
    assert code in (0, 2), code
    conn = sqlite3.connect("data/parsec.db")
    try:
        child = conn.execute(
            "SELECT session_id FROM sessions WHERE parent_session_id='frozen-refresh-parent'"
        ).fetchone()[0]
    finally:
        conn.close()
    return ["frozen-refresh-parent", child]


SCENARIOS = {
    "demo": record_demo,
    "full": record_full,
    "parallel": record_parallel,
    "refresh": record_refresh,
}


def main(out_dir: Path, only: list[str]) -> None:
    out_dir = out_dir.resolve()
    home = Path.cwd()
    for name, record in SCENARIOS.items():
        if only and name not in only:
            continue
        workdir = out_dir / name
        workdir.mkdir(parents=True)
        os.chdir(workdir)
        try:
            sessions = record()
            conn = sqlite3.connect("data/parsec.db")
            try:
                statuses = {
                    sid: conn.execute(
                        "SELECT status FROM sessions WHERE session_id=?", (sid,)
                    ).fetchone()[0]
                    for sid in sessions
                }
            finally:
                conn.close()
        finally:
            os.chdir(home)
        (workdir / "sessions.json").write_text(
            json.dumps({"sessions": sessions, "status": statuses}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{name}: {statuses}", file=sys.stderr)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
