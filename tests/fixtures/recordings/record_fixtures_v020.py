"""Recorder for the 0.2.0 frozen recording (`v020/`).

    python tests/fixtures/recordings/record_fixtures_v020.py <out_dir>

Same contract as record_fixtures.py, one release later: the committed
recording was made by the 0.2.0 source tree, and it exists so that every
LATER release proves it still replays what 0.2.0 recorded. Do not re-run it
against newer code — a new release gets a new recording beside this one.

One scripted, keyless, offline session through everything 0.2.0 added:
  - two subquestions with different target times (now, and "in 2019")
  - a premise dated by `as_of`, after one whose date its span does not state
    is rejected
  - a `computes` finding: a wrong conversion rejected with the expected
    value, then the right one verified
  - a writer draft with a wrong figure, caught by the sentence check and
    fixed in the repair round; a heading and a narrative sentence
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

import parsec.cli as cli
from parsec import ids
from parsec.gateway.fake_adapter import StreamFakeAdapter, scripted_response
from parsec.retrieval.span_indexer import index_spans

URL_GOV = "https://geo.example.gov/olympus"
URL_PRICE = "https://market.example/price"
PAGE_GOV = "Olympus Mons is 21.9 kilometers tall."
PAGE_PRICE = "The listed price was 10 dollars in 2019."
Y2019 = "2019-01-01/2019-12-31"
SESSION = "frozen-v020"

_n = 0


def call(name: str, **tool_input: object) -> object:
    global _n
    _n += 1
    block = {"type": "tool_use", "id": f"tu_{_n}", "name": name, "input": tool_input}
    return scripted_response([block], stop_reason="tool_use", index=_n)


def say(text: str) -> object:
    global _n
    _n += 1
    return scripted_response([{"type": "text", "text": text}], stop_reason="end_turn", index=_n)


def span_of(page: str) -> str:
    start, end = index_spans(page)[0]
    return ids.span_id(ids.doc_hash(page.encode()), start, end)


def record() -> None:
    p_height = ids.node_id(
        "Premise", {"text": PAGE_GOV, "span_refs": [span_of(PAGE_GOV)], "claim_class": "stable"}
    )
    p_price = ids.node_id(
        "Premise",
        {"text": PAGE_PRICE, "span_refs": [span_of(PAGE_PRICE)], "claim_class": "volatile",
         "target_time": Y2019},
    )

    def conversion(output: str) -> dict:
        return {
            "text": f"Olympus Mons is about {output} mi tall.",
            "premise_ids": [p_height],
            "edge_type": "computes",
            "operation": {
                "op": "convert", "operands": [{"premise_id": p_height, "value": "21.9"}],
                "unit_from": "km", "unit_to": "mi", "precision": "0.1", "output": output,
            },
        }

    f_miles = ids.node_id("Finding", {**conversion("13.6"), "status": "verified"})

    def draft(miles: str) -> str:
        return (
            "## Olympus Mons, and a price from 2019\n"
            f"Olympus Mons is 21.9 kilometers tall. [{p_height}]\n"
            f"That is about {miles} miles. [{f_miles}]\n"
            f"In 2019 the listed price was 10 dollars. [{p_price}]\n"
            "The price is the older of the figures given here. [narrative]"
        )

    price = {"text": PAGE_PRICE, "span_refs": [span_of(PAGE_PRICE)], "claim_class": "volatile"}
    scripts = {
        "orchestrator": [
            call("submit_subquestions",
                 scope="The height of Olympus Mons in miles, and a listed price in 2019.",
                 effort="standard",
                 subquestions=["how tall is olympus mons in miles",
                               "what was the listed price in 2019"],
                 target_times=["now", "in 2019"]),
            say(draft("14.2")),   # wrong figure -> the sentence check sends it back
            say(draft("13.6")),   # repaired
        ],
        "sq-1": [
            call("fetch", url=URL_GOV),
            call("record_premises", premises=[{"text": PAGE_GOV, "span_refs": [span_of(PAGE_GOV)]}]),
            call("submit_report", status="answered", findings=[conversion("14.2")]),  # rejected
            call("submit_report", status="answered", findings=[conversion("13.6")],
                 summary="Height converted to miles."),
        ],
        "sq-2": [
            call("fetch", url=URL_PRICE),
            call("record_premises", premises=[{**price, "as_of": "2021"}]),  # span says 2019
            call("record_premises", premises=[{**price, "as_of": "2019"}]),
            call("submit_report", status="answered", summary="One dated listing."),
        ],
    }
    pages = {URL_GOV: PAGE_GOV, URL_PRICE: PAGE_PRICE}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) in pages:
            return httpx.Response(
                200, text=pages[str(request.url)], headers={"content-type": "text/plain"}
            )
        return httpx.Response(404, text="not found")  # includes every robots.txt

    Path("queries.json").write_text("{}\n", encoding="utf-8")
    cli.adapter_factory = lambda config: StreamFakeAdapter(scripts)
    cli.fetch_transport = httpx.MockTransport(handler)
    code = cli.main(
        [
            "ask", "olympus mons height in miles, and the listed price in 2019",
            "--session-id", SESSION, "--adapter", "fake", "--model", "fake-model",
            "--cache-mode", "record",
            # relative paths: replay re-reads the search fixtures from the recorded path
            "--data-dir", "data", "--search-provider", "fixture",
            "--search-fixtures", "queries.json",
            "--max-turns", "30", "--max-turns-per-subagent", "6",
            "--max-gap-rounds", "0", "--max-coverage-gap-rounds", "0", "--json",
        ]
    )
    assert code == 0, code


def main(out_dir: Path) -> None:
    workdir = out_dir.resolve() / "v020"
    workdir.mkdir(parents=True)
    home = Path.cwd()
    os.chdir(workdir)
    try:
        record()
    finally:
        os.chdir(home)
    (workdir / "sessions.json").write_text(
        json.dumps({"sessions": [SESSION], "status": {SESSION: "done"}}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
