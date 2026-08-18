"""Built-in offline demo: a complete `ask` run with no API keys and no network.

The model is a scripted FakeAdapter and the web is a bundled two-page fixture
corpus served through the CLI's transport seam — the exact machinery the exit
tests use. The recorded session is real: it can be replayed, verified, forked,
and inspected like any live run, which is the point of the demo.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from parsec import ids
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.retrieval.extract import extract_text
from parsec.retrieval.span_indexer import index_spans

DEMO_QUERY = "how far is a parsec in light years"
DEMO_SESSION_PREFIX = "demo"

_PAGE_MAIN_URL = "https://demo.parsec.invalid/parallax"
_PAGE_UNIT_URL = "https://demo.parsec.invalid/unit"
_BLOCKED_URL = "https://demo.parsec.invalid/blocked/archive"

_PAGE_MAIN = (
    "<html><head><title>Stellar Parallax and the Parsec</title></head><body>"
    "<article><p>A parsec is defined as the distance at which one astronomical unit "
    "subtends an angle of one arcsecond of parallax. One parsec equals about 3.26 "
    "light-years, or roughly 31 trillion kilometres, and it remains the working "
    "distance unit of professional astronomy.</p>"
    "<p>The method is geometric: observe a nearby star from opposite sides of Earth's "
    "orbit, half a year apart, and the star's apparent shift against the background "
    "fixes its distance. Every distance claim traces back to that measured path of "
    "sight lines — the same discipline this harness applies to text.</p>"
    "</article></body></html>"
).encode()

_PAGE_UNIT = (
    "<html><head><title>Units: Parsec</title></head><body>"
    "<article><p>The word parsec is a contraction of “parallax of one arcsecond” "
    "and was coined by Herbert Hall Turner in 1913. The nearest star system, Alpha "
    "Centauri, lies about 1.3 parsecs from the Sun, which is roughly 4.2 light-years.</p>"
    "</article></body></html>"
).encode()

_ROBOTS = "User-agent: *\nDisallow: /blocked/\n"

_PREMISE_DISTANCE = (
    "One parsec equals about 3.26 light-years, or roughly 31 trillion kilometres."
)
_PREMISE_NEAREST = (
    "The nearest star system, Alpha Centauri, lies about 1.3 parsecs from the Sun, "
    "which is roughly 4.2 light-years."
)


def _span_id_containing(page: bytes, needle: str) -> str:
    text, _, _ = extract_text(page, "text/html")
    h = ids.doc_hash(page)
    for start, end in index_spans(text):
        if needle in text[start:end]:
            return ids.span_id(h, start, end)
    raise ValueError(f"demo fixture drifted: no span contains {needle!r}")


def demo_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url).rstrip("/")
        if url == "https://demo.parsec.invalid/robots.txt":
            return httpx.Response(200, text=_ROBOTS)
        if url == _PAGE_MAIN_URL:
            return httpx.Response(200, content=_PAGE_MAIN, headers={"content-type": "text/html"})
        if url == _PAGE_UNIT_URL:
            return httpx.Response(200, content=_PAGE_UNIT, headers={"content-type": "text/html"})
        return httpx.Response(404, content=b"not found")

    return httpx.MockTransport(handler)


def write_search_fixtures(data_dir: Path) -> Path:
    fixtures = {
        DEMO_QUERY: [
            {"title": "Stellar Parallax and the Parsec", "url": _PAGE_MAIN_URL,
             "snippet": "One parsec equals about 3.26 light-years."},
            {"title": "Units: Parsec", "url": _PAGE_UNIT_URL,
             "snippet": "Parallax of one arcsecond; coined 1913."},
        ]
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "demo_search_fixtures.json"
    path.write_text(json.dumps(fixtures, indent=2), encoding="utf-8")
    return path


def demo_adapter_factory(config) -> FakeAdapter:
    span_distance = _span_id_containing(_PAGE_MAIN, "3.26 light-years")
    span_nearest = _span_id_containing(_PAGE_UNIT, "Alpha Centauri")
    p_distance = ids.node_id(
        "Premise",
        {"text": _PREMISE_DISTANCE, "span_refs": [span_distance], "claim_class": "stable"},
    )
    p_nearest = ids.node_id(
        "Premise",
        {"text": _PREMISE_NEAREST, "span_refs": [span_nearest], "claim_class": "stable"},
    )
    answer = (
        "A parsec is about 3.26 light-years — the distance at which one astronomical "
        f"unit subtends one arcsecond of parallax. [{p_distance}]\n"
        "For scale, the nearest star system, Alpha Centauri, is about 1.3 parsecs "
        f"(roughly 4.2 light-years) away. [{p_nearest}]"
    )
    responses = [
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
              "input": {"subquestions": [DEMO_QUERY]}}], stop_reason="tool_use", index=0),
        scripted_response(
            [{"type": "tool_use", "id": "tu_s", "name": "search_broad",
              "input": {"query": DEMO_QUERY, "k": 5}}], stop_reason="tool_use", index=1),
        scripted_response(
            [
                {"type": "tool_use", "id": "tu_f1", "name": "fetch",
                 "input": {"url": _PAGE_MAIN_URL}},
                {"type": "tool_use", "id": "tu_f2", "name": "fetch",
                 "input": {"url": _PAGE_UNIT_URL}},
                {"type": "tool_use", "id": "tu_f3", "name": "fetch",
                 "input": {"url": _BLOCKED_URL}},
            ],
            stop_reason="tool_use", index=2),
        scripted_response(
            [{"type": "tool_use", "id": "tu_w", "name": "search_within",
              "input": {"query": "parsec light years distance", "k": 3}}],
            stop_reason="tool_use", index=3),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r", "name": "record_premises",
              "input": {"premises": [
                  {"text": _PREMISE_DISTANCE, "span_refs": [span_distance]},
                  {"text": _PREMISE_NEAREST, "span_refs": [span_nearest]},
              ]}}],
            stop_reason="tool_use", index=4),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use", index=5),
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=6),
    ]
    return FakeAdapter(responses)
