"""Shared M3 corpus for the CLI-driven exit tests (orchestrator, M2, M5):
pages, URLs, premise/finding texts, ID helpers, the scripted model flow,
and the `ask` invocation. Fixtures built on these live in conftest.py
(m3_transport / m3_fixtures_path / m3_scripted_adapter) so test modules
import plain names here without shadowing fixtures.
"""

from __future__ import annotations

import parsec.cli as cli
from parsec import ids
from parsec.gateway.fake_adapter import scripted_response
from parsec.retrieval.extract import extract_text
from parsec.retrieval.span_indexer import index_spans

QUERY = "what temperature does water boil at sea level, and what about on mount everest"
SQ1 = "boiling point at sea level"
SQ2 = "boiling point on mount everest"

# "(212 degrees Fahrenheit)" sits inside the first span's 200-char preview:
# it marks raw-document content that subagent contexts DO contain and
# orchestrator contexts must NOT.
RAW_DOC_SENTINEL = "212 degrees Fahrenheit"

PAGE_A = (
    "<html><head><title>Boiling Point</title></head><body>"
    "<p>Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard "
    "atmospheric pressure of one atmosphere, which is the pressure found at sea level. "
    "This has been the accepted reference value in thermodynamics for over a century and "
    "is used to calibrate thermometers worldwide. The Celsius scale itself was originally "
    "defined with the boiling point of water fixed at one hundred degrees exactly.</p>"
    "</body></html>"
).encode()

PAGE_B = (
    "<html><head><title>Altitude Effects</title></head><body>"
    "<p>At higher altitudes the boiling point of water decreases because atmospheric "
    "pressure is lower. In Denver, Colorado, at about 1600 meters of elevation, water "
    "boils at roughly 95 degrees Celsius. On the summit of Mount Everest water boils at "
    "only about 70 degrees Celsius, which makes cooking food by boiling significantly "
    "slower and less effective at extreme elevations.</p>"
    "</body></html>"
).encode()

URL_A = "https://example.test/boiling"
URL_B = "https://example.test/altitude"

PREMISE_A_TEXT = "Water boils at 100 degrees Celsius at standard atmospheric pressure at sea level."
PREMISE_B_TEXT = "On the summit of Mount Everest water boils at about 70 degrees Celsius."
FINDING_A_TEXT = "The sea-level boiling point of water is 100 degrees Celsius."


def page_span_ids(page: bytes) -> list[str]:
    """Compute span IDs exactly as the fetch pipeline will."""
    text, _, _ = extract_text(page, "text/html")
    h = ids.doc_hash(page)
    return [ids.span_id(h, s, e) for s, e in index_spans(text)]


def premise_id(text: str, span_ref: str) -> str:
    return ids.node_id("Premise", {"text": text, "span_refs": [span_ref], "claim_class": "stable"})


def finding_id(text: str, premise_ids: list[str], edge_type: str = "deduces") -> str:
    return ids.node_id("Finding", {"text": text, "premise_ids": premise_ids, "edge_type": edge_type})


def m3_script() -> tuple[str, list]:
    """The M3 orchestrator flow: decompose -> two subagents (search, fetch,
    record, submit) -> writer, plus one spare writer response for repair
    rounds. Returns (expected answer, scripted responses)."""
    span_a = page_span_ids(PAGE_A)[0]
    span_b = page_span_ids(PAGE_B)[0]
    p_a = premise_id(PREMISE_A_TEXT, span_a)
    p_b = premise_id(PREMISE_B_TEXT, span_b)
    f_a = finding_id(FINDING_A_TEXT, [p_a])
    answer = (
        "Here is what the sources say. [narrative]\n"
        f"Water boils at 100 degrees Celsius at sea level. [{f_a}] "
        f"On the summit of Mount Everest it boils at about 70 degrees Celsius. [{p_b}]"
    )
    responses = [
        # orchestrator: decomposition
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
              "input": {"subquestions": [SQ1, SQ2]}}],
            stop_reason="tool_use", index=0,
        ),
        # subagent sq-1
        scripted_response(
            [{"type": "tool_use", "id": "tu_s1", "name": "search_broad", "input": {"query": SQ1, "k": 5}}],
            stop_reason="tool_use", index=1,
        ),
        scripted_response(
            [{"type": "tool_use", "id": "tu_f1", "name": "fetch", "input": {"url": URL_A}}],
            stop_reason="tool_use", index=2,
        ),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r1", "name": "record_premises",
              "input": {"premises": [{"text": PREMISE_A_TEXT, "span_refs": [span_a]}]}}],
            stop_reason="tool_use", index=3,
        ),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub1", "name": "submit_report",
              "input": {
                  "status": "answered",
                  "findings": [{"text": FINDING_A_TEXT, "premise_ids": [p_a], "edge_type": "deduces"}],
                  "summary": "Sea-level boiling point established from one source.",
              }}],
            stop_reason="tool_use", index=4,
        ),
        # subagent sq-2
        scripted_response(
            [{"type": "tool_use", "id": "tu_s2", "name": "search_broad", "input": {"query": SQ2, "k": 5}}],
            stop_reason="tool_use", index=5,
        ),
        scripted_response(
            [{"type": "tool_use", "id": "tu_f2", "name": "fetch", "input": {"url": URL_B}}],
            stop_reason="tool_use", index=6,
        ),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r2", "name": "record_premises",
              "input": {"premises": [{"text": PREMISE_B_TEXT, "span_refs": [span_b]}]}}],
            stop_reason="tool_use", index=7,
        ),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub2", "name": "submit_report",
              "input": {"status": "answered"}}],
            stop_reason="tool_use", index=8,
        ),
        # orchestrator: writer
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=9),
        # spare for a repair round in negative scenarios
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=10),
    ]
    return answer, responses


def run_ask(data_dir, fixtures_path, session_id):
    return cli.main(
        [
            "ask",
            QUERY,
            "--session-id", session_id,
            "--adapter", "fake",
            "--model", "fake-model",
            "--cache-mode", "record",
            "--data-dir", str(data_dir),
            "--search-fixtures", str(fixtures_path),
            "--max-gap-rounds", "0",  # scripted runs budget no gap round; gap-fill has its own test
            "--json",
        ]
    )
