"""Shared M14 corpus: a two-subquestion question over a fake web whose price
page changes between observations. Pure helpers only — the fixtures that
record the parent session live in tests/integration/conftest.py."""

from __future__ import annotations

from parsec import ids
from parsec.gateway.fake_adapter import scripted_response
from parsec.retrieval.span_indexer import index_spans

URL_A = "https://geo.example/mountain"
URL_B = "https://market.example/price"
SENT_A = "Olympus Mons is 21.9 kilometers tall."
SENT_B1 = "The listed price is 10 dollars."
SENT_B2 = "The listed price is 12 dollars."


def _span(text: str) -> str:
    start, end = index_spans(text)[0]
    return ids.span_id(ids.doc_hash(text.encode()), start, end)


def _premise_id(sentence: str, claim_class: str = "stable") -> str:
    return ids.node_id(
        "Premise",
        {"text": sentence, "span_refs": [_span(sentence)], "claim_class": claim_class},
    )


def _parent_script() -> list:
    """decompose -> sq-1 (stable evidence) -> sq-2 (volatile evidence) -> write."""
    return [
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
              "input": {"scope": "Cover the mountain's height and the current listed price.",
                        "effort": "standard",
                        "subquestions": ["how tall is the mountain", "what is the listed price"]}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_f1", "name": "fetch", "input": {"url": URL_A}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r1", "name": "record_premises",
              "input": {"premises": [{"text": SENT_A, "span_refs": [_span(SENT_A)],
                                      "claim_class": "stable"}]}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_s1", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_f2", "name": "fetch", "input": {"url": URL_B}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r2", "name": "record_premises",
              "input": {"premises": [{"text": SENT_B1, "span_refs": [_span(SENT_B1)],
                                      "claim_class": "volatile"}]}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_s2", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use"),
        scripted_response(
            [{"type": "text", "text": _answer(SENT_B1)}], stop_reason="end_turn"),
    ]


def _refresh_script(price_sentence: str = SENT_B2, include_sq1: bool = False) -> list:
    """No decomposer call: the brief is seeded. sq-2 re-fetches the (changed)
    price page; sq-1 appears only under --all."""
    script: list = []
    if include_sq1:
        script += [
            scripted_response(
                [{"type": "tool_use", "id": "tu_rf0", "name": "fetch", "input": {"url": URL_A}}],
                stop_reason="tool_use"),
            scripted_response(
                [{"type": "tool_use", "id": "tu_rr0", "name": "record_premises",
                  "input": {"premises": [{"text": SENT_A, "span_refs": [_span(SENT_A)],
                                          "claim_class": "stable"}]}}],
                stop_reason="tool_use"),
            scripted_response(
                [{"type": "tool_use", "id": "tu_rs0", "name": "submit_report",
                  "input": {"status": "answered"}}], stop_reason="tool_use"),
        ]
    script += [
        scripted_response(
            [{"type": "tool_use", "id": "tu_rf1", "name": "fetch", "input": {"url": URL_B}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_rr1", "name": "record_premises",
              "input": {"premises": [{"text": price_sentence,
                                      "span_refs": [_span(price_sentence)],
                                      "claim_class": "volatile"}]}}],
            stop_reason="tool_use"),
        scripted_response(
            [{"type": "tool_use", "id": "tu_rs1", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use"),
        scripted_response(
            [{"type": "text", "text": _answer(price_sentence)}], stop_reason="end_turn"),
    ]
    return script


def _answer(price_sentence: str) -> str:
    p_a = _premise_id(SENT_A)
    p_b = _premise_id(price_sentence, "volatile")
    return f"Height and price are settled. [narrative]\n{SENT_A} [{p_a}]\n{price_sentence} [{p_b}]"
