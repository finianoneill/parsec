"""M5 exit test (§11): frozen corpora (replay mode), 3-axis scoring
(citation faithfulness mechanical / coverage vs. gold must-find list /
synthesis via judge), and a regression runner comparing two harness
versions on identical corpora.

The "two versions" here are two scripted writers over the SAME frozen
case: version A covers both gold items, version B silently drops one —
the regression runner must catch the drop mechanically.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import parsec.cli as cli
from parsec.config import RealClock
from parsec.evals.case import load_case, make_case_from_session
from parsec.evals.runner import run_cases
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.models.gateway import ModelResponse, Usage
from tests.integration.test_orchestrator_exit import (
    PAGE_A,
    PAGE_B,
    PREMISE_A_TEXT,
    PREMISE_B_TEXT,
    FINDING_A_TEXT,
    QUERY,
    SQ1,
    SQ2,
    URL_A,
    URL_B,
    finding_id,
    fixtures_path,
    page_span_ids,
    premise_id,
    run_ask,
    scripted_adapter,
    transport,
)

MUST_FIND = ["100 degrees Celsius", "re:everest.*70 degrees", "mariana trench"]  # third is a planted miss


class FakeJudge:
    """Judge adapter from a 'different family': returns a fixed score."""

    def __init__(self, score: int = 4):
        self.score = score
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        return ModelResponse(
            id="judge-1",
            model="fake-judge",
            content=[{"type": "text", "text": json.dumps({"synthesis_score": self.score, "rationale": "ok"})}],
            stop_reason="stop",
            usage=Usage(),
        )


def _eval_script(answer: str) -> list:
    """The same orchestrator flow as the recording, with a chosen writer answer."""
    span_a = page_span_ids(PAGE_A)[0]
    span_b = page_span_ids(PAGE_B)[0]
    p_a = premise_id(PREMISE_A_TEXT, span_a)
    p_b = premise_id(PREMISE_B_TEXT, span_b)
    return [
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
              "input": {"subquestions": [SQ1, SQ2]}}], stop_reason="tool_use", index=0),
        scripted_response(
            [{"type": "tool_use", "id": "tu_s1", "name": "search_broad", "input": {"query": SQ1, "k": 5}}],
            stop_reason="tool_use", index=1),
        scripted_response(
            [{"type": "tool_use", "id": "tu_f1", "name": "fetch", "input": {"url": URL_A}}],
            stop_reason="tool_use", index=2),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r1", "name": "record_premises",
              "input": {"premises": [{"text": PREMISE_A_TEXT, "span_refs": [span_a]}]}}],
            stop_reason="tool_use", index=3),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub1", "name": "submit_report",
              "input": {"status": "answered",
                        "findings": [{"text": FINDING_A_TEXT, "premise_ids": [p_a], "edge_type": "deduces"}]}}],
            stop_reason="tool_use", index=4),
        scripted_response(
            [{"type": "tool_use", "id": "tu_s2", "name": "search_broad", "input": {"query": SQ2, "k": 5}}],
            stop_reason="tool_use", index=5),
        scripted_response(
            [{"type": "tool_use", "id": "tu_f2", "name": "fetch", "input": {"url": URL_B}}],
            stop_reason="tool_use", index=6),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r2", "name": "record_premises",
              "input": {"premises": [{"text": PREMISE_B_TEXT, "span_refs": [span_b]}]}}],
            stop_reason="tool_use", index=7),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub2", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use", index=8),
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=9),
    ]


def _answers() -> tuple[str, str]:
    span_a = page_span_ids(PAGE_A)[0]
    span_b = page_span_ids(PAGE_B)[0]
    p_a = premise_id(PREMISE_A_TEXT, span_a)
    p_b = premise_id(PREMISE_B_TEXT, span_b)
    f_a = finding_id(FINDING_A_TEXT, [p_a])
    good = (
        "Here is what the sources say. [narrative]\n"
        f"Water boils at 100 degrees Celsius at sea level. [{f_a}] "
        f"On Mount Everest it boils at about 70 degrees Celsius. [{p_b}]"
    )
    degraded = (
        "Here is what the sources say. [narrative]\n"
        f"On Mount Everest water boils at about 70 degrees Celsius. [{p_b}]"
    )
    return good, degraded


@pytest.fixture
def frozen_case(tmp_path, transport, fixtures_path, scripted_adapter, capsys):
    """Record a real session, then snapshot it into a frozen case with gold."""
    data_dir = tmp_path / "record-data"
    assert run_ask(data_dir, fixtures_path, "recording-session") == 0
    capsys.readouterr()
    case_dir = tmp_path / "cases" / "boiling"
    make_case_from_session(
        data_dir, fixtures_path, case_dir, case_id="boiling", query=QUERY, must_find=MUST_FIND
    )
    return case_dir


def test_m5_exit(tmp_path, frozen_case, transport, capsys):
    good_answer, degraded_answer = _answers()
    judge = FakeJudge(score=4)
    clock = RealClock()

    case = load_case(frozen_case)
    assert case.must_find == MUST_FIND

    # --- version A: full harness behavior on the frozen corpus ---
    calls_before = transport["calls"]
    run_a = asyncio.run(
        run_cases(
            [frozen_case], tmp_path / "work-a",
            lambda config: FakeAdapter(_eval_script(good_answer)),
            clock, "fake-model", label="version-a",
            judge_adapter=judge, judge_model="fake-judge",
        )
    )
    assert transport["calls"] == calls_before  # frozen corpus: zero HTTP
    result_a = run_a.results[0]
    assert result_a.status == "done"
    assert result_a.scores.citation_faithfulness == 1.0   # mechanical axis
    assert abs(result_a.scores.coverage - 2 / 3) < 1e-9   # gold axis: planted miss stays missed
    assert result_a.scores.must_find_misses == ["mariana trench"]
    assert result_a.scores.synthesis == 0.75              # judge axis (advisory)
    assert judge.calls == 1

    # --- version B: same corpus, writer silently drops a covered item ---
    run_b = asyncio.run(
        run_cases(
            [frozen_case], tmp_path / "work-b",
            lambda config: FakeAdapter(_eval_script(degraded_answer)),
            clock, "fake-model", label="version-b",
            judge_adapter=judge, judge_model="fake-judge",
        )
    )
    result_b = run_b.results[0]
    assert abs(result_b.scores.coverage - 1 / 3) < 1e-9

    # --- regression runner: identical corpora, the drop is caught ---
    a_file = tmp_path / "a.json"
    b_file = tmp_path / "b.json"
    a_file.write_text(json.dumps(run_a.to_payload()))
    b_file.write_text(json.dumps(run_b.to_payload()))

    exit_code = cli.main(["eval", "compare", str(a_file), str(b_file), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert out["ok"] is False
    assert [r["axis"] for r in out["regressions"]] == ["coverage"]

    # comparing a run against itself is clean
    exit_code = cli.main(["eval", "compare", str(a_file), str(a_file), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0 and out["ok"] is True


def test_eval_cli_run_and_make_case(tmp_path, frozen_case, transport, monkeypatch, capsys):
    """The CLI path: parsec eval run over a cases root, results file written."""
    good_answer, _ = _answers()
    monkeypatch.setattr(cli, "adapter_factory", lambda config: FakeAdapter(_eval_script(good_answer)))

    out_file = tmp_path / "results" / "a.json"
    exit_code = cli.main(
        ["eval", "run", str(frozen_case.parent), "--out", str(out_file), "--label", "cli-a", "--model", "fake-model"]
    )
    capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(out_file.read_text())
    assert payload["label"] == "cli-a"
    assert payload["aggregate"]["citation_faithfulness"] == 1.0
    assert abs(payload["aggregate"]["coverage"] - round(2 / 3, 4)) < 1e-3
    assert payload["aggregate"]["synthesis"] is None  # judge=none on the CLI run
    assert payload["results"][0]["case_id"] == "boiling"
