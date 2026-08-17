"""M8 exit tests (v2 plan): a deliberately-degraded harness variant is
flagged "regressed" with correct CIs on the right axis; an unchanged
variant reads inconclusive; a case with hard negatives separates a good
retriever from a lucky one — via nugget rubrics, contradiction checks,
claim-support grading, and trajectory metrics.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import parsec.cli as cli
from parsec import ids
from parsec.config import RealClock
from parsec.evals.case import EvalCase, Nugget, make_case_from_session, save_case
from parsec.evals.runner import run_cases
from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
from parsec.retrieval.extract import extract_text
from parsec.retrieval.span_indexer import index_spans

QUERY = "what temperature does water boil at sea level and on everest"
SQ = "boiling point of water"

GOLD_PAGE_A = (
    "<html><head><title>Reference Tables</title></head><body>"
    "<article><p>Water boils at 100 degrees Celsius at standard atmospheric pressure at "
    "sea level, the reference value maintained by national metrology institutes and used "
    "to calibrate laboratory thermometers across every scientific discipline.</p></article>"
    "</body></html>"
).encode()
GOLD_PAGE_B = (
    "<html><head><title>Altitude Physics</title></head><body>"
    "<article><p>On the summit of Mount Everest water boils at about 70 degrees Celsius "
    "because atmospheric pressure at that extreme altitude is far below the sea-level "
    "standard, a canonical example of pressure-dependent phase transitions.</p></article>"
    "</body></html>"
).encode()
DISTRACTOR_PAGE = (
    "<html><head><title>Kitchen Myths Blog</title></head><body>"
    "<article><p>In my own kitchen tests water boils at 90 degrees Celsius on a normal "
    "stove, whatever the textbooks claim, and I have repeated the measurement many times "
    "with two different thermometers over several enthusiastic weekend afternoons.</p></article>"
    "</body></html>"
).encode()

URL_A = "https://reference.example/tables"
URL_B = "https://reference.example/altitude"
URL_BAD = "https://kitchenmyths.example/blog"

PREMISE_A = "Water boils at 100 degrees Celsius at standard atmospheric pressure at sea level."
PREMISE_B = "On the summit of Mount Everest water boils at about 70 degrees Celsius."
PREMISE_BAD = "Water boils at 90 degrees Celsius on a normal stove."

NUGGETS = [
    Nugget(
        text="sea-level boiling point is 100 degrees Celsius",
        weight="vital",
        patterns=["100 degrees Celsius"],
        contradiction_patterns=[r"re:9\d degrees celsius"],
    ),
    Nugget(
        text="everest boiling point is about 70 degrees Celsius",
        weight="okay",
        patterns=[r"re:everest.*70 degrees"],
    ),
]


def first_span(page: bytes) -> str:
    text, _, _ = extract_text(page, "text/html")
    h = ids.doc_hash(page)
    s, e = index_spans(text)[0]
    return ids.span_id(h, s, e)


def premise_id(text: str, span_ref: str) -> str:
    return ids.node_id("Premise", {"text": text, "span_refs": [span_ref], "claim_class": "stable"})


@pytest.fixture
def transport(monkeypatch):
    pages = {URL_A: GOLD_PAGE_A, URL_B: GOLD_PAGE_B, URL_BAD: DISTRACTOR_PAGE}
    counter = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["calls"] += 1
        page = pages.get(str(request.url).rstrip("/"))
        if page is None:
            return httpx.Response(404, content=b"not found")
        return httpx.Response(200, content=page, headers={"content-type": "text/html"})

    monkeypatch.setattr(cli, "fetch_transport", httpx.MockTransport(handler))
    return counter


@pytest.fixture
def fixtures_path(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps(
            {
                SQ: [
                    {"title": "Reference Tables", "url": URL_A, "snippet": "100C"},
                    {"title": "Altitude Physics", "url": URL_B, "snippet": "70C"},
                    {"title": "Kitchen Myths Blog", "url": URL_BAD, "snippet": "90C??"},
                ]
            }
        )
    )
    return path


def _flow(fetch_urls: list[str], premises: list[tuple[str, str]], answer: str) -> list:
    """decompose -> search -> fetch(es) -> record -> submit -> write."""
    responses = [
        scripted_response(
            [{"type": "tool_use", "id": "tu_dec", "name": "submit_subquestions",
              "input": {"subquestions": [SQ]}}], stop_reason="tool_use", index=0),
        scripted_response(
            [{"type": "tool_use", "id": "tu_s", "name": "search_broad",
              "input": {"query": SQ, "k": 5}}], stop_reason="tool_use", index=1),
        scripted_response(
            [{"type": "tool_use", "id": f"tu_f{i}", "name": "fetch", "input": {"url": u}}
             for i, u in enumerate(fetch_urls)],
            stop_reason="tool_use", index=2),
        scripted_response(
            [{"type": "tool_use", "id": "tu_r", "name": "record_premises",
              "input": {"premises": [{"text": t, "span_refs": [s]} for t, s in premises]}}],
            stop_reason="tool_use", index=3),
        scripted_response(
            [{"type": "tool_use", "id": "tu_sub", "name": "submit_report",
              "input": {"status": "answered"}}], stop_reason="tool_use", index=4),
        scripted_response([{"type": "text", "text": answer}], stop_reason="end_turn", index=5),
    ]
    return responses


def _good_script():
    sa, sb = first_span(GOLD_PAGE_A), first_span(GOLD_PAGE_B)
    pa, pb = premise_id(PREMISE_A, sa), premise_id(PREMISE_B, sb)
    answer = (
        "The reference values are clear. [narrative]\n"
        f"Water boils at 100 degrees Celsius at sea level. [{pa}] "
        f"On the summit of Mount Everest it boils at about 70 degrees Celsius. [{pb}]"
    )
    return _flow([URL_A, URL_B], [(PREMISE_A, sa), (PREMISE_B, sb)], answer)


def _lucky_script():
    sbad = first_span(DISTRACTOR_PAGE)
    pbad = premise_id(PREMISE_BAD, sbad)
    answer = (
        "One source had a clear answer. [narrative]\n"
        f"Water boils at 90 degrees Celsius on a normal stove. [{pbad}]"
    )
    return _flow([URL_BAD], [(PREMISE_BAD, sbad)], answer)


@pytest.fixture
def case_dirs(tmp_path, transport, fixtures_path, monkeypatch, capsys):
    """Record a corpus containing gold AND distractor pages, then snapshot it
    into two identical cases (n=2 gives the paired stats a variance estimate)."""
    sa, sb = first_span(GOLD_PAGE_A), first_span(GOLD_PAGE_B)
    record_script = _flow(
        [URL_A, URL_B, URL_BAD],  # corpus must contain the distractor too
        [(PREMISE_A, sa), (PREMISE_B, sb)],
        "Recorded corpus. [narrative]",
    )
    monkeypatch.setattr(cli, "adapter_factory", lambda config: FakeAdapter(record_script))
    data_dir = tmp_path / "record-data"
    exit_code = cli.main(
        [
            "ask", QUERY, "--session-id", "recording", "--adapter", "fake",
            "--model", "fake-model", "--cache-mode", "record",
            "--data-dir", str(data_dir), "--search-fixtures", str(fixtures_path),
            "--max-gap-rounds", "0", "--json",
        ]
    )
    capsys.readouterr()
    assert exit_code in (0, 3)  # narrative-only answer may score partial; corpus is what matters

    dirs = []
    for i in (1, 2):
        case_dir = tmp_path / "cases" / f"boiling-{i}"
        make_case_from_session(data_dir, fixtures_path, case_dir, case_id=f"boiling-{i}", query=QUERY)
        save_case(
            case_dir,
            EvalCase(
                case_id=f"boiling-{i}", query=QUERY, nuggets=NUGGETS,
                gold_docs=[URL_A, URL_B], distractor_docs=[URL_BAD],
            ),
        )
        dirs.append(case_dir)
    return dirs


def _run(case_dirs, workdir, script_factory, label):
    return asyncio.run(
        run_cases(
            case_dirs, workdir,
            lambda config: FakeAdapter(script_factory()),
            RealClock(), "fake-model", label=label,
        )
    )


def test_m8_exit(tmp_path, case_dirs, transport, capsys):
    # --- good retriever: fetches gold, covers both nuggets ---
    run_good = _run(case_dirs, tmp_path / "work-good", _good_script, "good")
    for r in run_good.results:
        assert r.status == "done"
        assert r.scores.nugget_recall == 1.0
        assert r.scores.claim_support == 1.0
        assert r.scores.nugget_contradictions == []
        assert r.trajectory.gold_fetch_fraction == 1.0
        assert r.trajectory.distractor_fetch_fraction == 0.0
        assert r.trajectory.fetches_to_first_gold == 1

    # --- lucky retriever: fetches only the planted distractor ---
    run_lucky = _run(case_dirs, tmp_path / "work-lucky", _lucky_script, "lucky")
    for r in run_lucky.results:
        assert r.status == "done"
        # its single claim IS supported by its (bad) source — support alone can't catch this
        assert r.scores.claim_support == 1.0
        # ...but the gold rubric can: contradiction fired, recall zero
        assert r.scores.nugget_recall == 0.0
        assert r.scores.nugget_contradictions == ["sea-level boiling point is 100 degrees Celsius"]
        assert r.trajectory.gold_fetch_fraction == 0.0
        assert r.trajectory.distractor_fetch_fraction == 1.0
        assert r.trajectory.fetches_to_first_gold is None

    # --- paired-difference regression: right axis, correct CI, exit 3 ---
    a_file, b_file = tmp_path / "a.json", tmp_path / "b.json"
    a_file.write_text(json.dumps(run_good.to_payload()))
    b_file.write_text(json.dumps(run_lucky.to_payload()))

    exit_code = cli.main(["eval", "compare", str(a_file), str(b_file), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert out["ok"] is False
    verdicts = {v["axis"]: v for v in out["verdicts"]}
    assert verdicts["nugget_recall"]["verdict"] == "regressed"
    assert verdicts["nugget_recall"]["n_cases"] == 2
    assert verdicts["nugget_recall"]["mean_delta"] == -1.0
    assert verdicts["nugget_recall"]["ci95"] == 0.0  # identical paired deltas
    assert verdicts["claim_support"]["verdict"] == "inconclusive"  # support did NOT regress
    assert verdicts["citation_faithfulness"]["verdict"] == "inconclusive"

    # --- unchanged variant reads inconclusive / no change, exit 0 ---
    exit_code = cli.main(["eval", "compare", str(a_file), str(a_file), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0 and out["ok"] is True
    assert all(v["verdict"] == "inconclusive" for v in out["verdicts"])


def test_multi_run_mean_aggregation(tmp_path, case_dirs, transport):
    run = asyncio.run(
        run_cases(
            case_dirs[:1], tmp_path / "work-runs",
            lambda config: FakeAdapter(_good_script()),
            RealClock(), "fake-model", label="runs3", runs=3,
        )
    )
    result = run.results[0]
    assert result.runs == 3
    assert len(result.per_run_scores) == 3
    assert result.scores.nugget_recall == 1.0  # mean of three identical deterministic runs