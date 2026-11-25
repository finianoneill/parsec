"""WS3 end to end: a source says 21.9 km, the subagent records a conversion
to miles, the writer cites it — and every check downstream agrees with every
other, where before the eval grader rejected the correct figure and the
harness accepted any figure at all."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import parsec.cli as cli
from parsec import ids
from parsec.config import RealClock
from parsec.db.connection import open_db
from parsec.evals.support import MechanicalSupportChecker, score_claim_support
from parsec.gateway.fake_adapter import StreamFakeAdapter, scripted_response
from parsec.loop.citations import UNVERIFIED_TAG
from parsec.replay import run_replay
from parsec.retrieval.span_indexer import index_spans
from parsec.store.blobs import BlobStore

URL = "https://geo.example.gov/olympus"
PAGE = "Olympus Mons is 21.9 kilometers tall."
QUERY = "how tall is olympus mons in miles"

_n = 0


def call(name: str, **tool_input) -> object:
    global _n
    _n += 1
    block = {"type": "tool_use", "id": f"tu_{_n}", "name": name, "input": tool_input}
    return scripted_response([block], stop_reason="tool_use", index=_n)


def say(text: str) -> object:
    global _n
    _n += 1
    return scripted_response([{"type": "text", "text": text}], stop_reason="end_turn", index=_n)


def _span() -> str:
    start, end = index_spans(PAGE)[0]
    return ids.span_id(ids.doc_hash(PAGE.encode()), start, end)


P_HEIGHT = ids.node_id("Premise", {"text": PAGE, "span_refs": [_span()], "claim_class": "stable"})


def convert(output: str) -> dict:
    return {
        "op": "convert", "operands": [{"premise_id": P_HEIGHT, "value": "21.9"}],
        "unit_from": "km", "unit_to": "mi", "precision": "0.1", "output": output,
    }


def finding(text: str, operation: dict, status: str | None) -> tuple[dict, str]:
    """(the draft as submitted, the id the harness will give its node)."""
    draft = {"text": text, "premise_ids": [P_HEIGHT], "edge_type": "computes", "operation": operation}
    payload = {**draft, "status": status}
    return draft, ids.node_id("Finding", payload)


@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"

    def _run(submissions: list[list[dict]], drafts: list[str]) -> dict:
        scripts = {
            "orchestrator": [
                call("submit_subquestions", effort="standard", subquestions=[QUERY]),
                *(say(d) for d in drafts),
            ],
            "sq-1": [
                call("fetch", url=URL),
                call("record_premises", premises=[{"text": PAGE, "span_refs": [_span()]}]),
                *(call("submit_report", status="answered", findings=f) for f in submissions),
            ],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == URL:
                return httpx.Response(200, text=PAGE, headers={"content-type": "text/plain"})
            return httpx.Response(404, text="not found")

        monkeypatch.setattr(cli, "adapter_factory", lambda config: StreamFakeAdapter(scripts))
        monkeypatch.setattr(cli, "fetch_transport", httpx.MockTransport(handler))
        fixtures = tmp_path / "queries.json"
        fixtures.write_text("{}")
        code = cli.main(
            [
                "ask", QUERY, "--session-id", "s-ws3", "--adapter", "fake", "--model", "fake-model",
                "--cache-mode", "record", "--data-dir", str(data_dir),
                "--search-provider", "fixture", "--search-fixtures", str(fixtures),
                "--max-turns", "20", "--max-turns-per-subagent", "6",
                "--max-gap-rounds", "0", "--max-coverage-gap-rounds", "0", "--json",
            ]
        )
        result = json.loads(capsys.readouterr().out)
        conn = open_db(data_dir / "parsec.db")
        return {"code": code, "result": result, "conn": conn, "data_dir": data_dir,
                "blobs": BlobStore(data_dir / "blobs")}

    return _run


def _verify(run: dict, capsys) -> tuple[int, dict]:
    code = cli.main(["verify", "s-ws3", "--data-dir", str(run["data_dir"]), "--json"])
    return code, json.loads(capsys.readouterr().out)


def _findings(run: dict) -> list[dict]:
    return [
        json.loads(r["payload_json"])
        for r in run["conn"].execute("SELECT payload_json FROM nodes WHERE tier=2")
    ]


def test_a_conversion_is_recomputed_cited_and_supported_end_to_end(run, capsys):
    text = "Olympus Mons is about 13.6 mi tall."
    draft, fid = finding(text, convert("13.6"), "verified")
    answer = f"Olympus Mons is 21.9 kilometers tall. [{P_HEIGHT}]\nThat is about 13.6 miles. [{fid}]"
    outcome = run([[draft]], [answer])

    assert outcome["result"]["status"] == "done" and outcome["code"] == cli.EXIT_OK
    assert UNVERIFIED_TAG not in outcome["result"]["answer"]  # WS0 accepts the derived figure
    stored = _findings(outcome)
    assert [(f["edge_type"], f["status"]) for f in stored] == [("computes", "verified")]
    assert stored[0]["operation"]["output"] == "13.6"
    edge_types = {r[0] for r in outcome["conn"].execute("SELECT edge_type FROM edges")}
    assert "computes" in edge_types

    code, report = _verify(outcome, capsys)
    assert code == 0 and report["violations"] == [] and report["advisories"] == []
    # the verified figure is as credible as the premise it was computed from
    assert report["credence"]["tiers"][fid] == report["credence"]["tiers"][P_HEIGHT]

    # eval grading: before this change the correct conversion graded "none",
    # because 13.6 appears in no span
    _, details = score_claim_support(outcome["conn"], "s-ws3", MechanicalSupportChecker())
    grades = {d.claim_text: d.grade for d in details}
    assert grades["That is about 13.6 miles."] != "none"

    replay = asyncio.run(run_replay(outcome["conn"], outcome["blobs"], RealClock(), "s-ws3"))
    assert replay.verified, replay.first_divergence


def test_a_wrong_conversion_comes_back_with_the_expected_value(run, capsys):
    wrong, _ = finding("Olympus Mons is about 14.2 mi tall.", convert("14.2"), None)
    right, fid = finding("Olympus Mons is about 13.6 mi tall.", convert("13.6"), "verified")
    answer = f"Olympus Mons is about 13.6 miles tall. [{fid}]"
    outcome = run([[wrong], [right]], [answer])  # one round trip fixes it

    assert outcome["result"]["status"] == "done"
    assert [f["operation"]["output"] for f in _findings(outcome)] == ["13.6"]
    # the correction the subagent was shown
    requests = [
        outcome["blobs"].get_text(json.loads(r["payload_json"])["request_blob"])
        for r in outcome["conn"].execute(
            "SELECT payload_json FROM events WHERE event_type='llm_request' AND stream_id='sq-1'"
            " ORDER BY stream_idx"
        )
    ]
    assert "21.9 km is 13.61 mi; the finding says 14.2" in requests[-1]
    assert _verify(outcome, capsys)[0] == 0


def test_a_free_text_derivation_is_unverified_flagged_and_shown_as_such(run, capsys):
    operation = {
        "op": "other", "operands": [{"premise_id": P_HEIGHT, "value": "21.9"}],
        "precision": "1", "output": "2.5", "description": "relative to Mount Everest, from memory",
    }
    draft, fid = finding("Olympus Mons is about 2.5 times the height of Everest.", operation, "unverified")
    answer = (
        f"Olympus Mons is 21.9 kilometers tall. [{P_HEIGHT}]\n"
        f"That is about 2.5 times the height of Everest. [{fid}]"
    )
    outcome = run([[draft]], [answer, answer])  # the repair round does not fix it

    result = outcome["result"]
    assert result["status"] == "partial"
    body, appendix = result["answer"].split("--- end of answer ---")
    lines = body.splitlines()
    assert UNVERIFIED_TAG not in lines[0]
    assert lines[1].endswith(UNVERIFIED_TAG)  # rendered as an unverified figure
    assert "unverified derivation" in appendix
    assert any("unverified derivation" in line for line in result["low_confidence"])  # flagged
    assert [f["status"] for f in _findings(outcome)] == ["unverified"]

    code, report = _verify(outcome, capsys)
    assert code != 0  # the sentence stating the figure is a sentence-support violation
    assert {v["check"] for v in report["violations"]} == {"sentence-support"}
    assert [a["check"] for a in report["advisories"]] == ["unverified-derivation"]
    assert report["credence"]["tiers"][fid] == "low"


def test_an_ordinary_finding_cannot_carry_a_figure_its_premises_lack(run, capsys):
    laundered = {
        "text": "Olympus Mons is about 13.6 miles tall.", "premise_ids": [P_HEIGHT],
        "edge_type": "deduces",
    }
    honest = {"text": "Olympus Mons is 21.9 kilometers tall.", "premise_ids": [P_HEIGHT],
              "edge_type": "deduces"}
    answer = f"Olympus Mons is 21.9 kilometers tall. [{P_HEIGHT}]"
    outcome = run([[laundered], [honest]], [answer])
    assert outcome["result"]["status"] == "done"
    assert [f["text"] for f in _findings(outcome)] == [honest["text"]]
    request = outcome["blobs"].get_text(
        json.loads(
            outcome["conn"].execute(
                "SELECT payload_json FROM events WHERE event_type='llm_request'"
                " AND stream_id='sq-1' ORDER BY stream_idx DESC LIMIT 1"
            ).fetchone()["payload_json"]
        )["request_blob"]
    )
    assert "number '13.6' in the finding is not in the premises it cites" in request
