"""The three defects the first GAIA question exposed, each as a full run.

The question (GAIA validation, level 1): how many thousand hours would
Eliud Kipchoge need, at his record marathon pace, to run the distance from
the Earth to the Moon at its minimum perigee? Expected answer: 17.

parsec recorded every input (356,400 km, 42.195 km, 2:01:09) and then:

  G9   could not state the figure: nothing after research can compute, a
       subagent may only compute from premises it recorded itself, and the
       operation set has no multiplication and no chaining
  G10  lost every subagent's report: all three hit the turn cap before
       calling submit_report
  G11  reported `done` with 0 of 3 subquestions answered

Each test asserts the CORRECT behaviour and is a strict expected failure
until its fix lands (see tests/regression/test_known_defects.py for the
convention). The fix's PR deletes the marker.

The scripted model here speaks the contracts the fixes will add; these tests
are where those contracts are first written down:

  G9   an orchestrator-stream call, after research and before the writer,
       answering with the tool `submit_computations`
  G10  the subagent's last permitted call carries
       tool_choice = {"type": "tool", "name": "submit_report"}
"""

from __future__ import annotations

import json
import re

import httpx
import pytest

import parsec.cli as cli
from parsec import ids
from parsec.db.connection import open_db
from parsec.gateway.fake_adapter import scripted_response
from parsec.loop.citations import UNVERIFIED_TAG
from parsec.models.gateway import ModelRequest, ModelResponse
from parsec.retrieval.span_indexer import index_spans
from parsec.store.event_log import CURRENT_STREAM

_n = 0


def call(name: str, **tool_input: object) -> ModelResponse:
    global _n
    _n += 1
    block = {"type": "tool_use", "id": f"tu_{_n}", "name": name, "input": tool_input}
    return scripted_response([block], stop_reason="tool_use", index=_n)


def say(text: str) -> ModelResponse:
    global _n
    _n += 1
    return scripted_response([{"type": "text", "text": text}], stop_reason="end_turn", index=_n)


class Script:
    """A scripted model, one list of replies per stream. Every request is
    kept, so a test can assert on what the harness sent. A reply may be a
    function of the request, for a reply that must cite an id the harness
    has only just assigned."""

    def __init__(self, replies: dict[str, list]):
        self.replies = {stream: list(entries) for stream, entries in replies.items()}
        self.requests: dict[str, list[ModelRequest]] = {}

    async def complete(self, request: ModelRequest) -> ModelResponse:
        stream = CURRENT_STREAM.get()
        self.requests.setdefault(stream, []).append(request)
        entries = self.replies.get(stream, [])
        if not entries:
            raise IndexError(f"script exhausted for stream {stream!r}")
        entry = entries.pop(0)
        return entry(request) if callable(entry) else entry


def span_of(page: str) -> str:
    start, end = index_spans(page)[0]
    return ids.span_id(ids.doc_hash(page.encode()), start, end)


def premise_of(page: str) -> str:
    """The id intake gives a premise that states a one-sentence page verbatim."""
    return ids.node_id(
        "Premise", {"text": page, "span_refs": [span_of(page)], "claim_class": "stable"}
    )


def record(page: str) -> ModelResponse:
    return call("record_premises", premises=[{"text": page, "span_refs": [span_of(page)]}])


@pytest.fixture
def ask(tmp_path, monkeypatch, capsys):
    """Run `parsec ask` offline against a scripted model and a fake web."""
    data_dir = tmp_path / "data"

    def _ask(query: str, pages: dict[str, str], script: Script, *extra: str) -> dict:
        def handler(request: httpx.Request) -> httpx.Response:
            page = pages.get(str(request.url))
            if page is None:
                return httpx.Response(404, text="not found")
            return httpx.Response(200, text=page, headers={"content-type": "text/plain"})

        monkeypatch.setattr(cli, "adapter_factory", lambda config: script)
        monkeypatch.setattr(cli, "fetch_transport", httpx.MockTransport(handler))
        fixtures = tmp_path / "queries.json"
        fixtures.write_text("{}")
        code = cli.main(
            [
                "ask", query, "--session-id", "s-gaia", "--adapter", "fake",
                "--model", "fake-model", "--cache-mode", "record", "--data-dir", str(data_dir),
                "--search-provider", "fixture", "--search-fixtures", str(fixtures),
                "--max-turns", "30", "--max-gap-rounds", "0", "--max-coverage-gap-rounds", "0",
                "--json", *extra,
            ]
        )
        result = json.loads(capsys.readouterr().out)
        return {"code": code, "result": result, "conn": open_db(data_dir / "parsec.db")}

    return _ask


def _findings(run: dict) -> list[dict]:
    return [
        json.loads(row["payload_json"])
        for row in run["conn"].execute("SELECT payload_json FROM nodes WHERE tier=2")
    ]


# -- G9: nothing can compute once research ends -------------------------------

URL_MOON = "https://moon.example.gov/orbit"
URL_RACE = "https://athletics.example.gov/marathon"
URL_RECORD = "https://athletics.example.gov/records"
PERIGEE = "The Moon's minimum perigee is 356,400 km."
MARATHON = "A marathon is 42.195 km long."
RECORD = "Eliud Kipchoge set the marathon world record of 2:01:09 in Berlin."
P_PERIGEE, P_MARATHON, P_RECORD = premise_of(PERIGEE), premise_of(MARATHON), premise_of(RECORD)

KIPCHOGE_QUERY = (
    "If Eliud Kipchoge could maintain his record marathon pace indefinitely, how many "
    "thousand hours would it take him to run the minimum perigee distance to the Moon?"
)
COMPUTED = (
    "At Eliud Kipchoge's record marathon pace, covering 356,400 km would take about "
    "17 thousand hours."
)
# 2:01:09 in hours, times the number of marathons in 356,400 km, to the
# nearest thousand hours, in thousands: 17,054.9 h -> 17,000 h -> 17.
KIPCHOGE_CHAIN = {
    "op": "chain",
    "steps": [
        {"op": "convert", "operands": [{"premise_id": P_RECORD, "value": "01"}],
         "unit_from": "min", "unit_to": "h"},
        {"op": "convert", "operands": [{"premise_id": P_RECORD, "value": "09"}],
         "unit_from": "s", "unit_to": "h"},
        {"op": "sum", "operands": [{"premise_id": P_RECORD, "value": "2"}, {"step": 0}, {"step": 1}]},
        {"op": "ratio", "operands": [
            {"premise_id": P_PERIGEE, "value": "356,400"},
            {"premise_id": P_MARATHON, "value": "42.195"},
        ]},
        {"op": "product", "operands": [{"step": 3}, {"step": 2}]},
        {"op": "round", "operands": [{"step": 4}], "precision": "1000"},
        {"op": "ratio", "operands": [{"step": 5}, {"scale": "1000"}]},
    ],
    "precision": "1",
    "output": "17",
}


def _kipchoge_writer(request: ModelRequest) -> ModelResponse:
    """Cites the computed finding if the writer was given one; otherwise
    says what the real run said."""
    given = re.findall(r"\[(finding:[0-9a-f]+)\]", json.dumps(request.messages))
    if given:
        return say(
            "At Eliud Kipchoge's record marathon pace, the run would take about "
            f"17 thousand hours. [{given[0]}]"
        )
    return say("No verified finding carries out that arithmetic. [narrative]")


@pytest.mark.xfail(strict=True, reason="G9: fixed by the compute step (compute_step 1)")
def test_g9_a_figure_is_computed_from_premises_recorded_by_different_subagents(ask):
    """The distance facts and the record time are recorded under different
    subquestions, and no subagent submits a computation. The harness's
    compute step derives 17 from the three premises, the writer cites it,
    and the sentence check accepts it."""
    script = Script(
        {
            "orchestrator": [
                call(
                    "submit_subquestions", effort="standard",
                    subquestions=[
                        "What is the Moon's minimum perigee, and how long is a marathon?",
                        "What is Eliud Kipchoge's marathon world record time?",
                    ],
                ),
                call("submit_computations", computations=[{"text": COMPUTED, "operation": KIPCHOGE_CHAIN}]),
                _kipchoge_writer,
            ],
            "sq-1": [
                call("fetch", url=URL_MOON), record(PERIGEE),
                call("fetch", url=URL_RACE), record(MARATHON),
                call("submit_report", status="answered"),
            ],
            "sq-2": [
                call("fetch", url=URL_RECORD), record(RECORD),
                call("submit_report", status="answered"),
            ],
        }
    )
    run = ask(
        KIPCHOGE_QUERY,
        {URL_MOON: PERIGEE, URL_RACE: MARATHON, URL_RECORD: RECORD},
        script,
        "--max-turns-per-subagent", "6",
    )

    answer = run["result"]["answer"]
    assert "17 thousand hours" in answer and UNVERIFIED_TAG not in answer
    assert run["result"]["status"] == "done" and run["code"] == cli.EXIT_OK
    computed = [f for f in _findings(run) if f["edge_type"] == "computes"]
    assert [(f["status"], f["operation"]["output"]) for f in computed] == [("verified", "17")]
    assert set(computed[0]["premise_ids"]) == {P_PERIGEE, P_MARATHON, P_RECORD}


# -- G10, G11: one page, one subquestion --------------------------------------

URL_BRIDGE = "https://bridges.example.gov/calder"
BRIDGE = "The Calder bridge is painted green."
P_BRIDGE = premise_of(BRIDGE)
BRIDGE_QUERY = "what colour is the Calder bridge, and how long is it"
BRIDGE_ANSWER = (
    f"The Calder bridge is painted green. [{P_BRIDGE}]\n"
    "Its length could not be established. [narrative]"
)
NO_LENGTH = "no source found for the length of the Calder bridge"


@pytest.mark.xfail(strict=True, reason="G10: fixed by the reserved submit turn (subagent_submit 1)")
def test_g10_the_last_subagent_turn_is_reserved_for_the_report(ask):
    """A subagent that spends every turn gathering is made to hand in what
    it has: its last permitted call can only be submit_report, and the
    report's findings reach the graph."""
    finding = {"text": BRIDGE, "premise_ids": [P_BRIDGE], "edge_type": "deduces"}
    script = Script(
        {
            "orchestrator": [
                call("submit_subquestions", effort="standard", subquestions=[BRIDGE_QUERY]),
                say(BRIDGE_ANSWER),
            ],
            "sq-1": [
                call("fetch", url=URL_BRIDGE),
                record(BRIDGE),
                call("fetch", url=URL_BRIDGE),  # still gathering with one turn left
                call("submit_report", status="partial", findings=[finding], dead_ends=[NO_LENGTH]),
            ],
        }
    )
    run = ask(BRIDGE_QUERY, {URL_BRIDGE: BRIDGE}, script, "--max-turns-per-subagent", "4")

    sent = script.requests["sq-1"]
    assert len(sent) == 4  # never more than the cap
    assert [request.tool_choice for request in sent] == [
        None, None, None, {"type": "tool", "name": "submit_report"},
    ]
    assert [f["text"] for f in _findings(run)] == [BRIDGE]
    row = run["conn"].execute("SELECT status, reason FROM coverage WHERE sq_id='sq-1'").fetchone()
    assert row["status"] == "partial"
    # the row says what is still missing, not that the report was lost
    assert NO_LENGTH in (row["reason"] or "")
    assert "before submit_report" not in row["reason"]


@pytest.mark.xfail(strict=True, reason="G11: fixed by the honest run status (run_status 1)")
def test_g11_a_run_with_an_unanswered_subquestion_does_not_end_done(ask):
    """Every sentence of the report passes its checks, but the only
    subquestion was reported partial. The question was not answered, so the
    run is `partial` and the exit code says so."""
    script = Script(
        {
            "orchestrator": [
                call("submit_subquestions", effort="standard", subquestions=[BRIDGE_QUERY]),
                say(BRIDGE_ANSWER),
            ],
            "sq-1": [
                call("fetch", url=URL_BRIDGE),
                record(BRIDGE),
                call("submit_report", status="partial", dead_ends=[NO_LENGTH]),
            ],
        }
    )
    run = ask(BRIDGE_QUERY, {URL_BRIDGE: BRIDGE}, script, "--max-turns-per-subagent", "6")

    assert run["result"]["coverage"] == {"sq-1": "partial"}
    assert run["result"]["unresolved"] == []  # the report itself is clean
    assert run["result"]["status"] == "partial"
    assert run["code"] == cli.EXIT_PARTIAL
