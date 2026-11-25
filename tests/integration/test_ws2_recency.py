"""WS2 end to end: the same 2019 observation, asked about today and asked
about 2019, in one run — and a refresh a week later that scores the new
run on its own observations."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

import parsec.cli as cli
from parsec import ids
from parsec.config import RealClock
from parsec.db.connection import open_db
from parsec.gateway.fake_adapter import StreamFakeAdapter, scripted_response
from parsec.replay import run_replay
from parsec.retrieval.span_indexer import index_spans
from parsec.store.blobs import BlobStore
from parsec.verify.credence import compute_credences

URL = "https://market.example/price"
Y2019 = "2019-01-01/2019-12-31"

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


def span_of(page: str) -> str:
    start, end = index_spans(page)[0]
    return ids.span_id(ids.doc_hash(page.encode()), start, end)


def premise_id(page: str, target: str | None = None) -> str:
    payload = {"text": page, "span_refs": [span_of(page)], "claim_class": "volatile"}
    if target:
        payload["target_time"] = target
    return ids.node_id("Premise", payload)


class SettableClock(RealClock):
    """Real monotonic time, a wall-clock date the test chooses."""

    today = "2026-06-01"

    def now_iso(self) -> str:
        return f"{self.today}T09:00:00.000+00:00"


@pytest.fixture
def harness(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    pages: dict[str, str] = {}
    fixtures = tmp_path / "queries.json"
    fixtures.write_text("{}")

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) in pages:
            return httpx.Response(
                200, text=pages[str(request.url)], headers={"content-type": "text/plain"}
            )
        return httpx.Response(404, text="not found")

    monkeypatch.setattr(cli, "fetch_transport", httpx.MockTransport(handler))
    monkeypatch.setattr(cli, "RealClock", SettableClock)

    def run(argv: list[str], scripts: dict) -> dict:
        monkeypatch.setattr(cli, "adapter_factory", lambda config: StreamFakeAdapter(scripts))
        code = cli.main([*argv, "--data-dir", str(data_dir), "--json"])
        return {"code": code, "result": json.loads(capsys.readouterr().out)}

    def ask(query: str, session_id: str, scripts: dict) -> dict:
        return run(
            [
                "ask", query, "--session-id", session_id, "--adapter", "fake",
                "--model", "fake-model", "--cache-mode", "record",
                "--search-provider", "fixture", "--search-fixtures", str(fixtures),
                "--max-turns", "30", "--max-gap-rounds", "0", "--max-coverage-gap-rounds", "0",
            ],
            scripts,
        )

    return {"pages": pages, "ask": ask, "run": run, "data_dir": data_dir}


def _credences(conn, session_id: str) -> dict[str, float]:
    return {
        r["node_id"]: r["credence"]
        for r in conn.execute("SELECT node_id, credence FROM nodes WHERE session_id=?", (session_id,))
    }


def test_one_observation_two_time_frames(harness):
    page = "The listed price was 10 dollars in 2019."
    harness["pages"][URL] = page
    p_now, p_2019 = premise_id(page), premise_id(page, Y2019)
    record = {"text": page, "span_refs": [span_of(page)], "claim_class": "volatile", "as_of": "2019"}
    subagent = [
        call("fetch", url=URL),
        call("record_premises", premises=[record]),
        call("submit_report", status="answered"),
    ]
    scripts = {
        "orchestrator": [
            call("submit_subquestions", effort="standard",
                 subquestions=["what is the listed price today", "what was the listed price in 2019"],
                 target_times=["now", "in 2019"]),
            say(
                f"One older listing gives a price of 10 dollars. [{p_now}]\n"
                f"In 2019 the listed price was 10 dollars. [{p_2019}]"
            ),
        ],
        "sq-1": list(subagent),
        "sq-2": list(subagent),
    }
    outcome = harness["ask"]("the listed price, today and in 2019", "s-paired", scripts)
    assert outcome["result"]["status"] == "done"

    conn = open_db(harness["data_dir"] / "parsec.db")
    blobs = BlobStore(harness["data_dir"] / "blobs")
    credence = _credences(conn, "s-paired")
    # the same observation: full strength for the year it was made in...
    assert credence[p_2019] == pytest.approx(0.6 * 0.85)
    # ...and worth almost nothing as an answer to "what is it today?"
    assert credence[p_now] < 1e-6

    # the plan recorded what time each subquestion asks about, and told the subagent
    planned = json.loads(
        conn.execute(
            "SELECT payload_json FROM events WHERE session_id='s-paired'"
            " AND event_type='subquestions_planned'"
        ).fetchone()["payload_json"]
    )["subquestions"]
    assert [s["target_time"] for s in planned] == ["now", Y2019]
    first_prompts = {}
    for row in conn.execute(
        "SELECT stream_id, payload_json FROM events WHERE session_id='s-paired'"
        " AND event_type='llm_request' ORDER BY idx"
    ):
        first_prompts.setdefault(
            row["stream_id"], blobs.get_text(json.loads(row["payload_json"])["request_blob"])
        )
    assert "Time frame: the present" in first_prompts["sq-1"]
    assert "Time frame: 2019-01-01 to 2019-12-31" in first_prompts["sq-2"]

    assert cli.main(["verify", "s-paired", "--data-dir", str(harness["data_dir"])]) == 0
    replay = asyncio.run(run_replay(conn, blobs, RealClock(), "s-paired"))
    assert replay.verified, replay.first_divergence


def test_a_refresh_a_week_later_scores_its_own_observations(harness):
    before = "On 2026-06-01 the listed price is 10 dollars."
    after = "On 2026-06-08 the listed price is 12 dollars."
    question = "what is the listed price"

    def scripts(page: str, with_plan: bool) -> dict:
        pid = premise_id(page)
        day = page.split()[1]
        return {
            "orchestrator": [
                *(
                    [call("submit_subquestions", effort="quick", subquestions=[question],
                          target_times=["now"])]
                    if with_plan
                    else []
                ),
                say(f"The listing of {day} gives the price. [{pid}]"),
            ],
            "sq-1": [
                call("fetch", url=URL),
                call("record_premises", premises=[
                    {"text": page, "span_refs": [span_of(page)], "claim_class": "volatile",
                     "as_of": day}]),
                call("submit_report", status="answered"),
            ],
        }

    harness["pages"][URL] = before
    SettableClock.today = "2026-06-01"
    assert harness["ask"](question, "s-parent", scripts(before, True))["result"]["status"] == "done"

    # a week passes; the page changes
    harness["pages"][URL] = after
    SettableClock.today = "2026-06-08"
    time.sleep(0.01)
    refreshed = harness["run"](["refresh", "s-parent"], scripts(after, False))["result"]
    child = refreshed["session_id"]
    assert refreshed["status"] == "done"

    conn = open_db(harness["data_dir"] / "parsec.db")
    config = json.loads(
        conn.execute("SELECT config_json FROM sessions WHERE session_id=?", (child,)).fetchone()[0]
    )
    assert config["as_of_ts"].startswith("2026-06-08")  # its own "now", not the parent's
    seeded = json.loads(
        conn.execute(
            "SELECT payload_json FROM events WHERE session_id=? AND event_type='refresh_seeded'",
            (child,),
        ).fetchone()["payload_json"]
    )
    assert [s["sq_id"] for s in seeded["researched"]] == ["sq-1"] and seeded["carried"] == []

    new_premise = premise_id(after)
    edge = json.loads(
        conn.execute(
            "SELECT payload_json FROM edges WHERE session_id=? AND src_node_id=?",
            (child, new_premise),
        ).fetchone()["payload_json"]
    )
    assert edge == {"observed_at": "2026-06-08/2026-06-08", "date_source": "as_of"}
    # observed on the day it is asked about: no decay
    assert _credences(conn, child)[new_premise] == pytest.approx(0.6 * 0.85)
    # the parent's week-old reading, judged from the refresh's "now", has decayed
    stale = compute_credences(
        conn, "s-parent", recency=2, as_of_ts=config["as_of_ts"], persist=False
    ).nodes[premise_id(before)]
    assert stale.credence == pytest.approx(0.6 * 0.85 * 0.5 ** (7 / 30)) and stale.stale
