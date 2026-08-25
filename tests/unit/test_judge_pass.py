import json

import pytest

from parsec.models.events import EventType
from parsec.models.gateway import ModelResponse, Usage
from parsec.store.dag import DagStore
from parsec.verify.judge_pass import judge_pass


class ScriptedJudge:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse(
            id="j", model="fake-judge", content=[{"type": "text", "text": self.replies.pop(0)}],
            stop_reason="stop", usage=Usage(),
        )


@pytest.fixture
def graph(db, event_log, sessions, config):
    sessions.create(config)
    sid = config.session_id
    dag = DagStore(db, event_log)
    p1 = dag.add_node(sid, "Premise", {"text": "A was founded in 1990.", "span_refs": ["doc:aaaaaaaaaaaa#0-1"], "claim_class": "stable"})
    p2 = dag.add_node(sid, "Premise", {"text": "B was founded in 2000.", "span_refs": ["doc:aaaaaaaaaaaa#0-1"], "claim_class": "stable"})
    f_ded = dag.add_node(sid, "Finding", {"text": "A predates B.", "premise_ids": [p1, p2], "edge_type": "deduces"})
    dag.add_edge(sid, f_ded, p1, "deduces")
    dag.add_edge(sid, f_ded, p2, "deduces")
    f_tmp = dag.add_node(sid, "Finding", {"text": "Order is temporal.", "premise_ids": [p1], "edge_type": "temporal"})
    dag.add_edge(sid, f_tmp, p1, "temporal")
    return dag, sid, f_ded, f_tmp


async def test_judge_pass_scores_deductions_only(graph, db, event_log):
    dag, sid, f_ded, f_tmp = graph
    judge = ScriptedJudge(['{"validity_score": 5, "rationale": "strictly follows"}'])
    judgments = await judge_pass(db, event_log, sid, judge, "fake-judge")

    assert len(judgments) == 1  # temporal edges are stage-2 territory, not judged
    assert judgments[0].finding_id == f_ded
    assert judgments[0].score == 1.0

    # the judge saw ONLY the local premise set
    body = judge.requests[0].messages[0]["content"]
    assert "A was founded in 1990." in body and "A predates B." in body
    assert "Order is temporal." not in body

    # advisory weight stored on the derivation edges
    rows = db.execute(
        "SELECT payload_json FROM edges WHERE session_id=? AND src_node_id=?", (sid, f_ded)
    ).fetchall()
    assert all(json.loads(r["payload_json"])["judge_score"] == 1.0 for r in rows)

    events = [e for e in event_log.read(sid) if e.event_type == EventType.JUDGE_SCORED]
    assert len(events) == 1 and events[0].payload["score"] == 1.0


async def test_judge_failure_degrades_to_none(graph, db, event_log):
    dag, sid, f_ded, _ = graph
    # bad reply, then the corrective retry (Phase 3) also fails -> None
    judge = ScriptedJudge(["not json at all", "still not json"])
    judgments = await judge_pass(db, event_log, sid, judge, "fake-judge")
    assert judgments[0].score is None
    assert len(judge.requests) == 2  # exactly one corrective retry
    assert "invalid" in judge.requests[1].messages[-1]["content"]
    rows = db.execute(
        "SELECT payload_json FROM edges WHERE session_id=? AND src_node_id=?", (sid, f_ded)
    ).fetchall()
    assert all("judge_score" not in json.loads(r["payload_json"]) for r in rows)


async def test_judge_retry_recovers_a_malformed_reply(graph, db, event_log):
    dag, sid, f_ded, _ = graph
    judge = ScriptedJudge(
        ["oops, no json here", '{"validity_score": 3, "rationale": "assumes an unstated step"}']
    )
    judgments = await judge_pass(db, event_log, sid, judge, "fake-judge")
    assert judgments[0].score == 0.5  # (3-1)/4 — the retry landed
    assert judgments[0].rationale == "assumes an unstated step"
