"""Judge pass (§6 stage 5): last, least trusted, advisory only.

A DIFFERENT model family than the generator scores derivation quality on
`deduces`/`induces` edges only, seeing ONLY the local premise set — no
extra context, no ability to grade its own homework. Scores are stored as
advisory weights on the finding's derivation edges and emitted as events;
they gate nothing (stages 1–4 need no judgment at all, which is the real
defense — §10.6).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from parsec.canonical import canonical_json
from parsec.gateway.base import ModelAdapter
from parsec.loop.structured import judged_json
from parsec.models.events import EventType
from parsec.models.gateway import ModelRequest
from parsec.store.event_log import EventLog

JUDGE_EDGE_SYSTEM = """You are auditing one derivation step in an evidence graph. You will see numbered premises and one derived statement, nothing else.

Score 1-5: does the derived statement actually follow from these premises alone? 5 = strictly follows; 3 = plausible but requires unstated assumptions; 1 = does not follow or contradicts them. Judge only the step — not whether the premises themselves are true.

Reply with ONLY a JSON object: {"validity_score": <1-5>, "rationale": "<one sentence>"}"""

_RETRY_INSTRUCTION = (
    'Reply with ONLY the JSON object: {"validity_score": <1-5>, "rationale": "<one sentence>"}'
)


class EdgeJudgeReply(BaseModel):
    """The judge's contract, validated instead of regex-scraped (Phase 3);
    one corrective retry before the advisory score degrades to None."""

    model_config = ConfigDict(extra="ignore")

    validity_score: float = Field(ge=1, le=5)
    rationale: str = ""


@dataclass
class EdgeJudgment:
    finding_id: str
    edge_type: str
    score: float | None  # normalized to [0,1]; None on judge failure
    rationale: str = ""


async def judge_pass(
    conn: sqlite3.Connection,
    event_log: EventLog,
    session_id: str,
    adapter: ModelAdapter,
    judge_model: str,
) -> list[EdgeJudgment]:
    payloads: dict[str, dict] = {}
    for row in conn.execute(
        "SELECT node_id, payload_json FROM nodes WHERE session_id=?", (session_id,)
    ):
        payloads[row["node_id"]] = json.loads(row["payload_json"])

    judgments: list[EdgeJudgment] = []
    findings = sorted(
        (nid, p) for nid, p in payloads.items()
        if nid.startswith("finding:") and p.get("edge_type") in ("deduces", "induces")
    )
    for finding_id, payload in findings:
        premises = [
            payloads[pid]["text"] for pid in payload["premise_ids"] if pid in payloads
        ]
        prompt_lines = [f"{i + 1}. {t}" for i, t in enumerate(premises)]
        prompt_lines += ["", f"Derived statement ({payload['edge_type']}): {payload['text']}"]
        request = ModelRequest(
            model=judge_model,
            max_tokens=300,
            system=[{"type": "text", "text": JUDGE_EDGE_SYSTEM}],
            messages=[{"role": "user", "content": "\n".join(prompt_lines)}],
        )
        reply = await judged_json(adapter, request, EdgeJudgeReply, _RETRY_INSTRUCTION)
        if reply is None:
            score, rationale = None, ""  # advisory: degrade, never fail
        else:
            score, rationale = (float(reply.validity_score) - 1.0) / 4.0, reply.rationale

        judgment = EdgeJudgment(finding_id, payload["edge_type"], score, rationale)
        judgments.append(judgment)
        event_log.append(
            session_id,
            "judge",
            EventType.JUDGE_SCORED,
            {"finding_id": finding_id, "edge_type": payload["edge_type"],
             "score": score, "rationale": rationale},
        )
        if score is not None:
            # store the advisory weight on the finding's derivation edges
            for row in conn.execute(
                "SELECT edge_id, payload_json FROM edges WHERE session_id=? AND src_node_id=?",
                (session_id, finding_id),
            ):
                merged = json.loads(row["payload_json"] or "{}")
                merged["judge_score"] = score
                conn.execute(
                    "UPDATE edges SET payload_json=? WHERE session_id=? AND edge_id=?",
                    (canonical_json(merged), session_id, row["edge_id"]),
                )
    return judgments
