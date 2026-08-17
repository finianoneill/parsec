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
import re
import sqlite3
from dataclasses import dataclass

from parsec.canonical import canonical_json
from parsec.gateway.base import ModelAdapter
from parsec.models.events import EventType
from parsec.models.gateway import ModelRequest
from parsec.store.event_log import EventLog

JUDGE_EDGE_SYSTEM = """You are auditing one derivation step in an evidence graph. You will see numbered premises and one derived statement, nothing else.

Score 1-5: does the derived statement actually follow from these premises alone? 5 = strictly follows; 3 = plausible but requires unstated assumptions; 1 = does not follow or contradicts them. Judge only the step — not whether the premises themselves are true.

Reply with ONLY a JSON object: {"validity_score": <1-5>, "rationale": "<one sentence>"}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class EdgeJudgment:
    finding_id: str
    edge_type: str
    score: float | None  # normalized to [0,1]; None on judge failure
    rationale: str = ""


def _parse(text: str) -> tuple[float | None, str]:
    m = _JSON_RE.search(text)
    if not m:
        return None, ""
    try:
        obj = json.loads(m.group(0))
        score = obj["validity_score"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None, ""
    if not isinstance(score, (int, float)) or not 1 <= score <= 5:
        return None, ""
    return (float(score) - 1.0) / 4.0, str(obj.get("rationale", ""))


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
        try:
            resp = await adapter.complete(request)
            score, rationale = _parse(resp.text)
        except Exception:
            score, rationale = None, ""  # advisory: degrade, never fail

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
