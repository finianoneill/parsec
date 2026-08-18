"""Omission detection (§6 stage 4): the inverted, bottom-up traversal.

Citation checks ask "is every stated claim supported?" — this asks the
question fluent output makes invisible: "what did the report FAIL to say?"

We walk bottom-up from consulted evidence: every document the session
fetched was chosen for a reason (the fetch itself is the relevance signal
at v1 — retrieval scores land with real search providers). A fetched
document none of whose spans has a path to any ReportClaim is a candidate
omission; so is a recorded premise no claim rests on. Both surface in the
report's "consulted but unused" appendix — never silently dropped.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from parsec.models.events import EventType
from parsec.store.event_log import EventLog


@dataclass
class OmissionReport:
    unused_documents: list[dict] = field(default_factory=list)  # {url, doc_hash}
    uncited_premises: list[dict] = field(default_factory=list)  # {node_id, text}

    @property
    def empty(self) -> bool:
        return not self.unused_documents and not self.uncited_premises

    def to_payload(self) -> dict:
        return {
            "unused_documents": self.unused_documents,
            "uncited_premises": self.uncited_premises,
        }


def detect_omissions(
    conn: sqlite3.Connection, event_log: EventLog, session_id: str
) -> OmissionReport:
    nodes: dict[str, dict] = {}
    for row in conn.execute(
        "SELECT node_id, node_type, payload_json FROM nodes WHERE session_id=?",
        (session_id,),
    ):
        nodes[row["node_id"]] = {
            "type": row["node_type"],
            "payload": json.loads(row["payload_json"]),
        }
    out_edges: dict[str, list[str]] = {}
    for row in conn.execute(
        "SELECT src_node_id, dst_node_id, edge_type FROM edges WHERE session_id=?",
        (session_id,),
    ):
        if row["edge_type"] != "contradicts":
            out_edges.setdefault(row["src_node_id"], []).append(row["dst_node_id"])

    # Everything reachable from any ReportClaim, walking support edges down.
    reachable: set[str] = set()
    stack = [nid for nid, n in nodes.items() if n["type"] == "ReportClaim"]
    while stack:
        cur = stack.pop()
        for dst in out_edges.get(cur, []):
            if dst in nodes and dst not in reachable:
                reachable.add(dst)
                stack.append(dst)

    used_doc_hashes = {
        nodes[nid]["payload"]["doc_hash"]
        for nid in reachable
        if nodes[nid]["type"] == "SourceSpan"
    }

    report = OmissionReport()

    # Every fetch this session performed is consulted evidence (v1 relevance
    # signal). Dedup by doc; sort by URL — first-fetch order varies with
    # cross-stream interleaving under concurrent subagents (M11).
    seen: set[str] = set()
    for ev in event_log.read(session_id):
        if ev.event_type != EventType.FETCH_PERFORMED:
            continue
        doc_hash = ev.payload["doc_hash"]
        if doc_hash in seen:
            continue
        seen.add(doc_hash)
        if doc_hash not in used_doc_hashes:
            report.unused_documents.append({"url": ev.payload["url"], "doc_hash": doc_hash})
    report.unused_documents.sort(key=lambda d: d["url"])

    for nid in sorted(nodes):
        node = nodes[nid]
        if node["type"] == "Premise" and nid not in reachable:
            report.uncited_premises.append({"node_id": nid, "text": node["payload"]["text"]})

    return report
