"""Evidence DAG persistence. At M1 only tiers 0 (SourceSpan) and 4
(ReportClaim) are written, linked directly by `extracts` edges — a
documented collapsed chain until M2/M3 insert tiers 1–3."""

from __future__ import annotations

import sqlite3

from parsec import ids
from parsec.canonical import canonical_json
from parsec.models.events import EventType
from parsec.models.nodes import NODE_TIERS
from parsec.store.event_log import EventLog


class DagStore:
    def __init__(self, conn: sqlite3.Connection, event_log: EventLog):
        self.conn = conn
        self.event_log = event_log

    def add_node(self, session_id: str, node_type: str, payload: dict) -> str:
        node_id = ids.node_id(node_type, payload)
        tier = NODE_TIERS[node_type]
        seq = self.event_log.append(
            session_id,
            "harness",
            EventType.NODE_ADDED,
            {"node_id": node_id, "node_type": node_type, "tier": tier, "payload": payload},
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO nodes (node_id, session_id, tier, node_type, payload_json, created_seq)"
            " VALUES (?,?,?,?,?,?)",
            (node_id, session_id, tier, node_type, canonical_json(payload), seq),
        )
        return node_id

    def add_edge(self, session_id: str, src_node_id: str, dst_node_id: str, edge_type: str, payload: dict | None = None) -> str:
        edge_id = ids.edge_id(src_node_id, dst_node_id, edge_type)
        seq = self.event_log.append(
            session_id,
            "harness",
            EventType.EDGE_ADDED,
            {"edge_id": edge_id, "src": src_node_id, "dst": dst_node_id, "edge_type": edge_type},
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO edges (edge_id, session_id, src_node_id, dst_node_id, edge_type, payload_json, created_seq)"
            " VALUES (?,?,?,?,?,?,?)",
            (edge_id, session_id, src_node_id, dst_node_id, edge_type, canonical_json(payload or {}), seq),
        )
        return edge_id

    def nodes_for_session(self, session_id: str, tier: int | None = None) -> list[sqlite3.Row]:
        if tier is None:
            return self.conn.execute(
                "SELECT * FROM nodes WHERE session_id=? ORDER BY created_seq", (session_id,)
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM nodes WHERE session_id=? AND tier=? ORDER BY created_seq",
            (session_id, tier),
        ).fetchall()

    def edges_for_session(self, session_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM edges WHERE session_id=? ORDER BY created_seq", (session_id,)
        ).fetchall()
