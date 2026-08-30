"""Refresh runner (M14.2): re-run a recorded session's query as a NEW
session seeded by the prior brief and coverage ledger, mutability classes
governing what gets re-researched.

A recorded session is one frozen observation of a moving world. `parsec
refresh <session>` re-observes it without re-planning: the new run reuses
the parent's persisted research brief (scope, effort, subquestions — no
decomposition model call), and per subquestion the coverage ledger plus
the recorded mutability classes decide whether the world could have moved
under it:

  carry forward — the parent answered it and every supporting premise is
      claim_class "stable": the evidence cannot have changed, so the
      parent's premises/findings (and the source spans under them) are
      re-recorded into the new session. Node IDs are content-derived
      (T4), so carried evidence keeps its cross-session identity — the
      final diff matches it by id.
  re-research — anything volatile or slow (the world may have moved),
      anything the parent left partial/blocked/dropped, an answered row
      with no premises of its own (e.g. one recovered by a coverage
      retry, whose evidence lives in the retry's stream), or everything
      under --all.

The refresh run fetches in RECORD mode — the point is to re-observe, so
cached bytes must not stand in for the live web. The parent's replay
stays intact because replay pins fetches to the doc hashes its own
events recorded (Fetcher.pinned_docs), not to the shared URL cache row
the refresh advances. The seed itself is a pure function of the
immutable parent recording plus the refresh_all flag, both frozen into
the new session's config (refresh_of / refresh_all), so a refreshed
session replays byte-identically: replay re-derives the same seed (T4).

Gates are stripped from the inherited config: the brief was already
approved in the parent, and refresh is built to be scriptable (M14.3's
`parsec watch` schedules it headless).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from parsec import __version__
from parsec.config import CacheMode, Clock, RunConfig
from parsec.gateway.base import ModelAdapter
from parsec.gateway.gateway import ModelGateway
from parsec.loop.agent import OrchestratorLoop, RunResult
from parsec.retrieval.embeddings import EmbeddingCache, HashedNgramEmbedder
from parsec.retrieval.fetcher import USER_AGENT, Fetcher
from parsec.retrieval.providers import build_search_provider
from parsec.retrieval.robots import RobotsPolicy
from parsec.store.blobs import BlobStore
from parsec.store.coverage import CoverageLedger
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.event_log import EventLog
from parsec.store.ledger import Ledger
from parsec.store.notebook import Notebook
from parsec.store.sessions import SessionStore
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.tools.fetch import FetchTool
from parsec.tools.record_premises import RecordPremisesTool
from parsec.tools.search_broad import SearchBroadTool
from parsec.tools.search_within import SearchWithinTool


@dataclass(frozen=True)
class CarriedSq:
    """One subquestion whose evidence carries forward instead of re-running."""

    sq_id: str
    premise_ids: tuple[str, ...]
    finding_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class RefreshSeed:
    """The plan a refresh run starts from — a pure function of the parent
    recording plus refresh_all, so replay re-derives it identically."""

    parent_session_id: str
    scope: str
    effort: str
    questions: tuple[str, ...]        # plan order; sq-N ids re-derive by position
    carried: dict[str, CarriedSq]     # sq_id -> evidence to carry forward
    research_reasons: dict[str, str]  # sq_id -> why it re-runs


def derive_seed(
    conn: sqlite3.Connection, parent_session_id: str, refresh_all: bool = False
) -> RefreshSeed:
    """Read the parent's recorded brief, plan, per-subquestion evidence, and
    coverage statuses, and split the plan into carried vs re-researched.
    Deterministic: the parent recording is immutable, so the same inputs
    always produce the same seed."""
    if (
        conn.execute(
            "SELECT 1 FROM sessions WHERE session_id=?", (parent_session_id,)
        ).fetchone()
        is None
    ):
        raise KeyError(f"unknown session: {parent_session_id}")

    scope, effort = "", "deep"
    planned: list[dict] | None = None
    completed: dict[str, dict] = {}
    for row in conn.execute(
        "SELECT event_type, payload_json FROM events WHERE session_id=? ORDER BY idx",
        (parent_session_id,),
    ):
        payload = json.loads(row["payload_json"])
        if row["event_type"] == "research_brief" and "limits" in payload:
            # The final (post-gate) brief; proposals carry status="proposed".
            scope, effort = payload.get("scope", ""), payload.get("effort", "deep")
        elif row["event_type"] == "subquestions_planned":
            planned = payload["subquestions"]
        elif row["event_type"] == "subagent_completed":
            completed.setdefault(payload["sq_id"], payload)
    if not planned:
        raise ValueError(
            f"session {parent_session_id} has no recorded subquestion plan to refresh from"
        )

    coverage = {
        r["sq_id"]: r
        for r in conn.execute(
            "SELECT sq_id, status, reason FROM coverage WHERE session_id=?",
            (parent_session_id,),
        )
    }
    classes = {
        r["node_id"]: json.loads(r["payload_json"]).get("claim_class", "stable")
        for r in conn.execute(
            "SELECT node_id, payload_json FROM nodes WHERE session_id=? AND tier=1",
            (parent_session_id,),
        )
    }

    carried: dict[str, CarriedSq] = {}
    research_reasons: dict[str, str] = {}
    for item in planned:
        sq_id = item["sq_id"]
        row = coverage.get(sq_id)
        status = row["status"] if row is not None else "open"
        reason = (row["reason"] or "") if row is not None else ""
        premises = tuple(completed.get(sq_id, {}).get("premises", []))
        findings = tuple(completed.get(sq_id, {}).get("findings", []))
        if refresh_all:
            research_reasons[sq_id] = "full refresh requested"
        elif status != "answered":
            research_reasons[sq_id] = f"was {status} in {parent_session_id}"
        elif reason.startswith("recovered by "):
            # The answer's evidence lives in the coverage-retry stream, not
            # this subquestion's — nothing of its own to carry.
            research_reasons[sq_id] = f"{reason} in {parent_session_id}"
        elif not premises:
            research_reasons[sq_id] = "answered without premises of its own"
        else:
            mutable = sorted({classes.get(p, "stable") for p in premises} - {"stable"})
            if mutable:
                research_reasons[sq_id] = (
                    "/".join(mutable) + " evidence must be re-observed"
                )
            else:
                carried[sq_id] = CarriedSq(
                    sq_id, premises, findings, "stable evidence carried forward"
                )
    return RefreshSeed(
        parent_session_id=parent_session_id,
        scope=scope,
        effort=effort,
        questions=tuple(item["question"] for item in planned),
        carried=carried,
        research_reasons=research_reasons,
    )


def copy_carried_evidence(
    conn: sqlite3.Connection,
    dag: DagStore,
    parent_sid: str,
    new_sid: str,
    carried: list[CarriedSq],
) -> dict[str, tuple[list[str], list[str]]]:
    """Re-record carried evidence into the new session: each premise/finding
    with the evidence nodes under it (spans first) and its derivation edges,
    in the parent's recorded order. Payloads are copied verbatim, so the
    content-derived node IDs are identical across sessions by construction.
    Contradicts edges copy last, and only when both endpoints made it.
    Returns sq_id -> (premise_ids, finding_ids) actually copied."""
    nodes = {
        r["node_id"]: (r["node_type"], json.loads(r["payload_json"]))
        for r in conn.execute(
            "SELECT node_id, node_type, payload_json FROM nodes WHERE session_id=?",
            (parent_sid,),
        )
    }
    out_edges: dict[str, list[sqlite3.Row]] = {}
    contradicts: list[sqlite3.Row] = []
    for e in conn.execute(
        "SELECT src_node_id, dst_node_id, edge_type, payload_json FROM edges"
        " WHERE session_id=? ORDER BY created_seq",
        (parent_sid,),
    ):
        if e["edge_type"] == "contradicts":
            contradicts.append(e)
        else:
            out_edges.setdefault(e["src_node_id"], []).append(e)

    copied: set[str] = set()

    def copy_with_evidence(nid: str) -> bool:
        if nid in copied:
            return True
        entry = nodes.get(nid)
        if entry is None:
            return False
        for e in out_edges.get(nid, []):
            copy_with_evidence(e["dst_node_id"])
        dag.add_node(new_sid, entry[0], entry[1])
        copied.add(nid)
        for e in out_edges.get(nid, []):
            if e["dst_node_id"] in copied:
                dag.add_edge(
                    new_sid, nid, e["dst_node_id"], e["edge_type"],
                    json.loads(e["payload_json"]) or None,
                )
        return True

    out: dict[str, tuple[list[str], list[str]]] = {}
    for c in carried:
        premise_ids = [pid for pid in c.premise_ids if copy_with_evidence(pid)]
        finding_ids = [fid for fid in c.finding_ids if copy_with_evidence(fid)]
        out[c.sq_id] = (premise_ids, finding_ids)
    for e in contradicts:
        if e["src_node_id"] in copied and e["dst_node_id"] in copied:
            dag.add_edge(
                new_sid, e["src_node_id"], e["dst_node_id"], "contradicts",
                json.loads(e["payload_json"]) or None,
            )
    return out


def _next_refresh_id(conn: sqlite3.Connection, original: str) -> str:
    n = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE parent_session_id=? AND session_id LIKE ?",
        (original, f"{original}-refresh-%"),
    ).fetchone()[0]
    return f"{original}-refresh-{n + 1}"


async def run_refresh(
    conn: sqlite3.Connection,
    blobs: BlobStore,
    clock: Clock,
    original_session_id: str,
    live_adapter: ModelAdapter,
    fetch_transport=None,
    refresh_all: bool = False,
) -> RunResult:
    event_log = EventLog(conn, clock)
    sessions = SessionStore(conn, clock)
    recorded_config = sessions.get_config(original_session_id)
    seed = derive_seed(conn, original_session_id, refresh_all)

    refresh_config = RunConfig(
        **{
            **recorded_config.model_dump(),
            "session_id": _next_refresh_id(conn, original_session_id),
            # Re-observe: cached bytes must not stand in for the live web.
            "cache_mode": CacheMode.RECORD,
            # The brief was approved in the parent; refresh runs headless.
            "brief_gate": False,
            "cost_gate_threshold": None,
            "refresh_of": original_session_id,
            "refresh_all": refresh_all,
            "parsec_version": __version__,
        }
    )

    ledger = Ledger(conn, clock)
    documents = DocumentStore(conn, clock)
    spans = SpanStore(conn)
    dag = DagStore(conn, event_log)
    gateway = ModelGateway(live_adapter, event_log, blobs, ledger, refresh_config)

    user_agent = USER_AGENT + (
        f" contact:{refresh_config.contact}" if refresh_config.contact else ""
    )
    robots = (
        RobotsPolicy(
            conn, clock, user_agent, refresh_config.robots_ttl_s, transport=fetch_transport
        )
        if refresh_config.respect_robots
        else None
    )
    fetcher = Fetcher(
        documents, blobs, clock, CacheMode.RECORD,
        transport=fetch_transport, robots=robots, user_agent=user_agent,
    )
    tools: list = [
        FetchTool(fetcher, spans),
        RecordPremisesTool(dag, spans, documents),
        SearchWithinTool(spans, EmbeddingCache(conn, HashedNgramEmbedder())),
    ]
    provider = build_search_provider(refresh_config, conn, clock, transport=fetch_transport)
    if provider is not None:
        tools.append(SearchBroadTool(provider))
    registry = ToolRegistry(tools)
    ctx = ToolContext(conn, blobs, event_log, ledger, refresh_config, clock)

    loop = OrchestratorLoop(
        refresh_config, gateway, registry, ctx, sessions, dag, spans, documents,
        CoverageLedger(conn, event_log), Notebook(conn, event_log, clock),
        parent_session_id=original_session_id,
    )
    loop.refresh_seed = seed
    return await loop.run()
