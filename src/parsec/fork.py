"""Fork runner: rewind a recorded session to call N and continue live.

The head of the run replays byte-exactly (prompt hashes asserted); the
tail runs against a live adapter with the cache in live-prefer-cache mode,
so already-fetched documents stay frozen and new fetches are allowed.
Recorded steering before the fork point is re-injected so head prompts
match; a `--steer` message can be added AT the fork point to redirect the
branch.
"""

from __future__ import annotations

import sqlite3

from parsec.config import CacheMode, Clock, RunConfig
from parsec.gateway.base import ModelAdapter
from parsec.gateway.fork_adapter import ForkAdapter
from parsec.gateway.gateway import ModelGateway
from parsec.gateway.replay_adapter import ReplayAdapter
from parsec.loop.agent import OrchestratorLoop, RunResult
from parsec.models.events import EventType
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


def _next_fork_id(conn: sqlite3.Connection, original: str, at_call: int) -> str:
    n = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE parent_session_id=?", (original,)
    ).fetchone()[0]
    return f"{original}-fork{at_call}-{n + 1}"


async def run_fork(
    conn: sqlite3.Connection,
    blobs: BlobStore,
    clock: Clock,
    original_session_id: str,
    at_call: int,
    live_adapter: ModelAdapter,
    fetch_transport=None,
    steer: str | None = None,
) -> RunResult:
    event_log = EventLog(conn, clock)
    sessions = SessionStore(conn, clock)
    recorded_config = sessions.get_config(original_session_id)
    if recorded_config.budgets.parallel_subagents > 1:
        raise ValueError(
            "fork --at-call requires a sequentially-recorded session: with "
            "parallel subagents, call order is per-stream and a single global "
            "fork point is ill-defined (record with parallel_subagents=1 to fork)"
        )

    fork_config = RunConfig(
        **{
            **recorded_config.model_dump(),
            "session_id": _next_fork_id(conn, original_session_id, at_call),
            "cache_mode": CacheMode.LIVE_PREFER_CACHE,
        }
    )

    ledger = Ledger(conn, clock)
    documents = DocumentStore(conn, clock)
    spans = SpanStore(conn)
    dag = DagStore(conn, event_log)

    replay = ReplayAdapter(event_log, blobs, original_session_id)
    adapter = ForkAdapter(replay, live_adapter, at_call)
    gateway = ModelGateway(adapter, event_log, blobs, ledger, fork_config)

    user_agent = USER_AGENT + (f" contact:{fork_config.contact}" if fork_config.contact else "")
    robots = (
        RobotsPolicy(conn, clock, user_agent, fork_config.robots_ttl_s, transport=fetch_transport)
        if fork_config.respect_robots
        else None
    )
    fetcher = Fetcher(
        documents, blobs, clock, CacheMode.LIVE_PREFER_CACHE,
        transport=fetch_transport, robots=robots, user_agent=user_agent,
    )
    tools: list = [
        FetchTool(fetcher, spans),
        RecordPremisesTool(dag, spans, documents),
        SearchWithinTool(spans, EmbeddingCache(conn, HashedNgramEmbedder())),
    ]
    provider = build_search_provider(fork_config, conn, clock, transport=fetch_transport)
    if provider is not None:
        tools.append(SearchBroadTool(provider))
    registry = ToolRegistry(tools)
    ctx = ToolContext(conn, blobs, event_log, ledger, fork_config, clock)

    loop = OrchestratorLoop(
        fork_config, gateway, registry, ctx, sessions, dag, spans, documents,
        CoverageLedger(conn, event_log), Notebook(conn, event_log, clock),
        parent_session_id=original_session_id,
    )
    # Head prompts must match the recording: re-inject recorded steering that
    # happened before the fork point (brief-gate messages via their own
    # channel). A new steer message lands AT the fork.
    scripted: dict[int, list[str]] = {}
    scripted_gates: dict[tuple[str, int], list[str]] = {}
    for ev in event_log.read(original_session_id):
        if ev.event_type == EventType.STEERING_INJECTED and ev.payload["turn_index"] < at_call:
            gate = ev.payload.get("gate")
            if gate:
                scripted_gates.setdefault((gate, ev.payload["turn_index"]), []).append(
                    ev.payload["text"]
                )
            else:
                scripted.setdefault(ev.payload["turn_index"], []).append(ev.payload["text"])
    if steer:
        scripted.setdefault(at_call, []).append(steer)
    loop.scripted_steering = scripted
    loop.scripted_gates = scripted_gates
    return await loop.run()
