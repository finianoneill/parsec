"""ReplayRunner: re-execute a recorded session against the frozen corpus (T4).

Rebuilds the recorded config, wires the ReplayAdapter and replay-mode cache
over the same DB/blob corpus, runs the identical loop into a child session,
then compares event-stream projections and answer bytes.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from parsec.config import CacheMode, Clock, RunConfig
from parsec.gateway.gateway import ModelGateway
from parsec.gateway.replay_adapter import ReplayAdapter
from parsec.loop.agent import OrchestratorLoop, RunResult
from parsec.retrieval.fetcher import Fetcher
from parsec.retrieval.search_provider import FixtureSearchProvider
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


@dataclass
class ReplayOutcome:
    result: RunResult
    projections_match: bool
    answers_match: bool
    first_divergence: str | None = None

    @property
    def verified(self) -> bool:
        return self.projections_match and self.answers_match


def _next_replay_id(conn: sqlite3.Connection, original: str) -> str:
    n = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE parent_session_id=?", (original,)
    ).fetchone()[0]
    return f"{original}-replay-{n + 1}"


async def run_replay(
    conn: sqlite3.Connection,
    blobs: BlobStore,
    clock: Clock,
    original_session_id: str,
) -> ReplayOutcome:
    event_log = EventLog(conn, clock)
    sessions = SessionStore(conn, clock)
    recorded_config = sessions.get_config(original_session_id)

    replay_config = RunConfig(
        **{
            **recorded_config.model_dump(),
            "session_id": _next_replay_id(conn, original_session_id),
            "adapter": "replay",
            "cache_mode": CacheMode.REPLAY,
        }
    )

    ledger = Ledger(conn, clock)
    documents = DocumentStore(conn, clock)
    spans = SpanStore(conn)
    dag = DagStore(conn, event_log)

    adapter = ReplayAdapter(event_log, blobs, original_session_id)
    gateway = ModelGateway(adapter, event_log, blobs, ledger, replay_config)

    fetcher = Fetcher(documents, blobs, clock, CacheMode.REPLAY)
    tools: list = [FetchTool(fetcher, spans), RecordPremisesTool(dag, spans, documents)]
    if replay_config.search_fixtures is not None:
        tools.append(SearchBroadTool(FixtureSearchProvider(replay_config.search_fixtures)))
    registry = ToolRegistry(tools)
    ctx = ToolContext(conn, blobs, event_log, ledger, replay_config, clock)

    coverage = CoverageLedger(conn, event_log)
    notebook = Notebook(conn, event_log, clock)
    loop = OrchestratorLoop(
        replay_config, gateway, registry, ctx, sessions, dag, spans, documents,
        coverage, notebook, parent_session_id=original_session_id,
    )
    result = await loop.run()

    proj_orig = event_log.projection(original_session_id)
    proj_replay = event_log.projection(replay_config.session_id)
    projections_match = proj_orig == proj_replay

    orig_row = sessions.get(original_session_id)
    replay_row = sessions.get(replay_config.session_id)
    answers_match = (
        orig_row["answer_blob"] is not None
        and orig_row["answer_blob"] == replay_row["answer_blob"]
    )

    first_divergence = None
    if not projections_match:
        for i, (a, b) in enumerate(
            zip(proj_orig.decode().splitlines(), proj_replay.decode().splitlines())
        ):
            if a != b:
                first_divergence = f"line {i}:\n  original: {a[:300]}\n  replay:   {b[:300]}"
                break
        else:
            first_divergence = "projections differ in length"

    return ReplayOutcome(result, projections_match, answers_match, first_divergence)
