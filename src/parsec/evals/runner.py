"""Eval runner: execute cases against their frozen corpora and score them.

Each case run: fork the corpus by file copy → run the full orchestrator
loop in replay cache mode (any fetch outside the corpus is a CacheMiss, so
the world is frozen) → score the three axes. The model itself is LIVE (or
a scripted fake in tests) — that is the point: same corpus, different
harness/model, measurable difference (T4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from parsec.config import Budgets, CacheMode, Clock, RunConfig
from parsec.evals.case import FIXTURES_FILE, EvalCase, copy_corpus, load_case
from parsec.evals.judge import judge_synthesis
from parsec.evals.scoring import AxisScores, score_session
from parsec.gateway.base import ModelAdapter
from parsec.gateway.gateway import ModelGateway
from parsec.loop.agent import OrchestratorLoop
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

AdapterFactory = Callable[[RunConfig], ModelAdapter]


@dataclass
class CaseResult:
    case_id: str
    session_id: str
    status: str
    scores: AxisScores
    turns: int = 0
    error: str | None = None

    def to_payload(self) -> dict:
        return {
            "case_id": self.case_id,
            "session_id": self.session_id,
            "status": self.status,
            "scores": self.scores.to_payload(),
            "turns": self.turns,
            "error": self.error,
        }


@dataclass
class EvalRun:
    label: str
    results: list[CaseResult] = field(default_factory=list)

    def to_payload(self) -> dict:
        return {
            "label": self.label,
            "results": [r.to_payload() for r in self.results],
            "aggregate": self.aggregate(),
        }

    def aggregate(self) -> dict:
        agg: dict[str, float | None] = {}
        for axis in ("citation_faithfulness", "coverage", "synthesis"):
            values = [
                getattr(r.scores, axis)
                for r in self.results
                if getattr(r.scores, axis) is not None
            ]
            agg[axis] = round(sum(values) / len(values), 4) if values else None
        return agg


async def run_case(
    case_dir: Path,
    workdir: Path,
    adapter_factory: AdapterFactory,
    clock: Clock,
    model: str,
    judge_adapter: ModelAdapter | None = None,
    judge_model: str = "",
) -> CaseResult:
    from parsec.db.connection import open_db

    case: EvalCase = load_case(case_dir)
    data_dir = copy_corpus(case_dir, workdir / case.case_id)
    session_id = f"eval-{case.case_id}"

    config = RunConfig(
        session_id=session_id,
        query=case.query,
        model=model,
        cache_mode=CacheMode.REPLAY,
        adapter="anthropic",  # informational; the factory decides
        budgets=Budgets(max_turns=case.max_turns),
        data_dir=data_dir,
        search_fixtures=case_dir / FIXTURES_FILE,
    )

    conn = open_db(data_dir / "parsec.db")
    blobs = BlobStore(data_dir / "blobs")
    event_log = EventLog(conn, clock)
    ledger = Ledger(conn, clock)
    sessions = SessionStore(conn, clock)
    documents = DocumentStore(conn, clock)
    spans = SpanStore(conn)
    dag = DagStore(conn, event_log)

    gateway = ModelGateway(adapter_factory(config), event_log, blobs, ledger, config)
    fetcher = Fetcher(documents, blobs, clock, CacheMode.REPLAY)
    registry = ToolRegistry(
        [
            FetchTool(fetcher, spans),
            RecordPremisesTool(dag, spans, documents),
            SearchBroadTool(FixtureSearchProvider(config.search_fixtures)),
        ]
    )
    ctx = ToolContext(conn, blobs, event_log, ledger, config, clock)
    loop = OrchestratorLoop(
        config, gateway, registry, ctx, sessions, dag, spans, documents,
        CoverageLedger(conn, event_log), Notebook(conn, event_log, clock),
    )

    try:
        result = await loop.run()
    except Exception as exc:
        return CaseResult(
            case.case_id, session_id, "error",
            AxisScores(None, None), error=f"{type(exc).__name__}: {exc}",
        )

    synthesis = None
    if judge_adapter is not None and result.answer:
        synthesis = await judge_synthesis(judge_adapter, judge_model, case.query, result.answer)
    scores = score_session(conn, blobs, session_id, case.must_find, synthesis)
    return CaseResult(case.case_id, session_id, result.status, scores, result.turns)


async def run_cases(
    case_dirs: list[Path],
    workdir: Path,
    adapter_factory: AdapterFactory,
    clock: Clock,
    model: str,
    label: str = "",
    judge_adapter: ModelAdapter | None = None,
    judge_model: str = "",
) -> EvalRun:
    run = EvalRun(label=label)
    for case_dir in case_dirs:
        run.results.append(
            await run_case(
                case_dir, workdir, adapter_factory, clock, model, judge_adapter, judge_model
            )
        )
    return run
