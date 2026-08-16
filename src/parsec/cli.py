"""Parsec CLI: ask / replay / sessions / spans."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from parsec.config import (
    Budgets,
    CacheMode,
    Clock,
    DEFAULT_MODEL,
    RealClock,
    RunConfig,
    make_session_id,
)
from parsec.db.connection import open_db
from parsec.gateway.gateway import ModelGateway
from parsec.loop.agent import SingleAgentLoop
from parsec.retrieval.fetcher import Fetcher
from parsec.retrieval.search_provider import FixtureSearchProvider
from parsec.store.blobs import BlobStore
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.event_log import EventLog
from parsec.store.ledger import Ledger
from parsec.store.sessions import SessionStore
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.tools.fetch import FetchTool
from parsec.tools.record_premises import RecordPremisesTool
from parsec.tools.search_broad import SearchBroadTool

console = Console()

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_PARTIAL = 3
EXIT_ERROR = 4


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="parsec", description="Local-first research harness")
    sub = p.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Run a research query")
    ask.add_argument("query")
    ask.add_argument("--session-id")
    ask.add_argument("--cache-mode", choices=[m.value for m in CacheMode], default=CacheMode.LIVE_PREFER_CACHE.value)
    ask.add_argument("--adapter", choices=["anthropic", "fake", "replay"], default="anthropic")
    ask.add_argument("--model", default=DEFAULT_MODEL)
    ask.add_argument("--max-usd", type=float, default=Budgets().max_usd)
    ask.add_argument("--max-tokens", type=int, default=Budgets().max_total_tokens)
    ask.add_argument("--max-seconds", type=int, default=Budgets().max_wall_seconds)
    ask.add_argument("--max-turns", type=int, default=Budgets().max_turns)
    ask.add_argument("--data-dir", type=Path, default=Path("data"))
    ask.add_argument("--search-fixtures", type=Path, default=None)
    ask.add_argument("--json", action="store_true", dest="as_json")

    replay = sub.add_parser("replay", help="Re-run a recorded session against the frozen corpus")
    replay.add_argument("session_id")
    replay.add_argument("--data-dir", type=Path, default=Path("data"))
    replay.add_argument("--no-verify", action="store_true")

    verify = sub.add_parser(
        "verify", help="Run stage-1 structural verification over a session's evidence DAG"
    )
    verify.add_argument("session_id")
    verify.add_argument("--data-dir", type=Path, default=Path("data"))
    verify.add_argument("--json", action="store_true", dest="as_json")

    sessions = sub.add_parser("sessions", help="Inspect sessions")
    sessions_sub = sessions.add_subparsers(dest="sessions_command", required=True)
    s_list = sessions_sub.add_parser("list")
    s_list.add_argument("--data-dir", type=Path, default=Path("data"))
    s_show = sessions_sub.add_parser("show")
    s_show.add_argument("session_id")
    s_show.add_argument("--data-dir", type=Path, default=Path("data"))

    spans = sub.add_parser("spans", help="Inspect spans")
    spans_sub = spans.add_subparsers(dest="spans_command", required=True)
    sp_show = spans_sub.add_parser("show")
    sp_show.add_argument("span_id")
    sp_show.add_argument("--data-dir", type=Path, default=Path("data"))

    return p


def _open(data_dir: Path):
    conn = open_db(data_dir / "parsec.db")
    blobs = BlobStore(data_dir / "blobs")
    return conn, blobs


# Test seams: the M1 exit test patches these to run the full CLI path with a
# scripted adapter and a mock HTTP transport, no network or keys required.
adapter_factory = None  # callable(config) -> ModelAdapter, or None for default
fetch_transport = None  # httpx.AsyncBaseTransport, or None for real HTTP


def make_adapter(config: RunConfig):
    if adapter_factory is not None:
        return adapter_factory(config)
    if config.adapter == "anthropic":
        from parsec.gateway.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter()
    raise SystemExit(
        f"adapter {config.adapter!r} is not runnable from `parsec ask` without a test adapter factory"
    )


def _build_loop(config: RunConfig, conn, blobs: BlobStore, clock: Clock) -> SingleAgentLoop:
    event_log = EventLog(conn, clock)
    ledger = Ledger(conn, clock)
    sessions = SessionStore(conn, clock)
    documents = DocumentStore(conn, clock)
    spans = SpanStore(conn)
    dag = DagStore(conn, event_log)

    adapter = make_adapter(config)
    gateway = ModelGateway(adapter, event_log, blobs, ledger, config)
    fetcher = Fetcher(documents, blobs, clock, config.cache_mode, transport=fetch_transport)
    tools: list = [FetchTool(fetcher, spans), RecordPremisesTool(dag, spans, documents)]
    if config.search_fixtures is not None:
        tools.append(SearchBroadTool(FixtureSearchProvider(config.search_fixtures)))
    registry = ToolRegistry(tools)
    ctx = ToolContext(conn, blobs, event_log, ledger, config, clock)
    return SingleAgentLoop(config, gateway, registry, ctx, sessions, dag, spans, documents)


def cmd_ask(args) -> int:
    clock = RealClock()
    conn, blobs = _open(args.data_dir)
    session_id = args.session_id or make_session_id(args.query, clock.now_iso())
    config = RunConfig(
        session_id=session_id,
        query=args.query,
        model=args.model,
        cache_mode=CacheMode(args.cache_mode),
        adapter=args.adapter,
        budgets=Budgets(
            max_usd=args.max_usd,
            max_total_tokens=args.max_tokens,
            max_wall_seconds=args.max_seconds,
            max_turns=args.max_turns,
        ),
        data_dir=args.data_dir,
        search_fixtures=args.search_fixtures,
    )
    loop = _build_loop(config, conn, blobs, clock)
    try:
        result = asyncio.run(loop.run())
    except KeyboardInterrupt:
        console.print("[red]aborted[/red]")
        return EXIT_ERROR

    ledger = Ledger(conn, clock)
    totals = ledger.totals(session_id)
    if args.as_json:
        print(
            json.dumps(
                {
                    "session_id": result.session_id,
                    "status": result.status,
                    "answer": result.answer,
                    "claims_total": result.claims_total,
                    "unresolved": result.unresolved,
                    "violations": result.violations,
                    "turns": result.turns,
                    "totals": totals,
                }
            )
        )
    else:
        console.print(result.answer)
        console.print(
            f"\n[dim]session {result.session_id} · {result.status} · {result.turns} turns · "
            f"{int(totals.get('input_tokens', 0))} in / {int(totals.get('output_tokens', 0))} out tokens · "
            f"${totals.get('usd', 0.0):.4f} · claims {result.claims_total}"
            + (f" · [red]{len(result.unresolved)} unresolved[/red]" if result.unresolved else "")
            + (f" · [red]{len(result.violations)} verification violations[/red]" if result.violations else "")
            + "[/dim]"
        )
    if result.status == "done":
        return EXIT_OK
    if result.status in ("partial", "halted_budget"):
        return EXIT_PARTIAL
    return EXIT_ERROR


def cmd_replay(args) -> int:
    from parsec.replay import run_replay

    clock = RealClock()
    conn, blobs = _open(args.data_dir)
    outcome = asyncio.run(run_replay(conn, blobs, clock, args.session_id))
    console.print(f"replayed as {outcome.result.session_id} · status {outcome.result.status}")
    if args.no_verify:
        return EXIT_OK
    if outcome.verified:
        console.print("[green]verified: projections and answer bytes identical[/green]")
        return EXIT_OK
    console.print("[red]replay verification FAILED[/red]")
    if not outcome.projections_match:
        console.print(f"projection divergence:\n{outcome.first_divergence}")
    if not outcome.answers_match:
        console.print("answer blobs differ")
    return EXIT_ERROR


def cmd_verify(args) -> int:
    from parsec.verify.structural import verify_session

    clock = RealClock()
    conn, blobs = _open(args.data_dir)
    if SessionStore(conn, clock).get(args.session_id) is None:
        console.print(f"[red]unknown session {args.session_id}[/red]")
        return EXIT_USAGE
    report = verify_session(conn, blobs, args.session_id)
    if args.as_json:
        print(json.dumps(report.to_payload()))
    else:
        console.print(
            f"checked {report.checked_claims} claims, {report.checked_premises} premises, "
            f"{report.checked_spans} spans"
        )
        if report.ok:
            console.print("[green]verification passed: every claim traces to intact spans[/green]")
        else:
            table = Table("check", "subject", "detail")
            for v in report.violations:
                table.add_row(v.check, v.subject, v.detail)
            console.print(table)
    return EXIT_OK if report.ok else EXIT_PARTIAL


def cmd_sessions(args) -> int:
    clock = RealClock()
    conn, blobs = _open(args.data_dir)
    store = SessionStore(conn, clock)
    ledger = Ledger(conn, clock)
    if args.sessions_command == "list":
        table = Table("session", "status", "query", "usd", "tokens", "created")
        for row in store.list():
            totals = ledger.totals(row["session_id"])
            tokens = int(sum(totals.get(c, 0) for c in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens")))
            table.add_row(
                row["session_id"],
                row["status"],
                row["query"][:50],
                f"{totals.get('usd', 0.0):.4f}",
                str(tokens),
                row["created_ts"],
            )
        console.print(table)
        return EXIT_OK
    row = store.get(args.session_id)
    if row is None:
        console.print(f"[red]unknown session {args.session_id}[/red]")
        return EXIT_USAGE
    console.print_json(row["config_json"])
    console.print(ledger.totals(args.session_id))
    n_events = conn.execute(
        "SELECT COUNT(*) FROM events WHERE session_id=?", (args.session_id,)
    ).fetchone()[0]
    console.print(f"{n_events} events")
    if row["answer_blob"]:
        console.print(blobs.get_text(row["answer_blob"]))
    return EXIT_OK


def cmd_spans(args) -> int:
    conn, blobs = _open(args.data_dir)
    spans = SpanStore(conn)
    row = spans.get(args.span_id)
    if row is None:
        console.print(f"[red]unknown span {args.span_id}[/red]")
        return EXIT_USAGE
    doc = DocumentStore(conn, RealClock()).get_document(row["doc_hash"])
    console.print(f"[bold]{args.span_id}[/bold] — {doc['url']} (fetched {doc['fetched_ts']})")
    console.print(row["text"])
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "ask": cmd_ask,
        "replay": cmd_replay,
        "verify": cmd_verify,
        "sessions": cmd_sessions,
        "spans": cmd_spans,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
