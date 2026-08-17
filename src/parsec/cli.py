"""Parsec CLI: ask / replay / sessions / spans."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
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
    ask.add_argument("--max-gap-rounds", type=int, default=Budgets().max_gap_rounds)
    ask.add_argument("--data-dir", type=Path, default=Path("data"))
    ask.add_argument("--search-fixtures", type=Path, default=None)
    ask.add_argument("--json", action="store_true", dest="as_json")
    ask.add_argument(
        "--live", action="store_true",
        help="Show a live progress view (state, coverage, DAG counts, spend)",
    )

    fork = sub.add_parser("fork", help="Rewind a recorded session to call N and continue live")
    fork.add_argument("session_id")
    fork.add_argument("--at-call", type=int, required=True, help="Model-call index to branch at (0-based)")
    fork.add_argument("--steer", default=None, help="Steering message injected at the fork point")
    fork.add_argument("--data-dir", type=Path, default=Path("data"))
    fork.add_argument("--json", action="store_true", dest="as_json")

    judge = sub.add_parser(
        "judge", help="Advisory judge pass over deduces/induces derivations (different model family)"
    )
    judge.add_argument("session_id")
    judge.add_argument("--judge-model", default=os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o"))
    judge.add_argument("--data-dir", type=Path, default=Path("data"))

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

    ev = sub.add_parser("eval", help="Eval harness: frozen-corpus runs, scoring, regression compare")
    ev_sub = ev.add_subparsers(dest="eval_command", required=True)
    ev_run = ev_sub.add_parser("run", help="Run all cases under a directory and score them")
    ev_run.add_argument("cases_root", type=Path)
    ev_run.add_argument("--out", type=Path, required=True, help="Write results JSON here")
    ev_run.add_argument("--label", default="")
    ev_run.add_argument("--model", default=DEFAULT_MODEL)
    ev_run.add_argument(
        "--judge", choices=["openai", "none"], default="none",
        help="Synthesis judge (axis 3, advisory). openai needs OPENAI_API_KEY.",
    )
    ev_run.add_argument("--judge-model", default=os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o"))
    ev_cmp = ev_sub.add_parser("compare", help="Compare two results files for regressions")
    ev_cmp.add_argument("results_a", type=Path)
    ev_cmp.add_argument("results_b", type=Path)
    ev_cmp.add_argument("--epsilon", type=float, default=0.05)
    ev_cmp.add_argument("--json", action="store_true", dest="as_json")
    ev_make = ev_sub.add_parser("make-case", help="Snapshot a recorded session's corpus into a frozen case")
    ev_make.add_argument("session_id")
    ev_make.add_argument("--data-dir", type=Path, default=Path("data"))
    ev_make.add_argument("--fixtures", type=Path, required=True, help="Search fixtures used for the recording")
    ev_make.add_argument("--out", type=Path, required=True, help="New case directory")
    ev_make.add_argument("--case-id", default=None)

    notebook = sub.add_parser("notebook", help="Print a session's notebook (append-only markdown)")
    notebook.add_argument("session_id")
    notebook.add_argument("--data-dir", type=Path, default=Path("data"))

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


def _build_loop(config: RunConfig, conn, blobs: BlobStore, clock: Clock) -> OrchestratorLoop:
    event_log = EventLog(conn, clock)
    ledger = Ledger(conn, clock)
    sessions = SessionStore(conn, clock)
    documents = DocumentStore(conn, clock)
    spans = SpanStore(conn)
    dag = DagStore(conn, event_log)
    coverage = CoverageLedger(conn, event_log)
    notebook = Notebook(conn, event_log, clock)

    adapter = make_adapter(config)
    gateway = ModelGateway(adapter, event_log, blobs, ledger, config)
    fetcher = Fetcher(documents, blobs, clock, config.cache_mode, transport=fetch_transport)
    tools: list = [FetchTool(fetcher, spans), RecordPremisesTool(dag, spans, documents)]
    if config.search_fixtures is not None:
        tools.append(SearchBroadTool(FixtureSearchProvider(config.search_fixtures)))
    registry = ToolRegistry(tools)
    ctx = ToolContext(conn, blobs, event_log, ledger, config, clock)
    return OrchestratorLoop(
        config, gateway, registry, ctx, sessions, dag, spans, documents, coverage, notebook
    )


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
            max_gap_rounds=args.max_gap_rounds,
        ),
        data_dir=args.data_dir,
        search_fixtures=args.search_fixtures,
    )
    loop = _build_loop(config, conn, blobs, clock)

    live_view = None
    if args.live and not args.as_json:
        from rich.live import Live

        def render(snapshot: dict):
            table = Table("state", "coverage", "premises", "findings", "claims", "turns", "tokens", "usd")
            table.add_row(
                snapshot["state"],
                " ".join(f"{k}:{v}" for k, v in sorted(snapshot["coverage"].items())) or "—",
                str(snapshot["premises"]), str(snapshot["findings"]), str(snapshot["claims"]),
                str(snapshot["turns"]), str(snapshot["tokens"]), f"${snapshot['usd']:.4f}",
            )
            return table

        live_view = Live(console=console, refresh_per_second=4)
        live_view.start()
        loop.reporter = lambda snapshot: live_view.update(render(snapshot))

    # Steering (§3): lines typed on stdin mid-run are injected into the next
    # model call without tearing down the turn.
    if sys.stdin.isatty() and not args.as_json:
        import threading

        def read_stdin():
            try:
                for line in sys.stdin:
                    text = line.strip()
                    if text:
                        loop.steer(text)
            except (ValueError, OSError):
                pass

        threading.Thread(target=read_stdin, daemon=True).start()

    try:
        result = asyncio.run(loop.run())
    except KeyboardInterrupt:
        console.print("[red]aborted[/red]")
        return EXIT_ERROR
    finally:
        if live_view is not None:
            live_view.stop()

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
                    "coverage": result.coverage,
                    "low_confidence": result.low_confidence,
                    "unused_sources": result.unused_sources,
                    "turns": result.turns,
                    "totals": totals,
                }
            )
        )
    else:
        console.print(result.answer)
        if result.coverage:
            console.print(
                "[dim]coverage: "
                + " · ".join(f"{sq} {status}" for sq, status in sorted(result.coverage.items()))
                + "[/dim]"
            )
        console.print(
            f"\n[dim]session {result.session_id} · {result.status} · {result.turns} turns · "
            f"{int(totals.get('input_tokens', 0))} in / {int(totals.get('output_tokens', 0))} out tokens · "
            f"${totals.get('usd', 0.0):.4f} · claims {result.claims_total}"
            + (f" · [red]{len(result.unresolved)} unresolved[/red]" if result.unresolved else "")
            + (f" · [red]{len(result.violations)} verification violations[/red]" if result.violations else "")
            + (f" · [yellow]{len(result.low_confidence)} low-confidence claims[/yellow]" if result.low_confidence else "")
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
    from parsec.verify.credence import compute_credences, render_tier
    from parsec.verify.omission import detect_omissions
    from parsec.verify.structural import verify_session

    clock = RealClock()
    conn, blobs = _open(args.data_dir)
    store = SessionStore(conn, clock)
    if store.get(args.session_id) is None:
        console.print(f"[red]unknown session {args.session_id}[/red]")
        return EXIT_USAGE
    report = verify_session(conn, blobs, args.session_id)
    session_config = store.get_config(args.session_id)
    credence = compute_credences(
        conn,
        args.session_id,
        source_tiers=session_config.source_tiers,
        stakes_threshold=session_config.stakes_threshold,
        volatile_penalty=session_config.volatile_penalty,
    )
    omissions = detect_omissions(conn, EventLog(conn, clock), args.session_id)
    if args.as_json:
        payload = report.to_payload()
        payload["credence"] = {
            "flagged_claims": credence.flagged_claims,
            "tiers": {nid: render_tier(nc.credence) for nid, nc in sorted(credence.nodes.items())},
        }
        payload["omissions"] = omissions.to_payload()
        print(json.dumps(payload))
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
        if credence.flagged_claims:
            console.print(
                f"[yellow]{len(credence.flagged_claims)} claims below the stakes threshold[/yellow]"
            )
        if not omissions.empty:
            for d in omissions.unused_documents:
                console.print(f"[yellow]consulted but unused:[/yellow] {d['url']}")
            for p in omissions.uncited_premises:
                console.print(f"[yellow]recorded but uncited:[/yellow] {p['text']}")
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
    coverage_rows = CoverageLedger(conn, EventLog(conn, clock)).all(args.session_id)
    if coverage_rows:
        table = Table("sq", "status", "question", "reason")
        for c in coverage_rows:
            table.add_row(c["sq_id"], c["status"], c["question"][:60], c["reason"] or "")
        console.print(table)
    if row["answer_blob"]:
        console.print(blobs.get_text(row["answer_blob"]))
    return EXIT_OK


def cmd_fork(args) -> int:
    from parsec.fork import run_fork

    clock = RealClock()
    conn, blobs = _open(args.data_dir)
    if SessionStore(conn, clock).get(args.session_id) is None:
        console.print(f"[red]unknown session {args.session_id}[/red]")
        return EXIT_USAGE
    config = SessionStore(conn, clock).get_config(args.session_id)
    live_adapter = make_adapter(config)
    result = asyncio.run(
        run_fork(
            conn, blobs, clock, args.session_id, args.at_call, live_adapter,
            fetch_transport=fetch_transport, steer=args.steer,
        )
    )
    if args.as_json:
        print(json.dumps({"session_id": result.session_id, "status": result.status, "answer": result.answer}))
    else:
        console.print(result.answer)
        console.print(f"[dim]forked as {result.session_id} · {result.status}[/dim]")
    return EXIT_OK if result.status == "done" else EXIT_PARTIAL


def cmd_judge(args) -> int:
    from parsec.gateway.openai_adapter import OpenAIAdapter
    from parsec.verify.judge_pass import judge_pass

    clock = RealClock()
    conn, blobs = _open(args.data_dir)
    if SessionStore(conn, clock).get(args.session_id) is None:
        console.print(f"[red]unknown session {args.session_id}[/red]")
        return EXIT_USAGE
    adapter = OpenAIAdapter()
    judgments = asyncio.run(
        judge_pass(conn, EventLog(conn, clock), args.session_id, adapter, args.judge_model)
    )
    if not judgments:
        console.print("no deduces/induces derivations to judge")
        return EXIT_OK
    table = Table("finding", "edge", "score", "rationale")
    for j in judgments:
        table.add_row(
            j.finding_id, j.edge_type,
            "—" if j.score is None else f"{j.score:.2f}", j.rationale[:80],
        )
    console.print(table)
    console.print("[dim]advisory only — judge scores gate nothing (§6)[/dim]")
    return EXIT_OK


def cmd_eval(args) -> int:
    if args.eval_command == "run":
        return _eval_run(args)
    if args.eval_command == "compare":
        return _eval_compare(args)
    return _eval_make_case(args)


def _eval_run(args) -> int:
    from parsec.evals.case import discover_cases
    from parsec.evals.runner import run_cases

    case_dirs = discover_cases(args.cases_root)
    if not case_dirs:
        console.print(f"[red]no cases found under {args.cases_root}[/red]")
        return EXIT_USAGE

    def factory(config: RunConfig):
        return make_adapter(config)

    judge_adapter = None
    if args.judge == "openai":
        from parsec.gateway.openai_adapter import OpenAIAdapter

        judge_adapter = OpenAIAdapter()

    clock = RealClock()
    with tempfile.TemporaryDirectory(prefix="parsec-eval-") as workdir:
        run = asyncio.run(
            run_cases(
                case_dirs, Path(workdir), factory, clock, args.model,
                label=args.label, judge_adapter=judge_adapter, judge_model=args.judge_model,
            )
        )
    payload = run.to_payload()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    table = Table("case", "status", "citation", "coverage", "synthesis", "turns")
    for r in run.results:
        s = r.scores
        fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
        table.add_row(r.case_id, r.status, fmt(s.citation_faithfulness), fmt(s.coverage), fmt(s.synthesis), str(r.turns))
    console.print(table)
    console.print(f"aggregate: {payload['aggregate']}  →  {args.out}")
    return EXIT_OK if all(r.error is None for r in run.results) else EXIT_ERROR


def _eval_compare(args) -> int:
    from parsec.evals.regression import compare_runs

    a = json.loads(args.results_a.read_text(encoding="utf-8"))
    b = json.loads(args.results_b.read_text(encoding="utf-8"))
    comparison = compare_runs(a, b, epsilon=args.epsilon)
    if args.as_json:
        print(json.dumps(comparison.to_payload()))
    else:
        table = Table("case", "axis", "before", "after", "delta", "regressed")
        for d in comparison.deltas:
            fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
            table.add_row(
                d.case_id, d.axis, fmt(d.before), fmt(d.after),
                "—" if d.delta is None else f"{d.delta:+.2f}",
                "[red]YES[/red]" if d.regressed else "no",
            )
        console.print(table)
        if comparison.only_in_a or comparison.only_in_b:
            console.print(f"only in A: {comparison.only_in_a}  only in B: {comparison.only_in_b}")
        if comparison.ok:
            console.print("[green]no regressions[/green]")
        else:
            console.print(f"[red]{len(comparison.regressions)} regressions (epsilon={args.epsilon})[/red]")
    return EXIT_OK if comparison.ok else EXIT_PARTIAL


def _eval_make_case(args) -> int:
    from parsec.evals.case import make_case_from_session

    clock = RealClock()
    conn, _ = _open(args.data_dir)
    row = SessionStore(conn, clock).get(args.session_id)
    if row is None:
        console.print(f"[red]unknown session {args.session_id}[/red]")
        return EXIT_USAGE
    conn.close()
    case = make_case_from_session(
        args.data_dir,
        args.fixtures,
        args.out,
        case_id=args.case_id or args.session_id,
        query=row["query"],
    )
    console.print(
        f"case [bold]{case.case_id}[/bold] created at {args.out} — "
        f"edit case.json to add the gold must_find list"
    )
    return EXIT_OK


def cmd_notebook(args) -> int:
    clock = RealClock()
    conn, blobs = _open(args.data_dir)
    if SessionStore(conn, clock).get(args.session_id) is None:
        console.print(f"[red]unknown session {args.session_id}[/red]")
        return EXIT_USAGE
    notebook = Notebook(conn, EventLog(conn, clock), clock)
    print(notebook.render(args.session_id))
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
        "fork": cmd_fork,
        "judge": cmd_judge,
        "eval": cmd_eval,
        "sessions": cmd_sessions,
        "notebook": cmd_notebook,
        "spans": cmd_spans,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
