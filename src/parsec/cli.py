"""Parsec CLI: ask / replay / sessions / spans."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import select
import sys
import tempfile
import threading
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

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

console = Console()

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_PARTIAL = 3
EXIT_ERROR = 4


def build_parser() -> argparse.ArgumentParser:
    from parsec import __version__

    p = argparse.ArgumentParser(
        prog="parsec",
        description="Local-first research harness. No subcommand starts the interactive shell.",
    )
    p.add_argument("--version", action="version", version=f"parsec {__version__}")
    p.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Data directory for the interactive shell (subcommands take their own)",
    )
    sub = p.add_subparsers(dest="command", required=False)

    demo = sub.add_parser(
        "demo",
        help="Run the built-in offline demo: a full recorded run, no API keys, no network",
    )
    demo.add_argument("--data-dir", type=Path, default=Path("data"))

    ask = sub.add_parser("ask", help="Run a research query")
    ask.add_argument("query")
    ask.add_argument("--session-id")
    ask.add_argument("--cache-mode", choices=[m.value for m in CacheMode], default=CacheMode.LIVE_PREFER_CACHE.value)
    ask.add_argument(
        "--adapter", choices=["anthropic", "bedrock", "fake", "replay"], default="anthropic"
    )
    ask.add_argument(
        "--aws-region", default=None,
        help="Bedrock region (adapter=bedrock); also settable as aws_region in config",
    )
    ask.add_argument(
        "--aws-profile", default=None,
        help="AWS credentials profile for Bedrock (e.g. the one okta-awscli writes)",
    )
    ask.add_argument("--model", default=DEFAULT_MODEL)
    ask.add_argument("--max-usd", type=float, default=Budgets().max_usd)
    ask.add_argument("--max-tokens", type=int, default=Budgets().max_total_tokens)
    ask.add_argument("--max-seconds", type=int, default=Budgets().max_wall_seconds)
    ask.add_argument("--max-turns", type=int, default=Budgets().max_turns)
    ask.add_argument("--max-gap-rounds", type=int, default=Budgets().max_gap_rounds)
    ask.add_argument(
        "--max-coverage-gap-rounds", type=int, default=Budgets().max_coverage_gap_rounds,
        help="Retries for subquestions still PARTIAL while budget headroom remains (0 disables)",
    )
    ask.add_argument(
        "--max-turns-per-subagent", type=int, default=Budgets().max_turns_per_subagent,
        help="Model calls each subagent may spend (search/fetch/record/report all count)",
    )
    ask.add_argument(
        "--parallel", type=int, default=Budgets().parallel_subagents,
        help="Concurrent subagents per wave (1-5); 1 = sequential (required for fork --at-call)",
    )
    ask.add_argument(
        "--brief-gate", action="store_true",
        help="Pause after the research brief for approval/edits via stdin steering (type 'approve' to dispatch)",
    )
    ask.add_argument("--data-dir", type=Path, default=Path("data"))
    ask.add_argument("--search-fixtures", type=Path, default=None)
    ask.add_argument(
        "--search-provider", choices=["fixture", "searxng", "brave", "serper"], default="fixture",
        help="Live providers need keys/urls: BRAVE_API_KEY / SERPER_API_KEY env, --searxng-url",
    )
    ask.add_argument("--searxng-url", default=None)
    ask.add_argument("--contact", default=None, help="Contact info appended to the fetch User-Agent")
    ask.add_argument("--no-robots", action="store_true", help="Skip robots.txt checks (not recommended)")
    ask.add_argument(
        "--nli-checker", choices=["lexical", "hhem", "none"], default="lexical",
        help="Grounded premise-support tier (advisory): hhem needs the `nli` extra",
    )
    ask.add_argument(
        "--calibration", type=Path, default=None,
        help="calibration.json from `parsec calibrate`; enables range-backed tiers (\"high (72–96%%)\")",
    )
    ask.add_argument(
        "--learned-reliability", action="store_true",
        help="Adjust source-tier priors ±cap by truth-discovery over the session's own graph",
    )
    ask.add_argument("--json", action="store_true", dest="as_json")
    ask.add_argument(
        "--live", action="store_true",
        help="Force the live activity view (on by default in a terminal; --json disables)",
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
        "verify",
        help="Run mechanical verification (structural + temporal + NLI advisories) over a session's evidence DAG",
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
    ev_run.add_argument(
        "--runs", type=int, default=1,
        help="Runs per case (scores are means; frozen corpora leave only agent sampling variance)",
    )
    ev_run.add_argument(
        "--support-checker", choices=["mechanical", "grounded"], default="mechanical",
        help="Claim-support axis grader: grounded adds the M9 lexical-NLI tier over the exact-match floor",
    )
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

    cal = sub.add_parser(
        "calibrate",
        help="Fit Platt scaling over labeled credences; report Brier, smooth ECE, risk-coverage",
    )
    cal.add_argument(
        "labels", nargs="+", type=Path,
        help="Labels JSON: a list of {credence, label} pairs, or eval results files carrying per-case labels",
    )
    cal.add_argument("--out", type=Path, default=None, help="Output path (default <data-dir>/calibration.json)")
    cal.add_argument("--data-dir", type=Path, default=Path("data"))

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
    if config.adapter == "bedrock":
        from parsec.gateway.bedrock_adapter import BedrockAdapter

        return BedrockAdapter(
            aws_region=config.aws_region or os.environ.get("AWS_REGION"),
            aws_profile=config.aws_profile,
        )
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
    user_agent = USER_AGENT + (f" contact:{config.contact}" if config.contact else "")
    robots = (
        RobotsPolicy(conn, clock, user_agent, config.robots_ttl_s, transport=fetch_transport)
        if config.respect_robots
        else None
    )
    fetcher = Fetcher(
        documents, blobs, clock, config.cache_mode,
        transport=fetch_transport, robots=robots, user_agent=user_agent,
    )
    embeddings = EmbeddingCache(conn, HashedNgramEmbedder())
    tools: list = [
        FetchTool(fetcher, spans),
        RecordPremisesTool(dag, spans, documents),
        SearchWithinTool(spans, embeddings),
    ]
    provider = build_search_provider(config, conn, clock, transport=fetch_transport)
    if provider is not None:
        tools.append(SearchBroadTool(provider))
    registry = ToolRegistry(tools)
    ctx = ToolContext(conn, blobs, event_log, ledger, config, clock)
    return OrchestratorLoop(
        config, gateway, registry, ctx, sessions, dag, spans, documents, coverage, notebook
    )


# A bracketed citation group ([premise:...] / [finding:...], possibly several
# comma-separated) or a [narrative] tag, as the writer emits them.
_CITATION_TAG_RE = re.compile(
    r"\[(?:\s*(?:premise|finding):[0-9a-f]{16}\s*,?)+\]|\[narrative\]"
)


def display_answer(answer: str) -> Text:
    """Answer text prepared for the terminal: citation tags stripped and the
    whitespace they leave before punctuation closed up. Returned as Text so
    printing is markup-safe — previously Rich happened to eat the tags as
    unknown markup, which also meant any bracketed text in an answer could
    style or break the output."""
    text = _CITATION_TAG_RE.sub("", answer)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    return Text(text)


class SteeringReader:
    """Reads steering lines from stdin only while a run is in flight.

    Polls with select() instead of blocking in a read so stop() can end the
    thread at run completion — a reader left blocked on stdin would race the
    interactive shell's readline for every later keystroke.
    """

    _POLL_SECONDS = 0.2

    def __init__(self, on_line, stdin=None):
        self._on_line = on_line
        self._stdin = stdin if stdin is not None else sys.stdin
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2 * self._POLL_SECONDS)

    def _read(self) -> None:
        try:
            while not self._stop.is_set():
                ready, _, _ = select.select([self._stdin], [], [], self._POLL_SECONDS)
                if not ready:
                    continue
                line = self._stdin.readline()
                if not line:  # EOF
                    return
                text = line.strip()
                if text:
                    self._on_line(text)
        except (ValueError, OSError):
            pass


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
        aws_region=args.aws_region,
        aws_profile=args.aws_profile,
        budgets=Budgets(
            max_usd=args.max_usd,
            max_total_tokens=args.max_tokens,
            max_wall_seconds=args.max_seconds,
            max_turns=args.max_turns,
            max_gap_rounds=args.max_gap_rounds,
            max_coverage_gap_rounds=args.max_coverage_gap_rounds,
            max_turns_per_subagent=args.max_turns_per_subagent,
            parallel_subagents=args.parallel,
        ),
        data_dir=args.data_dir,
        search_fixtures=args.search_fixtures,
        search_provider=args.search_provider,
        searxng_url=args.searxng_url,
        respect_robots=not args.no_robots,
        contact=args.contact,
        brief_gate=args.brief_gate,
        nli_checker=args.nli_checker,
        learned_reliability=args.learned_reliability,
        calibration=(
            json.loads(args.calibration.read_text(encoding="utf-8"))
            if args.calibration
            else None
        ),
    )
    loop = _build_loop(config, conn, blobs, clock)

    # Activity view: narrates the run from the event stream (thinking, searches,
    # fetches, subagents, phases). Default on a terminal; --live forces it on,
    # --json disables it.
    live_view = None
    if not args.as_json and (args.live or sys.stdout.isatty()):
        from parsec.activity import ActivityView

        live_view = ActivityView(console)
        live_view.__enter__()
        loop.event_log.listener = live_view.on_event
        loop.reporter = live_view.on_snapshot

    # Steering (§3): lines typed on stdin mid-run are injected into the next
    # model call without tearing down the turn.
    steering = None
    if sys.stdin.isatty() and not args.as_json:
        if config.brief_gate:
            console.print(
                "[dim]brief gate: type 'approve' to dispatch, 'edit' to open the "
                "proposed brief in $EDITOR, or any other text to request changes[/dim]"
            )
        steering = SteeringReader(lambda text: handle_steer_line(loop, text))
        steering.start()

    try:
        result = asyncio.run(loop.run())
    except KeyboardInterrupt:
        console.print("[red]aborted[/red]")
        return EXIT_ERROR
    finally:
        if steering is not None:
            steering.stop()
        if live_view is not None:
            live_view.__exit__()
            loop.event_log.listener = None

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
        console.print(display_answer(result.answer))
        if result.coverage:
            console.print(
                "[dim]coverage: "
                + " · ".join(f"{sq} {status}" for sq, status in sorted(result.coverage.items()))
                + "[/dim]"
            )
        cached = int(
            totals.get("cache_read_tokens", 0) + totals.get("cache_creation_tokens", 0)
        )
        spent_budget, budget_cap = ledger.spent_tokens(session_id), config.budgets.max_total_tokens
        budget_note = f"budget {spent_budget}/{budget_cap}"
        if spent_budget >= budget_cap:
            # Research stops at the cap but the writer runs in the grace, so
            # a finished run can land over — say so instead of hiding it.
            budget_note = f"[red]{budget_note} (over)[/red]"
        token_note = (
            f"{int(totals.get('input_tokens', 0))} in / {int(totals.get('output_tokens', 0))} out"
            + (f" / {cached} cache" if cached else "")
            + f" tokens · {budget_note}"
        )
        console.print(
            f"\n[dim]session {result.session_id} · {result.status} · {result.turns} turns · "
            f"{token_note} · "
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


def cmd_demo(args) -> int:
    global adapter_factory, fetch_transport
    from parsec import demo as demo_mod

    fixtures = demo_mod.write_search_fixtures(args.data_dir)
    clock = RealClock()
    session_id = make_session_id(demo_mod.DEMO_QUERY, clock.now_iso())
    console.print(
        "[dim]offline demo: scripted model + bundled fixture corpus — "
        "no API keys, no network. The recording is a real session.[/dim]\n"
    )
    prev = (adapter_factory, fetch_transport)
    adapter_factory = demo_mod.demo_adapter_factory
    fetch_transport = demo_mod.demo_transport()
    try:
        ask_args = build_parser().parse_args(
            [
                "ask", demo_mod.DEMO_QUERY,
                "--session-id", session_id,
                "--adapter", "fake",
                "--model", "fake-model",
                "--cache-mode", "record",
                "--search-provider", "fixture",
                "--search-fixtures", str(fixtures),
                "--max-gap-rounds", "0",
                "--max-coverage-gap-rounds", "0",
                "--data-dir", str(args.data_dir),
            ]
        )
        code = cmd_ask(ask_args)
    finally:
        adapter_factory, fetch_transport = prev
    if code == EXIT_OK:
        d = f" --data-dir {args.data_dir}" if args.data_dir != Path("data") else ""
        console.print(
            f"\n[dim]poke at the recording:[/dim]\n"
            f"  parsec replay {session_id}{d}\n"
            f"  parsec verify {session_id}{d}\n"
            f"  parsec notebook {session_id}{d}"
        )
    return code


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
    from parsec.verify.credence import annotate, compute_credences, render_tier
    from parsec.verify.nli import make_grounded_checker
    from parsec.verify.omission import detect_omissions
    from parsec.verify.structural import verify_session

    clock = RealClock()
    conn, blobs = _open(args.data_dir)
    store = SessionStore(conn, clock)
    if store.get(args.session_id) is None:
        console.print(f"[red]unknown session {args.session_id}[/red]")
        return EXIT_USAGE
    session_config = store.get_config(args.session_id)
    report = verify_session(
        conn, blobs, args.session_id,
        nli_checker=make_grounded_checker(session_config.nli_checker),
    )
    credence = compute_credences(
        conn,
        args.session_id,
        source_tiers=session_config.source_tiers,
        stakes_threshold=session_config.stakes_threshold,
        volatile_penalty=session_config.volatile_penalty,
        volatile_half_life_days=session_config.volatile_half_life_days,
        slow_half_life_days=session_config.slow_half_life_days,
        learned_reliability=session_config.learned_reliability,
    )
    omissions = detect_omissions(conn, EventLog(conn, clock), args.session_id)
    if args.as_json:
        payload = report.to_payload()
        payload["credence"] = {
            "flagged_claims": credence.flagged_claims,
            "tiers": {nid: render_tier(nc.credence) for nid, nc in sorted(credence.nodes.items())},
            # M10 uncertainty provenance: "conflicting sources", "possibly
            # stale", "superseded", "single source" — straight from the graph
            "provenance": {nid: annotate(nc) for nid, nc in sorted(credence.nodes.items())},
        }
        if credence.source_reliability:
            payload["credence"]["source_reliability"] = credence.source_reliability
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
        for v in report.advisories:
            console.print(f"[yellow]advisory ({v.check}):[/yellow] {v.subject} — {v.detail}")
        if credence.flagged_claims:
            console.print(
                f"[yellow]{len(credence.flagged_claims)} claims below the stakes threshold[/yellow]"
            )
        for nid, nc in sorted(credence.nodes.items()):
            if nc.conflicted or nc.superseded_by or nc.stale:
                console.print(f"[yellow]{annotate(nc)}:[/yellow] {nid}")
        for dom, provenance in sorted(credence.source_reliability.items()):
            console.print(f"[dim]reliability {dom}: {provenance}[/dim]")
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
        console.print(display_answer(blobs.get_text(row["answer_blob"])))
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
        console.print(display_answer(result.answer))
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
    from parsec.evals.support import make_support_checker

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
                runs=args.runs, support_checker=make_support_checker(args.support_checker),
            )
        )
    payload = run.to_payload()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    table = Table("case", "status", "citation", "coverage", "nuggets", "support", "synthesis", "gold%", "turns")
    for r in run.results:
        s = r.scores
        fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
        gold = fmt(r.trajectory.gold_fetch_fraction) if r.trajectory else "—"
        table.add_row(
            r.case_id, r.status, fmt(s.citation_faithfulness), fmt(s.coverage),
            fmt(s.nugget_recall), fmt(s.claim_support), fmt(s.synthesis), gold, str(r.turns),
        )
        if s.nugget_contradictions:
            console.print(f"[red]{r.case_id}: report contradicts gold: {s.nugget_contradictions}[/red]")
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
        table = Table("case", "axis", "before", "after", "delta", "flag")
        for d in comparison.deltas:
            fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
            table.add_row(
                d.case_id, d.axis, fmt(d.before), fmt(d.after),
                "—" if d.delta is None else f"{d.delta:+.2f}",
                "[red]drop[/red]" if d.regressed else "",
            )
        console.print(table)
        verdicts = Table("axis", "verdict", "n", "mean Δ", "±CI95")
        for v in comparison.verdicts:
            color = {"regressed": "red", "improved": "green"}.get(v.verdict, "yellow")
            verdicts.add_row(
                v.axis, f"[{color}]{v.verdict}[/{color}]", str(v.n_cases),
                "—" if v.mean_delta is None else f"{v.mean_delta:+.3f}",
                "—" if v.ci95 is None else f"{v.ci95:.3f}",
            )
        console.print(verdicts)
        if comparison.only_in_a or comparison.only_in_b:
            console.print(f"only in A: {comparison.only_in_a}  only in B: {comparison.only_in_b}")
        if comparison.ok:
            console.print("[green]no significant regressions[/green]")
        else:
            axes = ", ".join(v.axis for v in comparison.regressed_axes)
            console.print(f"[red]significant regression on: {axes} (epsilon={args.epsilon})[/red]")
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


def _extract_label_pairs(data) -> list[tuple[float, int]]:
    """Accept a bare list of {credence, label} pairs, an object with a
    top-level "labels" list, or an eval results file whose per-case results
    carry harvested labels."""
    if isinstance(data, dict):
        if "results" in data:
            items = [pair for result in data["results"] for pair in (result.get("labels") or [])]
        else:
            items = data.get("labels", [])
    else:
        items = data
    return [(float(item["credence"]), int(item["label"])) for item in items]


def cmd_calibrate(args) -> int:
    from parsec.verify.calibration import MIN_LABELS, RECOMMENDED_LABELS, calibration_report

    pairs: list[tuple[float, int]] = []
    for path in args.labels:
        pairs += _extract_label_pairs(json.loads(path.read_text(encoding="utf-8")))
    if len(pairs) < MIN_LABELS:
        console.print(
            f"[red]calibration needs at least {MIN_LABELS} labeled claims; got {len(pairs)}[/red]"
        )
        return EXIT_USAGE

    payload = calibration_report(pairs)
    out = args.out or (args.data_dir / "calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    console.print(f"fitted Platt scaling on {payload['n']} labeled claims")
    metrics = Table("metric", "raw heuristic", "calibrated")
    metrics.add_row("Brier", f"{payload['brier_raw']:.4f}", f"{payload['brier_calibrated']:.4f}")
    metrics.add_row(
        "smooth ECE", f"{payload['smooth_ece_raw']:.4f}", f"{payload['smooth_ece_calibrated']:.4f}"
    )
    console.print(metrics)
    ranges = ", ".join(f"{t} = {lo}–{hi}%" for t, (lo, hi) in payload["tier_ranges"].items())
    console.print(f"tier ranges (calibrated): {ranges}")
    rc = Table("coverage", "risk")
    for point in payload["risk_coverage"]:
        rc.add_row(f"{point['coverage']:.0%}", f"{point['risk']:.1%}")
    console.print(rc)
    if payload["underpowered"]:
        console.print(
            f"[yellow]only {payload['n']} labels (< {RECOMMENDED_LABELS}): the fit is weak — "
            "treat ranges as provisional and keep labeling[/yellow]"
        )
    console.print(f"→ {out}  (pass to `parsec ask --calibration {out}` for range-backed tiers)")
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


def compose_in_editor(initial: str = "") -> str | None:
    """Open $VISUAL/$EDITOR on a temp file seeded with `initial`; return the
    saved text (stripped), or None on abort/empty."""
    import shlex
    import subprocess
    import tempfile

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    fd, path = tempfile.mkstemp(suffix=".md", prefix="parsec-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(initial)
        code = subprocess.call([*shlex.split(editor), path])
        text = Path(path).read_text(encoding="utf-8").strip()
    finally:
        os.unlink(path)
    return text if code == 0 and text else None


def render_brief_for_editor(brief) -> str:
    questions = "".join(f"- {q}\n" for q in brief.questions)
    return (
        "<!-- Edit the research brief below, then save and quit to submit it\n"
        "     as one brief-edit steering message. Quit without saving (or save\n"
        "     empty) to abort the edit and keep waiting at the gate. -->\n\n"
        f"## Scope\n{brief.scope or '(none)'}\n\n"
        f"## Effort\n{brief.effort}\n\n"
        f"## Subquestions\n{questions}"
    )


def handle_steer_line(loop, text: str) -> None:
    """One stdin line during a live run. At the brief gate, `edit` opens
    $EDITOR seeded with the proposed brief; the saved text is what gets
    steered (and recorded), so replay semantics are untouched."""
    brief = getattr(loop, "current_brief", None)
    if text.lower() == "edit" and brief is not None:
        edited = compose_in_editor(render_brief_for_editor(brief))
        if edited:
            loop.steer(edited)
        else:
            console.print("[dim]edit aborted; still at the brief gate[/dim]")
        return
    loop.steer(text)


def main(argv: list[str] | None = None) -> int:
    from parsec.user_config import apply_config, load_user_config

    parser = build_parser()
    config, _sources = load_user_config()
    apply_config(parser, config)
    args = parser.parse_args(argv)
    if args.command is None:
        from parsec.interactive import run_interactive

        return run_interactive(args.data_dir, config_sources=_sources, user_config=config)
    handlers = {
        "ask": cmd_ask,
        "demo": cmd_demo,
        "replay": cmd_replay,
        "verify": cmd_verify,
        "fork": cmd_fork,
        "judge": cmd_judge,
        "eval": cmd_eval,
        "calibrate": cmd_calibrate,
        "sessions": cmd_sessions,
        "notebook": cmd_notebook,
        "spans": cmd_spans,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
