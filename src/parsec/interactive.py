"""Interactive shell: `parsec` with no subcommand lands here.

A thin REPL over the existing subcommands — bare text runs `ask` (live
adapter when ANTHROPIC_API_KEY is set, otherwise it points at /demo), and
slash commands map onto the CLI verbs so every path stays identical to the
scripted one.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from rich.console import Console

from parsec.banner import print_banner

HELP = """\
  <question>          run a research query (needs ANTHROPIC_API_KEY)
  /demo               run the built-in offline demo — no keys, no network
  /sessions           list recorded sessions
  /show <id>          show one session
  /replay <id>        re-run a recorded session against its frozen corpus
  /verify <id>        verification report for a session
  /notebook <id>      print a session's notebook
  /help               this help
  /exit               leave (also ctrl-d)"""


def has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def run_interactive(data_dir: Path, console: Console | None = None) -> int:
    # Imported here, not at module top: cli imports us for the no-args path.
    import parsec.cli as cli

    console = console or cli.console
    keyed = has_api_key()
    status = [
        f"data: {data_dir}",
        "model: anthropic (ANTHROPIC_API_KEY found)" if keyed
        else "model: none — no ANTHROPIC_API_KEY; /demo runs fully offline",
    ]
    print_banner(console, status)
    console.print("[dim]type a question, /demo for the offline tour, /help for commands[/dim]\n")

    while True:
        try:
            line = console.input("[bold #22d3ee]parsec ❯[/] ").strip()
        except (EOFError, KeyboardInterrupt, OSError):
            console.print()
            return cli.EXIT_OK
        if not line:
            continue
        if line.startswith("/"):
            parts = shlex.split(line)
            cmd, rest = parts[0].lstrip("/"), parts[1:]
            if cmd in ("exit", "quit", "q"):
                return cli.EXIT_OK
            if cmd == "help":
                console.print(HELP)
                continue
            argv = _slash_to_argv(cmd, rest, data_dir)
            if argv is None:
                console.print(f"[red]unknown command /{cmd}[/red] — /help lists them")
                continue
        else:
            if not keyed:
                console.print(
                    "[yellow]no ANTHROPIC_API_KEY set[/yellow] — live queries need one. "
                    "Try [bold]/demo[/bold] for the offline tour, or export a key and restart."
                )
                continue
            argv = ["ask", line, "--data-dir", str(data_dir), "--live"]

        try:
            code = cli.main(argv)
            if code not in (cli.EXIT_OK,):
                console.print(f"[dim]exit {code}[/dim]")
        except SystemExit as exc:  # argparse errors on bad slash-command args
            if exc.code not in (0, None):
                console.print("[red]bad arguments[/red] — /help shows usage")
        except Exception as exc:  # noqa: BLE001 — keep the shell alive
            console.print(f"[red]{type(exc).__name__}: {exc}[/red]")


def _slash_to_argv(cmd: str, rest: list[str], data_dir: Path) -> list[str] | None:
    d = ["--data-dir", str(data_dir)]
    match cmd:
        case "demo":
            return ["demo", *d]
        case "sessions":
            return ["sessions", "list", *d]
        case "show" if rest:
            return ["sessions", "show", rest[0], *d]
        case "replay" if rest:
            return ["replay", rest[0], *d]
        case "verify" if rest:
            return ["verify", rest[0], *d]
        case "notebook" if rest:
            return ["notebook", rest[0], *d]
        case _:
            return None
