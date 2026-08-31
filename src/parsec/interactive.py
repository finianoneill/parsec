"""Interactive shell: `parsec` with no subcommand lands here.

A thin REPL over the existing subcommands — bare text runs `ask` (live
adapter when ANTHROPIC_API_KEY is set, otherwise it points at /demo), and
slash commands map onto the CLI verbs so every path stays identical to the
scripted one.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

from rich.console import Console

from parsec.banner import print_banner

HISTORY_FILE = Path("~/.parsec_history").expanduser()

HELP = """\
  <question>          run a research query (needs ANTHROPIC_API_KEY)
  /edit               compose a query in $EDITOR (for long or multi-line asks)
  /demo               run the built-in offline demo — no keys, no network
  /sessions           list recorded sessions
  /show <id>          show one session
  /replay <id>        re-run a recorded session against its frozen corpus
  /verify <id>        verification report for a session
  /diff <a> <b>       claim-level diff of two sessions of the same question
  /refresh <id>       re-run a session's question now and diff against it (needs a key)
  /watch <id> [every] refresh on a schedule (e.g. /watch s-1 6h), report only material
                      diffs, label the prior claims for calibrate; one round without every
  /notebook <id>      print a session's notebook
  /help               this help
  /exit               leave (also ctrl-d)"""


def has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _init_history() -> None:
    """Arrow-key history across sessions. readline is absent or libedit on
    some builds — every step is best-effort."""
    try:
        import readline
    except ImportError:
        return
    try:
        readline.read_history_file(HISTORY_FILE)
    except OSError:
        pass
    readline.set_history_length(1000)


def _save_history() -> None:
    try:
        import readline

        readline.write_history_file(HISTORY_FILE)
    except (ImportError, OSError):
        pass


def _read_line(console: Console) -> str:
    if sys.stdin.isatty():
        # Plain input() so readline owns the line; \001/\002 mark the ANSI
        # codes as zero-width so history redraw stays aligned.
        return input("\001\x1b[1;38;2;34;211;238m\002parsec ❯\001\x1b[0m\002 ")
    return console.input("[bold #22d3ee]parsec ❯[/] ")


def run_interactive(
    data_dir: Path,
    console: Console | None = None,
    config_sources: list[Path] | None = None,
    user_config: dict | None = None,
) -> int:
    # Imported here, not at module top: cli imports us for the no-args path.
    import parsec.cli as cli

    console = console or cli.console
    bedrock = (user_config or {}).get("adapter") == "bedrock"
    keyed = bedrock or has_api_key()
    if bedrock:
        model_status = "model: bedrock (AWS credential chain — run okta-awscli/aws sso if expired)"
    elif keyed:
        model_status = "model: anthropic (ANTHROPIC_API_KEY found)"
    else:
        model_status = "model: none — no ANTHROPIC_API_KEY; /demo runs fully offline"
    status = [
        f"data: {data_dir}",
        *(
            [f"config: {' + '.join(str(p) for p in config_sources)}"]
            if config_sources
            else []
        ),
        model_status,
    ]
    print_banner(console, status)
    console.print("[dim]type a question, /demo for the offline tour, /help for commands[/dim]\n")
    _init_history()
    try:
        return _repl(cli, console, data_dir, keyed)
    finally:
        _save_history()


def _repl(cli, console: Console, data_dir: Path, keyed: bool) -> int:
    while True:
        try:
            line = _read_line(console).strip()
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
            if cmd == "edit":
                line = cli.compose_in_editor() or ""
                if not line:
                    console.print("[dim]nothing composed[/dim]")
                    continue
                console.print(f"[dim]query:[/dim] {line}")
                argv = None  # falls through to the bare-text path below
            else:
                argv = _slash_to_argv(cmd, rest, data_dir)
                if argv is None:
                    console.print(f"[red]unknown command /{cmd}[/red] — /help lists them")
                    continue
        else:
            argv = None
        if argv is None:
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
        case "diff" if len(rest) >= 2:
            return ["diff", rest[0], rest[1], *d]
        case "refresh" if rest:
            return ["refresh", rest[0], *d]
        case "watch" if rest:
            every = ["--every", rest[1]] if len(rest) > 1 else []
            return ["watch", rest[0], *every, *d]
        case "notebook" if rest:
            return ["notebook", rest[0], *d]
        case _:
            return None
