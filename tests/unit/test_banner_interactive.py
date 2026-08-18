from __future__ import annotations

import io

from rich.console import Console

import parsec.cli as cli
from parsec.banner import render_banner, print_banner


def test_banner_renders_wordmark_and_tagline():
    lines = render_banner(["data: data"])
    flat = "\n".join(line.plain for line in lines)
    assert "██████" in flat  # wordmark blocks
    assert "N O   C L A I M   W I T H O U T   A   P A T H" in flat
    assert "data: data" in flat


def test_banner_gradient_spans_indigo_to_cyan():
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, color_system="truecolor", width=120)
    print_banner(console)
    out = buf.getvalue()
    assert "\x1b[38;2;165;180;252" in out  # #a5b4fc at the left edge
    assert "\x1b[38;2;" in out


def test_no_args_enters_interactive_and_exits_on_eof(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # immediate EOF
    assert cli.main(["--data-dir", str(tmp_path / "data")]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "N O   C L A I M" in out


def test_interactive_slash_demo_runs_offline(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO("/demo\n/exit\n"))
    assert cli.main(["--data-dir", str(tmp_path / "data")]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "3.26 light-years" in out
