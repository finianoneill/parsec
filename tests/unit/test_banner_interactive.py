from __future__ import annotations

import io

from rich.console import Console

import parsec.cli as cli
from parsec.banner import print_banner, render_banner


def test_banner_renders_wordmark_and_tagline():
    lines = render_banner(["data: data"])
    flat = "\n".join(line.plain for line in lines)
    assert "██████" in flat  # wordmark blocks
    assert "N O   C L A I M   W I T H O U T   A   P A T H" in flat
    assert "data: data" in flat


def test_banner_gradient_spans_indigo_to_cyan():
    """Asserted on the styles render_banner() produces, not on ANSI bytes.

    rich caches a Style's escape codes from the first console that renders
    it, without keying on that console's colour system. Any earlier test that
    printed the banner through a 256-colour console (the CLI's own, under
    FORCE_COLOR without COLORTERM=truecolor) therefore left downgraded codes
    behind, and a byte-level assertion here failed depending on test order
    and environment."""
    wordmark = [line for line in render_banner() if "█" in line.plain]
    assert wordmark
    for line in wordmark:
        colours = [str(span.style) for span in line.spans]
        assert len(colours) == len(line.plain)  # one colour per character
        assert colours[0] == "#a5b4fc"  # indigo at the left edge
        assert len(set(colours)) > 10  # a gradient, not a flat fill
        # ends toward cyan: red falls and stays below blue all the way across
        reds = [int(c[1:3], 16) for c in colours]
        blues = [int(c[5:7], 16) for c in colours]
        assert reds == sorted(reds, reverse=True) and reds[-1] < reds[0] - 60
        assert all(blue > red for red, blue in zip(reds[1:], blues[1:], strict=True))


def test_print_banner_emits_colour_on_a_colour_terminal():
    buf = io.StringIO()
    console = Console(
        file=buf, force_terminal=True, color_system="truecolor", width=120, no_color=False
    )
    print_banner(console)
    assert "\x1b[" in buf.getvalue()  # styled at all; exact codes are rich's business


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


def test_slash_watch_maps_to_cli_argv(tmp_path):
    from parsec.interactive import _slash_to_argv

    d = ["--data-dir", str(tmp_path)]
    assert _slash_to_argv("watch", ["s-1"], tmp_path) == ["watch", "s-1", *d]
    assert _slash_to_argv("watch", ["s-1", "6h"], tmp_path) == ["watch", "s-1", "--every", "6h", *d]
    assert _slash_to_argv("watch", [], tmp_path) is None
