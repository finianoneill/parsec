"""Welcome banner: the parsec mark and wordmark rendered for the terminal.

Mirrors assets/logo-dark.svg — the parallax diagram (near star resolved by
sight lines from two vantage points on Earth's orbit), the indigo→cyan
gradient wordmark, and the tagline.
"""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

from parsec import __version__

# #a5b4fc → #22d3ee, the wordmark gradient from the SVG.
_GRAD_START = (0xA5, 0xB4, 0xFC)
_GRAD_END = (0x22, 0xD3, 0xEE)

_WORDMARK = [
    "██████   █████  ██████  ███████ ███████  ██████",
    "██   ██ ██   ██ ██   ██ ██      ██      ██     ",
    "██████  ███████ ██████  ███████ █████   ██     ",
    "██      ██   ██ ██   ██      ██ ██      ██     ",
    "██      ██   ██ ██   ██ ███████ ███████  ██████",
]

# The parallax mark: near star, crossing sight lines, orbit with two vantage
# points around the sun.
_MARK = [
    ("   ·        ✦        ·   ", "bright_black"),
    ("       ·   ╱ ╲   ·       ", "bright_black"),
    ("          ╱   ╲          ", "#818cf8"),
    ("         ╱     ╲         ", "#818cf8"),
    ("      ●─╱───☉───╲─●      ", "#c7d2fe"),
]

_TAGLINE = "N O   C L A I M   W I T H O U T   A   P A T H"


def _gradient_line(line: str) -> Text:
    text = Text()
    width = max(len(line), 1)
    for i, ch in enumerate(line):
        t = i / width
        r, g, b = (
            round(s + (e - s) * t) for s, e in zip(_GRAD_START, _GRAD_END)
        )
        text.append(ch, style=f"#{r:02x}{g:02x}{b:02x}")
    return text


def render_banner(status_lines: list[str] | None = None) -> list[Text]:
    """The banner as rich Text lines; status_lines render dim below the tagline."""
    lines: list[Text] = []
    for raw, style in _MARK:
        lines.append(Text(raw, style=style))
    lines.append(Text())
    for raw in _WORDMARK:
        lines.append(_gradient_line(raw))
    lines.append(Text())
    lines.append(Text(_TAGLINE, style="bold #94a3b8"))
    lines.append(Text(f"v{__version__} · local-first research harness", style="dim"))
    for status in status_lines or []:
        lines.append(Text(status, style="dim"))
    return lines


def print_banner(console: Console, status_lines: list[str] | None = None) -> None:
    body = render_banner(status_lines)
    width = max(line.cell_len for line in body)
    pad = max((console.width - width) // 2, 0) if console.is_terminal else 0
    console.print()
    for line in body:
        console.print(Text(" " * pad) + line)
    console.print()
