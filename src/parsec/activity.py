"""Live activity view: narrates a run from the event stream.

Purely a display tap — it subscribes to EventLog appends and the loop's
snapshot reporter, and renders a spinner (current phase), a rolling tail of
recent activity, and a stats footer. Nothing here feeds back into the run,
so recording and replay are untouched.
"""

from __future__ import annotations

from collections import deque
from urllib.parse import urlparse

from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from parsec.models.events import EventType

_TAIL = 8
_DIM = "dim"


def _short(text: str, limit: int = 70) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _host(url: str) -> str:
    return urlparse(url).netloc or _short(url, 40)


class ActivityView:
    """One live region per run: what parsec is doing, right now."""

    def __init__(self, console: Console):
        self._spinner = Spinner("dots", text=Text("starting…", style="#22d3ee"))
        self._tail: deque[Text] = deque(maxlen=_TAIL)
        self._stats = Text("")
        self._live = Live(
            self._render(), console=console, refresh_per_second=8, transient=True
        )

    def __enter__(self) -> "ActivityView":
        self._live.start()
        return self

    def __exit__(self, *exc) -> None:
        self._live.stop()

    # -- taps ---------------------------------------------------------------

    def on_event(self, event_type: EventType, payload: dict, stream_id: str) -> None:
        who = "" if stream_id == "orchestrator" else f"[{stream_id}] "
        line: Text | None = None
        match event_type:
            case EventType.LLM_REQUEST:
                self._busy(f"{who}thinking (model call {payload.get('call_index', '?')})…")
            case EventType.TOOL_INTENT:
                line = self._tool_line(who, payload)
            case EventType.FETCH_PERFORMED:
                outcome = payload.get("outcome", "?")
                style = _DIM if outcome == "ok" else "yellow"
                line = Text(f"  ↓ {who}{_host(payload.get('url', ''))} — {outcome}", style=style)
            case EventType.STATE_TRANSITION:
                to = str(payload.get("to_state", "")).replace("_", " ").lower()
                self._busy(f"{to}…")
                line = Text(f"  → {to}", style="#818cf8")
            case EventType.RESEARCH_BRIEF:
                n = len(payload.get("subquestions", []))
                line = Text(
                    f"  ▤ {who}research brief ({payload.get('effort', '?')}, "
                    f"{n} subquestion{'s' if n != 1 else ''})",
                    style="#a5b4fc",
                )
                if payload.get("status") == "proposed":
                    self._busy("brief proposed — waiting if gated…")
            case EventType.SUBAGENT_STARTED:
                line = Text(f"  ▸ {payload.get('sq_id', stream_id)} dispatched", style="#22d3ee")
            case EventType.SUBAGENT_JOINED:
                line = Text(f"  ✓ {payload.get('sq_id', stream_id)} joined", style="green")
            case EventType.GAP_FILL_STARTED:
                line = Text("  ↻ gap-fill: hunting the weakest premise", style="yellow")
            case EventType.CONTEXT_COMPACTED:
                line = Text(f"  ≋ {who}context compacted", style=_DIM)
            case EventType.STEERING_INJECTED:
                line = Text(f"  ⇨ steering: {_short(payload.get('text', ''), 60)}", style="magenta")
            case EventType.LLM_FAILED:
                line = Text(f"  ✗ {who}model call failed", style="red")
            case EventType.VERIFICATION_COMPLETED:
                line = Text("  ✓ verification complete", style="green")
            case EventType.ANSWER_EMITTED:
                self._busy("finalizing report…")
            case _:
                return
        if line is not None:
            self._tail.append(line)
        self._live.update(self._render())

    def on_snapshot(self, snap: dict) -> None:
        self._stats = Text(
            f"{snap.get('turns', 0)} turns · {snap.get('premises', 0)} premises · "
            f"{snap.get('claims', 0)} claims · {snap.get('tokens', 0)} tokens · "
            f"${snap.get('usd', 0.0):.4f}",
            style=_DIM,
        )
        self._live.update(self._render())

    # -- rendering ----------------------------------------------------------

    def _tool_line(self, who: str, payload: dict) -> Text | None:
        name = payload.get("tool_name", "")
        inp = payload.get("input") or {}
        match name:
            case "search_broad":
                return Text(f"  ⌕ {who}search: “{_short(inp.get('query', ''), 60)}”", style="cyan")
            case "search_within":
                return Text(
                    f"  ⌕ {who}search within corpus: “{_short(inp.get('query', ''), 50)}”",
                    style="cyan",
                )
            case "fetch":
                self._busy(f"{who}fetching {_host(inp.get('url', ''))}…")
                return None  # FETCH_PERFORMED reports the outcome
            case "record_premises":
                n = len(inp.get("premises", []))
                return Text(f"  + {who}record {n} premise{'s' if n != 1 else ''}", style="green")
            case _:
                return Text(f"  · {who}{name}", style=_DIM)

    def _busy(self, text: str) -> None:
        self._spinner.update(text=Text(text, style="#22d3ee"))

    def _render(self) -> Group:
        return Group(self._spinner, *self._tail, self._stats)
