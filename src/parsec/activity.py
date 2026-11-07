"""Live activity view: narrates a run from the event stream.

Purely a display tap — it subscribes to EventLog appends and the loop's
snapshot reporter, and renders a spinner (current phase), a rolling tail of
recent activity, and a stats footer. Nothing here feeds back into the run,
so recording and replay are untouched.
"""

from __future__ import annotations

import time
from collections import deque
from urllib.parse import urlparse

from rich._spinners import SPINNERS
from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from parsec.models.events import EventType

_TAIL = 8
_DIM = "dim"

# The parsec star: the banner's ✦ mark pulsing — swelling from a point to a
# full burst and back. Registered once at import; rich looks spinners up by
# name.
SPINNERS["parsec-star"] = {
    "interval": 140,
    "frames": ["·", "✦", "✶", "✹", "✺", "✹", "✶", "✦"],
}


def _short(text: str, limit: int = 70) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _host(url: str) -> str:
    return urlparse(url).netloc or _short(url, 40)


class ActivityView:
    """One live region per run: what parsec is doing, right now."""

    def __init__(self, console: Console):
        self._spinner = Spinner(
            "parsec-star", text=Text("starting…", style="#22d3ee"), style="#22d3ee"
        )
        self._tail: deque[Text] = deque(maxlen=_TAIL)
        self._stats = Text("")
        # Busy line state: the label, an optional start time for an elapsed
        # ticker (in-flight model calls — a frozen counter and a silent SDK
        # retry are otherwise indistinguishable), and the loop phase so
        # orchestrator calls get named by role instead of a bare per-stream
        # call index ("writer thinking" beats "model call 1" 39 turns in).
        self._busy_label = "starting…"
        self._busy_since: float | None = None
        self._phase = "PLANNING"
        # Retry state: once the journaled backoff has elapsed the re-attempt
        # is in flight, and the ticker must say so — "waiting to retry · 36s"
        # with a 2s delay reads as a hang (M15.8).
        self._retry_delay: float | None = None
        self._retry_attempt = 0
        self._retry_who = ""
        # get_renderable (not a static renderable): the auto-refresh thread
        # re-renders 8x/s, so the elapsed ticker advances between events.
        self._live = Live(
            get_renderable=self._render, console=console, refresh_per_second=8, transient=True
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
                self._busy(self._call_label(who, payload), timed=True)
            case EventType.LLM_RESPONSE:
                self._busy_since = None  # call landed; stop the ticker
                self._retry_delay = None
                if not payload.get("suspended_s"):
                    return
                line = self._slept_line(who, payload)
            case EventType.TOOL_INTENT:
                line = self._tool_line(who, payload)
            case EventType.FETCH_PERFORMED:
                outcome = payload.get("outcome", "?")
                style = _DIM if outcome == "ok" else "yellow"
                line = Text(f"  ↓ {who}{_host(payload.get('url', ''))} — {outcome}", style=style)
            case EventType.STATE_TRANSITION:
                self._phase = str(payload.get("to_state", ""))
                to = self._phase.replace("_", " ").lower()
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
            case EventType.GATE_PROPOSED:
                line = Text(
                    f"  ⏸ {payload.get('gate', '?')} gate: ${payload.get('spent_usd', 0.0):.4f} of "
                    f"${payload.get('cap_usd', 0.0):.2f} spent — approve to continue",
                    style="yellow",
                )
                self._busy(f"waiting at the {payload.get('gate', '?')} gate…")
            case EventType.LLM_RETRY:
                # The reason the busy ticker exists: a retry is now a visible
                # journaled event, not a silent SDK re-attempt.
                line = Text(
                    f"  ↻ {who}model {payload.get('error_kind', 'error')} — "
                    f"retry {payload.get('attempt', '?')} in {payload.get('delay_s', '?')}s",
                    style="yellow",
                )
                self._busy(f"{who}waiting to retry ({payload.get('error_kind', 'error')})…", timed=True)
                self._retry_delay = float(payload.get("delay_s") or 0.0)
                self._retry_attempt = int(payload.get("attempt") or 0)
                self._retry_who = who
                if payload.get("suspended_s"):
                    line.append(
                        f" — machine slept {self._fmt_elapsed(payload['suspended_s'])} during the call",
                        style="yellow",
                    )
            case EventType.LLM_FAILED:
                line = Text(f"  ✗ {who}model call failed", style="red")
                if payload.get("suspended_s"):
                    line.append(
                        f" — machine slept {self._fmt_elapsed(payload['suspended_s'])} during the call",
                        style="yellow",
                    )
            case EventType.VERIFICATION_COMPLETED:
                line = Text("  ✓ verification complete", style="green")
            case EventType.ANSWER_EMITTED:
                self._busy("finalizing report…")
            case _:
                return
        if line is not None:
            self._tail.append(line)
        self._live.refresh()

    def on_snapshot(self, snap: dict) -> None:
        self._stats = Text(
            f"{snap.get('turns', 0)} turns · {snap.get('premises', 0)} premises · "
            f"{snap.get('claims', 0)} claims · {snap.get('tokens', 0)} tokens · "
            f"${snap.get('usd', 0.0):.4f}",
            style=_DIM,
        )
        self._live.refresh()

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

    def _call_label(self, who: str, payload: dict) -> str:
        """Subagent calls keep their per-stream index (it counts that
        subagent's own turns); orchestrator calls are named by what the loop
        is doing, since their index is meaningless to a reader."""
        if who:
            return f"{who}thinking (model call {payload.get('call_index', '?')})…"
        match self._phase:
            case "PLANNING":
                return "decomposer thinking…"
            case "WRITING":
                return "writer thinking…"
            case "VERIFYING":
                return "writer repairing citations…"
        return f"thinking (model call {payload.get('call_index', '?')})…"

    def _busy(self, text: str, timed: bool = False) -> None:
        self._busy_label = text
        self._busy_since = time.monotonic() if timed else None
        self._retry_delay = None

    def _slept_line(self, who: str, payload: dict) -> Text:
        return Text(
            f"  ⏾ {who}machine slept {self._fmt_elapsed(payload['suspended_s'])} during the call",
            style="yellow",
        )

    @staticmethod
    def _fmt_elapsed(seconds: float) -> str:
        s = int(seconds)
        return f"{s}s" if s < 60 else f"{s // 60}m{s % 60:02d}s"

    def _render(self) -> Group:
        label = self._busy_label
        if self._busy_since is not None:
            elapsed = time.monotonic() - self._busy_since
            if self._retry_delay is not None and elapsed >= self._retry_delay:
                # Backoff over: the re-attempt is the thing in flight now.
                label = f"{self._retry_who}retrying (attempt {self._retry_attempt + 1})…"
                elapsed -= self._retry_delay
            label += f" · {self._fmt_elapsed(elapsed)}"
        self._spinner.update(text=Text(label, style="#22d3ee"))
        return Group(self._spinner, *self._tail, self._stats)
