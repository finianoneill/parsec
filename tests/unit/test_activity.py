from __future__ import annotations

import io

from rich.console import Console

from parsec.activity import ActivityView
from parsec.models.events import EventType
from parsec.store.event_log import EventLog


def test_event_log_listener_taps_appends(db, clock, sessions, config):
    sessions.create(config)
    log = EventLog(db, clock)
    seen = []
    log.listener = lambda et, payload, stream: seen.append((et, payload, stream))
    log.append(config.session_id, "harness", EventType.SESSION_STARTED, {"a": 1})
    assert seen == [(EventType.SESSION_STARTED, {"a": 1}, "orchestrator")]


def test_event_log_listener_errors_never_break_append(db, clock, sessions, config):
    sessions.create(config)
    log = EventLog(db, clock)
    log.listener = lambda *_: 1 / 0
    seq = log.append(config.session_id, "harness", EventType.SESSION_STARTED, {})
    assert seq is not None
    assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_activity_view_renders_every_event_shape():
    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    with ActivityView(console) as view:
        view.on_event(EventType.LLM_REQUEST, {"call_index": 0}, "orchestrator")
        view.on_event(
            EventType.RESEARCH_BRIEF,
            {"effort": "deep", "subquestions": ["q"], "status": "proposed"},
            "orchestrator",
        )
        view.on_event(EventType.STATE_TRANSITION, {"to_state": "DISPATCHING"}, "orchestrator")
        view.on_event(EventType.SUBAGENT_STARTED, {"sq_id": "sq-1"}, "orchestrator")
        view.on_event(
            EventType.TOOL_INTENT,
            {"tool_name": "search_broad", "input": {"query": "x" * 200}},
            "sq-1",
        )
        view.on_event(
            EventType.TOOL_INTENT, {"tool_name": "fetch", "input": {"url": "https://a.test/p"}},
            "sq-1",
        )
        view.on_event(
            EventType.FETCH_PERFORMED,
            {"url": "https://a.test/p", "outcome": "blocked_by_robots"},
            "sq-1",
        )
        view.on_event(
            EventType.TOOL_INTENT,
            {"tool_name": "record_premises", "input": {"premises": [{}, {}]}},
            "sq-1",
        )
        view.on_event(EventType.SUBAGENT_JOINED, {"sq_id": "sq-1"}, "orchestrator")
        view.on_event(EventType.NODE_ADDED, {}, "orchestrator")  # unmapped: ignored
        view.on_snapshot({"turns": 3, "premises": 2, "claims": 1, "tokens": 500, "usd": 0.01})


def test_busy_line_names_orchestrator_calls_by_phase():
    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    with ActivityView(console) as view:
        # before any transition, the orchestrator call is the decomposer
        view.on_event(EventType.LLM_REQUEST, {"call_index": 0}, "orchestrator")
        assert view._busy_label == "decomposer thinking…"
        view.on_event(EventType.STATE_TRANSITION, {"to_state": "WRITING"}, "orchestrator")
        view.on_event(EventType.LLM_REQUEST, {"call_index": 1}, "orchestrator")
        assert view._busy_label == "writer thinking…"
        view.on_event(EventType.STATE_TRANSITION, {"to_state": "VERIFYING"}, "orchestrator")
        view.on_event(EventType.LLM_REQUEST, {"call_index": 2}, "orchestrator")
        assert view._busy_label == "writer repairing citations…"
        # subagent calls keep their per-stream index
        view.on_event(EventType.LLM_REQUEST, {"call_index": 3}, "sq-1")
        assert view._busy_label == "[sq-1] thinking (model call 3)…"


def test_busy_line_elapsed_ticker_starts_and_stops():
    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    with ActivityView(console) as view:
        view.on_event(EventType.STATE_TRANSITION, {"to_state": "WRITING"}, "orchestrator")
        assert view._busy_since is None  # phase changes don't tick
        view.on_event(EventType.LLM_REQUEST, {"call_index": 1}, "orchestrator")
        assert view._busy_since is not None  # in-flight call ticks
        view._busy_since -= 61  # pretend the call has been in flight a while
        spinner_text = str(view._render().renderables[0].text)
        assert "writer thinking… · 1m01s" in spinner_text
        view.on_event(EventType.LLM_RESPONSE, {"call_index": 1}, "orchestrator")
        assert view._busy_since is None  # response landed: ticker stops


def test_fmt_elapsed():
    assert ActivityView._fmt_elapsed(3.9) == "3s"
    assert ActivityView._fmt_elapsed(59.9) == "59s"
    assert ActivityView._fmt_elapsed(61) == "1m01s"
    assert ActivityView._fmt_elapsed(600) == "10m00s"
