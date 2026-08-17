import pytest

from parsec.models.events import EventType
from parsec.store.coverage import CoverageLedger
from parsec.store.notebook import Notebook


@pytest.fixture
def coverage(db, event_log, sessions, config):
    sessions.create(config)
    return CoverageLedger(db, event_log)


@pytest.fixture
def notebook(db, event_log, sessions, config, clock):
    if sessions.get(config.session_id) is None:
        sessions.create(config)
    return Notebook(db, event_log, clock)


def test_coverage_lifecycle(coverage, config):
    sid = config.session_id
    coverage.create(sid, "sq-1", "first?")
    coverage.create(sid, "sq-2", "second?")
    assert [r["sq_id"] for r in coverage.open_items(sid)] == ["sq-1", "sq-2"]
    coverage.set_status(sid, "sq-1", "answered")
    coverage.set_status(sid, "sq-2", "blocked", "no sources")
    assert coverage.open_items(sid) == []
    assert coverage.summary(sid) == {"sq-1": "answered", "sq-2": "blocked"}


def test_blocked_requires_reason(coverage, config):
    sid = config.session_id
    coverage.create(sid, "sq-1", "q?")
    with pytest.raises(ValueError):
        coverage.set_status(sid, "sq-1", "blocked")
    with pytest.raises(ValueError):
        coverage.set_status(sid, "sq-1", "dropped", "")
    with pytest.raises(ValueError):
        coverage.set_status(sid, "sq-1", "nonsense-status")


def test_coverage_updates_are_events(coverage, config, event_log):
    sid = config.session_id
    coverage.create(sid, "sq-1", "q?")
    coverage.set_status(sid, "sq-1", "answered")
    updates = [e for e in event_log.read(sid) if e.event_type == EventType.COVERAGE_UPDATED]
    assert len(updates) == 2
    assert updates[1].payload["status"] == "answered"


def test_notebook_append_only_and_render(notebook, config, event_log):
    sid = config.session_id
    notebook.append(sid, "orchestrator", "# Plan\n- sq-1")
    notebook.append(sid, "subagent:sq-1", "## sq-1\nStatus: answered")
    rendered = notebook.render(sid)
    assert rendered.index("# Plan") < rendered.index("## sq-1")
    events = [e for e in event_log.read(sid) if e.event_type == EventType.NOTEBOOK_APPENDED]
    assert [e.payload["entry_idx"] for e in events] == [0, 1]
    assert events[0].actor == "orchestrator"
