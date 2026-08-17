import pytest

from parsec.evals.trajectory import compute_trajectory
from parsec.models.events import EventType


@pytest.fixture
def session(sessions, config):
    sessions.create(config)
    return config.session_id


def _intent(event_log, sid, tool, input_):
    event_log.append(sid, "model", EventType.TOOL_INTENT, {"tool_use_id": "t", "tool_name": tool, "input": input_})


def _fetch(event_log, sid, url):
    event_log.append(
        sid, "tool:fetch", EventType.FETCH_PERFORMED,
        {"cache_key": "k", "doc_hash": "h", "url": url, "status_code": 200,
         "outcome": "ok", "from_cache": False, "mode": "record"},
    )


def test_trajectory_counts(session, db, event_log, ledger):
    sid = session
    _intent(event_log, sid, "search_broad", {"query": "boiling point", "k": 5})
    _intent(event_log, sid, "search_broad", {"query": "Boiling  POINT", "k": 5})  # redundant (normalized)
    _intent(event_log, sid, "search_within", {"query": "altitude effects"})
    _intent(event_log, sid, "fetch", {"url": "https://gold.example/a"})
    _intent(event_log, sid, "fetch", {"url": "https://gold.example/a"})  # repeated identical call
    _fetch(event_log, sid, "https://bad.example/distractor")
    _fetch(event_log, sid, "https://gold.example/a")
    event_log.append(sid, "tool:fetch", EventType.TOOL_RESULT, {"tool_use_id": "t", "ok": False, "error": "boom"})
    ledger.debit(sid, "input_tokens", 500, "gateway:m")
    ledger.debit(sid, "usd", 0.02, "gateway:m")

    m = compute_trajectory(
        db, event_log, ledger, sid,
        gold_docs=["https://gold.example/a"],
        distractor_docs=["https://bad.example/distractor"],
    )
    assert m.searches == 3
    assert m.redundant_searches == 1
    assert m.repeated_tool_calls == 1
    assert m.tool_errors == 1
    assert m.fetches == 2
    assert m.gold_fetch_fraction == 0.5
    assert m.distractor_fetch_fraction == 0.5
    assert m.fetches_to_first_gold == 2
    assert m.tokens == 500
    assert abs(m.usd - 0.02) < 1e-9


def test_trajectory_no_gold_lists(session, db, event_log, ledger):
    sid = session
    _fetch(event_log, sid, "https://x.example/a")
    m = compute_trajectory(db, event_log, ledger, sid, [], [])
    assert m.gold_fetch_fraction is None
    assert m.distractor_fetch_fraction is None
    assert m.fetches_to_first_gold is None
