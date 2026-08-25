import json

from parsec.models.events import EventType


def _start_session(sessions, config):
    sessions.create(config)


def test_append_and_read_ordering(event_log, sessions, config):
    _start_session(sessions, config)
    sid = config.session_id
    event_log.append(sid, "harness", EventType.SESSION_STARTED, {"query": "q"})
    event_log.append(sid, "model", EventType.LLM_RESPONSE, {"call_index": 0})
    events = event_log.read(sid)
    assert [e.idx for e in events] == [0, 1]
    assert events[0].event_type == EventType.SESSION_STARTED


def test_projection_strips_ts_and_wall(event_log, sessions, config):
    _start_session(sessions, config)
    sid = config.session_id
    event_log.append(sid, "harness", EventType.SESSION_STARTED, {"query": "q", "ts": "x"})
    event_log.append(sid, "harness", EventType.BUDGET_DEBIT, {"category": "wall_ms", "amount": 123})
    event_log.append(sid, "harness", EventType.BUDGET_DEBIT, {"category": "usd", "amount": 0.01})
    proj = event_log.projection(sid).decode()
    lines = [json.loads(line) for line in proj.strip().splitlines()]
    assert len(lines) == 2  # wall_ms debit dropped
    assert all("ts" not in line["payload"] for line in lines)
    assert lines[1]["payload"]["category"] == "usd"


def test_projection_strips_version_and_transport_config(event_log, sessions, config, tmp_path):
    """parsec_version and model_max_retries are provenance/transport fields
    no recorded trajectory depends on — stripping them keeps pre-Phase-0
    recordings replayable (Phase 0)."""
    from parsec import __version__
    from tests.conftest import make_config

    cfg = make_config(tmp_path, session_id="s-ver")
    assert cfg.parsec_version == __version__  # auto-stamped at construction
    sessions.create(cfg)
    event_log.append(
        "s-ver", "harness", EventType.SESSION_STARTED,
        {"session_id": "s-ver", "query": cfg.query, "config": cfg.to_json_dict()},
    )
    line = json.loads(event_log.projection("s-ver").decode().strip())
    assert "parsec_version" not in line["payload"]["config"]
    assert "model_max_retries" not in line["payload"]["config"]


def test_projection_ignores_session_identity(event_log, sessions, config, tmp_path):
    from tests.conftest import make_config

    cfg_a = make_config(tmp_path, session_id="s-a")
    cfg_b = make_config(tmp_path, session_id="s-b")
    for cfg in (cfg_a, cfg_b):
        sessions.create(cfg)
        event_log.append(
            cfg.session_id,
            "harness",
            EventType.SESSION_STARTED,
            {"session_id": cfg.session_id, "query": cfg.query, "config": cfg.to_json_dict()},
        )
    assert event_log.projection("s-a") == event_log.projection("s-b")
