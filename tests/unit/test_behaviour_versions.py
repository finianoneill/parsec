"""Behaviour versions: legacy by default, current for new sessions, and
invisible to replay comparison."""

from __future__ import annotations

from parsec.config import (
    BEHAVIOUR_FIELDS,
    CURRENT_BEHAVIOUR,
    LEGACY_BEHAVIOUR,
    RunConfig,
    new_session_behaviour,
)
from parsec.models.events import EventType
from parsec.store.event_log import _strip_volatile


def test_field_defaults_are_the_legacy_versions():
    config = RunConfig(session_id="s", query="q")
    assert {key: getattr(config, key) for key in LEGACY_BEHAVIOUR} == LEGACY_BEHAVIOUR
    assert config.as_of_ts is None


def test_a_config_recorded_before_the_registry_loads_as_legacy():
    stored = RunConfig(session_id="s", query="q").to_json_dict()
    for key in BEHAVIOUR_FIELDS:
        del stored[key]
    loaded = RunConfig.model_validate(stored)
    assert {key: getattr(loaded, key) for key in LEGACY_BEHAVIOUR} == LEGACY_BEHAVIOUR


def test_new_sessions_record_current_behaviour_and_their_reference_time():
    overrides = new_session_behaviour("2026-11-25T12:00:00.000+00:00")
    assert set(overrides) == set(BEHAVIOUR_FIELDS)
    config = RunConfig(session_id="s", query="q", **overrides)
    assert {key: getattr(config, key) for key in CURRENT_BEHAVIOUR} == CURRENT_BEHAVIOUR
    assert config.as_of_ts == "2026-11-25T12:00:00.000+00:00"


def test_dev_versions_are_accepted():
    # Work merged between releases records "<n>-dev" until the release.
    assert RunConfig(session_id="s", query="q", claim_credence="2-dev").claim_credence == "2-dev"


def test_every_behaviour_key_is_a_config_field():
    assert set(BEHAVIOUR_FIELDS) <= set(RunConfig.model_fields)
    assert set(CURRENT_BEHAVIOUR) == set(LEGACY_BEHAVIOUR)


def test_projection_ignores_behaviour_keys():
    """A pre-registry SESSION_STARTED event has no behaviour keys; its replay
    carries them (as legacy defaults). The two must project identically."""
    new = RunConfig(
        session_id="s", query="q", **new_session_behaviour("2026-11-25T12:00:00.000+00:00")
    ).to_json_dict()
    old = {key: value for key, value in new.items() if key not in BEHAVIOUR_FIELDS}
    project = lambda config: _strip_volatile(  # noqa: E731
        EventType.SESSION_STARTED, {"session_id": "s", "query": "q", "config": dict(config)}
    )
    assert project(new) == project(old)
    assert not set(BEHAVIOUR_FIELDS) & set(project(new)["config"])
