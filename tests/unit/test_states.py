import pytest

from parsec.loop.states import AgentState, StateMachine
from parsec.models.events import EventType


@pytest.fixture
def sm(event_log, sessions, config):
    sessions.create(config)
    return StateMachine(event_log, config.session_id)


def test_happy_path(sm):
    for state in (
        AgentState.RESEARCHING,
        AgentState.ANSWERING,
        AgentState.CITE_CHECK,
        AgentState.DONE,
    ):
        sm.transition(state)
    assert sm.state == AgentState.DONE


def test_researching_self_loop(sm):
    sm.transition(AgentState.RESEARCHING)
    sm.transition(AgentState.RESEARCHING)


def test_halted_from_any_state(sm):
    sm.transition(AgentState.RESEARCHING)
    sm.transition(AgentState.HALTED)
    assert sm.state == AgentState.HALTED


def test_illegal_transition_raises(sm):
    with pytest.raises(RuntimeError):
        sm.transition(AgentState.DONE)  # INIT -> DONE not allowed
    sm.transition(AgentState.RESEARCHING)
    with pytest.raises(RuntimeError):
        sm.transition(AgentState.CITE_CHECK)  # must pass through ANSWERING


def test_terminal_states_locked(sm):
    sm.transition(AgentState.HALTED)
    with pytest.raises(RuntimeError):
        sm.transition(AgentState.RESEARCHING)


def test_transitions_are_logged(sm, event_log, config):
    sm.transition(AgentState.RESEARCHING, "start")
    events = event_log.read(config.session_id)
    assert events[-1].event_type == EventType.STATE_TRANSITION
    assert events[-1].payload == {"from_state": "INIT", "to_state": "RESEARCHING", "reason": "start"}
