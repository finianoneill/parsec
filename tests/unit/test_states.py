import pytest

from parsec.loop.states import AgentState, StateMachine
from parsec.models.events import EventType


@pytest.fixture
def sm(event_log, sessions, config):
    sessions.create(config)
    return StateMachine(event_log, config.session_id)


def test_happy_path(sm):
    for state in (
        AgentState.DISPATCHING,
        AgentState.COLLECTING,
        AgentState.DISPATCHING,  # next subquestion
        AgentState.COLLECTING,
        AgentState.WRITING,
        AgentState.VERIFYING,
        AgentState.DONE,
    ):
        sm.transition(state)
    assert sm.state == AgentState.DONE


def test_planning_straight_to_writing(sm):
    # no dispatchable subquestions (budget tripped in planning)
    sm.transition(AgentState.WRITING)
    assert sm.state == AgentState.WRITING


def test_halted_from_any_state(sm):
    sm.transition(AgentState.DISPATCHING)
    sm.transition(AgentState.HALTED)
    assert sm.state == AgentState.HALTED


def test_illegal_transition_raises(sm):
    with pytest.raises(RuntimeError):
        sm.transition(AgentState.DONE)  # PLANNING -> DONE not allowed
    sm.transition(AgentState.DISPATCHING)
    with pytest.raises(RuntimeError):
        sm.transition(AgentState.VERIFYING)  # must pass through WRITING


def test_terminal_states_locked(sm):
    sm.transition(AgentState.HALTED)
    with pytest.raises(RuntimeError):
        sm.transition(AgentState.PLANNING)


def test_transitions_are_logged(sm, event_log, config):
    sm.transition(AgentState.DISPATCHING, "sq-1")
    events = event_log.read(config.session_id)
    assert events[-1].event_type == EventType.STATE_TRANSITION
    assert events[-1].payload == {
        "from_state": "PLANNING",
        "to_state": "DISPATCHING",
        "reason": "sq-1",
    }
