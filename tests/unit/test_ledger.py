import pytest

from parsec.config import Budgets
from parsec.errors import BudgetExceeded


@pytest.fixture
def sid(sessions, config):
    sessions.create(config)
    return config.session_id


def test_debit_and_totals(ledger, sid):
    ledger.debit(sid, "input_tokens", 100, "gateway:m")
    ledger.debit(sid, "input_tokens", 50, "gateway:m")
    ledger.debit(sid, "usd", 0.01, "gateway:m")
    totals = ledger.totals(sid)
    assert totals["input_tokens"] == 150
    assert ledger.spent_tokens(sid) == 150
    assert abs(ledger.spent_usd(sid) - 0.01) < 1e-9


def test_check_caps_priority_usd_first(ledger, sid):
    budgets = Budgets(max_usd=0.01, max_total_tokens=10)
    ledger.debit(sid, "usd", 0.02, "gateway:m")
    ledger.debit(sid, "output_tokens", 100, "gateway:m")
    with pytest.raises(BudgetExceeded) as exc:
        ledger.check_caps(sid, budgets, wall_elapsed_s=0)
    assert exc.value.category == "usd"


def test_check_caps_tokens_and_wall(ledger, sid):
    budgets = Budgets(max_total_tokens=10, max_wall_seconds=5)
    ledger.debit(sid, "output_tokens", 100, "gateway:m")
    with pytest.raises(BudgetExceeded) as exc:
        ledger.check_caps(sid, budgets, wall_elapsed_s=0)
    assert exc.value.category == "tokens"

    fresh = Budgets(max_wall_seconds=5)
    with pytest.raises(BudgetExceeded) as exc:
        ledger.check_caps(sid, fresh, wall_elapsed_s=10)
    assert exc.value.category == "wall_seconds"


def test_under_caps_passes(ledger, sid):
    ledger.check_caps(sid, Budgets(), wall_elapsed_s=1)


def test_attribution_by_actor(ledger, sid):
    ledger.debit(sid, "input_tokens", 10, "gateway:a")
    ledger.debit(sid, "input_tokens", 20, "gateway:b")
    by_actor = ledger.totals_by_actor(sid)
    assert by_actor[("gateway:a", "input_tokens")] == 10
    assert by_actor[("gateway:b", "input_tokens")] == 20
