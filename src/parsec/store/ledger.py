"""Budget ledger (T5): every debit is a row; caps checked at every gate."""

from __future__ import annotations

import sqlite3

from parsec.config import Budgets, Clock
from parsec.errors import BudgetExceeded

TOKEN_CATEGORIES = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
)


class Ledger:
    def __init__(self, conn: sqlite3.Connection, clock: Clock):
        self.conn = conn
        self.clock = clock

    def debit(
        self,
        session_id: str,
        category: str,
        amount: float,
        actor: str,
        ref_seq: int | None = None,
        note: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO ledger (session_id, ts, category, amount, actor, ref_seq, note)"
            " VALUES (?,?,?,?,?,?,?)",
            (session_id, self.clock.now_iso(), category, amount, actor, ref_seq, note),
        )

    def totals(self, session_id: str) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT category, SUM(amount) AS total FROM ledger WHERE session_id=? GROUP BY category",
            (session_id,),
        ).fetchall()
        return {r["category"]: r["total"] for r in rows}

    def totals_by_actor(self, session_id: str) -> dict[tuple[str, str], float]:
        rows = self.conn.execute(
            "SELECT actor, category, SUM(amount) AS total FROM ledger"
            " WHERE session_id=? GROUP BY actor, category",
            (session_id,),
        ).fetchall()
        return {(r["actor"], r["category"]): r["total"] for r in rows}

    def spent_tokens(self, session_id: str) -> int:
        totals = self.totals(session_id)
        return int(sum(totals.get(c, 0.0) for c in TOKEN_CATEGORIES))

    def spent_usd(self, session_id: str) -> float:
        return self.totals(session_id).get("usd", 0.0)

    def check_caps(self, session_id: str, budgets: Budgets, wall_elapsed_s: float) -> None:
        """Raise BudgetExceeded on the first breached cap (priority: usd, tokens, wall)."""
        usd = self.spent_usd(session_id)
        if usd >= budgets.max_usd:
            raise BudgetExceeded("usd", usd, budgets.max_usd)
        tokens = self.spent_tokens(session_id)
        if tokens >= budgets.max_total_tokens:
            raise BudgetExceeded("tokens", tokens, budgets.max_total_tokens)
        if wall_elapsed_s >= budgets.max_wall_seconds:
            raise BudgetExceeded("wall_seconds", wall_elapsed_s, budgets.max_wall_seconds)
