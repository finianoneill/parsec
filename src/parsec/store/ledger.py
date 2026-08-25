"""Budget ledger (T5): every debit is a row; caps checked at every gate."""

from __future__ import annotations

import sqlite3

from parsec.config import Budgets, Clock
from parsec.errors import BudgetExceeded
from parsec.gateway.pricing import CACHE_READ_MULT, CACHE_WRITE_MULT
from parsec.store.event_log import CURRENT_STREAM

# max_total_tokens is a spend proxy, so each category counts at its price
# relative to a plain input token (cache reads bill at a tenth) — counting
# cache reads at full weight was exhausting the cap silently while the
# session footer showed only input/output.
TOKEN_WEIGHTS = {
    "input_tokens": 1.0,
    "output_tokens": 1.0,
    "cache_read_tokens": CACHE_READ_MULT,
    "cache_creation_tokens": CACHE_WRITE_MULT,
}


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
        # Stream attribution rides the same contextvar as events: a debit
        # issued inside a subagent's task lands in that subagent's stream
        # with no parameter threading, so per-subquestion spend survives the
        # process (gateway.stream_spend is in-memory only).
        self.conn.execute(
            "INSERT INTO ledger (session_id, ts, category, amount, actor, stream_id, ref_seq, note)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (
                session_id, self.clock.now_iso(), category, amount, actor,
                CURRENT_STREAM.get(), ref_seq, note,
            ),
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

    def totals_by_stream(self, session_id: str) -> dict[str, dict[str, float]]:
        """Per-stream category totals: which subquestion spent the budget."""
        rows = self.conn.execute(
            "SELECT stream_id, category, SUM(amount) AS total FROM ledger"
            " WHERE session_id=? GROUP BY stream_id, category",
            (session_id,),
        ).fetchall()
        out: dict[str, dict[str, float]] = {}
        for r in rows:
            out.setdefault(r["stream_id"], {})[r["category"]] = r["total"]
        return out

    def spent_tokens(self, session_id: str) -> int:
        totals = self.totals(session_id)
        return int(sum(totals.get(c, 0.0) * w for c, w in TOKEN_WEIGHTS.items()))

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
