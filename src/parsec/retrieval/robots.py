"""Robots policy (Politeness 2.0).

Identity-honest fetching: robots.txt is fetched once per domain (cached in
the DB with a TTL), parsed for our specific agent and for RSL `License:`
directives (Really Simple Licensing, 1.0 Dec 2025 — machine-readable
licensing terms embedded in robots.txt). A disallow is a typed outcome,
not an error, and licensed content is surfaced — never circumvented.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from parsec.config import Clock

ROBOTS_TIMEOUT_S = 10.0


class RobotsDecision:
    __slots__ = ("allowed", "license_url")

    def __init__(self, allowed: bool, license_url: str | None = None):
        self.allowed = allowed
        self.license_url = license_url


class RobotsPolicy:
    def __init__(
        self,
        conn: sqlite3.Connection,
        clock: Clock,
        user_agent: str,
        ttl_s: int = 24 * 3600,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.conn = conn
        self.clock = clock
        self.user_agent = user_agent
        self.ttl_s = ttl_s
        self._transport = transport

    async def check(self, url: str) -> RobotsDecision:
        parts = urlsplit(url)
        domain = parts.netloc.lower()
        robots_txt = await self._robots_txt(parts.scheme, domain)
        license_url = _parse_rsl_license(robots_txt)
        if not robots_txt.strip():
            return RobotsDecision(True, license_url)
        parser = RobotFileParser()
        parser.parse(robots_txt.splitlines())
        # match on the product token (before the slash), per convention
        agent_token = self.user_agent.split("/")[0]
        return RobotsDecision(parser.can_fetch(agent_token, url), license_url)

    async def _robots_txt(self, scheme: str, domain: str) -> str:
        row = self.conn.execute(
            "SELECT fetched_ts, robots_txt FROM robots_cache WHERE domain=?", (domain,)
        ).fetchone()
        if row is not None and self._fresh(row["fetched_ts"]):
            return row["robots_txt"]
        text = ""
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=ROBOTS_TIMEOUT_S,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            ) as client:
                resp = await client.get(f"{scheme}://{domain}/robots.txt")
            if resp.status_code == 200:
                text = resp.text
        except httpx.HTTPError:
            text = ""  # unreachable robots -> allow (standard practice)
        self.conn.execute(
            "INSERT OR REPLACE INTO robots_cache (domain, fetched_ts, robots_txt) VALUES (?,?,?)",
            (domain, self.clock.now_iso(), text),
        )
        return text

    def _fresh(self, fetched_ts: str) -> bool:
        try:
            fetched = datetime.fromisoformat(fetched_ts)
            now = datetime.fromisoformat(self.clock.now_iso())
        except ValueError:
            return False
        return (now - fetched).total_seconds() < self.ttl_s


def _parse_rsl_license(robots_txt: str) -> str | None:
    for line in robots_txt.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "license" and value.strip():
            return value.strip()
    return None
