"""Live smoke test: one tiny real query through the Anthropic adapter in
record mode, then replay it. Manual confidence check, not CI.

Run with:  uv run pytest -m live tests/integration/test_live_smoke.py
Requires ANTHROPIC_API_KEY. Costs a few cents.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

import parsec.cli as cli
from parsec.config import RealClock
from parsec.db.connection import open_db
from parsec.replay import run_replay
from parsec.store.blobs import BlobStore

pytestmark = pytest.mark.live


@pytest.mark.skipif("ANTHROPIC_API_KEY" not in os.environ, reason="needs ANTHROPIC_API_KEY")
def test_live_ask_and_replay(tmp_path, capsys):
    data_dir = tmp_path / "data"
    fixtures = tmp_path / "queries.json"
    # Point the search stub at a real page so the model can fetch something.
    fixtures.write_text(
        json.dumps(
            {
                "python programming language release history": [
                    {
                        "title": "History of Python - Wikipedia",
                        "url": "https://en.wikipedia.org/wiki/History_of_Python",
                        "snippet": "Python was created by Guido van Rossum...",
                    }
                ]
            }
        )
    )
    exit_code = cli.main(
        [
            "ask",
            "In what year was Python first released? Cite your source.",
            "--session-id", "live-smoke",
            "--cache-mode", "record",
            "--model", "claude-haiku-4-5",
            "--max-usd", "0.25",
            "--data-dir", str(data_dir),
            "--search-fixtures", str(fixtures),
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert exit_code in (0, 3), out
    assert out["answer"]

    conn = open_db(data_dir / "parsec.db")
    blobs = BlobStore(data_dir / "blobs")
    outcome = asyncio.run(run_replay(conn, blobs, RealClock(), "live-smoke"))
    assert outcome.verified, outcome.first_divergence
