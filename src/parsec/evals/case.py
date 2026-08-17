"""Eval cases: a frozen corpus + a query + gold expectations (§11 M5).

A case is a self-contained directory:

    <case_dir>/
      case.json      — case_id, query, must_find list, optional budgets
      queries.json   — search fixtures for the stubbed search provider
      data/
        parsec.db    — recorded corpus: documents, cache index, spans
        blobs/       — content-addressed raw bytes + extracted text

Eval runs never touch the case in place: the corpus is copied to a scratch
directory (SQLite's file-copy forkability, §8) and executed in replay cache
mode, so a case is a permanently frozen, replayable world.

must_find entries are matched against the run's non-narrative claim texts,
case-insensitive substring by default; prefix an entry with "re:" for a
regex.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CASE_FILE = "case.json"
FIXTURES_FILE = "queries.json"
DATA_DIR = "data"


class Nugget(BaseModel):
    """One binary gold rubric item (TREC-style nugget). `patterns` is the
    mechanical matcher tier: substrings or "re:" regexes matched against the
    run's claim texts. `contradiction_patterns` catch a report asserting the
    OPPOSITE of the gold key point (matched claims count as contradicted,
    which is worse than missing)."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)  # the key point, human-legible
    weight: Literal["vital", "okay"] = "vital"
    patterns: list[str] = Field(min_length=1)
    contradiction_patterns: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    # Strict legacy tier (kept): plain substring/"re:" matchers.
    must_find: list[str] = Field(default_factory=list)
    # Nugget tier: weighted binary rubric items with contradiction checks.
    nuggets: list[Nugget] = Field(default_factory=list)
    # Hard-negative bookkeeping: URLs verified to support the gold answer vs.
    # deliberately-planted distractors. Drives trajectory metrics.
    gold_docs: list[str] = Field(default_factory=list)
    distractor_docs: list[str] = Field(default_factory=list)
    max_turns: int = 20
    # Gap-fill rounds during eval runs; default 0 so a case's expected call
    # sequence stays fixed. Raise per case to eval the gap-fill loop itself.
    max_gap_rounds: int = 0
    notes: str = ""


def load_case(case_dir: Path) -> EvalCase:
    return EvalCase.model_validate_json((case_dir / CASE_FILE).read_text(encoding="utf-8"))


def save_case(case_dir: Path, case: EvalCase) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / CASE_FILE).write_text(
        json.dumps(case.model_dump(), indent=2) + "\n", encoding="utf-8"
    )


def discover_cases(root: Path) -> list[Path]:
    """Every direct or nested subdirectory containing a case.json, sorted."""
    return sorted(p.parent for p in root.rglob(CASE_FILE))


def copy_corpus(case_dir: Path, workdir: Path) -> Path:
    """Fork the frozen corpus by file copy into a scratch data dir."""
    dest = workdir / DATA_DIR
    shutil.copytree(case_dir / DATA_DIR, dest)
    return dest


def make_case_from_session(
    source_data_dir: Path,
    fixtures_path: Path,
    case_dir: Path,
    case_id: str,
    query: str,
    must_find: list[str] | None = None,
) -> EvalCase:
    """Snapshot a recorded session's corpus into a new frozen case."""
    if (case_dir / CASE_FILE).exists():
        raise FileExistsError(f"case already exists: {case_dir}")
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_data_dir, case_dir / DATA_DIR)
    shutil.copy(fixtures_path, case_dir / FIXTURES_FILE)
    case = EvalCase(case_id=case_id, query=query, must_find=must_find or [])
    save_case(case_dir, case)
    return case
