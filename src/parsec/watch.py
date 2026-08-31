"""Scheduled refresh with material-only reporting and mechanical outcome
labels (M14 phase 3: `parsec watch`).

A watch is a chain of refreshes. Each round refreshes the NEWEST session in
the chain — so stable evidence keeps carrying forward and the diff reads
"what changed since the last look", not "since the first" — reports the
claim-level diff only when it is material, and labels the prior
observation's claims by what time did to them. Research has no oracle
(T2); a later look at the same question is the closest thing to one:

  held        — held / strengthened / weakened in the later run: the claim
                still stands, whatever its credence did           -> 1
  overturned  — superseded (newer evidence replaced its support), or
                retracted by a refresh that finished `done` (full
                coverage: the absence is a finding, not a gap)     -> 0
  skipped     — retracted by a PARTIAL refresh (ambiguous: the claim may
                simply not have been re-researched) and `new` claims
                (nothing prior to score)

Each label pairs the claim's credence AS THE PRIOR RUN RECORDED IT with the
outcome, in the {credence, label} shape `parsec calibrate` reads, so a
standing watch feeds the calibration flywheel without manual grading.

The watch itself is orchestration, not a recording: every round is a
first-class refresh session that replays on its own (T4), and the labels
file is the watch's durable output. Deterministic given the world: the
schedule only decides WHEN each observation is taken.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import re
import sqlite3
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from parsec.config import Clock, RunConfig
from parsec.gateway.base import ModelAdapter
from parsec.loop.agent import RunResult
from parsec.store.blobs import BlobStore
from parsec.store.sessions import SessionStore
from parsec.verify.diff import DiffReport, diff_sessions

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$")
_UNIT_S = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_duration(text: str) -> float:
    """'30m', '6h', '1d', '90s', or bare seconds -> seconds. Must be > 0."""
    m = _DURATION.match(text)
    if m is None:
        raise ValueError(f"bad duration {text!r}: use e.g. 30m, 6h, 1d, or seconds")
    seconds = float(m.group(1)) * _UNIT_S[m.group(2)]
    if seconds <= 0:
        raise ValueError(f"bad duration {text!r}: must be positive")
    return seconds


def format_duration(seconds: float) -> str:
    for unit, size in (("d", 86400.0), ("h", 3600.0), ("m", 60.0)):
        if seconds >= size and seconds % size == 0:
            return f"{int(seconds // size)}{unit}"
    return f"{seconds:g}s"


def label_outcomes(report: DiffReport, refresh_status: str) -> list[dict]:
    """Mechanical (credence, outcome) labels for the PRIOR session's claims,
    read off a diff against its refresh (see the module docstring for the
    mapping). Pure: a function of the report and the refresh's status."""
    labels: list[dict] = []
    for c in report.claims:
        if c.a_id is None or c.credence_a is None:
            continue  # `new`: nothing prior to score
        if c.status in ("held", "strengthened", "weakened"):
            outcome = "held"
        elif c.status == "superseded":
            outcome = "overturned"
        elif c.status == "retracted" and refresh_status == "done":
            outcome = "overturned"
        else:
            continue
        labels.append(
            {
                "credence": round(c.credence_a, 6),
                "label": 1 if outcome == "held" else 0,
                "outcome": outcome,
                "status": c.status,
                "claim_id": c.a_id,
                "session": report.session_a,
                "refreshed": report.session_b,
                "text": c.text,
            }
        )
    return labels


def read_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("labels", []) if isinstance(data, dict) else data)


def _lock_backend() -> tuple[Callable, Callable]:
    """(acquire, release) over an open lock file: fcntl.flock on POSIX,
    msvcrt.locking on Windows — one of the two ships with every CPython.
    Never falls back to running unlocked: that would reintroduce the
    lost-labels race on exactly the platform that can't see it."""
    try:
        import fcntl
    except ImportError:
        fcntl = None
    if fcntl is not None:
        return (
            lambda f: fcntl.flock(f, fcntl.LOCK_EX),
            lambda f: fcntl.flock(f, fcntl.LOCK_UN),
        )
    try:
        import msvcrt
    except ImportError:
        raise RuntimeError(
            "parsec watch needs a process lock for the labels file (fcntl or msvcrt) "
            "and neither is available on this platform"
        ) from None

    # What the CRT reports when the byte is held by another process (LK_NBLCK)
    # or LK_LOCK's own retries gave up. Anything else — EBADF, EINVAL — is a
    # real failure and must surface, not spin forever.
    contention = {errno.EACCES, getattr(errno, "EDEADLOCK", errno.EDEADLK)}

    def acquire(f) -> None:
        # LK_NBLCK fails immediately when another process holds the byte;
        # poll rather than LK_LOCK, whose built-in retry gives up after ~10s.
        while True:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                if exc.errno not in contention:
                    raise
                time.sleep(0.05)

    return acquire, lambda f: msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Process-wide exclusive lock on `<path>.lock`: concurrent watches
    sharing one labels file — two cron entries for two questions —
    serialize their read-modify-write here."""
    acquire, release = _lock_backend()
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock:
        acquire(lock)
        try:
            yield
        finally:
            release(lock)


def append_labels(path: Path, labels: list[dict]) -> int:
    """Append to the labels file (created on first use) in the
    {"labels": [...]} shape `parsec calibrate` accepts; returns the total.
    The whole read-modify-write runs under the file lock, and the new
    content lands by atomic rename, so a concurrent reader never sees a
    torn file and a concurrent appender never loses labels."""
    with _locked(path):
        existing = read_labels(path) + labels
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps({"labels": existing}, indent=2) + "\n")
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    return len(existing)


@dataclass
class WatchRound:
    index: int                   # 1-based
    parent_session_id: str
    session_id: str
    status: str                  # the refresh run's status: done | partial
    material: bool               # claim-level change, or the refresh fell short of done
    report: DiffReport
    labels: list[dict] = field(default_factory=list)

    def to_payload(self) -> dict:
        return {
            "type": "round",
            "round": self.index,
            "parent": self.parent_session_id,
            "session_id": self.session_id,
            "status": self.status,
            "material": self.material,
            "counts": self.report.counts,
            "labels": self.labels,
            # Material-only reporting: a quiet round carries counts, not the diff.
            "diff": self.report.to_payload() if self.material else None,
        }


@dataclass
class WatchSummary:
    session_id: str
    rounds: list[WatchRound]
    labels_path: Path | None
    labels_total: int
    error: str | None = None     # a refresh ended halted_*: the watch stopped there

    @property
    def material(self) -> bool:
        return any(r.material for r in self.rounds)

    @property
    def latest_session_id(self) -> str:
        return self.rounds[-1].session_id if self.rounds else self.session_id

    def to_payload(self) -> dict:
        return {
            "type": "summary",
            "session_id": self.session_id,
            "latest": self.latest_session_id,
            "rounds": len(self.rounds),
            "material_rounds": sum(1 for r in self.rounds if r.material),
            "labels_path": None if self.labels_path is None else str(self.labels_path),
            "labels_total": self.labels_total,
            "error": self.error,
        }


async def run_watch(
    conn: sqlite3.Connection,
    blobs: BlobStore,
    clock: Clock,
    session_id: str,
    make_adapter: Callable[[RunConfig], ModelAdapter],
    *,
    fetch_transport=None,
    refresh_all: bool = False,
    epsilon: float = 0.05,
    every_s: float | None = None,
    rounds: int | None = None,
    labels_path: Path | None = None,
    on_round: Callable[[WatchRound], None] | None = None,
) -> WatchSummary:
    """Refresh the chain rooted at session_id: `rounds` times (default one
    round without `every_s`, unbounded with it), sleeping `every_s` between
    rounds. on_round fires after each round — before the sleep — so a CLI
    can report as it goes. Stops early when a refresh ends halted."""
    from parsec.refresh import run_refresh  # lazy: refresh imports the loop

    sessions = SessionStore(conn, clock)
    if sessions.get(session_id) is None:
        raise KeyError(f"unknown session: {session_id}")
    if rounds is None:
        rounds = 1 if every_s is None else 0  # 0 = until stopped
    total = len(read_labels(labels_path)) if labels_path is not None else 0
    summary = WatchSummary(session_id, [], labels_path, total)

    current = session_id
    index = 0
    while rounds == 0 or index < rounds:
        index += 1
        result: RunResult = await run_refresh(
            conn, blobs, clock, current, make_adapter(sessions.get_config(current)),
            fetch_transport=fetch_transport, refresh_all=refresh_all,
        )
        if result.status not in ("done", "partial"):
            summary.error = f"refresh {result.session_id} ended {result.status}"
            break
        report = diff_sessions(conn, current, result.session_id, epsilon=epsilon)
        labels = label_outcomes(report, result.status)
        if labels_path is not None and labels:
            summary.labels_total = append_labels(labels_path, labels)
        rnd = WatchRound(
            index, current, result.session_id, result.status,
            material=not report.unchanged or result.status != "done",
            report=report, labels=labels,
        )
        summary.rounds.append(rnd)
        if on_round is not None:
            on_round(rnd)
        current = result.session_id  # the newest observation seeds the next
        if every_s is not None and (rounds == 0 or index < rounds):
            await clock.sleep(every_s)
    return summary
