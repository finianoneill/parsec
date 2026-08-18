"""SteeringReader must deliver mid-run lines and die at run end — a reader
left blocked on stdin races the interactive shell's readline for keystrokes."""

from __future__ import annotations

import os
import time

from parsec.cli import SteeringReader


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


def test_delivers_lines_then_stops_cleanly():
    read_fd, write_fd = os.pipe()
    stdin = os.fdopen(read_fd, "r")
    seen: list[str] = []
    reader = SteeringReader(seen.append, stdin=stdin)
    reader.start()
    try:
        os.write(write_fd, b"approve\n")
        assert _wait_until(lambda: seen)
        assert seen == ["approve"]
    finally:
        reader.stop()
        os.close(write_fd)
        stdin.close()
    assert not reader._thread.is_alive()


def test_stops_without_any_input():
    read_fd, write_fd = os.pipe()
    stdin = os.fdopen(read_fd, "r")
    reader = SteeringReader(lambda _: None, stdin=stdin)
    reader.start()
    try:
        reader.stop()
        assert _wait_until(lambda: not reader._thread.is_alive())
    finally:
        os.close(write_fd)
        stdin.close()


def test_exits_on_eof():
    read_fd, write_fd = os.pipe()
    stdin = os.fdopen(read_fd, "r")
    reader = SteeringReader(lambda _: None, stdin=stdin)
    reader.start()
    try:
        os.close(write_fd)
        assert _wait_until(lambda: not reader._thread.is_alive())
    finally:
        reader.stop()
        stdin.close()
