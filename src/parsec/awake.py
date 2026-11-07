"""Keep the machine awake for a live run (M15.8, macOS).

A laptop that idles to sleep mid-run drops every in-flight model call and
pauses the monotonic clock the wall budget runs on — the harness sees a
"connection error" on each maintenance wake and a 12-minute run that took
99 minutes by the kitchen clock. `caffeinate -i` holds off idle sleep for
as long as this process lives (`-w` ties it to our pid, so it can never
outlive us); it does not prevent a closed lid. No-op anywhere else.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator


def available() -> bool:
    return sys.platform == "darwin" and shutil.which("caffeinate") is not None


@contextlib.contextmanager
def keep_awake(enabled: bool) -> Iterator[bool]:
    """Yields True while an idle-sleep assertion is held, False when disabled
    or unavailable (non-macOS, no caffeinate binary, spawn failure)."""
    if not enabled or not available():
        yield False
        return
    try:
        proc = subprocess.Popen(
            ["caffeinate", "-i", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        yield False
        return
    try:
        yield True
    finally:
        with contextlib.suppress(OSError):
            proc.terminate()
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            proc.wait(timeout=5)
