"""Content-addressed blob store.

Layout: <root>/<sha256[:2]>/<sha256>. Write-once: identical content is
stored exactly once; writes are atomic (temp file + rename). Holds raw
fetched bytes, extracted text, full LLM request/response bodies, full
tool results, and final answers. Rows elsewhere reference blobs by hash.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from parsec.canonical import sha256_hex


class BlobStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sha: str) -> Path:
        return self.root / sha[:2] / sha

    def put(self, data: bytes | str) -> str:
        if isinstance(data, str):
            data = data.encode("utf-8")
        sha = sha256_hex(data)
        dest = self._path(sha)
        if dest.exists():
            return sha
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return sha

    def get(self, sha: str) -> bytes:
        return self._path(sha).read_bytes()

    def get_text(self, sha: str) -> str:
        return self.get(sha).decode("utf-8")

    def exists(self, sha: str) -> bool:
        return self._path(sha).exists()
