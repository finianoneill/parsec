from parsec.canonical import sha256_hex
from parsec.store.blobs import BlobStore


def test_put_get_roundtrip(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    sha = store.put(b"hello world")
    assert sha == sha256_hex(b"hello world")
    assert store.get(sha) == b"hello world"
    assert store.get_text(sha) == "hello world"


def test_dedup_and_layout(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    sha1 = store.put("same content")
    sha2 = store.put("same content")
    assert sha1 == sha2
    path = tmp_path / "blobs" / sha1[:2] / sha1
    assert path.exists()
    files = [p for p in (tmp_path / "blobs").rglob("*") if p.is_file()]
    assert len(files) == 1


def test_exists(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    sha = store.put("x")
    assert store.exists(sha)
    assert not store.exists("0" * 64)
