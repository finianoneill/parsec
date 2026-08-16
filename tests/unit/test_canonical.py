from parsec.canonical import canonical_json, hash_obj, sha256_hex


def test_canonical_json_key_order_stable():
    a = {"b": 1, "a": [2, {"z": 3, "y": 4}]}
    b = {"a": [2, {"y": 4, "z": 3}], "b": 1}
    assert canonical_json(a) == canonical_json(b)


def test_canonical_json_compact():
    assert canonical_json({"a": 1, "b": "x"}) == '{"a":1,"b":"x"}'


def test_hash_obj_stable_across_orderings():
    assert hash_obj({"x": 1, "y": 2}) == hash_obj({"y": 2, "x": 1})


def test_sha256_hex_str_bytes_agree():
    assert sha256_hex("abc") == sha256_hex(b"abc")
    assert len(sha256_hex("abc")) == 64


def test_unicode_preserved():
    s = canonical_json({"t": "café — π"})
    assert "café" in s
