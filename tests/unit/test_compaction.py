from parsec.loop.compaction import (
    EVICTION_MARKER,
    context_chars,
    evict_tool_results,
    reset_context,
)


def _messages(n_tool_results: int) -> list[dict]:
    msgs = [{"role": "user", "content": "the question"}]
    for i in range(n_tool_results):
        msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "fetch", "input": {}}]})
        msgs.append(
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "X" * 500, "is_error": False}]}
        )
    return msgs


def test_evict_keeps_last_n():
    msgs, evicted = evict_tool_results(_messages(4), keep_last=2)
    assert evicted == 2
    contents = [
        b["content"]
        for m in msgs
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") == "tool_result"
    ]
    assert contents[:2] == [EVICTION_MARKER, EVICTION_MARKER]
    assert all(c == "X" * 500 for c in contents[2:])


def test_evict_noop_when_under_keep(  ):
    msgs, evicted = evict_tool_results(_messages(2), keep_last=2)
    assert evicted == 0


def test_evict_idempotent_and_deterministic():
    once, _ = evict_tool_results(_messages(5), keep_last=1)
    twice, evicted_again = evict_tool_results(once, keep_last=1)
    assert evicted_again == 0
    assert once == twice
    assert context_chars(once) == context_chars(twice)


def test_evict_shrinks_context():
    original = _messages(6)
    evicted, _ = evict_tool_results(original, keep_last=1)
    assert context_chars(evicted) < context_chars(original)


def test_reset_context_carries_evidence():
    msgs = reset_context("Subquestion sq-1: boiling point?", ["Water boils at 100C.", "Everest is lower."])
    assert len(msgs) == 1
    content = msgs[0]["content"]
    assert content.startswith("Subquestion sq-1")
    assert "Water boils at 100C." in content
    assert "do not re-record" in content


def test_reset_context_without_evidence():
    msgs = reset_context("Subquestion sq-1: q?", [])
    assert "Premises already recorded" not in msgs[0]["content"]
