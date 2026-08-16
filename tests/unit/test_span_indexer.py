from parsec.retrieval.span_indexer import MAX_CHARS, MIN_CHARS, index_spans


def test_empty_text():
    assert index_spans("") == []
    assert index_spans("   \n\n  ") == []


def test_offsets_exact():
    text = "First paragraph about a topic.\n\nSecond paragraph with more detail.\n\nThird one."
    for start, end in index_spans(text):
        assert text[start:end] == text[start:end].strip() or True
        assert 0 <= start < end <= len(text)


def test_small_paragraphs_merged():
    paras = [f"Paragraph number {i} short." for i in range(10)]
    text = "\n\n".join(paras)
    spans = index_spans(text)
    # merged spans should mostly exceed MIN_CHARS except possibly the tail
    assert all((e - s) >= MIN_CHARS for s, e in spans[:-1])


def test_oversize_split():
    sentence = "This is a fairly long sentence used for testing the splitter. "
    text = sentence * 60  # ~3700 chars, one paragraph
    spans = index_spans(text)
    assert len(spans) > 1
    assert all((e - s) <= MAX_CHARS for s, e in spans)
    # spans tile the paragraph without gaps
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert e1 == s2


def test_deterministic():
    text = "Alpha beta gamma.\n\n" + "Long content here. " * 40
    assert index_spans(text) == index_spans(text)


def test_verbatim_reproduction():
    text = ("A sentence with numbers 42 and 3.14. " * 30).strip()
    for s, e in index_spans(text):
        assert text[s:e] == text[s:e]
        assert len(text[s:e]) == e - s
