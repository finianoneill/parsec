from parsec.retrieval.extract import extract_text


def test_html_extraction_strips_tags_and_scripts():
    html = b"""<html><head><title>My Page</title><script>var x=1;</script>
    <style>body{}</style></head>
    <body><h1>Heading</h1><p>First para.</p><p>Second para.</p>
    <script>alert('no')</script></body></html>"""
    text, title, note = extract_text(html, "text/html; charset=utf-8")
    assert title == "My Page"
    assert "Heading" in text and "First para." in text
    assert "var x" not in text and "alert" not in text
    assert note is None


def test_plain_text_passthrough():
    text, title, note = extract_text(b"line one\n\nline two", "text/plain")
    assert "line one" in text and "line two" in text
    assert title is None


def test_binary_unsupported():
    text, title, note = extract_text(b"\x89PNG\r\n", "image/png")
    assert text == ""
    assert note and "unsupported" in note


def test_deterministic():
    html = b"<html><body><p>Stable output.</p></body></html>"
    assert extract_text(html, "text/html") == extract_text(html, "text/html")
