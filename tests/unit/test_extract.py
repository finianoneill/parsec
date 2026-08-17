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
    assert note == "stdlib fallback"  # page too small for main-content extraction


BOILERPLATE_PAGE = (
    "<html><head><title>Article Page</title></head><body>"
    "<nav><ul><li><a href='/'>Home</a></li><li><a href='/about'>About</a></li>"
    "<li><a href='/contact'>Contact</a></li></ul></nav>"
    "<article><h1>The Boiling Point of Water</h1>"
    "<p>Water boils at 100 degrees Celsius at standard atmospheric pressure, a value "
    "used to calibrate thermometers worldwide and taught in every introductory "
    "chemistry course as one of the fixed points of the Celsius scale.</p>"
    "<p>At higher altitudes the boiling point decreases; in Denver water boils at "
    "roughly 95 degrees Celsius because atmospheric pressure is lower there than at "
    "sea level, a difference that measurably affects cooking times.</p></article>"
    "<footer>Copyright 2026 Example Corp. All rights reserved. "
    "<a href='/privacy'>Privacy Policy</a> <a href='/terms'>Terms of Service</a></footer>"
    "</body></html>"
).encode()


def test_trafilatura_main_content_extraction():
    text, title, note = extract_text(BOILERPLATE_PAGE, "text/html")
    assert note is None  # trafilatura path, not fallback
    assert title == "Article Page"
    assert "100 degrees Celsius" in text and "95 degrees Celsius" in text
    # boilerplate stripped by main-content detection
    assert "Privacy Policy" not in text
    assert "All rights reserved" not in text


def test_trafilatura_deterministic():
    assert extract_text(BOILERPLATE_PAGE, "text/html") == extract_text(BOILERPLATE_PAGE, "text/html")


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
