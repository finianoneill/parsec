from parsec.verify.containment import check_containment, extract_numbers, extract_quotes


def test_extract_numbers():
    assert extract_numbers("boils at 100 degrees, 1,600 meters, pi is 3.14") == [
        "100",
        "1600",
        "3.14",
    ]


def test_extract_numbers_ignores_ids():
    # span ids like doc:ab12cd34#140-312 must not shed "numbers"
    assert extract_numbers("see doc:ab12cd34ef56#140-312 for detail") == []


def test_extract_quotes():
    assert extract_quotes('the report said "quality has improved" and “costs fell”') == [
        "quality has improved",
        "costs fell",
    ]


def test_matching_numbers_pass():
    problems = check_containment(
        "Water boils at 100 degrees Celsius.",
        ["Water boils at 100 degrees Celsius at sea level."],
    )
    assert problems == []


def test_separator_normalization():
    problems = check_containment(
        "Denver sits at 1600 meters.",
        ["Denver, Colorado, at about 1,600 meters of elevation."],
    )
    assert problems == []


def test_wrong_number_flagged():
    problems = check_containment(
        "Water boils at 90 degrees Celsius.",
        ["Water boils at 100 degrees Celsius."],
    )
    assert len(problems) == 1 and "'90'" in problems[0]


def test_transform_note_exempts_numbers():
    problems = check_containment(
        "Everest boiling point is about 158 degrees Fahrenheit.",
        ["water boils at only about 70 degrees Celsius"],
        transform_note="converted 70C to Fahrenheit",
    )
    assert problems == []


def test_quote_must_be_verbatim():
    ok = check_containment('The author called it "a landmark result".', ['This is a landmark result, they wrote.'])
    assert ok == []
    bad = check_containment('The author called it "a seminal result".', ["a landmark result"])
    assert len(bad) == 1 and "seminal" in bad[0]


def test_transform_note_does_not_exempt_quotes():
    problems = check_containment(
        'It was "totally invented".',
        ["nothing like that here"],
        transform_note="paraphrase",
    )
    assert len(problems) == 1
