from core.infer import text_normalize


def test_normalize_text_expands_simple_abbreviation():
    assert text_normalize.normalize_text("Mr. Smith") == "mister Smith"


def test_normalize_text_expands_abbreviations_case_insensitively():
    result = text_normalize.normalize_text("DR. Jones met MRS. Lee")
    assert "doctor" in result
    assert "missus" in result


def test_normalize_text_expands_integer():
    assert text_normalize.normalize_text("I have 42 apples") == "I have forty-two apples"


def test_normalize_text_expands_comma_separated_thousands():
    result = text_normalize.normalize_text("It cost 1,234 dollars")
    assert "1,234" not in result
    assert "thousand" in result


def test_normalize_text_expands_decimal():
    result = text_normalize.normalize_text("Pi is about 3.14")
    assert "point" in result
    assert "3.14" not in result


def test_normalize_text_collapses_whitespace():
    assert text_normalize.normalize_text("hello   world\n\tfoo") == "hello world foo"


def test_normalize_text_strips_leading_trailing_whitespace():
    assert text_normalize.normalize_text("  hi  ") == "hi"


def test_normalize_text_leaves_plain_text_unchanged():
    assert text_normalize.normalize_text("the quick brown fox") == "the quick brown fox"


def test_normalize_text_handles_negative_number():
    result = text_normalize.normalize_text("it was -5 degrees")
    assert "minus five" in result or "negative five" in result
    assert "-5" not in result


def test_normalize_text_handles_empty_string():
    assert text_normalize.normalize_text("") == ""
