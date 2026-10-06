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


def test_normalize_text_unwraps_bold_markdown():
    # Confirmed via a real run: leftover markdown reaching GPT-SoVITS's text frontend
    # as literal asterisks degraded that chunk's synthesis badly enough to fail the
    # WER gate outright.
    assert text_normalize.normalize_text("I'd like to discuss **global politics** today") == (
        "I'd like to discuss global politics today"
    )


def test_normalize_text_unwraps_italic_markdown():
    assert text_normalize.normalize_text("this is *very* important") == "this is very important"


def test_normalize_text_unwraps_underscore_bold_and_italic():
    assert text_normalize.normalize_text("__bold__ and _italic_ text") == "bold and italic text"


def test_normalize_text_does_not_pair_a_mid_word_underscore_as_an_italic_wrapper():
    # The paired _..._ pattern skips a mid-word underscore (it isn't a real italic
    # wrapper, so it must not pair up with some unrelated later underscore in the
    # same text) - the lone underscore itself still gets removed by the final
    # leftover-symbol cleanup, same as any other stray markdown character.
    assert text_normalize.normalize_text("the variable foo_bar holds it") == "the variable foobar holds it"


def test_normalize_text_unwraps_strikethrough_and_inline_code():
    result = text_normalize.normalize_text("~~old price~~ `new price`")
    assert result == "old price new price"


def test_normalize_text_strips_heading_markers():
    assert text_normalize.normalize_text("# Introduction") == "Introduction"


def test_normalize_text_unwraps_markdown_links_to_their_text():
    assert text_normalize.normalize_text("see [the docs](https://example.com) here") == "see the docs here"


def test_normalize_text_strips_unpaired_leftover_markdown_symbols():
    # _collapse_whitespace runs after markdown stripping, so the double spaces left
    # behind where "*"/"#" used to be get collapsed to single spaces too.
    assert text_normalize.normalize_text("weird * stray # symbols") == "weird stray symbols"
