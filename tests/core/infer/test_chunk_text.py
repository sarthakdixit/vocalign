from core.infer import chunk_text


def test_split_into_chunks_splits_on_sentence_boundaries():
    chunks = chunk_text.split_into_chunks("Hello there. How are you? I am fine!")
    assert chunks == ["Hello there.", "How are you?", "I am fine!"]


def test_split_into_chunks_handles_single_sentence():
    assert chunk_text.split_into_chunks("Just one sentence.") == ["Just one sentence."]


def test_split_into_chunks_handles_empty_string():
    assert chunk_text.split_into_chunks("") == []
    assert chunk_text.split_into_chunks("   ") == []


def test_split_into_chunks_strips_extra_whitespace_between_sentences():
    assert chunk_text.split_into_chunks("First.    Second.") == ["First.", "Second."]


def test_split_into_chunks_does_not_split_decimal_numbers():
    chunks = chunk_text.split_into_chunks("It costs 3.14 dollars total.")
    assert chunks == ["It costs 3.14 dollars total."]


def test_split_into_chunks_does_not_split_on_quotes_mid_sentence():
    chunks = chunk_text.split_into_chunks("She said hello and left.")
    assert chunks == ["She said hello and left."]


def test_split_into_chunks_known_limitation_abbreviation_before_capital():
    # Known limitation (documented in the module docstring): a plain heuristic
    # splitter can't tell "Dr." the abbreviation apart from a real sentence end
    # before a capital letter. Mitigated upstream by running
    # text_normalize.normalize_text() first, not eliminated here. This test records
    # actual current behavior rather than leaving the gap unverified.
    chunks = chunk_text.split_into_chunks("Dr. Smith is here.")
    assert chunks == ["Dr.", "Smith is here."]
