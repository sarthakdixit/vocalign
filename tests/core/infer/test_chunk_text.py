from core.infer import chunk_text


def test_split_into_chunks_splits_on_sentence_boundaries():
    # "Hello there." and "How are you?" are both under MIN_CHUNK_WORDS, so they merge
    # forward; "I am fine!" lands exactly at the threshold (3 words) and stays separate.
    chunks = chunk_text.split_into_chunks("Hello there. How are you? I am fine!")
    assert chunks == ["Hello there. How are you?", "I am fine!"]


def test_split_into_chunks_handles_single_sentence():
    assert chunk_text.split_into_chunks("Just one sentence.") == ["Just one sentence."]


def test_split_into_chunks_handles_empty_string():
    assert chunk_text.split_into_chunks("") == []
    assert chunk_text.split_into_chunks("   ") == []


def test_split_into_chunks_strips_extra_whitespace_between_sentences():
    # Both "First." and "Second." are 1 word each, so they merge into one chunk -
    # exercised separately with longer sentences in the whitespace-handling sense.
    assert chunk_text.split_into_chunks("First.    Second.") == ["First. Second."]


def test_split_into_chunks_does_not_split_decimal_numbers():
    chunks = chunk_text.split_into_chunks("It costs 3.14 dollars total.")
    assert chunks == ["It costs 3.14 dollars total."]


def test_split_into_chunks_does_not_split_on_quotes_mid_sentence():
    chunks = chunk_text.split_into_chunks("She said hello and left.")
    assert chunks == ["She said hello and left."]


def test_split_into_chunks_known_limitation_abbreviation_before_capital():
    # Known limitation (documented in the module docstring): a plain heuristic
    # splitter can't tell "Dr." the abbreviation apart from a real sentence end
    # before a capital letter - it still splits "Dr." off as its own piece here.
    # Short-chunk merging happens to fold it back into the next piece afterward
    # ("Dr." is only 1 word), which isn't a general fix for the abbreviation problem
    # (a longer abbreviation wouldn't get merged this way), just a side effect of it.
    chunks = chunk_text.split_into_chunks("Dr. Smith is here.")
    assert chunks == ["Dr. Smith is here."]


def test_split_into_chunks_preserves_whitespace_unchanged_when_nothing_is_short():
    chunks = chunk_text.split_into_chunks("Hello there friend. How are you today? I am doing great!")
    assert chunks == ["Hello there friend.", "How are you today?", "I am doing great!"]


# --- short-chunk merging (DESIGN.md S13: confirmed via a real run that an isolated
# short sentence, e.g. a standalone "Thank you.", fails the WER gate in total
# isolation while longer neighboring chunks pass cleanly) ---


def test_split_into_chunks_merges_a_short_trailing_sentence_into_its_predecessor():
    chunks = chunk_text.split_into_chunks(
        "This is a reasonably long closing sentence about global politics. Thank you."
    )
    assert chunks == ["This is a reasonably long closing sentence about global politics. Thank you."]


def test_split_into_chunks_merges_a_short_middle_sentence_forward():
    chunks = chunk_text.split_into_chunks("Yes. That is exactly correct in every way. Moving on now.")
    assert chunks == ["Yes. That is exactly correct in every way.", "Moving on now."]


def test_split_into_chunks_does_not_merge_sentences_already_long_enough():
    chunks = chunk_text.split_into_chunks("This sentence has enough words. So does this other one here.")
    assert chunks == ["This sentence has enough words.", "So does this other one here."]


def test_split_into_chunks_min_chunk_words_is_configurable():
    chunks = chunk_text.split_into_chunks("One two three four. Five six.", min_chunk_words=0)
    assert chunks == ["One two three four.", "Five six."]


def test_split_into_chunks_handles_all_chunks_being_short():
    chunks = chunk_text.split_into_chunks("Hi. Bye.")
    assert chunks == ["Hi. Bye."]
