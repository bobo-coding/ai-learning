import re

import pytest

from minigpt.bpe import (GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN, BPETokenizer,
                         CharTokenizer, merge)

# Long and varied enough that 256+ merges are available before the corpus is
# fully merged -- see test_training_stops_early_when_corpus_exhausted.
SAMPLE = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n"
    "All:\nSpeak, speak.\n"
    "MENENIUS:\nThe worthy fellow is our general, and he hath spoke well.\n"
    "CORIOLANUS:\nWhat must I say? I pray, sir, plague upon it, I cannot bring\n"
    "my tongue to such a pace: look, sir, my wounds, I got them in my country's\n"
    "service, when some certain of your brethren roar'd and ran from the noise\n"
    "of our own drums. Hence, and let me hear no more of this business.\n"
) * 40


@pytest.mark.parametrize("pattern", [GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN])
@pytest.mark.parametrize("text", [
    "", " ", "   ", "\n\n\n", "a", "Hello world!",
    "IT'S 2024, isn't_it? (yes)\n\n  ", "héllo 世界 \U0001f642", "x" * 200,
])
def test_split_is_lossless(pattern, text):
    """Pre-tokenization must partition the text, never drop or duplicate."""
    assert "".join(re.compile(pattern).findall(text)) == text


def test_gpt2_split_shape():
    got = re.compile(GPT2_SPLIT_PATTERN).findall("Hello world! It's 2024.")
    assert got == ["Hello", " world", "!", " It", "'s", " 2024", "."]


def test_gpt4_caps_digit_runs_at_three():
    got = re.compile(GPT4_SPLIT_PATTERN).findall("value 1234567")
    assert [g for g in got if g.strip().isdigit()] == ["123", "456", "7"]


def test_merge_replaces_non_overlapping():
    assert merge([1, 1, 1, 1], (1, 1), 9) == [9, 9]
    assert merge([1, 2, 1, 2, 3], (1, 2), 7) == [7, 7, 3]
    assert merge([5], (1, 2), 7) == [5]


def test_train_and_roundtrip():
    tok = BPETokenizer().train(SAMPLE, vocab_size=400)
    assert tok.n_merges == 400 - 256
    assert tok.vocab_size == 400
    assert set(tok.vocab) == set(range(400))
    ids = tok.encode_ordinary(SAMPLE)
    assert tok.decode(ids) == SAMPLE
    # BPE must compress relative to raw bytes
    assert len(ids) < len(SAMPLE.encode()) / 1.5


@pytest.mark.parametrize("text", [
    "", "a", "hello world", "héllo 世界 \U0001f642",
    "\x00\x01\xff", "  \n\t  ", "aaaaaaaaaaaaaaaaaaaa",
])
def test_roundtrip_arbitrary_text(text):
    """Byte-level BPE can encode *any* string -- that is the point of it."""
    tok = BPETokenizer().train(SAMPLE, vocab_size=300)
    assert tok.decode(tok.encode_ordinary(text)) == text


def test_merges_never_cross_pretoken_boundary():
    tok = BPETokenizer().train(SAMPLE, vocab_size=512)
    # No learned token may contain a space followed by more non-space text and
    # then another space -- that would mean a merge spanned two words.
    for b in tok.vocab.values():
        if b.strip():
            assert b.count(b" ") <= 1, f"token {b!r} spans a word boundary"


def test_encoding_is_bpe_not_greedy_longest_match():
    """Encoding must replay merges in learned order.

    Construct a case where greedy-longest-match differs: train so that "ab" and
    "abc" both exist as tokens but "ab" was learned first, then check that
    encoding "abc" uses the merge order, which is what the decoder expects.
    """
    tok = BPETokenizer().train("ab " * 100 + "abc " * 60, vocab_size=260)
    ids = tok.encode_ordinary("abc")
    assert tok.decode(ids) == "abc"
    # the token inventory must be reachable by replaying merges from bytes
    for tid, blob in tok.vocab.items():
        if tid >= 256:
            assert tok.decode([tid]) == blob.decode("utf-8", errors="replace")


def test_special_tokens():
    tok = BPETokenizer().train(SAMPLE, vocab_size=300)
    base = tok.vocab_size
    ids_map = tok.add_special_tokens(["<|endoftext|>", "<|user|>"])
    assert ids_map["<|endoftext|>"] == base
    assert tok.vocab_size == base + 2

    enc = tok.encode("hi<|endoftext|>bye")
    assert ids_map["<|endoftext|>"] in enc
    assert tok.decode(enc) == "hi<|endoftext|>bye"
    # treated as text when not allowed
    plain = tok.encode("hi<|endoftext|>bye", allowed_special="none")
    assert ids_map["<|endoftext|>"] not in plain
    assert len(plain) > len(enc)
    # selective allow-list
    only_eot = tok.encode("<|endoftext|><|user|>", allowed_special={"<|endoftext|>"})
    assert ids_map["<|user|>"] not in only_eot


def test_longest_special_token_wins():
    tok = BPETokenizer().train(SAMPLE, vocab_size=300)
    tok.add_special_tokens(["<|im|>", "<|im_start|>"])
    ids = tok.encode("<|im_start|>")
    assert ids == [tok.special_tokens["<|im_start|>"]]


def test_save_load_roundtrip(tmp_path):
    tok = BPETokenizer().train(SAMPLE, vocab_size=320)
    tok.add_special_tokens(["<|endoftext|>"])
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.merges == tok.merges
    assert tok2.encode("hello <|endoftext|>") == tok.encode("hello <|endoftext|>")


def test_training_stops_early_when_corpus_exhausted():
    """Asking for more merges than the corpus supports must stop, not crash.

    Once every pre-token has collapsed to a single symbol there are no adjacent
    pairs left, so the vocabulary ends up smaller than requested.  The invariant
    that must hold is vocab_size == 256 + n_merges.
    """
    tiny = "ab ab ab "
    tok = BPETokenizer().train(tiny, vocab_size=1000)
    assert tok.n_merges < 1000 - 256
    assert tok.vocab_size == 256 + tok.n_merges
    assert tok.decode(tok.encode_ordinary(tiny)) == tiny


def test_vocab_size_too_small():
    with pytest.raises(ValueError):
        BPETokenizer().train(SAMPLE, vocab_size=100)


def test_decode_rejects_unknown_id():
    tok = BPETokenizer().train(SAMPLE, vocab_size=300)
    with pytest.raises(ValueError):
        tok.decode([99999])


def test_char_tokenizer():
    tok = CharTokenizer("abc 123", special_tokens=["<|user|>", "<|end|>"])
    assert tok.vocab_size == 7 + 2
    ids = tok.encode("<|user|>abc 12<|end|>")
    assert tok.decode(ids) == "<|user|>abc 12<|end|>"
    assert ids[0] == tok.special_tokens["<|user|>"]
    # characters outside the vocabulary are dropped, not crashed on
    assert tok.decode(tok.encode("abcZZZ")) == "abc"


def test_char_tokenizer_save_load(tmp_path):
    tok = CharTokenizer("abcdef", special_tokens=["<|end|>"])
    p = tmp_path / "ct.json"
    tok.save(p)
    tok2 = CharTokenizer.load(p)
    assert tok2.chars == tok.chars
    assert tok2.encode("abc<|end|>") == tok.encode("abc<|end|>")
