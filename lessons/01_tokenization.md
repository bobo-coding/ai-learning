# Lesson 1 — Tokenization

Read: `minigpt/bpe.py` · Tests: `tests/test_tokenizer.py`

Tokenization is the part of an LLM that is pure engineering and no mathematics,
which is why it is skipped — and why so many mysterious model failures live
here. Digit arithmetic, rhyming, character counting, and "how many r's in
strawberry" are all tokenizer artefacts, not reasoning failures.

## Why byte-level

Start from the 256 possible byte values and every string on earth is
representable. No `UNK` token, no vocabulary-coverage problem, no
language-specific preprocessing. The cost is that a typical English word is 4–5
bytes, so a byte-level model would need 4–5× more positions for the same text —
which is what BPE fixes.

## BPE in three lines

Count adjacent symbol pairs, merge the most frequent into a new symbol, repeat.
`minigpt/bpe.py`:

```python
counts = Counter()
for seq, weight in zip(seqs, weights):
    get_pair_counts(seq, counts, weight)
pair = max(counts, key=lambda p: (counts[p], p))     # deterministic tie-break
seqs = [merge(s, pair, new_id) for s in seqs]
```

Two implementation notes that matter:

**Train on a frequency table, not on positions.** Natural text has a heavily
skewed fragment distribution, so counting pairs over ~10⁴ unique fragments
instead of ~10⁶ positions is a ~100× speedup with *identical* output. That is
why `train()` starts with `Counter(self._re.findall(text))`.

**Ties must break deterministically.** `max(counts, key=counts.__getitem__)`
depends on dict iteration order. `key=lambda p: (counts[p], p)` does not, so
training twice gives the same tokenizer.

## The first thing everyone gets wrong: pre-tokenization

Run BPE on raw text and it learns tokens that span word and punctuation
boundaries: `"dog."` and `"dog!"` become unrelated symbols that share nothing.
GPT-2's fix is to split the text with a regex first and run BPE *only inside*
each fragment, so a merge can never cross a boundary.

```python
>>> re.findall(GPT2_SPLIT_PATTERN, "Hello world! It's 2024.")
['Hello', ' world', '!', ' It', "'s", ' 2024', '.']
```

Note that the leading space attaches to the *following* word: `" world"` is one
token. That is why `"world"` and `" world"` are different token ids, and why a
prompt ending in a space quietly changes a model's behaviour.

The reference patterns use unicode properties (`\p{L}`, `\p{N}`) that need the
third-party `regex` module. `bpe.py` builds the identical classes out of stdlib
`re` primitives, so there is no dependency:

```python
_L     = r"[^\W\d_]"                    # one unicode letter   == \p{L}
_PUNCT = r"(?!" + _L + r")[^\s\d]"      # not letter, not digit, not space
```

`tests/test_tokenizer.py::test_split_is_lossless` checks that for both patterns
and a dozen inputs, the fragments concatenate back to the original — the
property a splitter must have and the one a hand-written regex usually breaks.

GPT-4's pattern differs in three ways worth knowing: contractions match
case-insensitively, digits are capped at runs of three (so `2024` is
`202` + `4`, not one token — which measurably changes arithmetic ability), and
the character before a word may be any non-alphanumeric, not just a space.

```python
>>> re.findall(GPT4_SPLIT_PATTERN, "value 1234567")
['value', ' ', '123', '456', '7']
```

## The second thing everyone gets wrong: encoding

Encoding is **not** greedy longest-match against the vocabulary. It is a replay
of the merges *in the order they were learned*:

```python
while len(ids) >= 2:
    best_id, best_pair = None, None
    for pair in zip(ids, ids[1:]):
        new_id = self.merges.get(pair)
        if new_id is not None and (best_id is None or new_id < best_id):
            best_id, best_pair = new_id, pair      # lowest id == earliest merge
    if best_pair is None:
        break
    ids = merge(ids, best_pair, best_id)
```

Greedy longest-match produces *different, usually longer* token sequences.
Since `new_id = 256 + merge_index`, the smallest id among the adjacent pairs is
exactly the earliest-learned merge — no separate rank table needed.

## Decoding: concatenate first, decode once

A single token's bytes can be an *incomplete* UTF-8 sequence — half of a CJK
character or an emoji. Decoding token by token mangles it. So `decode()`
concatenates all the bytes and calls `.decode("utf-8", errors="replace")` once
at the end. This is also why streaming output is fiddly in practice: a partial
token cannot be rendered.

## Special tokens

`<|endoftext|>`, `<|user|>` and friends are not learned by BPE. They are
appended after the vocabulary and matched by a separate regex *before* BPE runs,
longest-first so `<|im_start|>` beats a hypothetical `<|im_`.

`allowed_special` controls whether they are honoured or treated as ordinary
text. That matters for security: if you interpolate untrusted user text into a
prompt with `allowed_special="all"`, the user can inject a role marker and
impersonate the system. Use `"none"` on untrusted input.

## Measure it

```bash
python -m scripts.prepare_shakespeare --vocab-size 1024
```

On Tiny Shakespeare (1.1 MB), vocab 1024:

| vocab | bytes/token | tokens for the corpus |
|---|---|---|
| 256 (raw bytes) | 1.00 | 1,115,394 |
| 512 | 1.93 | 577,673 |
| 1024 | 2.40 | 465,482 |
| 2048 | 2.80 | 398,326 |

Compression grows sub-linearly: each doubling of the vocabulary buys ~15-20%
fewer tokens, with diminishing returns. Meanwhile the embedding and output layers grow linearly
in vocabulary. That trade-off is the whole vocabulary-size decision: GPT-2 chose
50k, Llama 32k, modern multilingual models 128k+ because non-English text
tokenizes far worse and the extra vocabulary buys more there.

## Exercises

1. **Measure the digit-tokenization effect.** Train two tokenizers on the same
   corpus with the GPT-2 and GPT-4 patterns. Compare how `"1234567"` tokenizes.
   Then reason about why capping digit runs at 3 helps arithmetic: what does the
   model have to learn if `1000` is one token and `1001` is two?

2. **Break the pre-tokenizer.** Train BPE with `pattern=r".+"` (no splitting at
   all) on Tiny Shakespeare. Find the longest learned token. How many distinct
   tokens contain more than one space?
   `test_merges_never_cross_pretoken_boundary` asserts this is zero with proper
   splitting — watch it fail.

3. **Greedy vs BPE.** Write a greedy longest-match encoder against
   `tok.vocab`, and compare token counts with `encode_ordinary` over the whole
   corpus. Which is shorter? Do they ever produce the same ids?

4. **The space problem.** Compare `tok.encode("Hello")` with
   `tok.encode(" Hello")`. Now think about a chat template that ends with
   `"Assistant:"` vs `"Assistant: "` — why does the second one often produce
   worse output?

5. **Unicode stress test.** Encode and decode a string containing an emoji with
   a skin-tone modifier (e.g. `"👋🏽"`). How many tokens? What does decoding
   each token *individually* produce, and why?

---

**Previous:** [Lesson 0](00_setup.md) · **Next:** [Lesson 2 — Attention](02_attention.md)
