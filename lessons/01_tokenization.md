# Lesson 1 — Tokenization

Read: `minigpt/bpe.py` · Tests: `tests/test_tokenizer.py` ·
Trace it: `python -m scripts.trace_bpe`

Tokenization is the part of an LLM that is pure engineering and no mathematics,
which is why it gets skipped — and why so many mysterious model failures live
here. Digit arithmetic, rhyming, character counting, and "how many r's in
strawberry" are all tokenizer artefacts, not reasoning failures.

Every number and every trace below is produced by
`python -m scripts.trace_bpe`, so you can re-run any claim in this lesson.

## Why byte-level

Start from the 256 possible byte values and every string on earth is
representable. No `UNK` token, no vocabulary-coverage problem, no
language-specific preprocessing. The cost is that a typical English word is 4–5
bytes, so a byte-level model needs 4–5× more positions for the same text — which
is what BPE buys back.

## The two data structures

Everything in `BPETokenizer` hangs off these:

```python
self.merges: dict[tuple[int, int], int]   # (left_id, right_id) -> new_id
self.vocab:  dict[int, bytes]             # id -> the bytes it expands to
```

`merges` is the *learned model*: "whenever you see token 115 next to token 116,
they may become token 256". `vocab` is the *decoder table*, built alongside it.

The id layout is what makes the whole thing work:

```
[0, 256)            one id per raw byte      vocab[97] = b'a'
[256, vocab_size)   one id per learned merge, in the order learned
[vocab_size, ...)   special tokens, appended afterwards
```

The critical consequence: because merge *i* gets id `256 + i`, **the id encodes
the rank**. A smaller id means an earlier-learned merge. That is why the encoder
never needs a separate rank table — it just takes the minimum id, which you will
see below.

`vocab` is built incrementally, one entry per merge:

```python
self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
```

Since `pair[0]` and `pair[1]` were themselves defined earlier, every entry
expands transitively to raw bytes. That single line is why decoding is a table
lookup instead of a recursive expansion.

## Training, step by step

Here is the whole loop from `train()`, with nothing removed:

```python
freqs = Counter(self._re.findall(text))                 # fragment -> count
seqs = [list(frag.encode("utf-8")) for frag in freqs]   # one symbol list per fragment
weights = list(freqs.values())                          # how many times each occurs

for i in range(num_merges):
    counts = Counter()
    for seq, w in zip(seqs, weights):
        if len(seq) >= 2:
            get_pair_counts(seq, counts, weight=w)      # count adjacent pairs

    pair = max(counts, key=lambda p: (counts[p], p))    # most frequent pair
    new_id = 256 + i

    seqs = [merge(s, pair, new_id) for s in seqs]       # rewrite every fragment
    self.merges[pair] = new_id                          # remember the rule
    self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
```

Four things are happening:

1. **`seqs` is the corpus, as symbol ids.** It starts as raw bytes and gets
   shorter with every merge, in place. At the end it is the corpus tokenized.
2. **`counts` is rebuilt from scratch each iteration**, because merging changes
   which pairs are adjacent. (Production tokenizers keep an incremental index
   instead; this is the readable version.)
3. **`merge(s, pair, new_id)`** rewrites one fragment, replacing every
   non-overlapping left-to-right occurrence:

   ```python
   while i < n:
       if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
           out.append(new_id); i += 2      # consume both
       else:
           out.append(ids[i]); i += 1
   ```

   "Non-overlapping" matters: merging `(a,a)` in `aaaa` gives `[aa, aa]`, not
   three overlapping candidates. `test_merge_replaces_non_overlapping` pins it.
4. **The tie-break is part of the algorithm.** `max(counts, key=counts.__getitem__)`
   depends on dict iteration order; `key=lambda p: (counts[p], p)` breaks ties on
   the pair itself, so training twice gives the same tokenizer.

### Watch it run

`python -m scripts.trace_bpe` trains on a deliberately tiny corpus —
`"low low low low low lower lower newest ×5 widest ×3"` — so every step fits on
screen.

Step 0, pre-tokenize and count:

```
  ' newest'    x5    bytes [32, 110, 101, 119, 101, 115, 116]
  ' low'       x4    bytes [32, 108, 111, 119]
  ' widest'    x3    bytes [32, 119, 105, 100, 101, 115, 116]
  ' lower'     x2    bytes [32, 108, 111, 119, 101, 114]
  'low'        x1    bytes [108, 111, 119]
```

Then the merges. Note how each one builds on the last:

```
  merge 1  ->  new token id 256
    top pairs: ('e','s')=8  ('s','t')=8  ('l','o')=7  ('o','w')=7
    chose (115, 116) = ('s', 't')  count=8
    vocab[256] = 'st'
    ' newest'    [32, 110, 101, 119, 101, 115, 116] -> [32, 110, 101, 119, 101, 256]

  merge 2  ->  new token id 257
    top pairs: ('e','st')=8  ('l','o')=7  ('o','w')=7  ('w','e')=7
    chose (101, 256) = ('e', 'st')  count=8        <- uses token 256
    vocab[257] = 'est'

  merge 3  ->  chose ('o','w')   vocab[258] = 'ow'
  merge 4  ->  chose ('l','ow')  vocab[259] = 'low'      <- uses token 258
  merge 5  ->  chose (' ','low') vocab[260] = ' low'     <- uses token 259
  merge 6  ->  chose ('w','est') vocab[261] = 'west'     <- uses token 257
```

`('e','s')` and `('s','t')` both had count 8; the tie broke on the pair tuple,
deterministically. And by merge 5 the tokenizer has learned `' low'` **with the
leading space** — a single token — which is the pre-tokenization behaviour the
next section explains.

### Why the frequency table, and what it actually saves

The loop above iterates over *distinct fragments* weighted by count, not over
every occurrence. In the toy corpus, 5 distinct fragments cover 87 byte
positions. On real text the gap is much larger, and it grows — distinct
fragments saturate (Heaps' law) while occurrences grow linearly:

| corpus | occurrences | distinct | ratio | weighted | positional | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 50 KB | 13,285 | 2,424 | 5.5× | 0.21 s | 0.64 s | **3.1×** |
| 200 KB | 53,282 | 5,635 | 9.5× | 0.50 s | 2.64 s | **5.3×** |
| 1.1 MB | 297,833 | 15,057 | 19.8× | 1.45 s | 16.23 s | **11.2×** |

(50 merges each, identical output both ways.) The realized speedup trails the
work ratio because of per-fragment overhead, and on a real pretraining corpus —
where the ratio is in the thousands — this is the difference between tractable
and not.

## The first thing everyone gets wrong: pre-tokenization

Run BPE on raw text and it learns tokens spanning word and punctuation
boundaries: `"dog."` and `"dog!"` become unrelated symbols sharing nothing.
GPT-2's fix is to split with a regex first and run BPE *only inside* each
fragment, so a merge can never cross a boundary.

```python
>>> re.findall(GPT2_SPLIT_PATTERN, "Hello world! It's 2024.")
['Hello', ' world', '!', ' It', "'s", ' 2024', '.']
```

The leading space attaches to the *following* word: `" world"` is one fragment,
and therefore one token. That is why `"world"` and `" world"` get different ids,
and why a prompt ending in a space quietly changes a model's behaviour — you
have handed it a token sequence it rarely saw in training.

The reference patterns use unicode properties (`\p{L}`, `\p{N}`) that need the
third-party `regex` module. `bpe.py` builds the identical classes from stdlib
`re` primitives instead, so there is no dependency:

```python
_L     = r"[^\W\d_]"                    # one unicode letter   == \p{L}
_PUNCT = r"(?!" + _L + r")[^\s\d]"      # not letter, not digit, not space
```

`_L` works because `\W` is "not word char" and word chars are letters, digits
and underscore; excluding digits and underscore from the negation leaves exactly
the letters. `_PUNCT` needs a negative lookahead because `re` has no
character-class subtraction — there is no way to write "any character except
this other class" directly.

`test_split_is_lossless` checks that for both patterns and a dozen inputs the
fragments concatenate back to the original. That is the property a splitter must
have and the one a hand-written regex usually breaks.

GPT-4's pattern differs in three ways worth knowing: contractions match
case-insensitively, digits are capped at runs of three, and the character before
a word may be any non-alphanumeric rather than just a space.

```python
>>> re.findall(GPT4_SPLIT_PATTERN, "value 1234567")
['value', ' ', '123', '456', '7']
```

The digit cap is the one with real consequences. With GPT-2's pattern a model
may see `2024` as one atomic token and `2025` as another, with no shared
structure; capping at three forces a positional decomposition that arithmetic
can actually generalize over.

## The second thing everyone gets wrong: encoding

Encoding is **not** greedy longest-match against the vocabulary. It replays the
merges *in the order they were learned*:

```python
while len(ids) >= 2:
    best_id, best_pair = None, None
    for pair in zip(ids, ids[1:]):          # every adjacent pair
        new_id = self.merges.get(pair)
        if new_id is not None and (best_id is None or new_id < best_id):
            best_id, best_pair = new_id, pair     # smallest id == earliest merge
    if best_pair is None:
        break                                # no learned merge applies
    ids = merge(ids, best_pair, best_id)
```

Each iteration scans all adjacent pairs, picks the **lowest-numbered** applicable
merge, applies it everywhere in the fragment, and repeats. `min` over ids works
as a rank lookup only because of the id layout described at the top.

Traced on `"lowest"` with the six merges learned above:

```
    start: [108, 111, 119, 101, 115, 116]  = ['l', 'o', 'w', 'e', 's', 't']
    step 1: applicable {(115,116):256, (111,119):258} -> lowest id 256 wins
            [108, 111, 119, 101, 256]  = ['l', 'o', 'w', 'e', 'st']
    step 2: applicable {(101,256):257, (111,119):258} -> lowest id 257 wins
            [108, 111, 119, 257]  = ['l', 'o', 'w', 'est']
    step 3: applicable {(111,119):258, (119,257):261} -> lowest id 258 wins
            [108, 258, 257]  = ['l', 'ow', 'est']
    step 4: applicable {(108,258):259} -> lowest id 259 wins
            [259, 257]  = ['low', 'est']
    no adjacent pair is a known merge -> done
```

Step 3 is the instructive one: both `('o','w')`→258 and `('w','est')`→261 apply,
and 258 wins purely because it was learned first. A greedy scheme picking the
longest match would have taken `west`.

**Cost.** Each iteration is O(L) over the fragment and removes at least one
symbol, so encoding a fragment of length L is O(L²) worst case. Fragments are
words, so L is small — but it is why `_encode_fragment` memoises into
`self._cache`, since natural text repeats fragments constantly.

### BPE really is different from greedy longest-match

Minimal case, from a corpus that teaches `'aa'`(256) then `'aaa'`(257):

```
  'aaaa'    bpe=['aa', 'aa']        greedy=['aaa', 'a']
  'aaaaa'   bpe=['aa', 'aaa']       greedy=['aaa', 'aa']
  'aaaaaa'  bpe=['aa', 'aa', 'aa']  greedy=['aaa', 'aaa']
```

BPE applies `(a,a)`→256 to every non-overlapping position *first*, because it is
the lowest rank, and only then looks for `aaa` — which is no longer there.

On real text, with a vocab-1024 tokenizer over Tiny Shakespeare's 15,057
distinct fragments:

- **69.5% segment identically**
- **30.5% differ** — of those, greedy is shorter 673 times, BPE shorter 334
  times, and 3,584 are the same length but different tokens

```
 ' savageness'  bpe=[' sa','v','ag','en','ess']   greedy=[' sa','v','age','ness']
 ' shamefully'  bpe=[' sha','m','ef','ull','y']   greedy=[' sha','me','ful','ly']
```

Two things to take from that. First, **greedy is not "worse" by token count** —
over these fragments it is marginally *shorter* (52,754 vs 53,131), and its
segmentations are often more morphologically sensible. Second, and this is the
only thing that matters: the model was trained on one convention. Encode with
the other at inference and you feed it 30% novel token sequences that happen to
decode to the right string. The output degrades and nothing errors.

## Decoding

```python
parts = [self.vocab[i] for i in ids]
return b"".join(parts).decode("utf-8", errors="replace")
```

Concatenate **all** the bytes, then decode once. A single token's bytes can be an
*incomplete* UTF-8 sequence — half a CJK character, or one byte of a 4-byte
emoji — so decoding token by token mangles it. This is also why streaming output
is fiddly: a partial token cannot be rendered, and a naive streaming decoder
emits replacement characters mid-word.

## Special tokens

`<|endoftext|>`, `<|user|>` and friends are not learned by BPE. They are
appended after the learned vocabulary and matched by a separate regex *before*
BPE runs, longest-first so `<|im_start|>` beats a hypothetical `<|im_`.

`allowed_special` controls whether they are honoured or treated as ordinary
text, and that is a security boundary: if you interpolate untrusted user text
into a prompt with `allowed_special="all"`, the user can inject a role marker
and impersonate the system. Use `"none"` on anything user-supplied.

## Choosing a vocabulary size

```bash
python -m scripts.prepare_shakespeare --vocab-size 1024
```

On Tiny Shakespeare (1.1 MB):

| vocab | bytes/token | tokens for the corpus |
|---|---|---|
| 256 (raw bytes) | 1.00 | 1,115,394 |
| 512 | 1.93 | 577,673 |
| 1024 | 2.40 | 465,482 |
| 2048 | 2.80 | 398,326 |

Each doubling buys ~15–20% fewer tokens, with clear diminishing returns — while
the embedding and output layers grow *linearly* in vocabulary. That trade-off is
the whole decision: GPT-2 chose 50k, Llama 32k, and modern multilingual models
128k+ because non-English text tokenizes far worse and the extra rows buy much
more there.

## Exercises

1. **Trace your own text.** `python -m scripts.trace_bpe --text "..." --merges 10
   --encode "..."`. Try a corpus with a repeated prefix (`"internet international
   interval"`) and watch the merge order.

2. **Break the pre-tokenizer.** Train with `pattern=r".+"` (no splitting) on Tiny
   Shakespeare. Find the longest learned token. How many learned tokens contain
   more than one space? `test_merges_never_cross_pretoken_boundary` asserts that
   is zero with proper splitting — watch it fail.

3. **Implement greedy longest-match** and reproduce the 69.5% / 30.5% split.
   Then take a model trained with BPE encoding and evaluate its perplexity using
   greedy encoding. Quantify the damage from a mismatched encoder.

4. **Make the tie-break non-deterministic** (`key=counts.__getitem__`), train
   twice, and diff the merge lists. How far into training does the first
   divergence appear, and how much does the final vocabulary differ?

5. **The space problem.** Compare `tok.encode("Hello")` with
   `tok.encode(" Hello")`. Then reason about a chat template ending in
   `"Assistant:"` versus `"Assistant: "` — why does the second often produce
   worse output?

6. **Unicode stress test.** Encode a string with a skin-tone emoji (`"👋🏽"`).
   How many tokens? Decode each token *individually* and explain what you see.

7. **Speed it up.** The training loop rebuilds `counts` from scratch each merge.
   Keep an incremental index instead — only fragments containing the merged pair
   can change — and measure the speedup at vocab 4096.

---

**Previous:** [Lesson 0](00_setup.md) · **Next:** [Lesson 2 — Attention](02_attention.md)
