"""Byte-level BPE, written from scratch (the tokenizer GPT-2/GPT-4/Llama all use).

Why byte-level?  If you start from the 256 possible bytes, every possible string
is encodable -- there is no `UNK` token and no unicode normalisation step.  BPE
then learns a vocabulary of *merges*: repeatedly fuse the most frequent adjacent
pair of symbols into a new symbol.

Two pieces of machinery matter and are easy to get wrong:

1. **Pre-tokenization.** Raw BPE over an entire document happily learns tokens
   that span word boundaries and punctuation ("dog." and "dog!" become
   unrelated symbols).  GPT-2 avoids this by first splitting text with a regex
   that keeps letters, digits, punctuation and runs of whitespace apart, and
   attaches a leading space to the following word (" dog" is one token).  BPE
   is then run *inside* each fragment only, and merges can never cross a
   fragment boundary.

2. **Encoding must replay merges in training order.** A greedy left-to-right
   longest-match is *not* BPE and gives different (usually longer) token
   sequences.  The correct algorithm repeatedly finds the adjacent pair with the
   lowest merge rank and applies it, until no adjacent pair is a known merge.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# --- Pre-tokenization patterns -------------------------------------------------
#
# The reference patterns are written with unicode *properties* (\p{L} letter,
# \p{N} number), which need the third-party `regex` module.  The stdlib `re`
# module can express exactly the same classes, so we build them up by name:
#
#   L   = a unicode letter                  == \p{L}
#   NL  = "not a letter"                    (negative lookahead + any char)
#   N   = a unicode digit                   == \p{N} (`re`'s \d is unicode-aware)
#
# `re` has no character-class subtraction, so "not whitespace, not letter, not
# digit" (\p{P} punctuation plus \p{S} symbols) is spelled as a negative
# lookahead for a letter in front of a class that already excludes the rest.
_L = r"[^\W\d_]"                 # one unicode letter
_PUNCT = r"(?!" + _L + r")[^\s\d]"  # one char: not letter, not digit, not space

# GPT-2 (r50k/p50k).  Reference:
#   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
GPT2_SPLIT_PATTERN = (
    r"'(?:s|t|re|ve|m|ll|d)"          # common English contractions
    r"| ?" + _L + r"+"                # optional leading space + letters
    r"| ?\d+"                         # optional leading space + digits
    r"| ?(?:" + _PUNCT + r")+"         # optional leading space + punctuation/symbols
    r"|\s+(?!\S)"                     # a whitespace run at end of string
    r"|\s+"                            # any other whitespace run
)

# GPT-4 (cl100k_base).  Reference:
#   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
#   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
# Differences that matter in practice:
#   * contractions match case-insensitively ("IT'S" tokenizes like "it's");
#   * digits are capped at runs of 3, so numbers split into 1-3 digit groups
#     instead of learning a token per common year/amount;
#   * the leading character before a word may be any non-letter/digit (not just
#     a space), so "(hello" is one fragment.
GPT4_SPLIT_PATTERN = (
    r"(?i:'(?:s|t|re|ve|m|ll|d))"
    r"|(?:(?!" + _L + r")[^\r\n\d])?" + _L + r"+"
    r"|\d{1,3}"
    r"| ?(?:" + _PUNCT + r")+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def get_pair_counts(ids: list[int], counts: Counter | None = None, weight: int = 1) -> Counter:
    """Count adjacent pairs in `ids`, adding `weight` per occurrence."""
    counts = Counter() if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """A byte-level BPE tokenizer with regex pre-tokenization and special tokens.

    Token ids are laid out as:
      ``[0, 256)``            raw bytes
      ``[256, vocab_size)``   learned merges, in the order they were learned
      ``[vocab_size, ...)``   special tokens, appended after training
    """

    def __init__(self, pattern: str = GPT2_SPLIT_PATTERN):
        self.pattern = pattern
        self._re = re.compile(pattern)
        # merges: (a, b) -> new_id.  Insertion order == merge rank.
        self.merges: dict[tuple[int, int], int] = {}
        # vocab: id -> bytes.  Built incrementally so decoding is a table lookup.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}
        self._special_re: re.Pattern | None = None
        self._cache: dict[str, list[int]] = {}

    # ---------------------------------------------------------------- properties

    @property
    def n_merges(self) -> int:
        return len(self.merges)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    # ------------------------------------------------------------------ training

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> BPETokenizer:
        """Learn `vocab_size - 256` merges from `text`.

        Implementation note: we collapse the corpus into a *fragment frequency
        table* first.  Natural text has a heavy-tailed fragment distribution, so
        counting pairs over ~10^4 unique fragments instead of ~10^6 positions is
        a 100x speedup with identical results.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 for a byte-level tokenizer")
        num_merges = vocab_size - 256

        freqs = Counter(self._re.findall(text))
        # Each unique fragment becomes one symbol sequence carrying a weight.
        seqs: list[list[int]] = [list(frag.encode("utf-8")) for frag in freqs]
        weights: list[int] = list(freqs.values())

        self.merges.clear()
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._cache.clear()

        for i in range(num_merges):
            counts: Counter = Counter()
            for seq, w in zip(seqs, weights):
                if len(seq) >= 2:
                    get_pair_counts(seq, counts, weight=w)
            if not counts:
                if verbose:
                    print(f"stopping early at {i} merges: corpus fully merged")
                break
            # max() ties break on the pair itself, which makes training deterministic.
            pair = max(counts, key=lambda p: (counts[p], p))
            new_id = 256 + i
            seqs = [merge(s, pair, new_id) if len(s) >= 2 else s for s in seqs]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (i < 5 or (i + 1) % 100 == 0):
                token = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"merge {i + 1:5d}/{num_merges}: {pair} -> {new_id} ({token!r}) count={counts[pair]}")
        return self

    def add_special_tokens(self, tokens: list[str]) -> dict[str, int]:
        """Append special tokens after the learned vocabulary and return the map."""
        base = len(self.vocab)
        for tok in tokens:
            if tok not in self.special_tokens:
                self.special_tokens[tok] = base + len(self.special_tokens)
        # Longest-first so "<|im_start|>" wins over a hypothetical "<|im_".
        if self.special_tokens:
            alts = sorted(self.special_tokens, key=len, reverse=True)
            self._special_re = re.compile("(" + "|".join(re.escape(t) for t in alts) + ")")
        return dict(self.special_tokens)

    # ------------------------------------------------------------------ encoding

    def _encode_fragment(self, frag: str) -> list[int]:
        """BPE a single pre-token fragment by replaying merges in rank order."""
        cached = self._cache.get(frag)
        if cached is not None:
            return cached
        ids = list(frag.encode("utf-8"))
        while len(ids) >= 2:
            # Find the adjacent pair with the *lowest* merge rank present.
            best_rank = None
            best_pair = None
            for pair in zip(ids, ids[1:]):
                rank = self.merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break  # no learned merge applies -- we are done
            ids = merge(ids, best_pair, best_rank)
        if len(self._cache) < 200_000:
            self._cache[frag] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special-token strings as ordinary text."""
        out: list[int] = []
        for frag in self._re.findall(text):
            out.extend(self._encode_fragment(frag))
        return out

    def encode(self, text: str, allowed_special: str | set[str] = "all") -> list[int]:
        """Encode text. `allowed_special` controls special-token handling.

        ``"all"``  -> recognise every registered special token (default)
        ``"none"`` -> treat them as ordinary text
        a set      -> recognise only those
        """
        if allowed_special == "none" or not self.special_tokens or self._special_re is None:
            return self.encode_ordinary(text)
        allowed = set(self.special_tokens) if allowed_special == "all" else set(allowed_special)
        out: list[int] = []
        for piece in self._special_re.split(text):
            if not piece:
                continue
            if piece in allowed:
                out.append(self.special_tokens[piece])
            else:
                out.extend(self.encode_ordinary(piece))
        return out

    # ------------------------------------------------------------------ decoding

    def decode(self, ids: list[int] | object) -> str:
        """Decode ids back to text.

        A single token's bytes may be an *incomplete* UTF-8 sequence, so we
        concatenate all the bytes first and decode once at the end; decoding
        token-by-token would mangle multi-byte characters.
        """
        if hasattr(ids, "tolist"):
            ids = ids.tolist()  # accept torch tensors / numpy arrays
        inv_special = {v: k for k, v in self.special_tokens.items()}
        parts: list[bytes] = []
        for i in ids:
            i = int(i)
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in inv_special:
                parts.append(inv_special[i].encode("utf-8"))
            else:
                raise ValueError(f"token id {i} out of range (vocab_size={self.vocab_size})")
        return b"".join(parts).decode("utf-8", errors="replace")

    # ----------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "pattern": self.pattern,
            # JSON keys must be strings; store merges as a flat ordered list.
            "merges": [[a, b] for (a, b) in self.merges],
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(blob), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(pattern=blob["pattern"])
        for i, (a, b) in enumerate(blob["merges"]):
            new_id = 256 + i
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        tok.special_tokens = {}
        tok.add_special_tokens(list(blob["special_tokens"]))
        return tok


class CharTokenizer:
    """Character-level tokenizer -- no training, tiny vocab, great for debugging.

    Use this when you want a model to fit in seconds and you do not want
    tokenization to be a variable in the experiment.  Special tokens are
    appended after the character vocabulary and matched before characters, so
    "<|user|>" is one token rather than eight.
    """

    def __init__(self, text: str = "", chars: list[str] | None = None,
                 special_tokens: list[str] | None = None):
        self.chars = list(chars) if chars is not None else sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.special_tokens: dict[str, int] = {}
        self._special_re: re.Pattern | None = None
        if special_tokens:
            self.add_special_tokens(special_tokens)

    def add_special_tokens(self, tokens: list[str]) -> dict[str, int]:
        base = len(self.chars)
        for tok in tokens:
            if tok not in self.special_tokens:
                self.special_tokens[tok] = base + len(self.special_tokens)
        alts = sorted(self.special_tokens, key=len, reverse=True)
        self._special_re = re.compile("(" + "|".join(re.escape(t) for t in alts) + ")")
        return dict(self.special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + len(self.special_tokens)

    def encode(self, text: str) -> list[int]:
        if self._special_re is None:
            return [self.stoi[c] for c in text if c in self.stoi]
        out: list[int] = []
        for piece in self._special_re.split(text):
            if not piece:
                continue
            if piece in self.special_tokens:
                out.append(self.special_tokens[piece])
            else:
                out.extend(self.stoi[c] for c in piece if c in self.stoi)
        return out

    def decode(self, ids) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        inv = {v: k for k, v in self.special_tokens.items()}
        return "".join(self.itos.get(int(i), inv.get(int(i), "")) for i in ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"chars": self.chars,
                                    "special_tokens": list(self.special_tokens)}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(chars=blob["chars"], special_tokens=blob["special_tokens"])
