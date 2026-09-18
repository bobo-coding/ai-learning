"""Download Tiny Shakespeare, train a tokenizer on it, write token bins.

Run:
    python -m scripts.prepare_shakespeare --vocab-size 1024

Produces in data/:
    shakespeare_train.bin   uint16 token stream (90%)
    shakespeare_val.bin     uint16 token stream (10%, the *tail* of the corpus)
    shakespeare_tok.json    the trained BPE merges
    shakespeare_meta.json   vocab_size and corpus statistics

Note the split is by position, not random: see `minigpt.data.train_val_split`.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from minigpt.bpe import BPETokenizer, CharTokenizer
from minigpt.data import write_tokens

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA = Path("data")


def fetch(path: Path) -> str:
    if not path.exists():
        print(f"downloading {URL}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as r:
            path.write_bytes(r.read())
    return path.read_text(encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=1024,
                    help="BPE vocabulary size (>=256); ignored for --char")
    ap.add_argument("--char", action="store_true", help="character-level instead of BPE")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--prefix", default="shakespeare")
    args = ap.parse_args(argv)

    text = fetch(DATA / "tinyshakespeare.txt")
    print(f"corpus: {len(text):,} characters, {len(set(text))} distinct")

    if args.char:
        tok = CharTokenizer(text)
        vocab_size = tok.vocab_size
        ids = tok.encode(text)
        meta_extra = {"kind": "char", "chars": tok.chars}
    else:
        tok = BPETokenizer()
        # Train on a prefix: merge statistics converge fast and this keeps the
        # script to a few seconds.  Encoding still covers the whole corpus.
        tok.train(text[:500_000], vocab_size=args.vocab_size, verbose=False)
        tok.add_special_tokens(["<|endoftext|>"])
        vocab_size = tok.vocab_size
        ids = tok.encode_ordinary(text)
        tok.save(DATA / f"{args.prefix}_tok.json")
        meta_extra = {"kind": "bpe", "tokenizer": f"{args.prefix}_tok.json",
                      "eos_id": tok.special_tokens["<|endoftext|>"]}

    n_val = int(len(ids) * args.val_fraction)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    write_tokens(train_ids, DATA / f"{args.prefix}_train.bin", vocab_size)
    write_tokens(val_ids, DATA / f"{args.prefix}_val.bin", vocab_size)

    meta = {
        "vocab_size": vocab_size,
        "n_train_tokens": len(train_ids),
        "n_val_tokens": len(val_ids),
        "bytes_per_token": len(text.encode()) / len(ids),
        **meta_extra,
    }
    (DATA / f"{args.prefix}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2)[:400])
    print(f"wrote {len(train_ids):,} train / {len(val_ids):,} val tokens to {DATA}/")


if __name__ == "__main__":
    main()
