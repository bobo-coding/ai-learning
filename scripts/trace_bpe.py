"""Print every step BPE takes, so the algorithm is visible rather than asserted.

    python -m scripts.trace_bpe                      # the lesson's worked example
    python -m scripts.trace_bpe --text "..." --merges 8
    python -m scripts.trace_bpe --encode "low lower"

Lesson 1 quotes this output. Re-run it to check the lesson, or point it at your
own text to watch the merge order change.
"""

from __future__ import annotations

import argparse
from collections import Counter

from minigpt.bpe import BPETokenizer, get_pair_counts, merge

# Small enough that every merge fits on screen, repetitive enough that the
# merges are obviously sensible. The classic BPE-paper example.
DEMO = "low low low low low lower lower newest newest newest newest newest widest widest widest"


def show(label: str, b: bytes) -> str:
    return b.decode("utf-8", errors="replace").replace("\n", "\\n")


def trace_training(text: str, num_merges: int) -> BPETokenizer:
    tok = BPETokenizer()

    # --- step 0: pre-tokenize into fragments, with their frequencies ---------
    freqs = Counter(tok._re.findall(text))
    print("=" * 74)
    print("STEP 0  pre-tokenize, then count each distinct fragment")
    print("=" * 74)
    for frag, n in freqs.most_common():
        print(f"  {frag!r:12s} x{n:<3d}  bytes {list(frag.encode())}")
    total_positions = sum(len(f.encode()) * n for f, n in freqs.items())
    print(f"\n  {len(freqs)} distinct fragments cover {total_positions} byte positions")
    print("  -> pair counting walks the fragments, weighted; not the positions")

    # --- the state BPE maintains --------------------------------------------
    seqs = [list(f.encode("utf-8")) for f in freqs]
    weights = list(freqs.values())
    vocab = {i: bytes([i]) for i in range(256)}
    merges: dict[tuple[int, int], int] = {}

    print()
    print("=" * 74)
    print(f"TRAINING  {num_merges} merges")
    print("=" * 74)

    for i in range(num_merges):
        counts: Counter = Counter()
        for seq, w in zip(seqs, weights):
            if len(seq) >= 2:
                get_pair_counts(seq, counts, weight=w)
        if not counts:
            print(f"  merge {i + 1}: no adjacent pairs left, stopping early")
            break

        pair = max(counts, key=lambda p: (counts[p], p))
        new_id = 256 + i
        top = counts.most_common(4)

        print(f"\n  merge {i + 1}  ->  new token id {new_id}")
        print("    top pairs:", "  ".join(
            f"({show('', vocab[a])!r},{show('', vocab[b])!r})={c}" for (a, b), c in top))
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        merges[pair] = new_id
        print(f"    chose {pair} = ({show('', vocab[pair[0]])!r}, {show('', vocab[pair[1]])!r})"
              f"  count={counts[pair]}")
        print(f"    vocab[{new_id}] = {show('', vocab[new_id])!r}")

        before = [list(s) for s in seqs]
        seqs = [merge(s, pair, new_id) if len(s) >= 2 else s for s in seqs]
        changed = [(f, b, a) for f, b, a in zip(freqs, before, seqs) if b != a]
        for frag, b, a in changed[:3]:
            print(f"    {frag!r:12s} {b} -> {a}")
        if len(changed) > 3:
            print(f"    ... and {len(changed) - 3} more fragment(s)")

    tok.merges = merges
    tok.vocab = vocab
    print()
    print(f"  final vocabulary: 256 bytes + {len(merges)} merges = {len(vocab)} tokens")
    print("  learned tokens:", ", ".join(
        f"{i}={show('', vocab[i])!r}" for i in sorted(vocab) if i >= 256))
    return tok


def trace_encoding(tok: BPETokenizer, text: str) -> None:
    print()
    print("=" * 74)
    print(f"ENCODING  {text!r}")
    print("=" * 74)
    frags = tok._re.findall(text)
    print(f"  fragments: {frags}")
    all_ids: list[int] = []

    for frag in frags:
        ids = list(frag.encode("utf-8"))
        print(f"\n  fragment {frag!r}")
        print(f"    start: {ids}  = {[show('', tok.vocab[i]) for i in ids]}")
        step = 0
        while len(ids) >= 2:
            # exactly the rule in BPETokenizer._encode_fragment
            candidates = {p: tok.merges[p] for p in zip(ids, ids[1:]) if p in tok.merges}
            if not candidates:
                print("    no adjacent pair is a known merge -> done")
                break
            best_pair = min(candidates, key=candidates.get)
            best_id = candidates[best_pair]
            step += 1
            print(f"    step {step}: applicable merges "
                  f"{ {p: i for p, i in sorted(candidates.items(), key=lambda kv: kv[1])} }"
                  f" -> lowest id {best_id} wins")
            ids = merge(ids, best_pair, best_id)
            print(f"            {ids}  = {[show('', tok.vocab[i]) for i in ids]}")
        all_ids.extend(ids)

    print(f"\n  token ids: {all_ids}")
    print(f"  as text:   {[show('', tok.vocab[i]) for i in all_ids]}")
    print(f"  {len(text.encode())} bytes -> {len(all_ids)} tokens "
          f"({len(text.encode()) / len(all_ids):.2f} bytes/token)")
    decoded = b"".join(tok.vocab[i] for i in all_ids).decode("utf-8", errors="replace")
    print(f"  decode round-trip exact: {decoded == text}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--text", default=DEMO, help="training corpus")
    ap.add_argument("--merges", type=int, default=6)
    ap.add_argument("--encode", default="lowest", help="string to trace through encode()")
    args = ap.parse_args(argv)

    tok = trace_training(args.text, args.merges)
    trace_encoding(tok, args.encode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
