"""Generate text from a checkpoint, and time the decoding paths.

    python -m scripts.sample --ckpt out/pretrain/best.pt --prompt "ROMEO:"
    python -m scripts.sample --ckpt out/pretrain/best.pt --bench

`--bench` is the more interesting mode: it measures the KV cache against
recomputing the prefix every step, which is the difference between O(T) and
O(T^2) total work, and it sweeps the sampling parameters so you can see what
temperature and top-p actually do to the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minigpt.bpe import BPETokenizer, CharTokenizer
from minigpt.generate import generate
from minigpt.model import GPT
from minigpt.utils import benchmark_repeat, human, pick_device, seed_everything


def load_tokenizer(meta_path: Path):
    meta = json.loads(meta_path.read_text())
    if meta["kind"] == "char":
        return CharTokenizer(chars=meta["chars"]), meta
    return BPETokenizer.load(meta_path.parent / meta["tokenizer"]), meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/pretrain/best.pt")
    ap.add_argument("--meta", default="data/shakespeare_meta.json")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bench", action="store_true", help="time the decoding paths instead")
    args = ap.parse_args(argv)

    device = torch.device(args.device) if args.device else pick_device()
    seed_everything(args.seed)

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    from minigpt.config import GPTConfig

    cfg = GPTConfig(**blob.get("model_cfg", blob.get("cfg")))
    model = GPT(cfg)
    model.load_state_dict(blob.get("model", blob.get("state_dict")))
    model = model.to(device).eval()
    tok, meta = load_tokenizer(Path(args.meta))
    print(f"{human(model.num_params())} params | vocab {cfg.vocab_size} | "
          f"block {cfg.block_size} | {device}")

    ids = tok.encode(args.prompt) or [0]
    prompt = torch.tensor([ids], dtype=torch.long, device=device)

    if args.bench:
        n = min(128, cfg.block_size - prompt.shape[1] - 1)
        stats = {}
        for label, use_cache in (("with KV cache", True), ("no cache (recompute)", False)):
            st = stats[label] = benchmark_repeat(
                lambda c=use_cache: generate(model, prompt, n, greedy=True, use_cache=c),
                device, repeats=5, warmup=2, iters=3)
            print(f"  {label:22s} {st['median'] * 1e3:7.1f} ms "
                  f"[{st['min'] * 1e3:.0f}-{st['max'] * 1e3:.0f}]  "
                  f"{n / st['median']:7.1f} tok/s  +/-{st['spread'] * 100:.0f}%")
        c, nc = stats["with KV cache"], stats["no cache (recompute)"]
        print(f"  -> cache speedup {nc['median'] / c['median']:.2f}x at {n} tokens "
              f"(range {nc['min'] / c['max']:.2f}-{nc['max'] / c['min']:.2f}x)")

        print("\n  effect of the sampling parameters (same seed, 80 tokens):")
        for label, kw in [("greedy", dict(greedy=True)),
                          ("T=0.5", dict(temperature=0.5)),
                          ("T=1.0", dict(temperature=1.0)),
                          ("T=1.0 top_k=40", dict(temperature=1.0, top_k=40)),
                          ("T=1.0 top_p=0.9", dict(temperature=1.0, top_p=0.9)),
                          ("T=1.5", dict(temperature=1.5))]:
            g = torch.Generator(device="cpu").manual_seed(args.seed)
            out = generate(model, prompt, 80, generator=g if device.type == "cpu" else None, **kw)
            text = tok.decode(out[0][len(ids):]).replace("\n", "\\n")
            print(f"    {label:16s} {text[:70]}")
        return 0

    for _ in range(args.samples):
        out = generate(model, prompt, args.tokens, temperature=args.temperature,
                       top_k=args.top_k, top_p=args.top_p)
        print("-" * 70)
        print(tok.decode(out[0]))
    print("-" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
