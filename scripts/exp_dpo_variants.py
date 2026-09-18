"""Why DPO needs care: a side-by-side comparison of the preference losses.

    python -m scripts.exp_dpo_variants          # needs out/alignment/sft.pt

Starting from the same SFT checkpoint, this runs several preference-optimization
configurations and evaluates each on the same held-out set.  The interesting
result is not which one wins; it is *how badly* plain DPO can fail when the
rejected samples all share one systematic error.

The mechanism: DPO's gradient is
``-beta * sigmoid(-beta*margin) * (grad log pi(y_w) - grad log pi(y_l))``.
The second term pushes down the rejected answer, but nothing says the freed
probability mass has to land on the chosen answer -- it lands wherever the model
finds it easiest to put.  When every rejected sample is `answer + 1`, "smaller"
is a direction the model can follow, and it happily overshoots to `answer - 1`,
which is just as wrong.

The three fixes this script measures:
  * `sft_weight > 0` (RPO): add the ordinary NLL of the chosen answer, which
    anchors its *absolute* likelihood instead of only the ratio;
  * a larger `beta`: a stronger pull back towards the reference;
  * IPO: target a finite margin instead of maximising it;
  * mixing the rejected samples so there is no single direction to follow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minigpt.bpe import CharTokenizer
from minigpt.config import GPTConfig
from minigpt.dpo import DPOConfig, encode_pairs, freeze_reference, train_dpo
from minigpt.eval import bootstrap_ci, generative_eval, mcnemar
from minigpt.model import GPT
from minigpt.sft import CHAT_SPECIALS, ChatTemplate
from minigpt.tasks import (TASK_CHARS, make_preference_pairs, reward_exact,
                           sample_examples, split_examples, stratified_sample)
from minigpt.utils import pick_device, seed_everything

OUT = Path("out/alignment")


def mixed_pairs(train_ex, seed: int):
    """Half systematically-wrong, half randomly-wrong rejected samples."""
    import random

    rng = random.Random(seed)
    sysp = make_preference_pairs(train_ex, seed=seed, systematic=True)
    randp = make_preference_pairs(train_ex, seed=seed, systematic=False)
    return [s if rng.random() < 0.5 else r for s, r in zip(sysp, randp)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-ckpt", default=str(OUT / "sft.pt"))
    ap.add_argument("--n-examples", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eval-per-task", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    ckpt = Path(args.sft_ckpt)
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not found -- run `python -m scripts.run_alignment` first")

    device = torch.device(args.device) if args.device else pick_device()
    seed_everything(args.seed)
    tok = CharTokenizer(chars=TASK_CHARS, special_tokens=CHAT_SPECIALS)
    template = ChatTemplate(tok)

    # Same seed and split as run_alignment, so the numbers are comparable.
    train_ex, val_ex = split_examples(sample_examples(args.n_examples, seed=args.seed), 0.12)
    # Stratified, and with the same seed as run_alignment so the two scripts
    # score the identical set.
    eval_ex = stratified_sample(val_ex, args.eval_per_task, seed=args.seed)

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**blob["model_cfg"])

    def fresh():
        m = GPT(cfg)
        m.load_state_dict(blob["model"])
        return m.to(device)

    sys_pairs = make_preference_pairs(train_ex, seed=args.seed + 1, systematic=True)
    mix_pairs = mixed_pairs(train_ex, args.seed + 1)

    runs = [
        ("DPO b=0.1",              dict(beta=0.1, lr=1e-5), sys_pairs),
        ("DPO b=0.5",              dict(beta=0.5, lr=1e-5), sys_pairs),
        ("DPO b=0.1 + NLL (RPO)",  dict(beta=0.1, lr=1e-5, sft_weight=1.0), sys_pairs),
        ("IPO b=0.5",              dict(beta=0.5, lr=1e-5, variant="ipo"), sys_pairs),
        ("SimPO b=2 g=0.5",        dict(beta=2.0, lr=5e-6, variant="simpo"), sys_pairs),
        ("DPO b=0.1, mixed reject", dict(beta=0.1, lr=1e-5), mix_pairs),
    ]

    def measure(model):
        res = generative_eval(model, template, eval_ex, reward_exact, max_new_tokens=14,
                              greedy=True, device=device, return_samples=len(eval_ex))
        correct = [int(r >= 1.0) for *_, r in res["samples"]]
        return res, correct

    print("evaluating the SFT starting point")
    sft_res, sft_correct = measure(fresh().eval())
    print(f"  SFT held-out exact match: {sft_res['overall']:.3f}")

    results = {"sft": {"overall": sft_res["overall"]}}
    for label, kw, pairs in runs:
        print(f"\n=== {label} ===")
        chosen, rejected = encode_pairs(pairs, template, max_len=args.block_size)
        policy = fresh()
        ref = freeze_reference(policy)
        hist = train_dpo(policy, chosen, rejected, DPOConfig(
            epochs=args.epochs, batch_size=args.batch_size,
            log_interval=max(1, len(pairs) // args.batch_size // 2),
            seed=args.seed, **kw,
        ), template.pad_id, reference=ref, device=device)
        res, correct = measure(policy.eval())
        mean, lo, hi = bootstrap_ci([float(c) for c in correct])
        test = mcnemar(sft_correct, correct)
        per_task = {k: v for k, v in res.items()
                    if k not in ("overall", "samples", "n", "n_by_task")}
        print(f"  held-out exact match {res['overall']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        nbt = res["n_by_task"]
        print(f"  overall n={res['n']}")
        print("  by task: " + "  ".join(
            f"{k}={v:.2f}(n={nbt[k]})" for k, v in sorted(per_task.items())))
        print(f"  vs SFT: fixed {int(test['b01'])}, broke {int(test['b10'])}, "
              f"p={test['p_value']:.4f}")
        print(f"  final: reward_acc {hist[-1]['reward_accuracy']:.3f} "
              f"margin {hist[-1]['reward_margin']:+.3f} "
              f"logp_chosen {hist[-1].get('policy_chosen_logp', float('nan')):+.2f}")
        results[label] = {
            "overall": res["overall"], "ci": [lo, hi], "per_task": per_task,
            "mcnemar_vs_sft": test,
            "final_reward_accuracy": hist[-1]["reward_accuracy"],
            "final_margin": hist[-1]["reward_margin"],
            "final_logp_chosen": hist[-1].get("policy_chosen_logp"),
        }

    print("\n" + "=" * 68)
    print(f"{'config':26s} {'exact match':>12s}  {'vs SFT':>18s}")
    print("=" * 68)
    print(f"{'SFT (start)':26s} {sft_res['overall']:12.3f}")
    for label, _, _ in runs:
        r = results[label]
        t = r["mcnemar_vs_sft"]
        delta = r["overall"] - sft_res["overall"]
        print(f"{label:26s} {r['overall']:12.3f}  {delta:+7.3f}  p={t['p_value']:.4f}")

    (OUT / "dpo_variants.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {OUT}/dpo_variants.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
