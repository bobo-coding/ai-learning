"""The full post-training pipeline on one small model, end to end.

    python -m scripts.run_alignment --help

Four stages, each measured on the same held-out set so the effect of each is
visible:

  0. **pretrain** on plain task text ("17 + 72 = 89") -- the model learns the
     patterns but has no idea it is supposed to answer questions.
  1. **SFT** on chat-formatted demonstrations, with loss on the completion only.
     The demonstrations are **30% wrong on purpose**: that is what real
     human-written data looks like, and it is what leaves room for the next two
     stages to do something.
  2. **DPO** on (correct, plausibly-wrong) preference pairs.  No verifier at
     training time -- just relative preferences, as with human labels.
  3. **GRPO / RLVR** with the exact-match verifier in the loop.

The point of the noisy SFT data: with clean demonstrations, SFT alone gets close
to the ceiling and the alignment stages have nothing to fix, which makes for a
flattering but uninformative demo.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from minigpt.bpe import CharTokenizer
from minigpt.config import GPTConfig
from minigpt.data import read_tokens, write_tokens
from minigpt.dpo import DPOConfig, encode_pairs, freeze_reference, train_dpo
from minigpt.eval import bootstrap_ci, generative_eval, mcnemar, pass_at_k_eval
from minigpt.grpo import GRPOConfig, train_grpo
from minigpt.model import GPT
from minigpt.sft import CHAT_SPECIALS, ChatTemplate, SFTConfig, train_sft
from minigpt.tasks import (TASK_CHARS, corrupt_answer, make_preference_pairs,
                           reward_exact, sample_examples, split_examples,
                           stratified_sample, systematic_corrupt)
from minigpt.train import TrainConfig, Trainer
from minigpt.utils import human, pick_device, seed_everything

DATA = Path("data")
OUT = Path("out/alignment")


def build_tokenizer() -> CharTokenizer:
    return CharTokenizer(chars=TASK_CHARS, special_tokens=CHAT_SPECIALS)


def stage0_pretrain(args, tok, train_ex, device):
    """Plain next-token pretraining on unformatted task text."""
    text = "".join(f"{e.prompt} {e.answer}\n" for e in train_ex)
    ids = tok.encode(text)
    n_val = len(ids) // 20
    write_tokens(ids[:-n_val], DATA / "tasks_train.bin", tok.vocab_size)
    write_tokens(ids[-n_val:], DATA / "tasks_val.bin", tok.vocab_size)
    print(f"pretrain corpus: {human(len(ids))} tokens, vocab {tok.vocab_size}")

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size,
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                    dropout=0.0)
    model = GPT(cfg)
    print(f"model: {human(model.num_params())} params")
    tr = read_tokens(DATA / "tasks_train.bin", tok.vocab_size)
    va = read_tokens(DATA / "tasks_val.bin", tok.vocab_size)
    trainer = Trainer(model, TrainConfig(
        batch_size=args.batch_size, max_steps=args.pretrain_steps, lr=args.pretrain_lr,
        warmup_steps=max(1, args.pretrain_steps // 20), eval_interval=max(1, args.pretrain_steps // 4),
        out_dir=str(Path(args.out_dir) / "base"), device=str(device),
        log_interval=max(1, args.pretrain_steps // 10),
    ), tr, va)
    trainer.fit()
    return trainer.raw_model


def make_sft_data(train_ex, noise: float, seed: int, systematic: bool = True):
    """Chat demonstrations, a `noise` fraction of which have wrong answers.

    `systematic=True` makes every wrong demonstration wrong in the *same* way
    (see minigpt.tasks.systematic_corrupt).  That is what leaves the later
    stages something to fix: random noise is averaged away by cross-entropy,
    a consistent bias is learned.
    """
    rng = random.Random(seed)
    rows = []
    n_bad = 0
    for e in train_ex:
        if rng.random() < noise:
            bad = systematic_corrupt(e) if systematic else corrupt_answer(e, rng)
            rows.append((e.prompt, bad))
            n_bad += 1
        else:
            rows.append((e.prompt, e.answer))
    return rows, n_bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-examples", type=int, default=20000)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--pretrain-steps", type=int, default=1500)
    ap.add_argument("--pretrain-lr", type=float, default=1e-3)
    ap.add_argument("--sft-epochs", type=int, default=3)
    ap.add_argument("--sft-lr", type=float, default=5e-4)
    ap.add_argument("--sft-noise", type=float, default=0.45)
    ap.add_argument("--random-noise", action="store_true",
                    help="use random wrong answers instead of a systematic bias; "
                         "cross-entropy averages these away, so SFT barely suffers")
    ap.add_argument("--dpo-epochs", type=int, default=2)
    ap.add_argument("--dpo-lr", type=float, default=2e-5)
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--dpo-variant", default="sigmoid",
                    choices=["sigmoid", "ipo", "hinge", "simpo"])
    ap.add_argument("--dpo-sft-weight", type=float, default=1.0,
                    help="weight on the NLL of the chosen answer (RPO). "
                         "Set to 0 for plain DPO and watch accuracy collapse -- "
                         "see scripts/exp_dpo_variants.py")
    ap.add_argument("--grpo-iters", type=int, default=60)
    ap.add_argument("--grpo-lr", type=float, default=2e-5)
    ap.add_argument("--grpo-group", type=int, default=8)
    ap.add_argument("--grpo-prompts", type=int, default=8)
    ap.add_argument("--eval-per-task", type=int, default=50,
                    help="evaluation examples per task. Stratified, because a "
                         "shuffled prefix follows the training mixture and can "
                         "leave a task with n=1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(OUT),
                    help="where checkpoints and report.json go; use a distinct "
                         "directory per experiment so runs do not clobber each other")
    ap.add_argument("--base-ckpt", default=None,
                    help="reuse this pretrained checkpoint instead of training one "
                         "(implies --skip-pretrain)")
    ap.add_argument("--skip-pretrain", action="store_true",
                    help="reuse <out-dir>/base/final.pt")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else pick_device()
    seed_everything(args.seed)
    tok = build_tokenizer()
    template = ChatTemplate(tok)

    # ---- data ------------------------------------------------------------
    all_ex = sample_examples(args.n_examples, seed=args.seed)
    train_ex, val_ex = split_examples(all_ex, val_fraction=0.12)
    eval_ex = stratified_sample(val_ex, args.eval_per_task, seed=args.seed)
    from collections import Counter
    counts = Counter(e.task for e in eval_ex)
    print(f"{len(train_ex)} train / {len(val_ex)} held-out prompts (disjoint)")
    print(f"evaluating on {len(eval_ex)} stratified: "
          + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    report: dict = {"args": vars(args), "stages": {}}

    def measure(model, label, with_pass_at_k=False):
        res = generative_eval(model, template, eval_ex, reward_exact,
                              max_new_tokens=14, greedy=True, device=device,
                              return_samples=4)
        if with_pass_at_k:
            res.update(pass_at_k_eval(model, template,
                                      stratified_sample(eval_ex, 8, seed=1), reward_exact,
                                      n=8, ks=(1, 4, 8), temperature=1.0,
                                      max_new_tokens=14, device=device))
        print(f"\n--- {label} ---")
        per_task = {k: v for k, v in res.items()
                    if k not in ("overall", "samples", "n", "n_by_task")
                    and not k.startswith("pass@")}
        print(f"  overall exact-match: {res['overall']:.3f}")
        nbt = res["n_by_task"]
        print(f"  overall n={res['n']}")
        print("  by task: " + "  ".join(
            f"{k}={v:.2f}(n={nbt[k]})" for k, v in sorted(per_task.items())))
        if with_pass_at_k:
            print("  " + "  ".join(f"{k}={res[k]:.3f}" for k in res if k.startswith("pass@")))
        for prompt, comp, ans, r in res["samples"]:
            print(f"    {prompt!r:26s} -> {comp!r:12s} (want {ans!r}) r={r}")
        return res

    def per_item_correct(model):
        """0/1 per eval item, for the paired significance test."""
        res = generative_eval(model, template, eval_ex, reward_exact, max_new_tokens=14,
                              greedy=True, device=device, return_samples=len(eval_ex))
        return [int(r >= 1.0) for *_, r in res["samples"]]

    # ---- stage 0: pretrain ----------------------------------------------
    t0 = time.perf_counter()
    base_path = Path(args.base_ckpt) if args.base_ckpt else out / "base" / "final.pt"
    if (args.skip_pretrain or args.base_ckpt) and base_path.exists():
        blob = torch.load(base_path, map_location="cpu", weights_only=False)
        base = GPT(GPTConfig(**blob["model_cfg"]))
        base.load_state_dict(blob["model"])
        base = base.to(device)
        print(f"loaded base model from {base_path}")
    else:
        base = stage0_pretrain(args, tok, train_ex, device)
    report["stages"]["pretrain"] = {"seconds": time.perf_counter() - t0}
    r_base = measure(base, "stage 0: BASE (pretrained, no chat format)")
    report["stages"]["pretrain"]["eval"] = {k: v for k, v in r_base.items() if k != "samples"}

    # ---- stage 1: SFT ----------------------------------------------------
    rows, n_bad = make_sft_data(train_ex, args.sft_noise, args.seed,
                                systematic=not args.random_noise)
    kind = "randomly" if args.random_noise else "systematically"
    print(f"\nSFT data: {len(rows)} demonstrations, {n_bad} ({n_bad / len(rows):.0%}) "
          f"{kind} wrong")
    enc = [template.encode(p, a, max_len=args.block_size) for p, a in rows]
    val_enc = [template.encode(e.prompt, e.answer, max_len=args.block_size) for e in val_ex]
    sft_model = GPT(base.cfg).to(device)
    sft_model.load_state_dict(base.state_dict())
    t0 = time.perf_counter()
    hist_sft = train_sft(sft_model, enc, SFTConfig(
        epochs=args.sft_epochs, batch_size=args.batch_size, lr=args.sft_lr,
        log_interval=max(1, len(enc) // args.batch_size // 2), seed=args.seed,
    ), template.pad_id, val_examples=val_enc, device=device)
    torch.save({"model": sft_model.state_dict(), "model_cfg": sft_model.cfg.to_dict()},
               out / "sft.pt")
    r_sft = measure(sft_model, "stage 1: SFT (noisy demonstrations)", with_pass_at_k=True)
    report["stages"]["sft"] = {"seconds": time.perf_counter() - t0, "n_bad": n_bad,
                               "eval": {k: v for k, v in r_sft.items() if k != "samples"},
                               "final_loss": hist_sft[-1]["loss"]}
    correct_sft = per_item_correct(sft_model)

    # ---- stage 2: DPO ----------------------------------------------------
    # The rejected side is the mistake the SFT model actually makes, which is
    # how preference data is really collected.
    pairs = make_preference_pairs(train_ex, seed=args.seed + 1,
                                  systematic=not args.random_noise)
    chosen, rejected = encode_pairs(pairs, template, max_len=args.block_size)
    print(f"\nDPO data: {len(pairs)} preference pairs")
    dpo_model = GPT(sft_model.cfg).to(device)
    dpo_model.load_state_dict(sft_model.state_dict())
    ref = freeze_reference(dpo_model)          # the reference IS the SFT checkpoint
    t0 = time.perf_counter()
    hist_dpo = train_dpo(dpo_model, chosen, rejected, DPOConfig(
        beta=args.dpo_beta, epochs=args.dpo_epochs, batch_size=args.batch_size,
        lr=args.dpo_lr, variant=args.dpo_variant, sft_weight=args.dpo_sft_weight,
        log_interval=max(1, len(pairs) // args.batch_size // 3), seed=args.seed,
    ), template.pad_id, reference=ref, device=device)
    torch.save({"model": dpo_model.state_dict(), "model_cfg": dpo_model.cfg.to_dict()},
               out / "dpo.pt")
    r_dpo = measure(dpo_model, "stage 2: DPO", with_pass_at_k=True)
    report["stages"]["dpo"] = {
        "seconds": time.perf_counter() - t0,
        "eval": {k: v for k, v in r_dpo.items() if k != "samples"},
        "final_reward_accuracy": hist_dpo[-1]["reward_accuracy"],
        "final_reward_margin": hist_dpo[-1]["reward_margin"],
    }
    correct_dpo = per_item_correct(dpo_model)

    # ---- stage 3: GRPO / RLVR -------------------------------------------
    grpo_model = GPT(dpo_model.cfg).to(device)
    grpo_model.load_state_dict(dpo_model.state_dict())
    grpo_ref = freeze_reference(grpo_model)
    rl_prompts = train_ex[:2000]
    quick = stratified_sample(val_ex, 10, seed=args.seed + 7)

    def eval_fn(m):
        return {"overall": generative_eval(m, template, quick, reward_exact,
                                           max_new_tokens=14, greedy=True,
                                           device=device)["overall"]}

    print(f"\nGRPO: {args.grpo_iters} iterations x {args.grpo_prompts} prompts "
          f"x {args.grpo_group} samples = "
          f"{args.grpo_iters * args.grpo_prompts * args.grpo_group} rollouts")
    t0 = time.perf_counter()
    hist_grpo = train_grpo(grpo_model, rl_prompts, template, reward_exact, GRPOConfig(
        iterations=args.grpo_iters, prompts_per_iter=args.grpo_prompts,
        group_size=args.grpo_group, lr=args.grpo_lr, kl_coef=0.02, temperature=1.0,
        max_new_tokens=14, log_interval=max(1, args.grpo_iters // 10), seed=args.seed,
    ), reference=grpo_ref, device=device, eval_fn=eval_fn)
    torch.save({"model": grpo_model.state_dict(), "model_cfg": grpo_model.cfg.to_dict()},
               out / "grpo.pt")
    r_grpo = measure(grpo_model, "stage 3: GRPO (RLVR)", with_pass_at_k=True)
    report["stages"]["grpo"] = {
        "seconds": time.perf_counter() - t0,
        "eval": {k: v for k, v in r_grpo.items() if k != "samples"},
        "reward_first": hist_grpo[0]["reward_mean"],
        "reward_last": hist_grpo[-1]["reward_mean"],
    }
    correct_grpo = per_item_correct(grpo_model)

    # ---- summary with error bars and a paired test -----------------------
    print("\n" + "=" * 70)
    print("SUMMARY  (held-out exact match, greedy decoding)")
    print("=" * 70)
    for label, res, correct in [("base", r_base, None), ("SFT", r_sft, correct_sft),
                                ("DPO", r_dpo, correct_dpo), ("GRPO", r_grpo, correct_grpo)]:
        line = f"  {label:6s} {res['overall']:.3f}"
        if correct is not None:
            m, lo, hi = bootstrap_ci([float(c) for c in correct])
            line += f"   95% CI [{lo:.3f}, {hi:.3f}]"
        if "pass@8" in res:
            line += f"   pass@1={res.get('pass@1', 0):.3f} pass@8={res['pass@8']:.3f}"
        print(line)
    print()
    for a_lab, a, b_lab, b in [("SFT", correct_sft, "DPO", correct_dpo),
                               ("DPO", correct_dpo, "GRPO", correct_grpo),
                               ("SFT", correct_sft, "GRPO", correct_grpo)]:
        t = mcnemar(a, b)
        print(f"  McNemar {a_lab} -> {b_lab}: {b_lab} fixed {int(t['b01'])}, "
              f"broke {int(t['b10'])}, p = {t['p_value']:.4f}"
              f"{'  (significant)' if t['p_value'] < 0.05 else ''}")
        report["stages"].setdefault("tests", {})[f"{a_lab}_to_{b_lab}"] = t

    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {out}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
