"""Measure, per task, how many GRPO groups carry no gradient at all.

    python -m scripts.probe_zero_variance --ckpt out/align_plaindpo/dpo.pt

Lesson 8 quotes this table. It is the difference between "RL struggled with this
task" and "RL received exactly zero signal about this task".

GRPO's advantage is `A_i = r_i - mean(r_1..r_G)`. If all `G` rollouts for a
prompt score the same -- all correct or, far more often, all wrong -- then every
advantage is 0 and the policy gradient contributed by that group is 0. No
learning rate, no number of iterations and no KL coefficient changes that.

So before blaming a reward or a hyperparameter, measure this. A task sitting at
`all-wrong = 1.00` cannot improve, and the fix is supervised data, not RL.
"""

from __future__ import annotations

import argparse

import torch

from minigpt.bpe import CharTokenizer
from minigpt.config import GPTConfig
from minigpt.grpo import sample_rollouts, zero_variance_fraction
from minigpt.model import GPT
from minigpt.sft import CHAT_SPECIALS, ChatTemplate
from minigpt.tasks import (TASK_CHARS, reward_exact, sample_examples,
                           split_examples)
from minigpt.utils import pick_device, seed_everything


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default="out/align_plaindpo/dpo.pt",
                    help="checkpoint to sample from")
    ap.add_argument("--group-size", type=int, default=8, help="rollouts per prompt (G)")
    ap.add_argument("--prompts-per-task", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="the exploration policy -- this is what creates the variance")
    ap.add_argument("--max-new-tokens", type=int, default=14)
    ap.add_argument("--n-examples", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = torch.device(args.device) if args.device else pick_device()
    seed_everything(args.seed)
    tok = CharTokenizer(chars=TASK_CHARS, special_tokens=CHAT_SPECIALS)
    template = ChatTemplate(tok)
    _, val = split_examples(sample_examples(args.n_examples, seed=args.seed), 0.12)

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**blob["model_cfg"]))
    model.load_state_dict(blob["model"])
    model = model.to(device).eval()

    print(f"{args.ckpt}: {args.group_size} rollouts x {args.prompts_per_task} prompts "
          f"per task at T={args.temperature}")
    print(f"\n{'task':9s} {'mean reward':>12s} {'zero-var':>10s} {'all-wrong':>11s} "
          f"{'all-right':>11s}")
    rows = []
    for task in sorted({e.task for e in val}):
        examples = [e for e in val if e.task == task][: args.prompts_per_task]
        rollouts = sample_rollouts(model, examples, template, reward_exact,
                                   group_size=args.group_size,
                                   max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature, device=device)
        rewards = torch.tensor([r.reward for r in rollouts])
        groups = rewards.view(-1, args.group_size)
        zv = zero_variance_fraction(rewards, args.group_size)
        all_wrong = float((groups.sum(1) == 0).float().mean())
        all_right = float((groups.sum(1) == args.group_size).float().mean())
        rows.append((task, float(rewards.mean()), zv, all_wrong, all_right))
        print(f"  {task:7s} {rewards.mean():12.3f} {zv:10.2f} {all_wrong:11.2f} "
              f"{all_right:11.2f}")

    dead = [t for t, _, _, aw, _ in rows if aw >= 0.99]
    if dead:
        print(f"\n  {', '.join(dead)}: every group is all-wrong, so the advantage is")
        print("  identically zero. RLVR cannot improve these at all -- not slowly, not")
        print("  at all. Fix the policy with supervised data first.")
    starved = [t for t, _, zv, aw, _ in rows if zv >= 0.8 and aw < 0.99]
    if starved:
        print(f"\n  {', '.join(starved)}: >=80% of groups carry no gradient. Learning")
        print("  here is possible but slow; raise G or the temperature, or drop solved")
        print("  prompts (DAPO-style dynamic sampling).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
