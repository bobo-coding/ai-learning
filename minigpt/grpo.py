"""RLVR with GRPO: reinforcement learning from a *verifier* instead of a human.

The setup that made reasoning models work: if you can check an answer
programmatically (arithmetic, unit tests, a proof checker), you do not need a
learned reward model at all.  Sample completions, score them with the checker,
and push up the log-probability of the ones that passed.

**GRPO's one idea.**  PPO needs a value network to estimate the baseline
``V(s)``, which for LLMs means a second model the same size as the policy, with
its own training instability.  GRPO replaces it with an empirical baseline: draw
G completions for the *same* prompt and use the group's own mean reward.

    A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

No value head, no GAE, no bootstrapping.  The variance reduction is real because
all G samples share the prompt, so the prompt's intrinsic difficulty cancels out
-- which is exactly what a value function was estimating.

**The objective** (PPO-clipped, per token):

    L = -E[ min( rho_t * A, clip(rho_t, 1-e, 1+e) * A ) ] + beta * KL(pi || pi_ref)
    rho_t = pi_theta(y_t | .) / pi_old(y_t | .)

with the low-variance, always-positive K3 estimator for the KL:

    KL_hat = exp(ref - pi) - (ref - pi) - 1

**The failure modes you will actually hit**, all visible in the metrics here:

* *Zero-variance groups.*  If all G completions are right (or all wrong), the
  advantage is exactly 0 and the group contributes no gradient.  Early on, most
  groups are all-wrong; late in training, most are all-right.  `zero_var_frac`
  is the metric that tells you your task difficulty is mismatched to your model.
* *Length bias.*  Averaging the loss per sequence lets a long wrong answer
  dilute its own penalty.  `loss_agg="token"` (sum over tokens, divide by total
  tokens in the batch) is the DAPO fix; `"sequence"` is the original GRPO.
* *Entropy collapse.*  Reward-only training sharpens the policy until it emits
  one answer per prompt and exploration stops.  Watch `entropy`.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import torch

from .generate import generate
from .model import GPT
from .optim import clip_grad_norm
from .sft import Encoded, collate
from .utils import cosine_lr, pick_device, seed_everything


# ---------------------------------------------------------------------------
# Advantages
# ---------------------------------------------------------------------------


def group_advantages(rewards: torch.Tensor, group_size: int, std_normalize: bool = True,
                     eps: float = 1e-4) -> torch.Tensor:
    """Group-relative advantages for a flat reward vector of length n*G.

    Rewards must be laid out group-major: ``[g0_0 .. g0_{G-1}, g1_0, ...]``.

    `std_normalize=False` gives the "Dr. GRPO" variant: dividing by the group
    std up-weights groups that happen to be low-variance (e.g. 7 of 8 correct),
    which is a difficulty bias nobody asked for.  Keeping only the mean
    subtraction removes it at the cost of a larger gradient scale.
    """
    if rewards.numel() % group_size != 0:
        raise ValueError(f"{rewards.numel()} rewards is not a multiple of group_size={group_size}")
    r = rewards.view(-1, group_size).float()
    adv = r - r.mean(dim=1, keepdim=True)
    if std_normalize:
        adv = adv / (r.std(dim=1, unbiased=False, keepdim=True) + eps)
    return adv.view(-1)


def zero_variance_fraction(rewards: torch.Tensor, group_size: int) -> float:
    """Fraction of groups whose completions all scored the same (no signal)."""
    r = rewards.view(-1, group_size).float()
    return (r.std(dim=1, unbiased=False) < 1e-8).float().mean().item()


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------


def per_token_logprobs(model: GPT, x, y) -> torch.Tensor:
    """log pi(y_t | x_<=t) for all positions: shape (B, T)."""
    logits, _ = model(x)
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)


def per_token_entropy(model: GPT, x) -> torch.Tensor:
    """Entropy of the predictive distribution at each position: shape (B, T)."""
    logits, _ = model(x)
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def kl_k3(policy_logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Schulman's K3 estimator of KL(pi || pi_ref), per token.

        k3 = exp(r) - r - 1,  r = log pi_ref - log pi

    It is unbiased for the KL *and* always non-negative, unlike the naive
    ``-r`` estimator which is unbiased but frequently negative on a single
    sample -- a negative KL penalty actively pushes the policy away from the
    reference, which is the opposite of the intent.
    """
    r = ref_logp - policy_logp
    return torch.exp(r) - r - 1.0


def grpo_loss(policy_logp, old_logp, advantages, mask, ref_logp=None, clip_eps: float = 0.2,
              kl_coef: float = 0.0, loss_agg: str = "token"):
    """PPO-clipped policy loss with group advantages, plus an optional KL penalty.

    `policy_logp`, `old_logp`, `ref_logp`, `mask` are all (B, T) aligned with
    the *targets*; `advantages` is (B,) -- one number per sampled sequence.

    `loss_agg`:
      ``"token"``    sum over all tokens / total token count  (DAPO)
      ``"sequence"`` mean over tokens within each sequence, then mean over
                     sequences (original GRPO); short sequences get more weight
                     per token.
    """
    m = mask.to(policy_logp.dtype)
    ratio = torch.exp(policy_logp - old_logp)
    adv = advantages.unsqueeze(1).to(policy_logp.dtype)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    # min() of the two is the pessimistic bound: it only ever *reduces* the
    # objective, which is what keeps a large ratio from taking a huge step.
    per_token = -torch.min(unclipped, clipped)

    if ref_logp is not None and kl_coef > 0:
        per_token = per_token + kl_coef * kl_k3(policy_logp, ref_logp)

    if loss_agg == "token":
        loss = (per_token * m).sum() / m.sum().clamp_min(1)
    elif loss_agg == "sequence":
        per_seq = (per_token * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
        loss = per_seq.mean()
    else:
        raise ValueError(f"unknown loss_agg {loss_agg!r}")

    with torch.no_grad():
        clipped_frac = (((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)) & mask).float().sum()
        metrics = {
            "loss": loss.item(),
            "ratio_mean": ((ratio * m).sum() / m.sum().clamp_min(1)).item(),
            "clip_frac": (clipped_frac / m.sum().clamp_min(1)).item(),
            "adv_mean": advantages.mean().item(),
            "adv_std": advantages.std(unbiased=False).item(),
        }
        if ref_logp is not None:
            metrics["kl"] = ((kl_k3(policy_logp, ref_logp) * m).sum() / m.sum().clamp_min(1)).item()
    return loss, metrics


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------


@dataclass
class Rollout:
    encoded: Encoded
    reward: float
    completion: str
    prompt: str


def extract_completion(ids: list[int], tokenizer, end_id: int) -> tuple[list[int], str]:
    """Trim generated ids at the first end-of-turn token and decode.

    The trimmed token list keeps the terminator, because the model must be
    credited (or blamed) for choosing to stop -- dropping it from the loss mask
    is how you end up with a policy that never terminates.
    """
    cut = len(ids)
    for i, t in enumerate(ids):
        if t == end_id:
            cut = i + 1
            break
    kept = ids[:cut]
    text = tokenizer.decode([t for t in kept if t != end_id])
    return kept, text


@torch.no_grad()
def sample_rollouts(model: GPT, examples, template, reward_fn, group_size: int,
                    max_new_tokens: int = 12, temperature: float = 1.0, top_k: int = 0,
                    device=None, generator: torch.Generator | None = None) -> list[Rollout]:
    """Draw `group_size` completions for each example and score them.

    Sampling temperature is a genuine hyperparameter here, not a cosmetic one:
    it *is* the exploration policy.  Too low and every group has zero variance
    (nothing to learn from); too high and the advantages are dominated by noise.
    """
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    out: list[Rollout] = []
    for ex in examples:
        head = template.tok.encode(template.prompt_text(ex.prompt))
        prompt_ids = torch.tensor([head] * group_size, dtype=torch.long, device=device)
        gen = generate(model, prompt_ids, max_new_tokens, temperature=temperature, top_k=top_k,
                       eos_id=template.end_id, generator=generator)
        for row in gen:
            new_ids = row[len(head):].tolist()
            kept, text = extract_completion(new_ids, template.tok, template.end_id)
            if not kept:                      # model emitted nothing at all
                kept, text = [template.end_id], ""
            enc = Encoded(head + kept, [0] * len(head) + [1] * len(kept))
            out.append(Rollout(enc, float(reward_fn(ex, text)), text, ex.prompt))
    if was_training:
        model.train()
    return out


# ---------------------------------------------------------------------------
# The training loop
# ---------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    iterations: int = 40
    prompts_per_iter: int = 8
    group_size: int = 8
    inner_epochs: int = 1          # PPO epochs over one batch of rollouts
    lr: float = 1e-5
    weight_decay: float = 0.0
    clip_eps: float = 0.2
    kl_coef: float = 0.02
    max_new_tokens: int = 12
    temperature: float = 1.0
    top_k: int = 0
    grad_clip: float = 1.0
    loss_agg: str = "token"
    std_normalize: bool = True
    warmup_frac: float = 0.05
    log_interval: int = 1
    seed: int = 0


def train_grpo(policy: GPT, examples, template, reward_fn, cfg: GRPOConfig,
               reference: GPT | None = None, device=None, verbose: bool = True,
               eval_fn=None):
    """Run GRPO.  Returns per-iteration metrics.

    `eval_fn(policy) -> dict` is called every `log_interval` iterations so you
    can watch held-out accuracy rather than training reward -- the two diverge
    as soon as the policy starts exploiting the reward.
    """
    from .dpo import freeze_reference

    device = torch.device(device) if device else pick_device()
    policy = policy.to(device)
    ref = None
    if cfg.kl_coef > 0:
        ref = (reference or freeze_reference(policy)).to(device)
    seed_everything(cfg.seed)
    opt = policy.configure_optimizers(lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = random.Random(cfg.seed)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    warmup = max(1, int(cfg.iterations * cfg.warmup_frac))
    history: list[dict] = []

    for it in range(cfg.iterations):
        t0 = time.perf_counter()
        batch_examples = [rng.choice(examples) for _ in range(cfg.prompts_per_iter)]
        rollouts = sample_rollouts(
            policy, batch_examples, template, reward_fn, cfg.group_size,
            max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
            top_k=cfg.top_k, device=device,
            generator=gen if device.type == "cpu" else None,
        )
        rewards = torch.tensor([r.reward for r in rollouts], device=device)
        adv = group_advantages(rewards, cfg.group_size, std_normalize=cfg.std_normalize)
        zvf = zero_variance_fraction(rewards, cfg.group_size)

        x, y, mask = collate([r.encoded for r in rollouts], template.pad_id, device)
        # `old_logp` must come from the policy *as it was when it sampled*.  With
        # inner_epochs=1 the first inner step has ratio exactly 1; recomputing it
        # here (under no_grad) keeps that identity exact.
        with torch.no_grad():
            old_logp = per_token_logprobs(policy, x, y)
            ref_logp = per_token_logprobs(ref, x, y) if ref is not None else None

        lr = cosine_lr(it, base_lr=cfg.lr, warmup=warmup, total=cfg.iterations, min_ratio=0.2)
        for g in opt.param_groups:
            g["lr"] = lr

        last: dict = {}
        for _ in range(cfg.inner_epochs):
            policy.train()
            logp = per_token_logprobs(policy, x, y)
            loss, metrics = grpo_loss(logp, old_logp, adv, mask, ref_logp=ref_logp,
                                      clip_eps=cfg.clip_eps, kl_coef=cfg.kl_coef,
                                      loss_agg=cfg.loss_agg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = clip_grad_norm([p for gp in opt.param_groups for p in gp["params"]],
                                   cfg.grad_clip)
            opt.step()
            metrics["grad_norm"] = gnorm
            last = metrics

        with torch.no_grad():
            ent = per_token_entropy(policy, x)
            entropy = float((ent * mask).sum() / mask.sum().clamp_min(1))

        rec = {
            "iter": it, "lr": lr, "reward_mean": rewards.mean().item(),
            "reward_max": rewards.max().item(), "zero_var_frac": zvf,
            "entropy": entropy, "mean_len": float(mask.sum(1).float().mean()),
            "dt": time.perf_counter() - t0, **last,
        }
        if eval_fn is not None and (it % cfg.log_interval == 0 or it == cfg.iterations - 1):
            rec.update({f"eval_{k}": v for k, v in eval_fn(policy).items()})
        history.append(rec)
        if verbose and (it % cfg.log_interval == 0 or it == cfg.iterations - 1):
            msg = (f"grpo it {it:3d} | reward {rec['reward_mean']:.3f} | zero-var {zvf:.2f} "
                   f"| kl {rec.get('kl', 0):.4f} | clip {rec.get('clip_frac', 0):.3f} "
                   f"| H {entropy:.3f} | {rec['dt']:.1f}s")
            if "eval_overall" in rec:
                msg += f" | eval {rec['eval_overall']:.3f}"
            print(msg)
    return history
