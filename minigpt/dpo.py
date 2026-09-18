"""Direct Preference Optimization, and the family of losses around it.

**Where it comes from.**  RLHF's objective is

    max_pi  E_{y ~ pi(.|x)}[ r(x, y) ]  -  beta * KL( pi(.|x) || pi_ref(.|x) )

whose exact solution is the Gibbs distribution

    pi*(y|x) = pi_ref(y|x) * exp(r(x, y) / beta) / Z(x).

Invert it and the reward is recovered from the optimal policy:

    r(x, y) = beta * log( pi*(y|x) / pi_ref(y|x) ) + beta * log Z(x).

Now put that into the Bradley-Terry preference model
``P(y_w > y_l) = sigmoid(r(x, y_w) - r(x, y_l))``.  The intractable ``log Z(x)``
cancels because it depends only on x, leaving a loss you can minimise directly
on preference pairs -- no reward model, no sampling, no RL:

    L = -log sigmoid( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))
                             - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )

**What the gradient does.**  d/dtheta L is
``-beta * sigmoid(-beta * margin) * [ grad log pi(y_w) - grad log pi(y_l) ]``.
The scalar ``sigmoid(-beta*margin)`` is an automatic difficulty weight: pairs
the model already gets right contribute almost nothing.  That is what makes DPO
stable without a value function -- and also what makes it quietly stop learning
once the margin is large, which is why `reward_margin` is the metric to watch,
not the loss.

**What it is not.**  DPO optimises a *relative* objective.  Both log-probs can
fall as long as the rejected one falls faster -- and in practice they do.  The
`chosen_logp` metric going steadily negative while accuracy rises is the normal,
documented behaviour, and the reason SFT loss is often mixed back in (see
`dpo_loss(..., sft_weight=...)`).
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import GPT
from .optim import clip_grad_norm
from .sft import Encoded, collate
from .utils import cosine_lr, pick_device, seed_everything


def freeze_reference(model: GPT) -> GPT:
    """A deep copy of `model` in eval mode with gradients off.

    The reference must be the *SFT* checkpoint you start from.  Using the
    pretrained base instead lets DPO undo the SFT formatting; using an EMA of
    the policy makes the KL term meaningless.
    """
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def masked_logprob_sum(model: GPT, x, y, mask, average: bool = False) -> torch.Tensor:
    """Sum of log p(y_t | x_<=t) over positions where `mask` is true: shape (B,)."""
    logits, _ = model(x)
    logp = torch.log_softmax(logits.float(), dim=-1)
    # gather needs valid indices everywhere, so clamp the padding then mask it out.
    tok_logp = logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    m = mask.to(tok_logp.dtype)
    total = (tok_logp * m).sum(dim=-1)
    return total / m.sum(dim=-1).clamp_min(1) if average else total


def dpo_loss(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp,
             beta: float = 0.1, label_smoothing: float = 0.0, variant: str = "sigmoid"):
    """The DPO objective and its diagnostics.

    `variant`:
      ``"sigmoid"`` -- standard DPO (Bradley-Terry).  With
          `label_smoothing > 0` this becomes conservative DPO (cDPO), which
          assumes a fraction of the preference labels are wrong and stops the
          loss from pushing any single pair to infinity.
      ``"ipo"``     -- Identity-PO: a squared loss that targets a *specific*
          margin ``1/(2*beta)`` instead of maximising it.  Notably less prone
          to the "both log-probs collapse" failure, at the cost of needing beta
          tuned to the data.
      ``"hinge"``   -- SLiC-style max-margin; ignores pairs already beyond the
          margin entirely.

    Returns (loss, metrics).  The metrics are the *implicit rewards*
    ``beta * log(pi/pi_ref)``, which is what DPO is secretly doing reward
    modelling with.
    """
    pi_logratio = policy_chosen_logp - policy_rejected_logp
    ref_logratio = ref_chosen_logp - ref_rejected_logp
    logits = pi_logratio - ref_logratio                     # the "margin"

    if variant == "sigmoid":
        if label_smoothing > 0:
            # cDPO: mix in the flipped-label loss with weight `label_smoothing`.
            loss = (-F.logsigmoid(beta * logits) * (1 - label_smoothing)
                    - F.logsigmoid(-beta * logits) * label_smoothing)
        else:
            loss = -F.logsigmoid(beta * logits)
    elif variant == "ipo":
        loss = (logits - 1.0 / (2.0 * beta)) ** 2
    elif variant == "hinge":
        loss = torch.relu(1.0 - beta * logits)
    else:
        raise ValueError(f"unknown DPO variant {variant!r}")

    with torch.no_grad():
        chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)
        rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)
        metrics = {
            "loss": loss.mean().item(),
            # Fraction of pairs the implicit reward model now orders correctly.
            # This, not the loss, is the number that tells you DPO is working.
            "reward_accuracy": (chosen_reward > rejected_reward).float().mean().item(),
            "reward_margin": (chosen_reward - rejected_reward).mean().item(),
            "chosen_reward": chosen_reward.mean().item(),
            "rejected_reward": rejected_reward.mean().item(),
            "policy_chosen_logp": policy_chosen_logp.mean().item(),
            "policy_rejected_logp": policy_rejected_logp.mean().item(),
        }
    return loss.mean(), metrics


def simpo_loss(policy_chosen_logp, policy_rejected_logp, chosen_len, rejected_len,
               beta: float = 2.0, gamma: float = 0.5):
    """SimPO: reference-free, length-normalised preference loss.

        L = -log sigmoid( beta * (logp_w/|y_w| - logp_l/|y_l|) - gamma )

    Two changes from DPO, both aimed at real failure modes: dividing by length
    removes DPO's bias towards long answers (a longer sequence has a more
    negative log-prob, so raw sums favour brevity in the rejected slot), and
    dropping the reference halves the memory and removes a hyperparameter.  The
    price is that nothing anchors the policy to its starting point, so it drifts
    more and needs a smaller LR.
    """
    avg_chosen = policy_chosen_logp / chosen_len.clamp_min(1)
    avg_rejected = policy_rejected_logp / rejected_len.clamp_min(1)
    logits = beta * (avg_chosen - avg_rejected) - gamma
    loss = -F.logsigmoid(logits)
    with torch.no_grad():
        metrics = {
            "loss": loss.mean().item(),
            "reward_accuracy": (avg_chosen > avg_rejected).float().mean().item(),
            "reward_margin": (beta * (avg_chosen - avg_rejected)).mean().item(),
        }
    return loss.mean(), metrics


@dataclass
class DPOConfig:
    beta: float = 0.1              # KL strength: higher = stays closer to ref
    epochs: int = 1
    batch_size: int = 16
    lr: float = 5e-6               # DPO needs a *much* smaller LR than SFT
    weight_decay: float = 0.0
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    variant: str = "sigmoid"   # "sigmoid" | "ipo" | "hinge" | "simpo"
    label_smoothing: float = 0.0
    simpo_gamma: float = 0.5   # target margin for the reference-free SimPO loss
    sft_weight: float = 0.0        # add SFT loss on the chosen response (RPO-style)
    log_interval: int = 10
    seed: int = 0


def encode_pairs(pairs: list[dict], template, max_len: int | None = None):
    """Encode (prompt, chosen, rejected) triples into two aligned example lists."""
    chosen = [template.encode(p["prompt"], p["chosen"], max_len) for p in pairs]
    rejected = [template.encode(p["prompt"], p["rejected"], max_len) for p in pairs]
    return chosen, rejected


def train_dpo(policy: GPT, pairs_chosen: list[Encoded], pairs_rejected: list[Encoded],
              cfg: DPOConfig, pad_id: int, reference: GPT | None = None, device=None,
              verbose: bool = True):
    """Run DPO.  `reference` defaults to a frozen copy of `policy` at entry."""
    device = torch.device(device) if device else pick_device()
    policy = policy.to(device)
    # SimPO needs no reference model at all -- half the memory of DPO.
    ref = None if cfg.variant == "simpo" else (reference or freeze_reference(policy)).to(device)
    seed_everything(cfg.seed)
    opt = policy.configure_optimizers(lr=cfg.lr, weight_decay=cfg.weight_decay)

    n = len(pairs_chosen)
    steps_per_epoch = math.ceil(n / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total * cfg.warmup_frac))
    rng = random.Random(cfg.seed)
    order = list(range(n))
    history: list[dict] = []
    step = 0
    policy.train()

    for epoch in range(cfg.epochs):
        rng.shuffle(order)
        for i in range(0, n, cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            # Chosen and rejected are collated separately (different lengths),
            # then concatenated into one forward pass -- halving the launches
            # and keeping batch statistics identical for both sides.
            xc, yc, mc = collate([pairs_chosen[j] for j in idx], pad_id, device)
            xr, yr, mr = collate([pairs_rejected[j] for j in idx], pad_id, device)

            lr = cosine_lr(step, base_lr=cfg.lr, warmup=warmup, total=total, min_ratio=0.1)
            for g in opt.param_groups:
                g["lr"] = lr

            t0 = time.perf_counter()
            pc = masked_logprob_sum(policy, xc, yc, mc)
            pr = masked_logprob_sum(policy, xr, yr, mr)

            if cfg.variant == "simpo":
                # Reference-free: no second model to run, so this is also the
                # cheapest of the family.
                loss, metrics = simpo_loss(pc, pr, mc.sum(-1).float(), mr.sum(-1).float(),
                                           beta=cfg.beta, gamma=cfg.simpo_gamma)
                metrics.setdefault("policy_chosen_logp", pc.mean().item())
                metrics.setdefault("policy_rejected_logp", pr.mean().item())
            else:
                with torch.no_grad():
                    rc = masked_logprob_sum(ref, xc, yc, mc)
                    rr = masked_logprob_sum(ref, xr, yr, mr)
                loss, metrics = dpo_loss(pc, pr, rc, rr, beta=cfg.beta,
                                         label_smoothing=cfg.label_smoothing,
                                         variant=cfg.variant)
            if cfg.sft_weight > 0:
                # RPO / "DPO + NLL": keeps the absolute likelihood of good
                # answers from sliding while the ratio improves.
                _, nll = policy(xc, targets=yc, loss_mask=mc)
                loss = loss + cfg.sft_weight * nll
                metrics["sft_loss"] = nll.item()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = clip_grad_norm([p for gp in opt.param_groups for p in gp["params"]],
                                   cfg.grad_clip)
            opt.step()

            metrics.update(step=step, epoch=epoch, lr=lr, grad_norm=gnorm,
                           dt=time.perf_counter() - t0, total_loss=loss.item())
            history.append(metrics)
            if verbose and step % cfg.log_interval == 0:
                print(f"dpo step {step:4d}/{total} | loss {metrics['loss']:.4f} "
                      f"| acc {metrics['reward_accuracy']:.3f} "
                      f"| margin {metrics['reward_margin']:+.3f} "
                      f"| logp_w {metrics['policy_chosen_logp']:+.2f} "
                      f"logp_l {metrics['policy_rejected_logp']:+.2f}")
            step += 1
    return history
