"""AdamW and gradient clipping, written out so the update rule is visible.

`torch.optim.AdamW` is three lines of math hidden behind a lot of machinery.
Those three lines explain most of what you tune, so they are worth writing once:

    m_t = b1 * m_{t-1} + (1 - b1) * g              first moment  (momentum)
    v_t = b2 * v_{t-1} + (1 - b2) * g^2            second moment (per-parameter scale)
    p  <- p * (1 - lr * wd)                        decoupled weight decay
    p  <- p - lr * m_t / (1 - b1^t) / (sqrt(v_t / (1 - b2^t)) + eps)

Reading it:

* The update is *scale-invariant* in g: multiply every gradient by 10 and the
  step barely changes, because v grows too.  That is why Adam works without
  per-layer LR tuning, and why gradient clipping (which changes the scale) has
  a much weaker effect than you would expect.
* Bias correction matters only for the first ~1/(1-b2) ~ 1000 steps.  Without
  it, v starts near zero, sqrt(v) is tiny, and the first steps are enormous --
  which is a large part of why warmup exists even when you do have correction.
* `weight_decay` here is *decoupled*: a plain multiplicative shrink, not an
  L2 term added to the gradient.  Adding L2 to g would get divided by sqrt(v),
  making the effective decay depend on the gradient history -- the bug that
  "AdamW" fixed in "Adam + L2".
"""

from __future__ import annotations

import math

import torch


class AdamW(torch.optim.Optimizer):
    """AdamW, matching `torch.optim.AdamW` numerically (see tests/test_optim.py)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0 or eps < 0 or not (0 <= betas[0] < 1) or not (0 <= betas[1] < 1):
            raise ValueError("invalid hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                # Decoupled weight decay, applied to the parameter directly.
                if wd != 0:
                    p.mul_(1 - lr * wd)

                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                bc1 = 1 - b1**t
                bc2 = 1 - b2**t
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)
        return loss


class SGD(torch.optim.Optimizer):
    """SGD with (optionally Nesterov) momentum -- useful as a contrast to Adam.

    Try it on the pretraining loop: it needs a ~10x larger LR, is far more
    sensitive to that LR, and stalls on the embedding layer, whose gradients are
    extremely sparse and badly scaled.  Per-parameter scaling is what Adam buys.
    """

    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0.0, nesterov=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                                      nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mu, wd, nesterov = (group["lr"], group["momentum"],
                                    group["weight_decay"], group["nesterov"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)   # coupled L2, the classic formulation
                if mu != 0:
                    buf = self.state[p].get("momentum_buffer")
                    if buf is None:
                        buf = self.state[p]["momentum_buffer"] = g.clone()
                    else:
                        buf.mul_(mu).add_(g)
                    g = g.add(buf, alpha=mu) if nesterov else buf
                p.add_(g, alpha=-lr)
        return loss


@torch.no_grad()
def clip_grad_norm(params, max_norm: float) -> float:
    """Rescale all gradients so their *global* L2 norm is at most `max_norm`.

    Two things people get wrong:
      * it is one norm over all parameters concatenated, not per-tensor -- that
        is what preserves the *direction* of the update;
      * it must run after the last `backward()` of a gradient-accumulation
        cycle and before `step()`, otherwise you clip a partial gradient.

    Returns the norm *before* clipping.  Log it: a sudden spike is the earliest
    warning of a bad batch or a divergent run, several hundred steps before the
    loss curve shows anything.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    device = grads[0].device
    total = torch.norm(torch.stack([g.detach().norm(2).to(device) for g in grads]), 2)
    if max_norm and max_norm > 0:
        # clamp_max(1.0) means "only ever shrink"; a small-gradient step is left alone.
        scale = (max_norm / (total + 1e-6)).clamp(max=1.0)
        for g in grads:
            g.mul_(scale.to(g.device))
    return float(total)


def wsd_lr(step: int, *, base_lr: float, warmup: int, total: int, decay_frac: float = 0.1,
           min_ratio: float = 0.0) -> float:
    """Warmup-Stable-Decay: constant LR in the middle, linear decay at the end.

    Increasingly the default for large runs because the stable phase gives you a
    checkpoint that is usable at *any* point -- you decide the token budget
    afterwards and only then spend the decay.  Cosine, by contrast, bakes the
    total step count into the schedule.
    """
    decay_steps = max(1, int(total * decay_frac))
    stable_end = total - decay_steps
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step < stable_end:
        return base_lr
    frac = (step - stable_end) / decay_steps
    return base_lr * (1.0 - frac * (1.0 - min_ratio))
