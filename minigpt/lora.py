"""LoRA and QLoRA: fine-tuning without touching the base weights.

**The observation.**  A fine-tune moves the weights by a small amount, and that
update turns out to be close to low-rank.  So instead of learning
``W + Delta W`` with ``Delta W`` full rank, learn

    W' = W + (alpha / r) * B @ A,      A: (r, in),  B: (out, r),  r << min(in, out)

and train only A and B.  For a 4096x4096 matrix at r=8 that is 65k trainable
parameters instead of 16.7M -- a 256x reduction.

**Why it saves so much more memory than it saves parameters.**  The weights
were never the problem: Adam keeps *two* fp32 moments per trainable parameter,
so optimizer state alone is 8 bytes per trainable weight.  Freezing the base
removes that, and it also removes the need to store gradients for it.  What you
still pay is activations, because backprop through the frozen ``W`` still needs
them -- which is why gradient checkpointing and LoRA are complementary, not
alternatives.

**The two initialisation rules that make it work.**
* ``B = 0`` so the adapter is exactly zero at step 0.  The fine-tune therefore
  starts from the base model's behaviour rather than from noise.
* ``A ~ N(0, 1/r)`` (or Kaiming) so that once B moves, the update has sensible
  scale.  If both were zero the product's gradient would be identically zero
  and nothing would ever train.

**What ``alpha`` is for.**  The update is scaled by ``alpha/r``, so raising r
does not change the initial update magnitude.  That means you can sweep r
without re-tuning the LR -- the reason the convention exists. In practice people
set ``alpha = 2r`` or ``alpha = r``.

**QLoRA** = this, on top of an NF4-quantized frozen base.  The base is 4-bit
(so a 7B model fits in ~4 GB), the adapters are bf16, and gradients flow
*through* the dequantized weights to A and B.  The base is never updated, so its
quantization error is a fixed bias rather than something that compounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import QuantizedLinear


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` (or `QuantizedLinear`) with a trainable low-rank update."""

    def __init__(self, base: nn.Module, r: int = 8, alpha: float | None = None,
                 dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("rank must be positive")
        if isinstance(base, QuantizedLinear):
            in_f, out_f = base.in_features, base.out_features
        else:
            out_f, in_f = base.weight.shape
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = r
        self.alpha = alpha if alpha is not None else 2.0 * r
        self.scaling = self.alpha / r
        # Dropout on the *input to the adapter only* -- the base path stays
        # deterministic, which is what LoRA's regularisation actually is.
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        device = (base.q.device if isinstance(base, QuantizedLinear) else base.weight.device)
        dtype = (torch.float32 if isinstance(base, QuantizedLinear) else base.weight.dtype)
        self.lora_A = nn.Parameter(torch.empty(r, in_f, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=device, dtype=dtype))
        # Kaiming-uniform on A, zeros on B: adapter output is exactly 0 at init.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if self.merged:
            return out
        # (x @ A^T) @ B^T -- two skinny matmuls, never forming the (out, in) product.
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return out + self.scaling * delta

    @torch.no_grad()
    def merge(self):
        """Fold `scaling * B @ A` into the base weight: zero inference overhead.

        Only possible for an unquantized base.  Merging into a 4-bit base would
        require re-quantizing, which changes the result -- so QLoRA adapters
        stay separate at inference (and that is why serving many LoRAs off one
        base model is cheap).
        """
        if self.merged:
            return self
        if isinstance(self.base, QuantizedLinear):
            raise RuntimeError("cannot merge into a quantized base without re-quantizing")
        self.base.weight.data += self.scaling * (self.lora_B @ self.lora_A)
        self.merged = True
        return self

    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return self
        self.base.weight.data -= self.scaling * (self.lora_B @ self.lora_A)
        self.merged = False
        return self

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}, merged={self.merged}"


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float | None = None
    dropout: float = 0.0
    # Which projections to adapt.  The original paper used only q_proj/v_proj;
    # later work (and QLoRA) found adapting *every* linear layer is better at
    # equal trainable-parameter budget, so that is the default here.
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down")
    train_norms: bool = False   # unfreezing norm gains is cheap and often helps


def apply_lora(model: nn.Module, cfg: LoRAConfig | None = None) -> nn.Module:
    """Freeze `model`, then wrap every targeted Linear with a LoRA adapter."""
    cfg = cfg or LoRAConfig()
    for p in model.parameters():
        p.requires_grad_(False)

    def recurse(module: nn.Module):
        for name, child in list(module.named_children()):
            if name in cfg.targets and isinstance(child, (nn.Linear, QuantizedLinear)):
                setattr(module, name, LoRALinear(child, cfg.r, cfg.alpha, cfg.dropout))
            else:
                recurse(child)

    recurse(model)

    if cfg.train_norms:
        for name, p in model.named_parameters():
            if "norm" in name:
                p.requires_grad_(True)
    return model


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if p.requires_grad]


def lora_state_dict(model: nn.Module) -> dict:
    """Only the adapter tensors -- a few MB instead of a few GB per fine-tune."""
    return {n: p.detach().cpu() for n, p in model.named_parameters()
            if "lora_" in n or (p.requires_grad and "norm" in n)}


def merge_all(model: nn.Module) -> nn.Module:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()
    return model


def unmerge_all(model: nn.Module) -> nn.Module:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()
    return model


def trainable_summary(model: nn.Module) -> dict[str, float]:
    """Trainable vs total weights, and the optimizer-state saving it implies.

    Quantized weights live in *buffers*, not parameters, so counting only
    `model.parameters()` would report a QLoRA model as ~90% trainable.  We count
    the quantized weight buffers as part of the total, and report bytes as well
    as counts -- with a 4-bit base the byte ratio is the number that matters.
    """
    total = sum(p.numel() for p in model.parameters())
    base_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            total += m.q.numel()
            mb = m.memory_bytes()
            base_bytes += mb["total_bytes"]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": float(total),
        "trainable": float(trainable),
        "trainable_pct": 100.0 * trainable / max(1, total),
        "weight_bytes": float(base_bytes),
        # AdamW: 2 fp32 moments + 1 fp32 gradient per trainable parameter.
        "adam_state_mb_full": total * 12 / 1e6,
        "adam_state_mb_lora": trainable * 12 / 1e6,
    }
