"""Quantization: trading precision for memory bandwidth.

**Why it works at all.**  Decoding one token is memory-bound: you read every
weight and do two FLOPs with it.  So time-per-token is roughly
``bytes_of_weights / memory_bandwidth``, and halving the bytes nearly halves the
latency -- even though you spend extra ALU cycles dequantizing.  This is also
why quantization does almost nothing for *training* throughput, which is
compute-bound.

**The core operation.**  Map a float block to `b`-bit integers:

    scale = max|w| / q_max            (symmetric, "absmax")
    q     = round(w / scale)          in [-q_max, q_max]
    w_hat = q * scale

Everything else is a choice about *granularity* and *shape*:

* **Granularity.**  One scale for the whole tensor is cheap and bad -- a single
  outlier sets the scale and crushes everything else.  Per-output-channel
  (per-row) is nearly free and much better.  Per-group (e.g. 64 or 128 weights
  share a scale) is what int4 methods actually use.
* **Symmetric vs asymmetric.**  Symmetric has no zero-point, so the matmul
  stays a plain integer matmul.  Asymmetric fits skewed distributions better
  but adds a correction term.
* **Shape of the grid.**  Uniform spacing is wrong for weights, which are
  roughly Gaussian.  NF4 uses quantiles of a normal distribution, so its 16
  levels are dense where the mass is.  That is ~0.5 bits of free accuracy.

**Where the error goes.**  Weight quantization error is benign and roughly
additive noise.  *Activation* error is not: transformer activations have
systematic outlier channels (a handful of dimensions with 20x the magnitude),
which is why naive int8 activation quantization collapses and why LLM.int8()
splits those channels out into fp16.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Integer quantization primitives
# ---------------------------------------------------------------------------


def quantize_symmetric(w: torch.Tensor, bits: int = 8, dim: int | None = -1,
                       group_size: int | None = None):
    """Symmetric absmax quantization.  Returns (q, scale, meta).

    `dim=None`   -> one scale for the whole tensor (per-tensor)
    `dim=-1`     -> one scale per row (per-output-channel for an nn.Linear weight)
    `group_size` -> one scale per contiguous group of `group_size` elements
                    along the last dim (what int4 schemes use)
    """
    q_max = 2 ** (bits - 1) - 1
    orig_shape = w.shape
    if group_size:
        if w.shape[-1] % group_size != 0:
            raise ValueError(f"last dim {w.shape[-1]} not divisible by group_size {group_size}")
        wg = w.reshape(-1, group_size)
        scale = wg.abs().amax(dim=-1, keepdim=True) / q_max
        scale = scale.clamp_min(1e-12)
        q = torch.round(wg / scale).clamp(-q_max - 1, q_max)
        return q.reshape(orig_shape), scale, {"bits": bits, "group_size": group_size,
                                              "shape": tuple(orig_shape), "symmetric": True}
    if dim is None:
        scale = (w.abs().amax() / q_max).clamp_min(1e-12).reshape(1)
    else:
        scale = (w.abs().amax(dim=dim, keepdim=True) / q_max).clamp_min(1e-12)
    q = torch.round(w / scale).clamp(-q_max - 1, q_max)
    return q, scale, {"bits": bits, "group_size": None, "shape": tuple(orig_shape),
                      "symmetric": True}


def dequantize_symmetric(q: torch.Tensor, scale: torch.Tensor, meta: dict) -> torch.Tensor:
    if meta.get("group_size"):
        g = meta["group_size"]
        return (q.reshape(-1, g) * scale).reshape(meta["shape"])
    return q * scale


def quantize_affine(w: torch.Tensor, bits: int = 8, dim: int = -1,
                    group_size: int | None = None):
    """Asymmetric (affine) quantization with a zero-point: q = round(w/s) + z.

    Fits a distribution that is not centred on zero -- activations after ReLU,
    or weights of a layer with a strong bias.  The cost at inference is a
    correction term: ``(q - z) * s`` expands to an integer matmul plus a rank-1
    term ``z * sum(x)``, which is cheap but must not be forgotten.
    """
    levels = 2**bits - 1
    if group_size:
        wg = w.reshape(-1, group_size)
        w_min = wg.amin(dim=-1, keepdim=True)
        w_max = wg.amax(dim=-1, keepdim=True)
    else:
        wg = w
        w_min = w.amin(dim=dim, keepdim=True)
        w_max = w.amax(dim=dim, keepdim=True)
    scale = ((w_max - w_min) / levels).clamp_min(1e-12)
    zero = torch.round(-w_min / scale)
    q = torch.round(wg / scale) + zero
    q = q.clamp(0, levels)
    meta = {"bits": bits, "group_size": group_size, "shape": tuple(w.shape), "symmetric": False}
    return q.reshape(w.shape) if not group_size else q, scale, zero, meta


def dequantize_affine(q, scale, zero, meta):
    if meta.get("group_size"):
        g = meta["group_size"]
        return ((q.reshape(-1, g) - zero) * scale).reshape(meta["shape"])
    return (q - zero) * scale


# ---------------------------------------------------------------------------
# NF4 -- the QLoRA data type
# ---------------------------------------------------------------------------

# The 16 NF4 levels: quantiles of a standard normal, rescaled to [-1, 1] and
# forced to contain an exact zero (so that a zero weight stays exactly zero,
# which matters for masked/pruned weights).  These are the published constants.
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
])


def nf4_levels_from_normal(n: int = 16) -> torch.Tensor:
    """Derive the NF4 grid from first principles, to show where it comes from.

    Take the `n` "equal-information" points of a standard normal -- the inverse
    CDF at evenly spaced probabilities -- then normalise to [-1, 1].  The
    published table splits the negative and positive halves so that 0 is
    represented exactly; we reproduce that here, which is why the result matches
    `NF4_LEVELS` to ~1e-3 rather than exactly.
    """
    from math import erf, sqrt

    def ppf(p):  # inverse normal CDF by bisection on erf -- no scipy needed
        lo, hi = -10.0, 10.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if 0.5 * (1 + erf(mid / sqrt(2))) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    offset = 0.9677083  # the QLoRA paper's offset: 1 - 1/(2*15) style margin
    n_neg, n_pos = n // 2, n // 2 + 1
    neg = [ppf(p) for p in torch.linspace(1 - offset, 0.5, n_neg).tolist()]
    pos = [ppf(p) for p in torch.linspace(0.5, offset, n_pos).tolist()]
    vals = sorted(set(neg + pos))
    v = torch.tensor(vals)
    return v / v.abs().max()


def quantize_nf4(w: torch.Tensor, group_size: int = 64, levels: torch.Tensor | None = None):
    """Blockwise NF4: per-group absmax scale, then nearest of 16 normal levels.

    The group scale is stored in fp16/fp32.  At group_size=64 that is
    16 bits / 64 weights = 0.25 bits of overhead, so NF4 costs ~4.25 bits per
    weight.  QLoRA's "double quantization" then quantizes the scales themselves
    to save another 0.37 bits.
    """
    levels = NF4_LEVELS.to(w.device, torch.float32) if levels is None else levels.to(w.device)
    if w.numel() % group_size != 0:
        raise ValueError(f"numel {w.numel()} not divisible by group_size {group_size}")
    wg = w.reshape(-1, group_size).float()
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    normed = wg / scale                                        # now in [-1, 1]
    # Nearest level: one (G, group, 16) comparison.  Fine for offline packing.
    idx = (normed.unsqueeze(-1) - levels.view(1, 1, -1)).abs().argmin(dim=-1)
    meta = {"group_size": group_size, "shape": tuple(w.shape), "levels": levels}
    return idx.to(torch.uint8), scale, meta


def dequantize_nf4(idx: torch.Tensor, scale: torch.Tensor, meta: dict) -> torch.Tensor:
    levels = meta["levels"].to(idx.device)
    return (levels[idx.long()] * scale).reshape(meta["shape"])


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack values in [0, 15] two-per-byte.  The actual memory saving happens here.

    Everything above stores 4-bit values in int8/uint8 tensors, which is
    convenient but saves nothing.  Real kernels read packed bytes and unpack in
    registers; this function is what makes the on-disk/in-VRAM size honest.
    """
    if q.min() < 0 or q.max() > 15:
        raise ValueError("pack_int4 expects values in [0, 15]")
    flat = q.reshape(-1).to(torch.uint8)
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
    return (flat[0::2] << 4) | flat[1::2]


def unpack_int4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    hi = (packed >> 4) & 0xF
    lo = packed & 0xF
    return torch.stack([hi, lo], dim=1).reshape(-1)[:numel]


# ---------------------------------------------------------------------------
# Quantized linear layers
# ---------------------------------------------------------------------------


class QuantizedLinear(nn.Module):
    """Weight-only quantized `nn.Linear`: store ints, dequantize, matmul in fp.

    This is "fake quantization" in the sense that the matmul is still floating
    point -- but it is *not* a simulation: the weights genuinely occupy 4 or 8
    bits, which is the part that buys decode latency.  A true integer-matmul
    kernel additionally quantizes activations and is what W8A8 inference does.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 bits: int = 8, scheme: str = "symmetric", group_size: int | None = None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.bits, self.scheme, self.group_size = bits, scheme, group_size
        self.register_buffer("q", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features, 1))
        self.register_buffer("zero", torch.zeros(out_features, 1))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.meta: dict = {}

    @classmethod
    def from_linear(cls, lin: nn.Linear, bits: int = 8, scheme: str = "symmetric",
                    group_size: int | None = None) -> QuantizedLinear:
        out_f, in_f = lin.weight.shape
        m = cls(in_f, out_f, bias=lin.bias is not None, bits=bits, scheme=scheme,
                group_size=group_size)
        w = lin.weight.data.float()
        if scheme == "symmetric":
            q, scale, meta = quantize_symmetric(w, bits, dim=-1, group_size=group_size)
            m.q = q.to(torch.int8)
            m.scale = scale
            m.meta = meta
        elif scheme == "affine":
            q, scale, zero, meta = quantize_affine(w, bits, dim=-1, group_size=group_size)
            # uint8 range stored in int8 storage: shift by 128 so it fits.
            m.q = (q - 128).to(torch.int8)
            m.scale, m.zero, m.meta = scale, zero, meta
        elif scheme == "nf4":
            idx, scale, meta = quantize_nf4(w, group_size=group_size or 64)
            m.q = idx.to(torch.int8)
            m.scale, m.meta = scale, meta
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        if lin.bias is not None:
            m.bias.data = lin.bias.data.clone()
        return m

    def dequantized_weight(self) -> torch.Tensor:
        if self.scheme == "symmetric":
            return dequantize_symmetric(self.q.float(), self.scale, self.meta)
        if self.scheme == "affine":
            return dequantize_affine(self.q.float() + 128, self.scale, self.zero, self.meta)
        return dequantize_nf4(self.q.to(torch.uint8), self.scale, self.meta)

    def forward(self, x):
        w = self.dequantized_weight().to(x.dtype)
        return F.linear(x, w, self.bias)

    def memory_bytes(self) -> dict[str, float]:
        """Honest byte accounting, assuming int4 values are packed two per byte."""
        n = self.q.numel()
        eff_bits = 4 if (self.bits == 4 or self.scheme == "nf4") else self.bits
        weight_bytes = n * eff_bits / 8
        scale_bytes = self.scale.numel() * 4 + (self.zero.numel() * 4 if self.scheme == "affine" else 0)
        return {
            "weight_bytes": weight_bytes,
            "scale_bytes": float(scale_bytes),
            "total_bytes": weight_bytes + scale_bytes,
            "fp32_bytes": float(n * 4),
            "compression": n * 4 / (weight_bytes + scale_bytes),
            "effective_bits": (weight_bytes + scale_bytes) * 8 / n,
        }


def quantize_model(model: nn.Module, bits: int = 8, scheme: str = "symmetric",
                   group_size: int | None = None, skip: tuple[str, ...] = ("lm_head",)) -> nn.Module:
    """Replace every `nn.Linear` with a `QuantizedLinear`, in place.

    `skip` exists because not all layers are equal.  The lm_head and the
    embedding are the most quantization-sensitive parts of a small model: their
    output feeds directly into a softmax over the whole vocabulary, so the error
    is not averaged away by later layers.  Keeping them in fp16 costs a few
    percent of the memory and recovers most of the quality.
    """
    for name, child in list(model.named_children()):
        if any(s in name for s in skip):
            continue
        if isinstance(child, nn.Linear):
            setattr(model, name, QuantizedLinear.from_linear(child, bits, scheme, group_size))
        else:
            quantize_model(child, bits, scheme, group_size, skip)
    return model


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def quantization_error(w: torch.Tensor, bits: int = 8, scheme: str = "symmetric",
                       group_size: int | None = None) -> dict[str, float]:
    """Reconstruction error of one weight tensor under a given scheme."""
    w = w.float()
    if scheme == "symmetric":
        q, s, meta = quantize_symmetric(w, bits, dim=-1, group_size=group_size)
        wq = dequantize_symmetric(q, s, meta)
    elif scheme == "affine":
        q, s, z, meta = quantize_affine(w, bits, dim=-1, group_size=group_size)
        wq = dequantize_affine(q, s, z, meta)
    elif scheme == "per_tensor":
        q, s, meta = quantize_symmetric(w, bits, dim=None)
        wq = dequantize_symmetric(q, s, meta)
    elif scheme == "nf4":
        idx, s, meta = quantize_nf4(w, group_size=group_size or 64)
        wq = dequantize_nf4(idx, s, meta)
    else:
        raise ValueError(scheme)
    err = wq - w
    return {
        "rel_l2": float(err.norm() / w.norm()),
        "max_abs": float(err.abs().max()),
        "snr_db": float(20 * math.log10(w.norm() / err.norm().clamp_min(1e-20))),
    }


@torch.no_grad()
def kv_cache_quant_error(k: torch.Tensor, bits: int = 8, per_token: bool = True) -> dict[str, float]:
    """Quantization error for a KV cache, per-token vs per-channel.

    The KV cache is often bigger than the weights at long context, and it
    quantizes well -- but *only* along the right axis.  Keys have persistent
    outlier channels (the same dimensions are large for every token), so a
    per-token scale is dominated by them.  Per-channel scales fix it, and that
    asymmetry (per-channel for K, per-token for V) is what production KV
    quantization actually does.
    """
    dim = -1 if per_token else -2
    q, s, meta = quantize_symmetric(k.float(), bits, dim=dim)
    kq = dequantize_symmetric(q, s, meta)
    err = kq - k.float()
    return {"rel_l2": float(err.norm() / k.float().norm()),
            "axis": "per_token" if per_token else "per_channel"}
