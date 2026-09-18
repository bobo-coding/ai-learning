"""Attention, from the textbook formula up to a working FlashAttention.

Everything here uses the tensor layout ``(B, H, T, D)``:
  B = batch, H = number of heads, T = sequence length, D = head dimension.

Five implementations of the same function, in increasing sophistication:

1. `attention_reference`  -- one query at a time, Python loops.  Unmistakably
   the definition; O(T^2) memory and very slow.  This is the oracle every other
   implementation is tested against.
2. `attention_batched`    -- the usual `softmax(QK^T/sqrt(D) + mask) V`.
3. `online_softmax`       -- the streaming softmax trick, in isolation.
4. `flash_attention`      -- tiled forward *and* backward built on the online
   softmax, never materialising the T x T score matrix.
5. `F.scaled_dot_product_attention` -- PyTorch's fused kernel, for timing.

Plus the two positional schemes you actually meet in modern models: learned
absolute embeddings (in `model.py`) and RoPE (here).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. The definition
# ---------------------------------------------------------------------------


def attention_reference(q, k, v, causal: bool = True) -> torch.Tensor:
    """Attention computed one (batch, head, query) at a time.

    For query position i, attention returns a convex combination of the value
    vectors, with weights given by a softmax over scaled dot products:

        a_ij = exp(q_i . k_j / sqrt(D)) / sum_{j'} exp(q_i . k_{j'} / sqrt(D))
        o_i  = sum_j a_ij v_j

    With `causal=True` the sum runs only over j <= i, so position i cannot see
    the future.  This is what makes a decoder trainable on all T positions at
    once: one forward pass gives T independent next-token predictions.
    """
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros_like(q)
    for b in range(B):
        for h in range(H):
            for i in range(T):
                hi = i + 1 if causal else T
                scores = (q[b, h, i] @ k[b, h, :hi].transpose(-1, -2)) * scale  # (hi,)
                weights = torch.softmax(scores, dim=-1)
                out[b, h, i] = weights @ v[b, h, :hi]
    return out


# ---------------------------------------------------------------------------
# 2. The batched form everyone writes
# ---------------------------------------------------------------------------


def causal_mask(T: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Additive mask of shape (1, 1, T, T): 0 where allowed, -inf where not."""
    m = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
    return torch.triu(m, diagonal=1).view(1, 1, T, T)


def attention_batched(q, k, v, causal: bool = True, dropout_p: float = 0.0, training: bool = False):
    """`softmax(QK^T / sqrt(D) + mask) @ V` as three batched matmuls.

    Note the two places the scale factor could go.  We divide the *scores*, not
    Q, which is numerically identical here but matters once you write kernels:
    scaling Q once is D*T multiplies, scaling S is T^2.

    Memory is the problem: the score matrix S is (B, H, T, T).  At B=8, H=12,
    T=4096 in fp32 that is 6.4 GB -- for a single layer.  That is the entire
    motivation for FlashAttention.
    """
    D = q.size(-1)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(D)
    if causal:
        Tq, Tk = q.size(-2), k.size(-2)
        # Offset handles the incremental-decoding case Tq < Tk: query i of this
        # chunk is really at absolute position i + (Tk - Tq).
        mask = torch.ones(Tq, Tk, device=q.device, dtype=torch.bool).tril(diagonal=Tk - Tq)
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0 and training:
        attn = F.dropout(attn, p=dropout_p)
    return attn @ v


# ---------------------------------------------------------------------------
# 3. Online softmax
# ---------------------------------------------------------------------------


def online_softmax(x: torch.Tensor, block: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Streaming softmax over the last dim, processed `block` elements at a time.

    The naive stable softmax needs three passes over x: max, sum of exp, divide.
    Online softmax does it in one pass by carrying a running max `m` and a
    running denominator `l`, and *rescaling* `l` whenever the max grows:

        m_new = max(m, max(x_block))
        l_new = l * exp(m - m_new) + sum(exp(x_block - m_new))

    The rescale factor exp(m - m_new) is the whole idea, and it is exactly what
    lets FlashAttention keep an output accumulator instead of a score matrix.

    Returns (softmax(x), logsumexp(x)).
    """
    *lead, N = x.shape
    m = torch.full(lead, float("-inf"), device=x.device, dtype=x.dtype)
    l = torch.zeros(lead, device=x.device, dtype=x.dtype)
    for s in range(0, N, block):
        xb = x[..., s : s + block]
        m_new = torch.maximum(m, xb.amax(dim=-1))
        l = l * torch.exp(m - m_new) + torch.exp(xb - m_new.unsqueeze(-1)).sum(dim=-1)
        m = m_new
    lse = m + torch.log(l)
    return torch.exp(x - lse.unsqueeze(-1)), lse


# ---------------------------------------------------------------------------
# 4. FlashAttention: tiled forward and backward
# ---------------------------------------------------------------------------


class _FlashAttention(torch.autograd.Function):
    """FlashAttention in pure PyTorch: same tiling a CUDA/Triton kernel uses.

    The forward never materialises S = QK^T.  It walks key/value tiles and keeps
    three small accumulators per query tile: the running max `m`, the running
    denominator `l`, and the unnormalised output `acc`.

    The backward is the part people skip, and it is where the memory saving is
    actually paid for.  We saved only `lse` (one float per query), so S and P
    must be *recomputed* tile by tile.  The softmax Jacobian collapses to

        dS = P * (dP - rowsum(dO * O))

    where `rowsum(dO * O)` is the same for every key tile, so it is computed
    once up front as `delta`.  Recomputation costs one extra QK^T; it buys an
    O(T) instead of O(T^2) memory footprint, which is why every trained
    transformer since 2022 uses it.
    """

    @staticmethod
    def forward(ctx, q, k, v, causal, q_block, kv_block):
        B, H, Tq, D = q.shape
        Tk = k.shape[2]
        scale = 1.0 / math.sqrt(D)
        # Accumulate in fp32 even if inputs are bf16 -- this is what real kernels
        # do, because the running sum `l` and the accumulator lose too much to
        # rounding in 16 bits.  fp64 inputs keep fp64 so gradcheck can be exact.
        acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        out = torch.empty((B, H, Tq, D), device=q.device, dtype=q.dtype)
        lse = torch.empty((B, H, Tq), device=q.device, dtype=acc_dtype)

        for i0 in range(0, Tq, q_block):
            i1 = min(i0 + q_block, Tq)
            qi = q[:, :, i0:i1].to(acc_dtype) * scale                 # (B,H,bq,D)
            m = torch.full((B, H, i1 - i0), float("-inf"), device=q.device, dtype=acc_dtype)
            l = torch.zeros((B, H, i1 - i0), device=q.device, dtype=acc_dtype)
            accum = torch.zeros((B, H, i1 - i0, D), device=q.device, dtype=acc_dtype)

            # Under causal masking, query block [i0,i1) can only see keys up to
            # i1-1 + (Tk-Tq); every later key tile is entirely masked, so we
            # simply never load it.  That is the 2x saving of causal flash attn.
            kv_limit = Tk if not causal else min(Tk, i1 + (Tk - Tq))
            for j0 in range(0, kv_limit, kv_block):
                j1 = min(j0 + kv_block, kv_limit)
                kj = k[:, :, j0:j1].to(acc_dtype)
                vj = v[:, :, j0:j1].to(acc_dtype)
                s = qi @ kj.transpose(-1, -2)                          # (B,H,bq,bk)
                if causal:
                    qpos = torch.arange(i0, i1, device=q.device) + (Tk - Tq)
                    kpos = torch.arange(j0, j1, device=q.device)
                    s = s.masked_fill(kpos[None, :] > qpos[:, None], float("-inf"))
                m_new = torch.maximum(m, s.amax(dim=-1))
                # A fully-masked row keeps m = -inf; guard so exp() sees 0, not nan.
                m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
                p = torch.exp(s - m_safe.unsqueeze(-1))
                rescale = torch.exp(m - m_safe)
                rescale = torch.where(torch.isinf(m), torch.zeros_like(rescale), rescale)
                l = l * rescale + p.sum(dim=-1)
                accum = accum * rescale.unsqueeze(-1) + p @ vj
                m = m_new

            out[:, :, i0:i1] = (accum / l.clamp_min(1e-20).unsqueeze(-1)).to(q.dtype)
            lse[:, :, i0:i1] = m + torch.log(l.clamp_min(1e-20))

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal, ctx.q_block, ctx.kv_block = causal, q_block, kv_block
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        causal, q_block, kv_block = ctx.causal, ctx.q_block, ctx.kv_block
        B, H, Tq, D = q.shape
        Tk = k.shape[2]
        scale = 1.0 / math.sqrt(D)
        f32 = torch.float64 if q.dtype == torch.float64 else torch.float32

        dq = torch.zeros_like(q, dtype=f32)
        dk = torch.zeros_like(k, dtype=f32)
        dv = torch.zeros_like(v, dtype=f32)
        dout32 = dout.to(f32)
        # delta_i = sum_d dO_id * O_id -- the "row dot" of the softmax Jacobian.
        delta = (dout32 * out.to(f32)).sum(dim=-1)                      # (B,H,Tq)

        for i0 in range(0, Tq, q_block):
            i1 = min(i0 + q_block, Tq)
            qi = q[:, :, i0:i1].to(f32)
            doi = dout32[:, :, i0:i1]
            lse_i = lse[:, :, i0:i1].unsqueeze(-1)
            del_i = delta[:, :, i0:i1].unsqueeze(-1)
            kv_limit = Tk if not causal else min(Tk, i1 + (Tk - Tq))
            for j0 in range(0, kv_limit, kv_block):
                j1 = min(j0 + kv_block, kv_limit)
                kj = k[:, :, j0:j1].to(f32)
                vj = v[:, :, j0:j1].to(f32)
                s = (qi @ kj.transpose(-1, -2)) * scale
                if causal:
                    qpos = torch.arange(i0, i1, device=q.device) + (Tk - Tq)
                    kpos = torch.arange(j0, j1, device=q.device)
                    s = s.masked_fill(kpos[None, :] > qpos[:, None], float("-inf"))
                p = torch.exp(s - lse_i)                                # recomputed P
                dv[:, :, j0:j1] += p.transpose(-1, -2) @ doi
                dp = doi @ vj.transpose(-1, -2)
                ds = p * (dp - del_i) * scale
                dq[:, :, i0:i1] += ds @ kj
                dk[:, :, j0:j1] += ds.transpose(-1, -2) @ qi

        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


def flash_attention(q, k, v, causal: bool = True, q_block: int = 64, kv_block: int = 64):
    """Differentiable tiled attention.  Identical output to `attention_batched`."""
    return _FlashAttention.apply(q, k, v, causal, q_block, kv_block)


# ---------------------------------------------------------------------------
# Rotary position embeddings (RoPE)
# ---------------------------------------------------------------------------


def rope_tables(head_dim: int, max_pos: int, base: float = 10000.0, device=None, dtype=torch.float32):
    """Precompute (cos, sin) tables of shape (max_pos, head_dim // 2).

    RoPE treats each consecutive pair of channels as a point in the plane and
    rotates it by an angle proportional to the token's position:

        theta_i(pos) = pos * base ** (-2i / head_dim),  i = 0 .. D/2 - 1

    Channel pair 0 rotates once per token; the last pair rotates once per
    ~`base` tokens.  So the pairs form a multi-resolution positional code, much
    like the binary digits of `pos`.

    The point of doing it this way: the dot product of two rotated vectors
    depends on their positions only through the *difference* m - n, so the model
    gets relative position for free and can be evaluated at positions it never
    saw during training (that is what "RoPE scaling" then extrapolates).
    """
    if head_dim % 2 != 0:
        raise ValueError("RoPE needs an even head_dim")
    half = head_dim // 2
    # Build the tables in fp32 or better even for a bf16 model: `pos * inv_freq`
    # for the slowest channel at pos=100k needs more mantissa than bf16 has, and
    # a wrong angle is a silent quality bug.
    inv_freq = base ** (-torch.arange(0, half, device=device, dtype=dtype) * 2.0 / head_dim)
    pos = torch.arange(max_pos, device=device, dtype=dtype)
    angles = torch.outer(pos, inv_freq)                                 # (max_pos, half)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0):
    """Rotate `x` of shape (B, H, T, D) using the interleaved convention.

    Interleaved (RoFormer/Llama reference): channel pairs are (0,1), (2,3), ...
    The other convention you will see (HuggingFace `rotate_half`) pairs channel
    i with i + D/2.  The two are related by a fixed permutation of the Q/K
    output columns, so a model is only compatible with the one it was trained
    with -- mixing them silently degrades quality instead of erroring.

    `offset` is the absolute position of x[..., 0, :]; during incremental
    decoding you pass the number of tokens already in the KV cache.
    """
    B, H, T, D = x.shape
    c = cos[offset : offset + T].to(x.dtype).view(1, 1, T, D // 2)
    s = sin[offset : offset + T].to(x.dtype).view(1, 1, T, D // 2)
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * c - x_odd * s
    out[..., 1::2] = x_even * s + x_odd * c
    return out


def apply_rope_complex(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0):
    """Reference RoPE via complex multiplication -- used to test `apply_rope`."""
    B, H, T, D = x.shape
    dt = torch.float64 if x.dtype == torch.float64 else torch.float32
    # Pair up channels (0,1), (2,3), ... as complex numbers; RoPE is then a
    # single complex multiply by e^{i theta}, which is all `apply_rope` unrolls.
    xc = torch.view_as_complex(x.to(dt).reshape(B, H, T, D // 2, 2).contiguous())
    rot = torch.complex(
        cos[offset : offset + T].to(dt), sin[offset : offset + T].to(dt)
    ).view(1, 1, T, D // 2)
    return torch.view_as_real(xc * rot).reshape(B, H, T, D).to(x.dtype)


# ---------------------------------------------------------------------------
# The module used by the model
# ---------------------------------------------------------------------------


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand (B, H_kv, T, D) to (B, H_kv * n_rep, T, D) for grouped-query attention.

    GQA gives each group of `n_rep` query heads one shared K/V head.  Quality
    loss is small; the KV cache -- which is what actually limits batch size at
    inference -- shrinks by `n_rep`.  n_rep = H is multi-query attention (MQA).
    """
    if n_rep == 1:
        return x
    B, Hkv, T, D = x.shape
    return x[:, :, None].expand(B, Hkv, n_rep, T, D).reshape(B, Hkv * n_rep, T, D)


class KVCache:
    """Per-layer key/value cache for incremental decoding.

    Preallocating to `max_seq_len` and writing in place keeps generation free of
    reallocation.  Size is
        2 * n_layer * n_kv_head * head_dim * max_seq_len * batch * bytes,
    which for a 7B model at 4k context in fp16 is ~2 GB per sequence.  That
    number, not the weights, is why GQA and KV-cache quantization exist.
    """

    def __init__(self, batch, n_kv_head, max_seq_len, head_dim, device, dtype=torch.float32):
        shape = (batch, n_kv_head, max_seq_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0

    def append(self, k, v):
        """Write new keys/values and return views of the whole cache so far."""
        T = k.shape[2]
        if self.length + T > self.k.shape[2]:
            raise ValueError("KV cache overflow: increase max_seq_len")
        self.k[:, :, self.length : self.length + T] = k
        self.v[:, :, self.length : self.length + T] = v
        self.length += T
        return self.k[:, :, : self.length], self.v[:, :, : self.length]

    def reset(self):
        self.length = 0


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with GQA, RoPE and an optional KV cache.

    `impl` selects the kernel, which is the knob you use when you want to check
    that your hand-written version matches the fused one:
      "sdpa"    -- F.scaled_dot_product_attention (fused; the default)
      "math"    -- explicit batched matmuls
      "flash"   -- the tiled implementation in this file
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_kv_head: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
        rope: bool = True,
        impl: str = "sdpa",
    ):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        if n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        self.n_rep = n_head // self.n_kv_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        self.rope = rope
        self.impl = impl

        # One fused projection for Q, K, V: one matmul instead of three.  With
        # GQA the three slices have different widths.
        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_head * self.head_dim, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, cos=None, sin=None, cache: KVCache | None = None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        offset = cache.length if cache is not None else 0
        if self.rope:
            if cos is None or sin is None:
                raise ValueError("rope=True requires cos/sin tables")
            q = apply_rope(q, cos, sin, offset=offset)
            k = apply_rope(k, cos, sin, offset=offset)

        if cache is not None:
            k, v = cache.append(k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # With a cache, T queries attend to `offset + T` keys; the causal mask
        # must be shifted accordingly.  Tq == 1 in the common decode step, where
        # every key is visible and no mask is needed at all.
        is_causal = k.shape[2] > 1 and q.shape[2] > 1
        if self.impl == "sdpa":
            if q.shape[2] == k.shape[2]:
                y = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
                )
            else:
                # Tq != Tk: build the shifted mask explicitly, since is_causal
                # would align the mask to the top-left corner instead.
                Tq, Tk = q.shape[2], k.shape[2]
                m = torch.ones(Tq, Tk, device=x.device, dtype=torch.bool).tril(diagonal=Tk - Tq)
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=m, dropout_p=self.dropout if self.training else 0.0
                )
        elif self.impl == "math":
            y = attention_batched(q, k, v, causal=True, dropout_p=self.dropout, training=self.training)
        elif self.impl == "flash":
            y = flash_attention(q, k, v, causal=True)
        else:
            raise ValueError(f"unknown impl {self.impl!r}")

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.resid_dropout(self.o_proj(y))
