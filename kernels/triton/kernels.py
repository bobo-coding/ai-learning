"""Triton kernels: the same six problems, at a much higher level of abstraction.

Triton's bargain: you write code over **blocks** and the compiler handles what
you spent the CUDA lesson doing by hand -- assigning work to threads, staging
tiles in shared memory, choosing vector widths, software-pipelining the loads.
You still own the things that need a human: the tiling *strategy*, the memory
access *pattern*, and every mask.

Concretely, compared to `kernels/cuda/kernels.cu`:

    CUDA, by hand                        Triton
    ---------------------------------    ---------------------------------
    threadIdx / per-thread scalars       blocks; no thread index exists
    __shared__ tile + 2 __syncthreads    implicit: `tl.load` of a block
    manual bounds checks per thread      `mask=` on load/store
    __shfl_down_sync reduction tree      `tl.sum(x, axis=1)`
    hand-tuned TILE, unrolling           `BLOCK: tl.constexpr` + autotune
    one output element per thread         a whole output tile per program

The result is typically 80-95% of a hand-written CUDA kernel's performance for
a fraction of the code -- and for fused ops like softmax and layernorm it beats
what most people write by hand, because the compiler's pipelining is better
than theirs.

This file imports real Triton when available and falls back to `tritonsim`
(a pure-PyTorch interpreter) otherwise, so it runs on a Mac.
"""

from __future__ import annotations

try:  # pragma: no cover - depends on the machine
    import triton
    import triton.language as tl

    HAVE_REAL_TRITON = True
except ImportError:  # pragma: no cover
    import tritonsim as triton
    import tritonsim.language as tl

    HAVE_REAL_TRITON = False

import torch


# ---------------------------------------------------------------------------
# 1. Elementwise add -- the whole programming model in 6 lines
# ---------------------------------------------------------------------------


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # One program instance handles BLOCK contiguous elements.  There is no
    # thread index: `offs` is a *block* of BLOCK offsets, and every operation
    # below is on the whole block at once.
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # The mask is not optional.  n_elements is almost never a multiple of BLOCK,
    # so the last program's block runs off the end of the tensor.
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block: int = 1024) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, block),)
    add_kernel[grid](x.contiguous(), y.contiguous(), out, n, BLOCK=block)
    return out


# ---------------------------------------------------------------------------
# 2. Fused softmax -- one program per row, the whole row in registers
# ---------------------------------------------------------------------------


@triton.jit
def softmax_kernel(x_ptr, out_ptr, row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    # `other=-inf` is what makes the masked lanes vanish from the max and
    # contribute exp(-inf) = 0 to the sum.  With other=0 (the default) a row
    # shorter than BLOCK would get spurious mass at every padded position --
    # a bug that produces plausible-looking wrong probabilities.
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    out = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * row_stride + cols, out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    x2 = x.reshape(-1, x.shape[-1])
    rows, cols = x2.shape
    out = torch.empty_like(x2)
    # BLOCK must cover the whole row: this kernel does no looping, so the row
    # has to fit in registers.  Past ~64k columns you need a two-pass or online
    # variant -- which is exactly the flash-attention algorithm applied to a row.
    block = triton.next_power_of_2(cols) if hasattr(triton, "next_power_of_2") else 1 << (cols - 1).bit_length()
    softmax_kernel[(rows,)](x2, out, x2.stride(0), cols, BLOCK=block)
    return out.reshape(x.shape)


# ---------------------------------------------------------------------------
# 3. Fused RMSNorm, forward and backward
# ---------------------------------------------------------------------------


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, out_ptr, rstd_ptr, row_stride, n_cols,
                       eps, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Saving rstd (one float per row) is the classic forward/backward trade:
    # recomputing it in the backward would cost another full read of x.
    tl.store(rstd_ptr + row, rstd)
    tl.store(out_ptr + row * row_stride + cols, x * rstd * w, mask=mask)


@triton.jit
def rmsnorm_bwd_kernel(dout_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_ptr,
                       row_stride, n_cols, BLOCK: tl.constexpr):
    """dx for one row, and a partial dw accumulated atomically.

    The math: with r = rstd(x) = (mean(x^2) + eps)^(-1/2) and y = x*r*w,

        dx = r * (dout*w  -  x * r^2 * mean(dout*w*x))

    The second term is the correction for r depending on every element of x --
    dropping it is the most common hand-written-norm-backward bug, and it
    produces gradients that are *nearly* right, so tests with loose tolerances
    miss it.

    dw sums over rows, so every program instance adds into the same dw vector:
    `tl.atomic_add` is mandatory. (Note: the tritonsim interpreter runs programs
    serially, so it would NOT catch a missing atomic -- this is the one bug
    class you have to reason about rather than test for.)
    """
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    base = row * row_stride + cols
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    dout = tl.load(dout_ptr + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    xhat = x * rstd
    wdy = dout * w
    mean_wdy_xhat = tl.sum(wdy * xhat, axis=0) / n_cols
    dx = (wdy - xhat * mean_wdy_xhat) * rstd
    tl.store(dx_ptr + base, dx, mask=mask)
    tl.atomic_add(dw_ptr + cols, dout * xhat, mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    x = x.contiguous()
    x2 = x.reshape(-1, x.shape[-1])
    rows, cols = x2.shape
    out = torch.empty_like(x2)
    rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
    block = 1 << (cols - 1).bit_length()
    rmsnorm_fwd_kernel[(rows,)](x2, weight.contiguous(), out, rstd,
                                x2.stride(0), cols, eps, BLOCK=block)
    return out.reshape(x.shape), rstd


def rmsnorm_backward(dout: torch.Tensor, x: torch.Tensor, weight: torch.Tensor,
                     rstd: torch.Tensor):
    x2 = x.contiguous().reshape(-1, x.shape[-1])
    d2 = dout.contiguous().reshape(-1, x.shape[-1])
    rows, cols = x2.shape
    dx = torch.empty_like(x2)
    dw = torch.zeros(cols, device=x.device, dtype=torch.float32)
    block = 1 << (cols - 1).bit_length()
    rmsnorm_bwd_kernel[(rows,)](d2, x2, weight.contiguous(), rstd, dx, dw,
                                x2.stride(0), cols, BLOCK=block)
    return dx.reshape(x.shape), dw


# ---------------------------------------------------------------------------
# 4. Matmul -- blocked, with an L2-friendly program ordering
# ---------------------------------------------------------------------------


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr):
    # --- program ID -> output tile, reordered for L2 reuse -----------------
    # The naive `pid_m = pid // grid_n` walks a whole row of C before moving
    # down, so consecutive programs share a row of A but sweep *all* of B --
    # and B falls out of L2 before the next row reuses it.  Grouping GROUP_M
    # rows into a "super-row" traversed column-major means the programs running
    # concurrently touch a GROUP_M x BLOCK_N corner of A and B, which fits.
    # Worth 10-20% on large matmuls, for pure index arithmetic.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 2-D blocks of pointers, built by broadcasting.  This is Triton's whole
    # trick: `offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak` is a
    # (BLOCK_M, BLOCK_K) array of addresses.
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # The accumulator stays fp32 even for fp16 inputs.  Accumulating in fp16
    # over a K=4096 reduction loses ~3 bits and is a real source of error.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < N),
                    other=0.0)
        # Zero-padding the out-of-range lanes (rather than masking the dot) is
        # what lets a single `tl.dot` handle the ragged last tile: 0 * b = 0.
        acc += tl.dot(a, b)
        a_ptrs = a_ptrs + BLOCK_K * stride_ak
        b_ptrs = b_ptrs + BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor, block_m: int = 64, block_n: int = 64,
           block_k: int = 32, group_m: int = 8) -> torch.Tensor:
    a, b = a.contiguous(), b.contiguous()
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} @ {tuple(b.shape)}")
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
    )
    return c


# ---------------------------------------------------------------------------
# 5. FlashAttention forward -- tiled over queries AND keys
# ---------------------------------------------------------------------------


@triton.jit
def flash_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
                     stride_qb, stride_qt, stride_qd,
                     T, D, sm_scale, CAUSAL: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    """One program per (batch*head, query tile).  Streams K/V tiles past it.

    This is the real FlashAttention shape, unlike the one-query-per-block Metal
    and CUDA versions: because BLOCK_M queries share each loaded K/V tile, the
    inner loop is two `tl.dot`s on tensor cores instead of a reduction, and the
    kernel becomes compute-bound.  That is the entire performance story.

    We also write out `lse` (the log-sum-exp per query), which is exactly what
    the backward pass needs to recompute P without ever storing the T x T
    matrix -- see minigpt.attention._FlashAttention.backward for that math.
    """
    pid_m = tl.program_id(axis=0)
    bh = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    m_mask = offs_m < T

    q_base = bh * stride_qb
    q = tl.load(q_ptr + q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    # Running online-softmax state, one entry per query in the tile.
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Under causal masking, queries in this tile can see at most
    # (pid_m+1)*BLOCK_M keys, so later tiles are skipped entirely -- the 2x
    # saving that makes causal flash attention cheaper, not just smaller.
    hi = T
    if CAUSAL:
        hi = tl.minimum(T, (pid_m + 1) * BLOCK_M)

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < T
        k = tl.load(k_ptr + q_base + offs_n[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                    mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        v = tl.load(v_ptr + q_base + offs_n[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                    mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        keep = n_mask[None, :]
        if CAUSAL:
            keep = keep & (offs_n[None, :] <= offs_m[:, None])
        s = tl.where(keep, s, -float("inf"))

        # --- the online softmax update ---
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # A fully-masked row keeps m = -inf; guard it so exp() sees 0, not nan.
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m_safe))
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    tl.store(o_ptr + q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
             out, mask=m_mask[:, None] & d_mask[None, :])
    m_store = tl.where(m_i == -float("inf"), 0.0, m_i)
    tl.store(lse_ptr + bh * T + offs_m, m_store + tl.log(l_safe), mask=m_mask)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
                    block_m: int = 32, block_n: int = 32):
    """Attention over (B, H, T, D) tensors.  Returns (out, logsumexp)."""
    import math

    B, H, T, D = q.shape
    q, k, v = (t.contiguous().reshape(B * H, T, D) for t in (q, k, v))
    out = torch.empty_like(q)
    lse = torch.empty((B * H, T), device=q.device, dtype=torch.float32)
    block_d = 1 << (D - 1).bit_length()
    grid = (triton.cdiv(T, block_m), B * H)
    flash_fwd_kernel[grid](
        q, k, v, out, lse,
        q.stride(0), q.stride(1), q.stride(2),
        T, D, 1.0 / math.sqrt(D), causal,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d,
    )
    return out.reshape(B, H, T, D), lse.reshape(B, H, T)


# ---------------------------------------------------------------------------
# 6. Fused cross-entropy -- the one that actually saves an LLM fine-tune
# ---------------------------------------------------------------------------


@triton.jit
def cross_entropy_kernel(logits_ptr, labels_ptr, loss_ptr, dlogits_ptr,
                         n_cols, row_stride, IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    """Loss and gradient for one row, in one pass, without materialising softmax.

    Why this kernel matters more than the others: the logits tensor is
    (batch, seq, vocab).  At vocab=128k, batch*seq=8k that is 4 GB in fp32 --
    and eager PyTorch allocates it *again* for log_softmax and *again* for the
    gradient.  Computing the gradient in place, right here, is often the
    difference between a fine-tune that fits and one that OOMs.

    The gradient of mean-reduced cross-entropy w.r.t. the logits is just
    ``softmax(x) - onehot(label)`` -- so once we have lse we are one line away.
    """
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    label = tl.load(labels_ptr + row)

    x = tl.load(logits_ptr + row * row_stride + cols, mask=mask,
                other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    z = tl.sum(tl.exp(x - m), axis=0)
    lse = m + tl.log(z)

    # ignore_index rows contribute zero loss and zero gradient.
    keep = label != IGNORE
    x_label = tl.sum(tl.where(cols == label, x, 0.0), axis=0)
    loss = tl.where(keep, lse - x_label, 0.0)
    tl.store(loss_ptr + row, loss)

    probs = tl.exp(x - lse)
    grad = probs - tl.where(cols == label, 1.0, 0.0)
    grad = tl.where(keep, grad, 0.0)
    tl.store(dlogits_ptr + row * row_stride + cols, grad, mask=mask)


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100):
    """Returns (per_row_loss, dlogits) with dlogits already the softmax gradient."""
    logits = logits.contiguous()
    flat = logits.reshape(-1, logits.shape[-1])
    lab = labels.contiguous().reshape(-1).to(torch.int32)
    rows, cols = flat.shape
    loss = torch.empty(rows, device=logits.device, dtype=torch.float32)
    dlogits = torch.empty_like(flat, dtype=torch.float32)
    block = 1 << (cols - 1).bit_length()
    cross_entropy_kernel[(rows,)](flat, lab, loss, dlogits, cols, flat.stride(0),
                                  IGNORE=ignore_index, BLOCK=block)
    return loss, dlogits.reshape(logits.shape)
