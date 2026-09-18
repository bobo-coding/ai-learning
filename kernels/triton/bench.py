"""Verify the Triton kernels -- under real Triton on a GPU, or `tritonsim` here.

    python -m kernels.triton.bench

On a Mac this runs every kernel through the interpreter and checks it against
PyTorch. That is a genuine correctness check on the *kernel logic*: the masks,
the offsets, the online-softmax algebra, the gradient formulas. It is not a
performance check -- the interpreter is ~1000x slower than a GPU -- so sizes
here are small on purpose.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from minigpt.utils import allclose_report, rel_err

from . import kernels as tk


def main():
    torch.manual_seed(0)
    backend = "REAL TRITON (GPU)" if tk.HAVE_REAL_TRITON else "tritonsim interpreter (CPU)"
    dev = "cuda" if tk.HAVE_REAL_TRITON and torch.cuda.is_available() else "cpu"
    # The interpreter is slow, so scale the problems down when it is in use.
    big = tk.HAVE_REAL_TRITON
    print(f"backend: {backend}   device: {dev}")
    print("=" * 74)
    ok = True

    # 1. add
    n = (1 << 20) if big else 5000          # deliberately not a multiple of BLOCK
    x, y = torch.randn(n, device=dev), torch.randn(n, device=dev)
    ok &= allclose_report(tk.add(x, y, block=1024), x + y, "add (ragged tail)", atol=0, rtol=0)

    # 2. softmax
    rows, cols = (512, 1024) if big else (7, 100)
    a = torch.randn(rows, cols, device=dev) * 5
    ok &= allclose_report(tk.softmax(a), torch.softmax(a, -1), "softmax", atol=1e-6, rtol=1e-5)
    hot = torch.tensor([[0.0, 100.0, 300.0, 800.0]], device=dev)
    ok &= allclose_report(tk.softmax(hot), torch.softmax(hot, -1), "softmax (logit 800)",
                          atol=1e-7, rtol=1e-6)

    # 3. rmsnorm forward and backward, checked against autograd
    rows, cols = (256, 512) if big else (6, 64)
    xx = torch.randn(rows, cols, device=dev, requires_grad=True)
    ww = torch.randn(cols, device=dev, requires_grad=True)
    eps = 1e-5

    def ref_rmsnorm(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    ref = ref_rmsnorm(xx, ww)
    got, rstd = tk.rmsnorm(xx.detach(), ww.detach(), eps)
    ok &= allclose_report(got, ref, "rmsnorm forward", atol=1e-5, rtol=1e-4)
    ok &= allclose_report(rstd, torch.rsqrt(xx.detach().pow(2).mean(-1) + eps),
                          "rmsnorm saved rstd", atol=1e-6, rtol=1e-5)

    g = torch.randn_like(ref)
    ref.backward(g)
    dx, dw = tk.rmsnorm_backward(g, xx.detach(), ww.detach(), rstd)
    ok &= allclose_report(dx, xx.grad, "rmsnorm dx vs autograd", atol=1e-5, rtol=1e-4)
    ok &= allclose_report(dw, ww.grad, "rmsnorm dw vs autograd", atol=1e-4, rtol=1e-3)

    # 4. matmul, including shapes that do not divide the block sizes
    shapes = [(256, 256, 256), (512, 384, 256), (129, 77, 53)] if big \
        else [(64, 64, 64), (65, 33, 47)]
    for M, K, N in shapes:
        A = torch.randn(M, K, device=dev)
        B = torch.randn(K, N, device=dev)
        e = rel_err(tk.matmul(A, B, 32, 32, 16, 4), A @ B)
        good = e < 1e-4
        ok &= good
        print(f"{'OK  ' if good else 'FAIL'} matmul {M}x{K}x{N}{' (ragged)' if M % 32 else ''}"
              f"{'':<12} rel_l2={e:.3e}")

    # 5. flash attention forward, vs sdpa, causal and not
    cases = [(1, 2, 128, 64), (2, 2, 96, 32)] if big else [(1, 2, 40, 16), (1, 1, 33, 8)]
    for B_, H, T, D in cases:
        q, k, v = (torch.randn(B_, H, T, D, device=dev) for _ in range(3))
        for causal in (True, False):
            out, lse = tk.flash_attention(q, k, v, causal=causal, block_m=16, block_n=16)
            ref_o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            ok &= allclose_report(out, ref_o, f"flash B{B_}H{H}T{T}D{D} causal={causal}",
                                  atol=1e-4, rtol=1e-3)
            # lse must equal logsumexp of the masked scores -- this is what the
            # backward pass depends on, so check it separately.
            s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(D)
            if causal:
                m = torch.ones(T, T, device=dev, dtype=torch.bool).tril()
                s = s.masked_fill(~m, float("-inf"))
            ok &= allclose_report(lse, torch.logsumexp(s, -1), f"flash lse causal={causal}",
                                  atol=1e-4, rtol=1e-4)

    # 6. fused cross entropy: loss and gradient
    rows, vocab = (256, 1024) if big else (8, 64)
    logits = torch.randn(rows, vocab, device=dev, requires_grad=True)
    labels = torch.randint(0, vocab, (rows,), device=dev)
    labels[0] = -100                                   # exercise ignore_index
    loss, dlogits = tk.cross_entropy(logits.detach(), labels)
    ref_each = F.cross_entropy(logits, labels, ignore_index=-100, reduction="none")
    ok &= allclose_report(loss, ref_each, "cross_entropy per-row loss", atol=1e-5, rtol=1e-4)

    n_valid = int((labels != -100).sum())
    ref_mean = F.cross_entropy(logits, labels, ignore_index=-100)
    ref_mean.backward()
    # the kernel returns d(sum)/dlogits; mean-reduced autograd divides by n_valid
    ok &= allclose_report(dlogits / n_valid, logits.grad, "cross_entropy gradient",
                          atol=1e-6, rtol=1e-5)
    print(f"     (ignored row gradient is exactly zero: "
          f"{float(dlogits[0].abs().max()) == 0.0})")

    print("=" * 74)
    print("ALL CORRECTNESS CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    if not tk.HAVE_REAL_TRITON:
        print("NOTE: run this on an NVIDIA GPU with `pip install triton` for real timings.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
