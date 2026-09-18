"""Verify and benchmark the Metal kernels against PyTorch.

    python -m kernels.metal.bench

Every kernel is checked against a PyTorch reference first -- a fast wrong kernel
is worth nothing -- and only then timed.  Read the numbers as "how close to
PyTorch's hand-tuned MPS kernels can 200 lines of MSL get", not as a claim that
hand-writing kernels is a good idea for these ops.
"""

from __future__ import annotations


import torch

from minigpt.utils import allclose_report, benchmark, rel_err

from . import kernels as mk


def gbps(bytes_moved: float, seconds: float) -> float:
    return bytes_moved / seconds / 1e9


def tflops(flops: float, seconds: float) -> float:
    return flops / seconds / 1e12


def main():
    if not torch.backends.mps.is_available():
        print("no MPS device; skipping")
        return
    dev = "mps"
    torch.manual_seed(0)
    ok = True

    print("=" * 78)
    print("1. vector_add -- pure bandwidth")
    print("=" * 78)
    n = 1 << 22
    a, b = torch.randn(n, device=dev), torch.randn(n, device=dev)
    ok &= allclose_report(mk.vector_add(a, b), a + b, "vector_add", atol=0, rtol=0)
    ok &= allclose_report(mk.vector_add(a, b, stride_loop=True), a + b, "vector_add_stride", atol=0, rtol=0)
    moved = 3 * n * 4          # read a, read b, write out
    for label, fn in [("metal 1-thread/elem", lambda: mk.vector_add(a, b)),
                      ("metal grid-stride", lambda: mk.vector_add(a, b, stride_loop=True)),
                      ("torch a+b", lambda: a + b)]:
        t = benchmark(fn, dev)
        print(f"  {label:22s} {t * 1e6:8.1f} us   {gbps(moved, t):7.1f} GB/s")

    print()
    print("=" * 78)
    print("2. row_sum -- reductions, tree vs SIMD shuffle")
    print("=" * 78)
    x = torch.randn(4096, 1024, device=dev)
    ok &= allclose_report(mk.row_sum(x, simd=False), x.sum(-1), "row_sum (tree)", atol=1e-3, rtol=1e-4)
    ok &= allclose_report(mk.row_sum(x, simd=True), x.sum(-1), "row_sum (simd)", atol=1e-3, rtol=1e-4)
    moved = x.numel() * 4
    for label, fn in [("metal tree reduction", lambda: mk.row_sum(x, simd=False)),
                      ("metal simd_sum", lambda: mk.row_sum(x, simd=True)),
                      ("torch x.sum(-1)", lambda: x.sum(-1))]:
        t = benchmark(fn, dev)
        print(f"  {label:22s} {t * 1e6:8.1f} us   {gbps(moved, t):7.1f} GB/s")

    print()
    print("=" * 78)
    print("3. softmax_rows -- stable two-pass reduction")
    print("=" * 78)
    x = torch.randn(8192, 1024, device=dev) * 5
    ok &= allclose_report(mk.softmax_rows(x), torch.softmax(x, -1), "softmax_rows", atol=1e-6, rtol=1e-4)
    # the stability test: a row with huge logits must not overflow
    hot = torch.tensor([[0.0, 100.0, 200.0, 800.0]], device=dev)
    got = mk.softmax_rows(hot, group_size=32)
    ok &= allclose_report(got, torch.softmax(hot, -1), "softmax stability (logit 800)", atol=1e-7, rtol=1e-5)
    moved = 2 * x.numel() * 4
    for label, fn in [("metal softmax", lambda: mk.softmax_rows(x)),
                      ("torch softmax", lambda: torch.softmax(x, -1))]:
        t = benchmark(fn, dev)
        print(f"  {label:22s} {t * 1e6:8.1f} us   {gbps(moved, t):7.1f} GB/s")

    print()
    print("=" * 78)
    print("4. rmsnorm -- fusion: 1 kernel vs 5")
    print("=" * 78)
    x = torch.randn(8192, 1024, device=dev)
    g = torch.randn(1024, device=dev)

    def torch_rmsnorm(x, g, eps=1e-5):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g

    ok &= allclose_report(mk.rmsnorm(x, g), torch_rmsnorm(x, g), "rmsnorm", atol=1e-5, rtol=1e-4)
    moved = 2 * x.numel() * 4
    for label, fn in [("metal fused", lambda: mk.rmsnorm(x, g)),
                      ("torch eager (5 kernels)", lambda: torch_rmsnorm(x, g))]:
        t = benchmark(fn, dev)
        print(f"  {label:22s} {t * 1e6:8.1f} us   {gbps(moved, t):7.1f} GB/s")

    print()
    print("=" * 78)
    print("5. matmul -- naive vs tiled vs PyTorch")
    print("=" * 78)
    for M, K, N in [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]:
        A = torch.randn(M, K, device=dev)
        B = torch.randn(K, N, device=dev)
        ref = A @ B
        # fp32 matmul reassociation means ~1e-4 relative error is expected, not a bug
        e_naive = rel_err(mk.matmul(A, B, tiled=False), ref)
        e_tiled = rel_err(mk.matmul(A, B, tiled=True), ref)
        flops = 2.0 * M * N * K
        # default args bind the current A/B rather than closing over the loop var
        t_naive = benchmark(lambda A=A, B=B: mk.matmul(A, B, tiled=False), dev)
        t_tiled = benchmark(lambda A=A, B=B: mk.matmul(A, B, tiled=True), dev)
        t_torch = benchmark(lambda A=A, B=B: A @ B, dev)
        print(f"  {M}x{K}x{N}:")
        print(f"    naive  {t_naive * 1e6:8.1f} us  {tflops(flops, t_naive):6.2f} TFLOP/s  rel_err {e_naive:.2e}")
        print(f"    tiled  {t_tiled * 1e6:8.1f} us  {tflops(flops, t_tiled):6.2f} TFLOP/s  rel_err {e_tiled:.2e}"
              f"   ({t_naive / t_tiled:.2f}x over naive)")
        print(f"    torch  {t_torch * 1e6:8.1f} us  {tflops(flops, t_torch):6.2f} TFLOP/s"
              f"   ({t_tiled / t_torch:.2f}x slower than torch)")
        ok &= e_tiled < 1e-4 and e_naive < 1e-4

    print()
    print("=" * 78)
    print("6. flash_attention -- online softmax, no T x T matrix")
    print("=" * 78)
    for B_, H, T, D in [(1, 4, 128, 64), (2, 8, 256, 64), (1, 8, 512, 64)]:
        q, k, v = (torch.randn(B_, H, T, D, device=dev) for _ in range(3))
        for causal in (True, False):
            ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
            got = mk.flash_attention(q, k, v, causal=causal)
            ok &= allclose_report(got, ref, f"flash B{B_} H{H} T{T} causal={causal}",
                                  atol=1e-4, rtol=1e-3)
        t_mine = benchmark(lambda q=q, k=k, v=v: mk.flash_attention(q, k, v, True), dev)
        t_sdpa = benchmark(
            lambda q=q, k=k, v=v:
                torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True), dev)
        # causal attention does ~half the T^2 work
        flops = 2 * 2 * B_ * H * T * T * D / 2
        print(f"  B{B_} H{H} T{T} D{D}: mine {t_mine * 1e3:6.2f} ms ({tflops(flops, t_mine):5.2f} TFLOP/s)"
              f" | sdpa {t_sdpa * 1e3:6.2f} ms ({tflops(flops, t_sdpa):5.2f} TFLOP/s)")

    print()
    print("=" * 78)
    print("ALL CORRECTNESS CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
