"""Build and verify the CUDA kernels.  Needs an NVIDIA GPU.

    python -m kernels.cuda.bench

On a Mac this exits immediately with a message.  See kernels/cuda/README.md for
how to run it on a rented GPU or in Colab; the checks here are deliberately the
same ones as `kernels/metal/bench.py`, so the two outputs are comparable.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

_EXT = None


def load_extension(verbose: bool = False):
    """JIT-compile kernels.cu with nvcc via torch's cpp_extension loader.

    The first call takes ~1 minute and caches the .so under
    ~/.cache/torch_extensions.  `TORCH_CUDA_ARCH_LIST` is worth setting to your
    exact architecture (e.g. "8.0" for A100, "8.9" for L4/4090, "9.0" for H100)
    -- otherwise torch compiles for every architecture it knows, which is slow.
    """
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device available")
    from torch.utils.cpp_extension import load

    src = Path(__file__).with_name("kernels.cu")
    _EXT = load(
        name="minigpt_cuda_kernels",
        sources=[str(src)],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        extra_cflags=["-O3"],
        verbose=verbose,
    )
    return _EXT


def main():
    if not torch.cuda.is_available():
        print("No CUDA device found -- this module is for reading on a Mac, and for")
        print("running on an NVIDIA GPU.  The Metal equivalents in kernels/metal/")
        print("implement the same six kernels and DO run here:")
        print("    python -m kernels.metal.bench")
        return 0

    from minigpt.utils import allclose_report, benchmark, rel_err

    ext = load_extension(verbose=bool(os.environ.get("VERBOSE")))
    dev = "cuda"
    torch.manual_seed(0)
    ok = True

    print(f"device: {torch.cuda.get_device_name(0)}")

    n = 1 << 22
    a, b = torch.randn(n, device=dev), torch.randn(n, device=dev)
    ok &= allclose_report(ext.vector_add(a, b), a + b, "vector_add", atol=0, rtol=0)
    ok &= allclose_report(ext.vector_add(a, b, True), a + b, "vector_add_stride", atol=0, rtol=0)
    moved = 3 * n * 4
    for label, fn in [("cuda 1-thread/elem", lambda: ext.vector_add(a, b)),
                      ("cuda grid-stride", lambda: ext.vector_add(a, b, True)),
                      ("torch a+b", lambda: a + b)]:
        t = benchmark(fn, dev)
        print(f"  {label:22s} {t * 1e6:8.1f} us   {moved / t / 1e9:7.1f} GB/s")

    x = torch.randn(4096, 1024, device=dev)
    ok &= allclose_report(ext.row_sum(x), x.sum(-1), "row_sum", atol=1e-3, rtol=1e-4)
    x = torch.randn(8192, 1024, device=dev) * 5
    ok &= allclose_report(ext.softmax_rows(x), torch.softmax(x, -1), "softmax_rows",
                          atol=1e-6, rtol=1e-4)
    hot = torch.tensor([[0.0, 100.0, 200.0, 800.0]], device=dev)
    ok &= allclose_report(ext.softmax_rows(hot), torch.softmax(hot, -1),
                          "softmax stability", atol=1e-7, rtol=1e-5)

    g = torch.randn(1024, device=dev)
    xr = torch.randn(8192, 1024, device=dev)

    def torch_rmsnorm(x, g, eps=1e-5):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g

    ok &= allclose_report(ext.rmsnorm(xr, g), torch_rmsnorm(xr, g), "rmsnorm", atol=1e-5, rtol=1e-4)
    moved = 2 * xr.numel() * 4
    for label, fn in [("cuda fused", lambda: ext.rmsnorm(xr, g)),
                      ("torch eager", lambda: torch_rmsnorm(xr, g))]:
        t = benchmark(fn, dev)
        print(f"  {label:22s} {t * 1e6:8.1f} us   {moved / t / 1e9:7.1f} GB/s")

    for M, K, N in [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]:
        A = torch.randn(M, K, device=dev)
        B = torch.randn(K, N, device=dev)
        ref = A @ B
        e_n, e_t = rel_err(ext.matmul(A, B, False), ref), rel_err(ext.matmul(A, B, True), ref)
        flops = 2.0 * M * N * K
        # default args bind the current A/B rather than closing over the loop var
        tn = benchmark(lambda A=A, B=B: ext.matmul(A, B, False), dev)
        tt = benchmark(lambda A=A, B=B: ext.matmul(A, B, True), dev)
        tc = benchmark(lambda A=A, B=B: A @ B, dev)
        print(f"  {M}^3 matmul: naive {flops / tn / 1e12:6.2f} | tiled {flops / tt / 1e12:6.2f} "
              f"| cuBLAS {flops / tc / 1e12:6.2f} TFLOP/s  (err {e_n:.1e}/{e_t:.1e})")
        ok &= e_n < 1e-4 and e_t < 1e-4

    for B_, H, T, D in [(1, 4, 128, 64), (2, 8, 256, 64), (1, 8, 512, 64)]:
        q, k, v = (torch.randn(B_, H, T, D, device=dev) for _ in range(3))
        for causal in (True, False):
            ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
            ok &= allclose_report(ext.flash_attention(q, k, v, causal), ref,
                                  f"flash T{T} causal={causal}", atol=1e-4, rtol=1e-3)

    print("ALL CORRECTNESS CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
