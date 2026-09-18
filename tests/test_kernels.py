"""Correctness tests for the Metal and Triton kernels.

The Metal tests are skipped without an Apple GPU; the Triton tests always run,
either against real Triton or against the `tritonsim` interpreter.  The CUDA
kernels have no test here because they cannot be compiled on this machine --
`kernels/cuda/bench.py` runs the identical checks where nvcc exists.
"""

import math
import warnings

import pytest
import torch
import torch.nn.functional as F

from minigpt.utils import rel_err

HAS_MPS = torch.backends.mps.is_available()
mps_only = pytest.mark.skipif(not HAS_MPS, reason="needs an Apple Silicon GPU")


# ===========================================================================
# tritonsim: the interpreter itself
# ===========================================================================

import tritonsim as tsim
import tritonsim.language as tsl


def test_cdiv_and_next_power_of_2():
    assert tsim.cdiv(10, 3) == 4
    assert tsim.cdiv(9, 3) == 3
    assert [tsim.next_power_of_2(n) for n in (1, 2, 3, 17, 64, 65)] == [1, 2, 4, 32, 64, 128]


def test_program_id_covers_the_grid():
    seen = []

    @tsim.jit
    def k(dummy):
        seen.append((tsl.program_id(0), tsl.program_id(1), tsl.num_programs(0)))

    k[(3, 2)](torch.zeros(1))
    assert len(seen) == 6
    assert {(x, y) for x, y, _ in seen} == {(x, y) for x in range(3) for y in range(2)}
    assert all(n == 3 for _, _, n in seen)


def test_program_id_outside_a_launch_raises():
    with pytest.raises(RuntimeError):
        tsl.program_id(0)


def test_kernels_must_be_launched_with_a_grid():
    @tsim.jit
    def k(p):
        pass

    with pytest.raises(RuntimeError):
        k(torch.zeros(1))


def test_unmasked_out_of_bounds_load_is_caught():
    """The interpreter's main value: a GPU would silently return garbage here."""
    @tsim.jit
    def buggy(x_ptr, out_ptr, n, BLOCK: tsl.constexpr):
        offs = tsl.program_id(0) * BLOCK + tsl.arange(0, BLOCK)
        tsl.store(out_ptr + offs, tsl.load(x_ptr + offs), mask=offs < n)

    x = torch.randn(1000)
    out = torch.empty(1000)
    with pytest.raises(IndexError, match="out of bounds"):
        buggy[(tsim.cdiv(1000, 128),)](x, out, 1000, BLOCK=128)


def test_mask_that_disagrees_with_the_offsets_is_caught():
    @tsim.jit
    def off_by_one(x_ptr, out_ptr, n, BLOCK: tsl.constexpr):
        offs = tsl.program_id(0) * BLOCK + tsl.arange(0, BLOCK)
        v = tsl.load(x_ptr + offs, mask=offs <= n)          # <= instead of <
        tsl.store(out_ptr + offs, v, mask=offs < n)

    with pytest.raises(IndexError, match="mask"):
        off_by_one[(8,)](torch.randn(1000), torch.empty(1000), 1000, BLOCK=128)


def test_out_of_bounds_store_is_caught():
    @tsim.jit
    def bad_store(out_ptr, BLOCK: tsl.constexpr):
        offs = tsl.arange(0, BLOCK)
        tsl.store(out_ptr + offs, tsl.zeros([BLOCK], dtype=tsl.float32))

    with pytest.raises(IndexError):
        bad_store[(1,)](torch.empty(8), BLOCK=16)


def test_non_power_of_two_block_warns():
    """Real Triton will not compile it, so the interpreter must say so."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tsl.arange(0, 100)
    assert w and "power of two" in str(w[0].message)


def test_non_contiguous_tensors_are_rejected():
    @tsim.jit
    def k(p):
        tsl.load(p + tsl.arange(0, 4))

    with pytest.raises(ValueError, match="contiguous"):
        k[(1,)](torch.randn(8, 8).t())


def test_masked_load_uses_other():
    @tsim.jit
    def k(x_ptr, out_ptr, n, BLOCK: tsl.constexpr):
        offs = tsl.arange(0, BLOCK)
        v = tsl.load(x_ptr + offs, mask=offs < n, other=-float("inf"))
        tsl.store(out_ptr + offs, v)

    out = torch.empty(8)
    k[(1,)](torch.ones(5), out, 5, BLOCK=8)
    assert out[:5].tolist() == [1.0] * 5
    assert bool(torch.isinf(out[5:]).all()) and bool((out[5:] < 0).all())


def test_atomic_add_accumulates_across_programs():
    @tsim.jit
    def k(out_ptr, BLOCK: tsl.constexpr):
        offs = tsl.arange(0, BLOCK)
        tsl.atomic_add(out_ptr + offs, tsl.full([BLOCK], 1.0, dtype=tsl.float32))

    out = torch.zeros(4)
    k[(5,)](out, BLOCK=4)
    assert out.tolist() == [5.0] * 4


def test_dot_accumulates_in_fp32():
    a = torch.randn(16, 16, dtype=torch.float16)
    b = torch.randn(16, 16, dtype=torch.float16)
    assert tsl.dot(a, b).dtype == torch.float32
    assert torch.allclose(tsl.dot(a, b), a.float() @ b.float(), atol=1e-3)


def test_constexpr_is_transparent():
    c = tsl.constexpr(8)
    assert int(c) == 8
    assert tsl.zeros([c], dtype=tsl.float32).shape == (8,)


def test_autotune_and_heuristics_decorators_run():
    @tsim.autotune(configs=[tsim.Config({"BLOCK": 8}, num_warps=4)], key=["n"])
    @tsim.jit
    def k(out_ptr, n, BLOCK: tsl.constexpr):
        offs = tsl.arange(0, BLOCK)
        tsl.store(out_ptr + offs, tsl.full([BLOCK], float(BLOCK), dtype=tsl.float32),
                  mask=offs < n)

    out = torch.zeros(8)
    k[(1,)](out, 8)
    assert out.tolist() == [8.0] * 8


def test_callable_grid_receives_meta():
    captured = {}

    @tsim.jit
    def k(out_ptr, n, BLOCK: tsl.constexpr):
        pass

    def grid(meta):
        captured.update(meta)
        return (tsim.cdiv(100, meta["BLOCK"]),)

    k[grid](torch.zeros(1), 100, BLOCK=32)
    assert captured["BLOCK"] == 32


# ===========================================================================
# Triton kernels (run under real Triton or the interpreter)
# ===========================================================================

from kernels.triton import kernels as tk


def test_triton_add_with_a_ragged_tail():
    x, y = torch.randn(5000), torch.randn(5000)     # not a multiple of BLOCK
    assert torch.equal(tk.add(x, y, block=1024), x + y)


@pytest.mark.parametrize("shape", [(7, 100), (1, 1), (3, 64)])
def test_triton_softmax(shape):
    x = torch.randn(*shape) * 5
    assert torch.allclose(tk.softmax(x), torch.softmax(x, -1), atol=1e-6)


def test_triton_softmax_is_numerically_stable():
    """`other=-inf` on the masked lanes is what makes this work."""
    hot = torch.tensor([[0.0, 100.0, 300.0, 800.0]])
    got = tk.softmax(hot)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, torch.softmax(hot, -1), atol=1e-7)


def test_triton_rmsnorm_forward_and_saved_rstd():
    x = torch.randn(6, 64)
    w = torch.randn(64)
    out, rstd = tk.rmsnorm(x, w, 1e-5)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    assert torch.allclose(out, ref, atol=1e-5)
    assert torch.allclose(rstd, torch.rsqrt(x.pow(2).mean(-1) + 1e-5), atol=1e-6)


def test_triton_rmsnorm_backward_matches_autograd():
    """Catches the classic dropped-correction-term bug in a hand-written norm."""
    x = torch.randn(6, 64, requires_grad=True)
    w = torch.randn(64, requires_grad=True)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    g = torch.randn_like(ref)
    ref.backward(g)
    _, rstd = tk.rmsnorm(x.detach(), w.detach(), 1e-5)
    dx, dw = tk.rmsnorm_backward(g, x.detach(), w.detach(), rstd)
    assert torch.allclose(dx, x.grad, atol=1e-5)
    assert torch.allclose(dw, w.grad, atol=1e-4)


@pytest.mark.parametrize("M,K,N", [(64, 64, 64), (65, 33, 47), (16, 128, 16)])
def test_triton_matmul(M, K, N):
    a, b = torch.randn(M, K), torch.randn(K, N)
    assert rel_err(tk.matmul(a, b, 32, 32, 16, 4), a @ b) < 1e-5


@pytest.mark.parametrize("B,H,T,D", [(1, 2, 40, 16), (1, 1, 33, 8), (2, 1, 16, 16)])
@pytest.mark.parametrize("causal", [True, False])
def test_triton_flash_attention(B, H, T, D, causal):
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    out, lse = tk.flash_attention(q, k, v, causal=causal, block_m=16, block_n=16)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3)

    # lse is what the backward pass needs; check it independently
    s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(D)
    if causal:
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    assert torch.allclose(lse, torch.logsumexp(s, -1), atol=1e-4)


def test_triton_cross_entropy_loss_and_gradient():
    torch.manual_seed(0)
    logits = torch.randn(8, 64, requires_grad=True)
    labels = torch.randint(0, 64, (8,))
    labels[0] = -100                                   # ignore_index
    loss, dlogits = tk.cross_entropy(logits.detach(), labels)
    ref = F.cross_entropy(logits, labels, ignore_index=-100, reduction="none")
    assert torch.allclose(loss, ref, atol=1e-5)

    n_valid = int((labels != -100).sum())
    F.cross_entropy(logits, labels, ignore_index=-100).backward()
    assert torch.allclose(dlogits / n_valid, logits.grad, atol=1e-6)
    assert float(dlogits[0].abs().max()) == 0.0        # ignored row is exactly zero


# ===========================================================================
# Metal kernels (Apple GPU only)
# ===========================================================================


@mps_only
def test_metal_vector_add():
    from kernels.metal import kernels as mk
    a, b = torch.randn(10_000, device="mps"), torch.randn(10_000, device="mps")
    assert torch.equal(mk.vector_add(a, b), a + b)
    assert torch.equal(mk.vector_add(a, b, stride_loop=True), a + b)


@mps_only
@pytest.mark.parametrize("simd", [True, False])
def test_metal_row_sum(simd):
    from kernels.metal import kernels as mk
    x = torch.randn(129, 777, device="mps")
    assert rel_err(mk.row_sum(x, simd=simd), x.sum(-1)) < 1e-5


@mps_only
def test_metal_softmax_and_stability():
    from kernels.metal import kernels as mk
    x = torch.randn(64, 513, device="mps") * 5
    assert torch.allclose(mk.softmax_rows(x), torch.softmax(x, -1), atol=1e-6)
    hot = torch.tensor([[0.0, 100.0, 200.0, 800.0]], device="mps")
    got = mk.softmax_rows(hot, group_size=32)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, torch.softmax(hot, -1), atol=1e-7)


@mps_only
def test_metal_rmsnorm():
    from kernels.metal import kernels as mk
    x = torch.randn(97, 256, device="mps")
    g = torch.randn(256, device="mps")
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * g
    assert torch.allclose(mk.rmsnorm(x, g), ref, atol=1e-5)


@mps_only
@pytest.mark.parametrize("M,K,N", [(64, 64, 64), (100, 37, 53), (256, 128, 256)])
@pytest.mark.parametrize("tiled", [True, False])
def test_metal_matmul(M, K, N, tiled):
    from kernels.metal import kernels as mk
    a = torch.randn(M, K, device="mps")
    b = torch.randn(K, N, device="mps")
    assert rel_err(mk.matmul(a, b, tiled=tiled), a @ b) < 1e-4


@mps_only
@pytest.mark.parametrize("causal", [True, False])
def test_metal_flash_attention(causal):
    from kernels.metal import kernels as mk
    q, k, v = (torch.randn(2, 4, 96, 32, device="mps") for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    assert torch.allclose(mk.flash_attention(q, k, v, causal=causal), ref,
                          atol=1e-4, rtol=1e-3)


@mps_only
def test_metal_input_validation():
    from kernels.metal import kernels as mk
    with pytest.raises(ValueError, match="mps"):
        mk.vector_add(torch.randn(8), torch.randn(8))
    # MPS has no float64 at all, so use fp16 to exercise the dtype guard
    with pytest.raises(ValueError, match="float32"):
        mk.vector_add(torch.randn(8, device="mps").half(),
                      torch.randn(8, device="mps").half())
    with pytest.raises(ValueError, match="contiguous"):
        mk.row_sum(torch.randn(8, 8, device="mps").t())
    with pytest.raises(ValueError, match="matmul"):
        mk.matmul(torch.randn(4, 5, device="mps"), torch.randn(6, 4, device="mps"))
