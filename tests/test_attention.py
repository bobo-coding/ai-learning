import math

import pytest
import torch
import torch.nn.functional as F

from minigpt.attention import (CausalSelfAttention, KVCache, attention_batched,
                               attention_reference, apply_rope, apply_rope_complex,
                               causal_mask, flash_attention, online_softmax,
                               repeat_kv, rope_tables)

F64 = dict(dtype=torch.float64)
TIGHT = dict(atol=1e-10, rtol=1e-8)


@pytest.fixture
def qkv():
    torch.manual_seed(0)
    return [torch.randn(2, 3, 24, 8, **F64) for _ in range(3)]


@pytest.mark.parametrize("causal", [True, False])
def test_batched_matches_definition(qkv, causal):
    q, k, v = qkv
    assert torch.allclose(attention_batched(q, k, v, causal),
                          attention_reference(q, k, v, causal), **TIGHT)


@pytest.mark.parametrize("causal", [True, False])
def test_sdpa_matches_definition(qkv, causal):
    q, k, v = qkv
    assert torch.allclose(F.scaled_dot_product_attention(q, k, v, is_causal=causal),
                          attention_reference(q, k, v, causal), **TIGHT)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("bq,bk", [(7, 5), (24, 24), (1, 3), (32, 64)])
def test_flash_forward_matches_definition(qkv, causal, bq, bk):
    """Tiling must not change the answer, for any tile size."""
    q, k, v = qkv
    assert torch.allclose(flash_attention(q, k, v, causal, bq, bk),
                          attention_reference(q, k, v, causal), **TIGHT)


@pytest.mark.parametrize("causal", [True, False])
def test_flash_backward_matches_autograd(qkv, causal):
    q, k, v = qkv
    torch.manual_seed(1)
    g = torch.randn_like(q)

    def grads(fn):
        a, b, c = (t.clone().requires_grad_(True) for t in (q, k, v))
        fn(a, b, c).backward(g)
        return a.grad, b.grad, c.grad

    ref = grads(lambda a, b, c: attention_batched(a, b, c, causal))
    got = grads(lambda a, b, c: flash_attention(a, b, c, causal, 7, 5))
    for r, o in zip(ref, got):
        assert torch.allclose(o, r, atol=1e-9, rtol=1e-7)


def test_flash_gradcheck():
    """The strongest correctness statement available for a custom autograd.Function."""
    torch.manual_seed(2)
    args = tuple(torch.randn(1, 1, 6, 4, **F64, requires_grad=True) for _ in range(3))
    assert torch.autograd.gradcheck(
        lambda a, b, c: flash_attention(a, b, c, True, 3, 2), args, eps=1e-6, atol=1e-7)


def test_flash_incremental_shapes():
    """Tq < Tk, as during decoding: the mask must be bottom-right aligned."""
    torch.manual_seed(3)
    q = torch.randn(1, 2, 3, 8, **F64)
    k = torch.randn(1, 2, 10, 8, **F64)
    v = torch.randn(1, 2, 10, 8, **F64)
    assert torch.allclose(flash_attention(q, k, v, True, 2, 4),
                          attention_batched(q, k, v, True), **TIGHT)


@pytest.mark.parametrize("block", [1, 3, 17, 64])
def test_online_softmax(block):
    torch.manual_seed(4)
    x = torch.randn(5, 17, **F64) * 30
    p, lse = online_softmax(x, block=block)
    assert torch.allclose(p, torch.softmax(x, -1), atol=1e-12)
    assert torch.allclose(lse, torch.logsumexp(x, -1), atol=1e-12)


def test_causal_mask_shape():
    m = causal_mask(4)[0, 0]
    assert m.shape == (4, 4)
    finite = torch.isfinite(m)
    # additive mask: 0 on and below the diagonal, -inf strictly above
    assert torch.equal(finite, torch.ones(4, 4, dtype=torch.bool).tril())
    assert (m[finite] == 0).all()
    assert (m[~finite] == float("-inf")).all()


def test_rope_matches_complex_reference():
    cos, sin = rope_tables(8, 64, dtype=torch.float64)
    torch.manual_seed(5)
    x = torch.randn(2, 3, 16, 8, **F64)
    assert torch.allclose(apply_rope(x, cos, sin), apply_rope_complex(x, cos, sin), **TIGHT)


def test_rope_preserves_norm():
    """RoPE is a rotation, so every channel pair keeps its magnitude."""
    cos, sin = rope_tables(16, 32, dtype=torch.float64)
    x = torch.randn(1, 2, 32, 16, **F64)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), **TIGHT)


def test_rope_depends_only_on_relative_position():
    """<RoPE(q, m), RoPE(k, n)> must be a function of m - n alone.

    This is the entire justification for RoPE, and it is the property that
    breaks if you mix the interleaved and split-half conventions.
    """
    cos, sin = rope_tables(8, 128, dtype=torch.float64)
    torch.manual_seed(6)
    q = torch.randn(1, 1, 1, 8, **F64)
    k = torch.randn(1, 1, 1, 8, **F64)

    def dot(m, n):
        return float((apply_rope(q, cos, sin, offset=m) * apply_rope(k, cos, sin, offset=n)).sum())

    base = dot(5, 2)
    for m in (3, 10, 40, 100):
        assert abs(dot(m, m - 3) - base) < 1e-12


def test_rope_offset_is_a_slice():
    cos, sin = rope_tables(8, 64, dtype=torch.float64)
    x = torch.randn(1, 1, 20, 8, **F64)
    full = apply_rope(x, cos, sin)
    assert torch.allclose(apply_rope(x[:, :, 5:9], cos, sin, offset=5), full[:, :, 5:9], **TIGHT)


def test_rope_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        rope_tables(7, 16)


def test_repeat_kv():
    x = torch.arange(2 * 2 * 3 * 2, **F64).view(2, 2, 3, 2)
    r = repeat_kv(x, 3)
    assert r.shape == (2, 6, 3, 2)
    for g in range(2):
        for j in range(3):
            assert torch.equal(r[:, g * 3 + j], x[:, g])
    assert torch.equal(repeat_kv(x, 1), x)


@pytest.mark.parametrize("n_kv_head", [3, 1])
def test_module_impls_agree(n_kv_head):
    outs = {}
    for impl in ("sdpa", "math", "flash"):
        torch.manual_seed(7)
        att = CausalSelfAttention(24, 3, n_kv_head=n_kv_head, impl=impl).double().eval()
        cos, sin = rope_tables(8, 64, dtype=torch.float64)
        torch.manual_seed(8)
        x = torch.randn(2, 12, 24, **F64)
        outs[impl] = att(x, cos, sin)
    assert torch.allclose(outs["math"], outs["sdpa"], **TIGHT)
    assert torch.allclose(outs["flash"], outs["sdpa"], **TIGHT)


def test_module_is_causal():
    torch.manual_seed(9)
    att = CausalSelfAttention(24, 3).double().eval()
    cos, sin = rope_tables(8, 64, dtype=torch.float64)
    x = torch.randn(1, 10, 24, **F64)
    o1 = att(x, cos, sin)
    x2 = x.clone()
    x2[0, 6] += 5.0
    o2 = att(x2, cos, sin)
    assert torch.equal(o1[:, :6], o2[:, :6])          # past unchanged
    assert not torch.allclose(o1[:, 6:], o2[:, 6:])   # present/future changed


@pytest.mark.parametrize("prefill", [1, 4, 10])
def test_kv_cache_equals_full_forward(prefill):
    torch.manual_seed(10)
    att = CausalSelfAttention(24, 3, n_kv_head=1).double().eval()
    cos, sin = rope_tables(8, 64, dtype=torch.float64)
    x = torch.randn(1, 10, 24, **F64)
    full = att(x, cos, sin)

    cache = KVCache(1, 1, 32, 8, device="cpu", dtype=torch.float64)
    parts = [att(x[:, :prefill], cos, sin, cache=cache)]
    for t in range(prefill, 10):
        parts.append(att(x[:, t:t + 1], cos, sin, cache=cache))
    assert torch.allclose(torch.cat(parts, 1), full, **TIGHT)


def test_kv_cache_overflow_raises():
    cache = KVCache(1, 1, 4, 2, device="cpu")
    k = torch.zeros(1, 1, 3, 2)
    cache.append(k, k)
    with pytest.raises(ValueError):
        cache.append(k, k)


def test_attention_config_validation():
    with pytest.raises(ValueError):
        CausalSelfAttention(25, 4)               # n_embd not divisible by n_head
    with pytest.raises(ValueError):
        CausalSelfAttention(24, 4, n_kv_head=3)  # n_head not divisible by n_kv_head
    with pytest.raises(ValueError):
        CausalSelfAttention(24, 4, impl="nope")(torch.randn(1, 2, 24), *rope_tables(6, 8))


def test_attention_scale_is_one_over_sqrt_d():
    """A regression guard on the one constant everybody eventually mistypes."""
    torch.manual_seed(11)
    q = torch.randn(1, 1, 4, 16, **F64)
    k = torch.randn(1, 1, 4, 16, **F64)
    v = torch.eye(4, **F64).view(1, 1, 4, 4).expand(1, 1, 4, 4).contiguous()
    got = attention_batched(q, k, v, causal=False)
    want = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(16), dim=-1) @ v
    assert torch.allclose(got, want, **TIGHT)
