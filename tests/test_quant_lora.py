import pytest
import torch
import torch.nn as nn

from minigpt.config import GPTConfig
from minigpt.lora import (LoRAConfig, LoRALinear, apply_lora, lora_state_dict,
                          merge_all, trainable_summary, unmerge_all)
from minigpt.model import GPT
from minigpt.quant import (NF4_LEVELS, QuantizedLinear, dequantize_affine,
                           dequantize_nf4, dequantize_symmetric, kv_cache_quant_error,
                           nf4_levels_from_normal, pack_int4, quantization_error,
                           quantize_affine, quantize_model, quantize_nf4,
                           quantize_symmetric, unpack_int4)

SMALL = dict(vocab_size=64, block_size=32, n_layer=3, n_head=4, n_embd=64)


# ------------------------------------------------------------- quantization


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("group_size", [None, 64])
def test_symmetric_error_is_at_most_half_a_grid_step(bits, group_size):
    """The definitive check on any round-to-nearest quantizer."""
    torch.manual_seed(0)
    w = torch.randn(32, 256)
    q, scale, meta = quantize_symmetric(w, bits, dim=-1, group_size=group_size)
    wq = dequantize_symmetric(q, scale, meta)
    q_max = 2 ** (bits - 1) - 1
    assert int(q.abs().max()) <= q_max + 1
    per_elem_scale = (scale.reshape(-1, 1).expand(-1, group_size).reshape(w.shape)
                      if group_size else scale.expand_as(w))
    assert bool(((wq - w).abs() <= 0.5 * per_elem_scale + 1e-6).all())


def test_symmetric_per_tensor_vs_per_channel_vs_group():
    """Finer granularity must never be worse, and an outlier must prove why."""
    torch.manual_seed(1)
    w = torch.randn(64, 256)
    e_tensor = quantization_error(w, 8, "per_tensor")["rel_l2"]
    e_row = quantization_error(w, 8, "symmetric")["rel_l2"]
    e_group = quantization_error(w, 8, "symmetric", 64)["rel_l2"]
    assert e_group <= e_row <= e_tensor

    w[0, 0] = 60.0                        # one 60-sigma outlier
    assert quantization_error(w, 8, "per_tensor")["rel_l2"] > \
        5 * quantization_error(w, 8, "symmetric", 64)["rel_l2"]


def test_affine_handles_a_shifted_distribution():
    """Asymmetric quantization should beat symmetric on non-zero-centred data."""
    torch.manual_seed(2)
    w = torch.rand(32, 128) * 2 + 5.0     # all positive, far from zero
    assert quantization_error(w, 4, "affine")["rel_l2"] < \
        quantization_error(w, 4, "symmetric")["rel_l2"]
    q, scale, zero, meta = quantize_affine(w, 4)
    assert q.min() >= 0 and q.max() <= 15
    assert torch.allclose(dequantize_affine(q, scale, zero, meta), w, atol=float(scale.max()))


def test_nf4_grid_matches_published_table():
    """Deriving the grid from the normal quantiles must reproduce the constants."""
    derived = nf4_levels_from_normal(16)
    assert derived.numel() == 16
    assert float((derived - NF4_LEVELS).abs().max()) < 1e-3
    assert bool((NF4_LEVELS == 0).any())          # zero must be exactly representable
    assert float(NF4_LEVELS.min()) == -1.0 and float(NF4_LEVELS.max()) == 1.0
    assert bool((NF4_LEVELS.sort().values == NF4_LEVELS).all())


def test_nf4_preserves_exact_zeros():
    z = torch.zeros(128)
    idx, scale, meta = quantize_nf4(z, 64)
    assert bool((dequantize_nf4(idx, scale, meta) == 0).all())


def test_nf4_beats_uniform_int4_on_gaussian_weights():
    """The whole claim of NF4: normal-shaped levels fit normal-shaped weights."""
    torch.manual_seed(3)
    w = torch.randn(64, 256)
    assert quantization_error(w, 4, "nf4", 64)["rel_l2"] < \
        quantization_error(w, 4, "symmetric", 64)["rel_l2"]


def test_nf4_requires_divisible_size():
    with pytest.raises(ValueError):
        quantize_nf4(torch.randn(100), group_size=64)


def test_int4_packing_roundtrip():
    torch.manual_seed(4)
    q = torch.randint(0, 16, (1001,), dtype=torch.uint8)
    packed = pack_int4(q)
    assert packed.numel() == 501                  # odd length is padded
    assert bool((unpack_int4(packed, 1001) == q).all())
    with pytest.raises(ValueError):
        pack_int4(torch.tensor([16], dtype=torch.uint8))


@pytest.mark.parametrize("scheme,bits,gs", [
    ("symmetric", 8, None), ("affine", 8, None), ("symmetric", 4, 64), ("nf4", 4, 64)])
def test_quantized_linear_output_error_is_small(scheme, bits, gs):
    torch.manual_seed(5)
    lin = nn.Linear(256, 128, bias=True)
    x = torch.randn(8, 256)
    with torch.no_grad():
        ref = lin(x)
    ql = QuantizedLinear.from_linear(lin, bits, scheme, gs)
    with torch.no_grad():
        err = float((ql(x) - ref).norm() / ref.norm())
    assert err < (0.02 if bits == 8 else 0.15)
    assert torch.equal(ql.bias.data, lin.bias.data)


@pytest.mark.parametrize("scheme,bits,gs,max_bits", [
    ("symmetric", 8, None, 8.2), ("symmetric", 4, 64, 4.6), ("nf4", 4, 64, 4.6)])
def test_quantized_linear_memory_accounting(scheme, bits, gs, max_bits):
    lin = nn.Linear(512, 256, bias=False)
    ql = QuantizedLinear.from_linear(lin, bits, scheme, gs)
    mb = ql.memory_bytes()
    assert mb["effective_bits"] <= max_bits
    assert mb["compression"] == pytest.approx(32 / mb["effective_bits"], rel=1e-6)
    assert mb["fp32_bytes"] == 512 * 256 * 4


def test_quantize_model_replaces_linears_and_skips_head():
    torch.manual_seed(6)
    m = GPT(GPTConfig(**SMALL)).eval()
    idx = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        ref, _ = m(idx)
    n_linear = sum(isinstance(x, nn.Linear) for x in m.modules())
    quantize_model(m, bits=8, scheme="symmetric")
    n_quant = sum(isinstance(x, QuantizedLinear) for x in m.modules())
    assert n_quant == n_linear - 1                # lm_head is skipped by default
    assert isinstance(m.lm_head, nn.Linear)
    with torch.no_grad():
        got, _ = m(idx)
    assert float((got - ref).norm() / ref.norm()) < 0.05


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        QuantizedLinear.from_linear(nn.Linear(8, 8), scheme="nope")
    with pytest.raises(ValueError):
        quantization_error(torch.randn(4, 8), scheme="nope")


def test_kv_cache_quantizes_better_per_channel_when_outliers_are_per_channel():
    """Keys have persistent outlier channels, so the scale axis matters."""
    torch.manual_seed(7)
    k = torch.randn(1, 4, 128, 64)
    k[..., 7] *= 25
    per_token = kv_cache_quant_error(k, 8, per_token=True)["rel_l2"]
    per_chan = kv_cache_quant_error(k, 8, per_token=False)["rel_l2"]
    assert per_chan < per_token


# --------------------------------------------------------------------- LoRA


def _lora_model(r=4, quantized=False, **cfg_over):
    torch.manual_seed(8)
    m = GPT(GPTConfig(**{**SMALL, **cfg_over})).eval()
    if quantized:
        quantize_model(m, bits=4, scheme="nf4", group_size=64)
    return apply_lora(m, LoRAConfig(r=r))


def test_lora_is_identity_at_init():
    """B=0 means the adapted model starts *exactly* at the base model."""
    torch.manual_seed(8)
    base = GPT(GPTConfig(**SMALL)).eval()
    idx = torch.randint(0, 64, (2, 8))
    adapted = _lora_model()
    with torch.no_grad():
        ref, _ = base(idx)
        got, _ = adapted(idx)
    assert torch.allclose(got, ref, atol=1e-6)


def test_lora_A_is_nonzero_so_gradients_can_flow():
    m = _lora_model()
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            assert float(mod.lora_A.detach().abs().max()) > 0
            assert float(mod.lora_B.detach().abs().max()) == 0


def test_merge_is_exact_and_reversible():
    m = _lora_model()
    for n, p in m.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.05)
    idx = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        unmerged, _ = m(idx)
        merge_all(m)
        merged, _ = m(idx)
        assert torch.allclose(merged, unmerged, atol=1e-5)
        unmerge_all(m)
        restored, _ = m(idx)
    assert torch.allclose(restored, unmerged, atol=1e-5)


def test_only_adapters_receive_gradients():
    m = _lora_model()
    idx = torch.randint(0, 64, (2, 8))
    m(idx, targets=idx)[1].backward()
    for n, p in m.named_parameters():
        assert (p.grad is not None) == ("lora_" in n), n


def test_alpha_over_r_keeps_the_update_scale_constant():
    """Doubling r must not double the update magnitude -- that is what alpha is for."""
    outs = []
    for r in (4, 8):
        torch.manual_seed(8)
        base = GPT(GPTConfig(**SMALL)).eval()
        m = apply_lora(base, LoRAConfig(r=r))
        for mod in m.modules():
            if isinstance(mod, LoRALinear):
                assert mod.scaling == 2.0            # alpha defaults to 2r
                # A fixed-magnitude B gives an update that grows like sqrt(r),
                # not like r, because scaling cancels the explicit r dependence.
                mod.lora_B.data.fill_(0.01)
        idx = torch.randint(0, 64, (1, 6))
        with torch.no_grad():
            outs.append(float(m(idx)[0].abs().mean()))
    assert 0.3 < outs[0] / outs[1] < 3.0


def test_lora_state_dict_is_small():
    m = _lora_model()
    sd = lora_state_dict(m)
    assert sd and all("lora_" in k for k in sd)
    adapter_bytes = sum(v.numel() * 4 for v in sd.values())
    full_bytes = sum(p.numel() * 4 for p in m.parameters())
    assert adapter_bytes < full_bytes / 2


def test_trainable_summary_counts_quantized_buffers():
    """Without this, a QLoRA model reports ~90% trainable, which is nonsense."""
    full = trainable_summary(GPT(GPTConfig(**SMALL)))
    lora = trainable_summary(_lora_model(r=8))
    qlora = trainable_summary(_lora_model(r=8, quantized=True))
    assert full["trainable_pct"] == 100.0
    assert lora["trainable_pct"] < 30.0
    assert qlora["total"] == lora["total"]            # same weight *count*
    assert qlora["weight_bytes"] < lora["weight_bytes"] / 2   # far fewer bytes
    assert lora["adam_state_mb_lora"] < full["adam_state_mb_full"] / 3


def test_qlora_gradients_flow_through_a_4bit_base():
    m = _lora_model(r=8, quantized=True)
    idx = torch.randint(0, 64, (2, 8))
    m(idx, targets=idx)[1].backward()
    trainable = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    assert trainable
    assert all(p.grad is not None and float(p.grad.detach().abs().sum()) >= 0 for _, p in trainable)


def test_merging_into_a_quantized_base_is_refused():
    m = _lora_model(r=8, quantized=True)
    with pytest.raises(RuntimeError):
        merge_all(m)


def test_lora_rejects_zero_rank():
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 4), r=0)


def test_train_norms_option():
    torch.manual_seed(8)
    m = apply_lora(GPT(GPTConfig(**SMALL)), LoRAConfig(r=4, train_norms=True))
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert any("norm" in n for n in names)
    assert any("lora_" in n for n in names)
