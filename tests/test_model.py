import math

import pytest
import torch
import torch.nn.functional as F

from minigpt.config import PRESETS, GPTConfig, preset
from minigpt.model import GPT, LayerNorm, RMSNorm

SMALL = dict(vocab_size=41, block_size=24, n_layer=2, n_head=2, n_embd=32)


def test_rmsnorm_matches_formula():
    x = torch.randn(3, 7, 16, dtype=torch.float64)
    got = RMSNorm(16).double()(x)
    want = x / (x.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
    assert torch.allclose(got, want, atol=1e-12)


def test_rmsnorm_has_no_mean_subtraction():
    """The defining difference from LayerNorm: a constant shift is NOT removed."""
    x = torch.randn(1, 4, 8, dtype=torch.float64)
    rn = RMSNorm(8).double()
    assert not torch.allclose(rn(x + 10.0), rn(x), atol=1e-6)
    ln = LayerNorm(8).double()
    assert torch.allclose(ln(x + 10.0), ln(x), atol=1e-10)


def test_rmsnorm_upcasts_only_low_precision():
    """fp16/bf16 must be accumulated in fp32; fp64 must stay fp64."""
    x = torch.randn(2, 8, dtype=torch.float64)
    assert RMSNorm(8).double()(x).dtype == torch.float64
    xb = torch.randn(2, 8, dtype=torch.bfloat16)
    assert RMSNorm(8).to(torch.bfloat16)(xb).dtype == torch.bfloat16


def test_layernorm_matches_functional():
    x = torch.randn(3, 5, 16, dtype=torch.float64)
    assert torch.allclose(LayerNorm(16).double()(x), F.layer_norm(x, (16,)), atol=1e-12)
    assert LayerNorm(16, bias=False).bias is None


@pytest.mark.parametrize("name", [n for n in PRESETS if n != "llama7b"])
def test_analytic_param_count_matches_module(name):
    """The parameter formula in GPTConfig must match the real module exactly."""
    cfg = preset(name)
    m = GPT(cfg)
    pc = cfg.param_count()
    assert m.num_params() == pc["total"]
    assert m.num_params(non_embedding=True) == pc["non_embedding"]


def test_llama7b_param_count():
    """Sanity-check the formula against the published 6.74B figure."""
    total = preset("llama7b").param_count()["total"]
    assert abs(total - 6.74e9) / 6.74e9 < 0.01


def test_swiglu_hidden_dim_matches_llama():
    assert preset("llama7b").hidden_dim == 11008


def test_weight_tying_shares_storage():
    m = GPT(GPTConfig(**SMALL, tie_embeddings=True))
    assert m.lm_head.weight is m.tok_emb.weight
    m2 = GPT(GPTConfig(**SMALL, tie_embeddings=False))
    assert m2.lm_head.weight is not m2.tok_emb.weight
    assert m2.num_params() > m.num_params()


def test_initial_loss_is_near_log_vocab():
    """At init the model is ~uniform, so next-token loss ~= ln(V).

    The first number to check on any new training run.  Much higher means the
    init scale is wrong; much lower means labels are leaking.

    Note that `targets` are NOT shifted inside `forward` -- the caller aligns
    them, exactly as the data loader does.  Passing `targets=idx` unshifted
    would ask the model to predict the token it can already see.
    """
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=1000, block_size=32, n_layer=4, n_head=4, n_embd=128)
    m = GPT(cfg).eval()
    idx = torch.randint(0, 1000, (8, 32))
    _, loss = m(idx[:, :-1], targets=idx[:, 1:])
    assert abs(loss.item() - math.log(1000)) < 0.4


def test_tied_embeddings_bias_toward_copying_at_init():
    """Documents why an unshifted-target loss looks suspiciously good.

    With tied embeddings the logits are `residual @ E^T`, and the residual
    stream at position t still contains `E[x_t]`.  So at init the largest logit
    is the *current* token, and a model scored on unshifted targets appears to
    have learned something before any training.  This is a real trap when
    debugging a data pipeline: a loss well below ln(V) at step 0 means the
    targets are misaligned.
    """
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=1000, block_size=32, n_layer=4, n_head=4, n_embd=128)
    m = GPT(cfg).eval()
    idx = torch.randint(0, 1000, (8, 32))
    _, shifted = m(idx[:, :-1], targets=idx[:, 1:])
    _, unshifted = m(idx, targets=idx)
    assert unshifted.item() < shifted.item() - 1.0


def test_residual_projections_are_downscaled():
    """GPT-2's 1/sqrt(2L) init on residual outputs keeps deep models trainable."""
    cfg = GPTConfig(**{**SMALL, "n_layer": 8}, init_std=0.02)
    m = GPT(cfg)
    expected = 0.02 / math.sqrt(2 * 8)
    for name, p in m.named_parameters():
        if name.endswith("o_proj.weight") or name.endswith("down.weight"):
            assert p.std().item() < 0.02 * 0.75
            assert abs(p.std().item() - expected) / expected < 0.35


def test_forward_shapes_and_loss():
    m = GPT(GPTConfig(**SMALL)).eval()
    idx = torch.randint(0, 41, (3, 12))
    logits, loss = m(idx)
    assert logits.shape == (3, 12, 41)
    assert loss is None
    _, loss = m(idx, targets=idx)
    assert loss.dim() == 0 and loss.item() > 0


def test_loss_mask_selects_positions():
    torch.manual_seed(1)
    m = GPT(GPTConfig(**SMALL)).eval()
    idx = torch.randint(0, 41, (2, 12))
    logits, _ = m(idx)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[:, 5] = True
    _, masked = m(idx, targets=idx, loss_mask=mask)
    manual = F.cross_entropy(logits[:, 5].float(), idx[:, 5])
    assert torch.allclose(masked, manual, atol=1e-6)


def test_ignore_index_is_minus_100():
    m = GPT(GPTConfig(**SMALL)).eval()
    idx = torch.randint(0, 41, (2, 6))
    tgt = idx.clone()
    tgt[:, :3] = -100
    _, a = m(idx, targets=tgt)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 3:] = True
    _, b = m(idx, targets=idx, loss_mask=mask)
    assert torch.allclose(a, b, atol=1e-6)


def test_model_is_causal():
    torch.manual_seed(2)
    m = GPT(GPTConfig(**SMALL)).eval()
    idx = torch.randint(0, 41, (1, 12))
    o1, _ = m(idx)
    idx2 = idx.clone()
    idx2[0, 7] = (int(idx2[0, 7]) + 5) % 41
    o2, _ = m(idx2)
    assert torch.equal(o1[:, :7], o2[:, :7])
    assert not torch.allclose(o1[:, 7:], o2[:, 7:])


@pytest.mark.parametrize("pos", ["rope", "learned", "none"])
@pytest.mark.parametrize("mlp", ["swiglu", "gelu"])
@pytest.mark.parametrize("norm", ["rms", "layer"])
def test_kv_cache_equals_full_forward(pos, mlp, norm):
    torch.manual_seed(3)
    cfg = GPTConfig(**{**SMALL, "n_kv_head": 1}, pos=pos, mlp=mlp, norm=norm,
                    bias=(norm == "layer"))
    m = GPT(cfg).double().eval()
    idx = torch.randint(0, 41, (1, 12))
    full, _ = m(idx)
    caches = m.make_caches(1, 24, dtype=torch.float64)
    parts = [m(idx[:, :5], caches=caches)[0]]
    for t in range(5, 12):
        parts.append(m(idx[:, t:t + 1], caches=caches)[0])
    assert torch.allclose(torch.cat(parts, 1), full, atol=1e-9, rtol=1e-7)


def test_block_size_is_enforced():
    m = GPT(GPTConfig(**SMALL))
    with pytest.raises(ValueError):
        m(torch.zeros(1, 25, dtype=torch.long))


def test_optimizer_groups_split_by_dimension():
    m = GPT(GPTConfig(**SMALL))
    opt = m.configure_optimizers(lr=1e-3, weight_decay=0.1)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    n = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n == m.num_params()          # tied weights counted exactly once


def test_config_validation():
    with pytest.raises(ValueError):
        GPTConfig(n_embd=30, n_head=4)
    with pytest.raises(ValueError):
        GPTConfig(n_embd=32, n_head=4, n_kv_head=3)
    for bad in [dict(norm="x"), dict(pos="x"), dict(mlp="x")]:
        with pytest.raises(ValueError):
            GPTConfig(**bad)


def test_flops_and_kv_cache_accounting():
    cfg = preset("llama7b")
    fl = cfg.flops_per_token()
    n = cfg.param_count()["total"]
    # The 2N rule of thumb should be within ~10% of the dense term.
    assert 0.9 < fl["dense"] / (2 * n) < 1.15
    assert 0.0 < fl["fraction_attention"] < 0.2
    # attention's share must grow with context length
    assert cfg.flops_per_token(32768)["fraction_attention"] > fl["fraction_attention"]
    # GQA/MQA shrink the KV cache proportionally to n_kv_head
    full = cfg.kv_cache_bytes(1)
    mqa = preset("llama7b", n_kv_head=1).kv_cache_bytes(1)
    assert full / mqa == cfg.n_head


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(4)
    m = GPT(GPTConfig(**SMALL)).eval()
    idx = torch.randint(0, 41, (2, 8))
    before, _ = m(idx)
    p = tmp_path / "m.pt"
    m.save(p)
    m2 = GPT.load(p).eval()
    after, _ = m2(idx)
    assert torch.equal(before, after)


def test_crop_block_size():
    m = GPT(GPTConfig(**SMALL, pos="learned"))
    m.crop_block_size(12)
    assert m.cfg.block_size == 12
    assert m.pos_emb.weight.shape[0] == 12
    m(torch.zeros(1, 12, dtype=torch.long))


def test_estimate_mfu_is_a_fraction():
    m = GPT(GPTConfig(**SMALL))
    mfu = m.estimate_mfu(tokens_per_step=1024, dt=0.01, peak_flops=1e12)
    assert 0 < mfu < 1
