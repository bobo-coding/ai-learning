import numpy as np
import pytest
import torch

from minigpt.data import (TokenBatcher, iter_eval_batches, pack_documents,
                          read_tokens, token_dtype, train_val_split, write_tokens)
from minigpt.optim import AdamW, SGD, clip_grad_norm, wsd_lr
from minigpt.utils import cosine_lr


# ---------------------------------------------------------------- optimizers


def _run(make_opt, steps=30, wd=0.1):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(6, 4, dtype=torch.float64))
    q = torch.nn.Parameter(torch.randn(4, dtype=torch.float64))
    opt = make_opt([{"params": [p], "weight_decay": wd},
                    {"params": [q], "weight_decay": 0.0}])
    torch.manual_seed(5)
    grads = [(torch.randn(6, 4, dtype=torch.float64), torch.randn(4, dtype=torch.float64))
             for _ in range(steps)]
    for gp, gq in grads:
        p.grad, q.grad = gp.clone(), gq.clone()
        opt.step()
    return p.detach(), q.detach()


def test_adamw_matches_torch():
    """Our AdamW must agree with torch's to float64 rounding, over many steps.

    The 2-D (weight-decayed) tensor comes out bitwise identical; the 1-D one
    differs by about one ULP because torch's `_foreach_` path fuses the ops
    slightly differently.  30 steps of accumulation without drift is the real
    statement -- a wrong bias correction or a coupled weight decay would show up
    as a relative error of 1e-2, not 1e-16.
    """
    mine = _run(lambda g: AdamW(g, lr=3e-3, betas=(0.9, 0.95), eps=1e-8))
    ref = _run(lambda g: torch.optim.AdamW(g, lr=3e-3, betas=(0.9, 0.95), eps=1e-8))
    assert torch.equal(mine[0], ref[0])
    for a, b in zip(mine, ref):
        assert float((a - b).abs().max()) < 8 * 2.3e-16 * float(b.abs().max())


def test_adamw_weight_decay_is_decoupled():
    """With zero gradient, AdamW must still shrink the weight by exactly (1 - lr*wd)."""
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.float64))
    opt = AdamW([p], lr=0.1, weight_decay=0.5)
    p.grad = torch.zeros(4, dtype=torch.float64)
    opt.step()
    assert torch.allclose(p.detach(), torch.full((4,), 1 - 0.1 * 0.5, dtype=torch.float64))


def test_adamw_is_scale_invariant_in_the_gradient():
    """Multiplying every gradient by 100 barely changes the step -- that is Adam."""
    def final(scale):
        torch.manual_seed(1)
        p = torch.nn.Parameter(torch.zeros(8, dtype=torch.float64))
        opt = AdamW([p], lr=1e-2, weight_decay=0.0)
        torch.manual_seed(2)
        for _ in range(20):
            p.grad = torch.randn(8, dtype=torch.float64) * scale
            opt.step()
        return p.detach()
    assert torch.allclose(final(1.0), final(100.0), atol=1e-6)


@pytest.mark.parametrize("nesterov", [False, True])
def test_sgd_matches_torch(nesterov):
    mine = _run(lambda g: SGD(g, lr=1e-2, momentum=0.9, nesterov=nesterov), wd=0.05)
    ref = _run(lambda g: torch.optim.SGD(g, lr=1e-2, momentum=0.9, nesterov=nesterov), wd=0.05)
    for a, b in zip(mine, ref):
        assert torch.equal(a, b)


def test_adamw_rejects_bad_hyperparameters():
    p = torch.nn.Parameter(torch.zeros(2))
    for bad in [dict(lr=-1.0), dict(eps=-1.0), dict(betas=(1.0, 0.9)), dict(betas=(0.9, 1.5))]:
        with pytest.raises(ValueError):
            AdamW([p], **bad)


def test_clip_grad_norm_matches_torch_and_is_global():
    torch.manual_seed(2)
    ps = [torch.nn.Parameter(torch.randn(5, 3, dtype=torch.float64)) for _ in range(3)]
    for p in ps:
        p.grad = torch.randn_like(p) * 4
    ref = [torch.nn.Parameter(p.detach().clone()) for p in ps]
    for a, b in zip(ps, ref):
        b.grad = a.grad.clone()

    mine = clip_grad_norm(ps, 1.0)
    theirs = float(torch.nn.utils.clip_grad_norm_(ref, 1.0))
    assert abs(mine - theirs) < 1e-10
    for a, b in zip(ps, ref):
        assert torch.allclose(a.grad, b.grad, atol=1e-12)
    # The *global* norm, not any per-tensor norm, is what lands at max_norm.
    # It lands just below, because the scale factor is max_norm/(norm + 1e-6) --
    # the same epsilon torch uses to avoid dividing by zero.
    post = float(torch.cat([p.grad.flatten() for p in ps]).norm())
    assert 1.0 - 1e-6 < post <= 1.0


def test_clip_grad_norm_leaves_small_gradients_alone():
    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.full((4,), 1e-4)
    before = p.grad.clone()
    clip_grad_norm([p], 1.0)
    assert torch.equal(before, p.grad)


def test_clip_grad_norm_handles_no_gradients():
    p = torch.nn.Parameter(torch.zeros(4))
    assert clip_grad_norm([p], 1.0) == 0.0


# ----------------------------------------------------------------- schedules


def test_cosine_schedule_endpoints():
    kw = dict(base_lr=1.0, warmup=10, total=100, min_ratio=0.1)
    assert cosine_lr(0, **kw) == pytest.approx(0.1)          # first step, not zero
    assert cosine_lr(9, **kw) == pytest.approx(1.0)          # peak at end of warmup
    assert cosine_lr(10, **kw) == pytest.approx(1.0)
    assert cosine_lr(55, **kw) == pytest.approx(0.55, abs=0.01)
    assert cosine_lr(200, **kw) == pytest.approx(0.1)        # clamped past the end


def test_cosine_is_monotone_after_warmup():
    kw = dict(base_lr=1.0, warmup=10, total=100)
    vals = [cosine_lr(s, **kw) for s in range(10, 100)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))


def test_wsd_schedule_has_a_stable_plateau():
    kw = dict(base_lr=1.0, warmup=10, total=100, decay_frac=0.1)
    assert wsd_lr(9, **kw) == pytest.approx(1.0)
    assert wsd_lr(50, **kw) == pytest.approx(1.0)            # plateau
    assert wsd_lr(89, **kw) == pytest.approx(1.0)
    assert wsd_lr(95, **kw) == pytest.approx(0.5)            # linear decay
    assert wsd_lr(99, **kw) == pytest.approx(0.1)


# --------------------------------------------------------------------- data


def test_token_dtype_choice():
    assert token_dtype(50257) == np.uint16
    assert token_dtype(65535) == np.uint16
    assert token_dtype(65536) == np.uint32


def test_write_read_roundtrip(tmp_path):
    ids = list(range(1000))
    p = write_tokens(ids, tmp_path / "t.bin", 1024)
    back = read_tokens(p, 1024)
    assert back.dtype == np.uint16
    assert back.tolist() == ids


def test_write_rejects_out_of_range_token(tmp_path):
    with pytest.raises(ValueError):
        write_tokens([0, 1, 5000], tmp_path / "t.bin", 100)


def test_batcher_shapes_and_shift():
    tokens = np.arange(1000, dtype=np.uint16)
    b = TokenBatcher(tokens, block_size=16, batch_size=4, seed=0)
    x, y = b()
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert x.dtype == torch.int64
    # y must be x shifted left by exactly one token
    assert torch.equal(y[:, :-1], x[:, 1:])
    # and the windows must be real slices of the corpus
    for row in x:
        assert torch.equal(row, torch.arange(int(row[0]), int(row[0]) + 16))


def test_batcher_is_reproducible_and_varies():
    tokens = np.arange(500, dtype=np.uint16)
    a = TokenBatcher(tokens, 8, 4, seed=7)()
    b = TokenBatcher(tokens, 8, 4, seed=7)()
    c = TokenBatcher(tokens, 8, 4, seed=8)()
    assert torch.equal(a[0], b[0])
    assert not torch.equal(a[0], c[0])


def test_batcher_rejects_tiny_corpus():
    with pytest.raises(ValueError):
        TokenBatcher(np.arange(5, dtype=np.uint16), block_size=16, batch_size=1)


def test_eval_batches_are_deterministic_and_cover_the_corpus():
    tokens = np.arange(200, dtype=np.uint16)
    runs = [[(x.tolist(), y.tolist()) for x, y in iter_eval_batches(tokens, 16, 3)]
            for _ in range(2)]
    assert runs[0] == runs[1]
    # Each yielded batch holds 3 windows, so consecutive batches start 3*16
    # apart; within a batch the windows step by the stride.
    batches = list(iter_eval_batches(tokens, 16, 3))
    assert [int(x[0][0]) for x, _ in batches] == [0, 48, 96, 144]
    assert [int(r[0]) for r in batches[0][0]] == [0, 16, 32]
    # every window is a contiguous slice, and y is x shifted by one
    for x, y in batches:
        assert torch.equal(y[:, :-1], x[:, 1:])


def test_eval_batches_stride_overlaps():
    tokens = np.arange(100, dtype=np.uint16)
    full = [int(x[0][0]) for x, _ in iter_eval_batches(tokens, 16, 1, stride=4)]
    assert full[:3] == [0, 4, 8]


def test_pack_documents():
    docs = [[1, 2], [3], [4, 5, 6]]
    rows = pack_documents(docs, eos_id=0, block_size=3)
    # flat = 1 2 0 3 0 4 5 6 0  -> 3 rows of 3
    assert rows.tolist() == [[1, 2, 0], [3, 0, 4], [5, 6, 0]]


def test_pack_documents_needs_enough_tokens():
    with pytest.raises(ValueError):
        pack_documents([[1]], eos_id=0, block_size=10)
    padded = pack_documents([[1]], eos_id=0, block_size=10, drop_last=False)
    assert padded.shape == (1, 10)


def test_train_val_split_is_positional_not_random():
    tokens = np.arange(100, dtype=np.uint16)
    tr, va = train_val_split(tokens, 0.1)
    assert tr.tolist() == list(range(90))
    assert va.tolist() == list(range(90, 100))
    with pytest.raises(ValueError):
        train_val_split(tokens, 0.0001)
