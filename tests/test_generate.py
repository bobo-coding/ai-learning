import math

import pytest
import torch
import torch.nn.functional as F

from minigpt.config import GPTConfig
from minigpt.generate import (apply_temperature, generate, min_p_filter,
                              repetition_penalty, sample_from_logits,
                              sequence_logprob, speculative_decode, token_logprobs,
                              top_k_filter, top_p_filter)
from minigpt.model import GPT

SMALL = dict(vocab_size=64, block_size=48, n_layer=2, n_head=2, n_embd=32, n_kv_head=1)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(GPTConfig(**SMALL)).eval()


# ---------------------------------------------------------------- processors


def test_temperature_scales_logits():
    lg = torch.tensor([[2.0, 1.0]])
    assert torch.allclose(apply_temperature(lg, 2.0), lg / 2)
    with pytest.raises(ValueError):
        apply_temperature(lg, 0.0)


def test_top_k_keeps_exactly_k():
    lg = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
    out = top_k_filter(lg.clone(), 2)
    assert int(torch.isfinite(out).sum()) == 2
    assert torch.equal(out[0, :2], lg[0, :2])
    # no-ops outside the useful range
    assert torch.equal(top_k_filter(lg.clone(), 0), lg)
    assert torch.equal(top_k_filter(lg.clone(), 99), lg)


def test_top_p_is_the_smallest_set_with_mass_at_least_p():
    """Nucleus sampling keeps the *smallest* set whose mass reaches p.

    So the token that pushes the cumulative mass over the threshold is kept.
    Getting this boundary wrong by one is the classic top-p bug and it silently
    changes the sampling distribution.  This matches HuggingFace's
    `TopPLogitsWarper` exactly.
    """
    lg = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
    probs = F.softmax(lg, -1)[0]
    assert probs[0] > 0.4                        # the top token alone exceeds p=0.4
    assert int(torch.isfinite(top_p_filter(lg.clone(), 0.4)).sum()) == 1

    p2 = float(probs[:2].sum())
    # just below the two-token mass: two tokens suffice
    assert int(torch.isfinite(top_p_filter(lg.clone(), p2 - 1e-4)).sum()) == 2
    # just above it: two are not enough, so the third is pulled in
    assert int(torch.isfinite(top_p_filter(lg.clone(), p2 + 1e-4)).sum()) == 3


def test_top_p_always_keeps_at_least_one_token():
    """A distribution more peaked than p must still be sampleable."""
    peak = torch.tensor([[50.0, 0.0, 0.0]])
    assert int(torch.isfinite(top_p_filter(peak.clone(), 0.5)).sum()) == 1


def test_min_p_is_relative_to_the_top_token():
    lg = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    probs = F.softmax(lg, -1)[0]
    out = min_p_filter(lg.clone(), 0.3)
    kept = torch.isfinite(out)[0]
    for i, k in enumerate(kept):
        assert bool(k) == bool(probs[i] >= 0.3 * probs.max())


def test_repetition_penalty_handles_both_signs():
    out = repetition_penalty(torch.tensor([[2.0, -2.0, 1.0]]), torch.tensor([[0, 1]]), 2.0)
    # positive logits are divided (reduced), negative ones multiplied (reduced too)
    assert out.tolist() == [[1.0, -4.0, 1.0]]
    untouched = repetition_penalty(torch.tensor([[2.0, -2.0]]), torch.tensor([[0]]), 1.0)
    assert untouched.tolist() == [[2.0, -2.0]]


def test_greedy_sampling_is_argmax():
    lg = torch.tensor([[1.0, 5.0, 2.0]])
    assert int(sample_from_logits(lg, greedy=True)) == 1
    assert int(sample_from_logits(lg, temperature=0)) == 1


def test_top_k_with_ties_keeps_more_than_k():
    """Documented edge case: `top_k` cannot break exact ties.

    The filter keeps everything at or above the k-th largest logit, so a
    uniform distribution survives top_k=2 intact.  That is the right call --
    arbitrarily dropping tied tokens would bias sampling -- but it means
    "top_k=2" is an upper bound on selectivity, not a guarantee of 2 tokens.
    """
    flat = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    assert int(torch.isfinite(top_k_filter(flat.clone(), 2)).sum()) == 4


def test_sampling_respects_the_filter():
    """A filtered-out token must never be drawn, however many samples we take."""
    lg = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    g = torch.Generator().manual_seed(0)
    draws = {int(sample_from_logits(lg, top_k=2, generator=g)) for _ in range(300)}
    assert draws == {0, 1}


# ------------------------------------------------------------------ decoding


def test_greedy_cache_matches_no_cache(model):
    prompt = torch.randint(0, 64, (3, 5))
    a = generate(model, prompt, 20, greedy=True, use_cache=True)
    b = generate(model, prompt, 20, greedy=True, use_cache=False)
    assert torch.equal(a, b)


def test_sampled_cache_matches_no_cache(model):
    prompt = torch.randint(0, 64, (2, 4))
    g1 = torch.Generator().manual_seed(4)
    g2 = torch.Generator().manual_seed(4)
    a = generate(model, prompt, 15, temperature=0.9, top_k=10, use_cache=True, generator=g1)
    b = generate(model, prompt, 15, temperature=0.9, top_k=10, use_cache=False, generator=g2)
    assert torch.equal(a, b)


def test_generate_preserves_the_prompt(model):
    prompt = torch.randint(0, 64, (2, 6))
    out = generate(model, prompt, 10, greedy=True)
    assert torch.equal(out[:, :6], prompt)
    assert out.shape == (2, 16)


def test_eos_freezes_finished_rows(model):
    """Once a row emits EOS, every later token in that row must also be EOS."""
    prompt = torch.randint(0, 64, (4, 3))
    # pick the token greedy decoding actually produces first, so EOS definitely fires
    logits, _ = model(prompt)
    eos = int(logits[0, -1].argmax())
    out = generate(model, prompt, 15, greedy=True, eos_id=eos)
    assert (out[:, 3:] == eos).any()
    for row in out:
        hits = (row == eos).nonzero().flatten().tolist()
        hits = [h for h in hits if h >= 3]
        if hits:
            assert all(int(row[i]) == eos for i in range(hits[0], len(row)))


def test_generate_stops_at_block_size(model):
    prompt = torch.randint(0, 64, (1, 40))
    out = generate(model, prompt, 100, greedy=True)
    assert out.shape[1] <= model.cfg.block_size


def test_generate_restores_training_mode(model):
    model.train()
    generate(model, torch.randint(0, 64, (1, 3)), 2, greedy=True)
    assert model.training


# ------------------------------------------------------- speculative decoding


@pytest.fixture
def target_draft():
    torch.manual_seed(2)
    target = GPT(GPTConfig(vocab_size=8, block_size=16, n_layer=2, n_head=2, n_embd=32)).eval()
    torch.manual_seed(99)
    draft = GPT(GPTConfig(vocab_size=8, block_size=16, n_layer=1, n_head=2, n_embd=16)).eval()
    return target, draft


def test_speculative_decoding_is_distribution_preserving(target_draft):
    """The central claim: a *bad* draft must not change the sampled distribution.

    We run many single-token generations and compare the empirical distribution
    to the target model's exact next-token distribution with a chi-square test.
    If the accept/reject rule or the residual distribution were wrong, this is
    the test that catches it -- the output would still look plausible.
    """
    target, draft = target_draft
    prompt = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        logits, _ = target(prompt)
        p = F.softmax(logits[0, -1].float(), -1)

    N = 6000
    g = torch.Generator().manual_seed(0)
    counts = torch.zeros(8)
    for _ in range(N):
        out, _ = speculative_decode(target, draft, prompt, 1, lookahead=3, generator=g)
        counts[int(out[0, 3])] += 1

    expected = p * N
    chi2 = float((((counts - expected) ** 2) / expected).sum())
    # 7 degrees of freedom; the 99.9th percentile of chi2(7) is ~24.3
    assert chi2 < 24.3, f"chi2={chi2:.1f}: speculative decoding is biased"


def test_speculative_accepts_everything_when_draft_equals_target(target_draft):
    target, _ = target_draft
    prompt = torch.tensor([[1, 2, 3]])
    g = torch.Generator().manual_seed(1)
    acc = prop = 0
    for _ in range(40):
        _, st = speculative_decode(target, target, prompt, 4, lookahead=4, generator=g)
        acc += st["accepted"]
        prop += st["proposed"]
    assert acc == prop                      # p/q == 1 everywhere, so always accept


def test_speculative_produces_the_requested_length(target_draft):
    target, draft = target_draft
    prompt = torch.tensor([[1, 2, 3]])
    g = torch.Generator().manual_seed(3)
    out, st = speculative_decode(target, draft, prompt, 6, lookahead=3, generator=g)
    assert out.shape[1] >= prompt.shape[1] + 6
    assert 0.0 <= st["acceptance_rate"] <= 1.0
    assert st["tokens_per_round"] > 0


def test_speculative_rejects_batched_input(target_draft):
    target, draft = target_draft
    with pytest.raises(ValueError):
        speculative_decode(target, draft, torch.zeros(2, 3, dtype=torch.long), 2)


# ------------------------------------------------------------------- scoring


def test_token_logprobs_matches_manual_gather(model):
    idx = torch.randint(0, 64, (2, 9))
    got = token_logprobs(model, idx)
    with torch.no_grad():
        logits, _ = model(idx)
    want = torch.log_softmax(logits[:, :-1].float(), -1).gather(
        -1, idx[:, 1:].unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got, want, atol=1e-6)


def test_sequence_logprob_sum_and_mean(model):
    idx = torch.randint(0, 64, (2, 9))
    lp = token_logprobs(model, idx)
    mask = torch.zeros_like(lp)
    mask[:, 3:6] = 1
    assert torch.allclose(sequence_logprob(model, idx, mask), lp[:, 3:6].sum(-1), atol=1e-6)
    assert torch.allclose(sequence_logprob(model, idx, mask, average=True),
                          lp[:, 3:6].mean(-1), atol=1e-6)


def test_cross_entropy_equals_negative_mean_logprob(model):
    """The identity that makes perplexity = exp(loss) true."""
    idx = torch.randint(0, 64, (2, 9))
    lp = token_logprobs(model, idx)
    _, loss = model(idx[:, :-1], targets=idx[:, 1:])
    assert abs(loss.item() + lp.mean().item()) < 1e-5
    assert abs(math.exp(loss.item()) - math.exp(-lp.mean().item())) < 1e-4
