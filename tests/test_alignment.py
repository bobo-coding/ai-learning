"""Tests for SFT, DPO and GRPO -- the loss functions and the data plumbing."""

import math
import random

import pytest
import torch

from minigpt.config import GPTConfig
from minigpt.dpo import (dpo_loss, freeze_reference, masked_logprob_sum, simpo_loss)
from minigpt.grpo import (extract_completion, grpo_loss, group_advantages, kl_k3,
                          per_token_logprobs, sample_rollouts, zero_variance_fraction)
from minigpt.model import GPT
from minigpt.sft import (CHAT_SPECIALS, ChatTemplate, END, collate,
                         resize_token_embeddings)
from minigpt.bpe import CharTokenizer
from minigpt.tasks import (GENERATORS, TASK_CHARS, accuracy_by_task, corrupt_answer,
                           make_preference_pairs, reward_exact, reward_shaped,
                           sample_examples, split_examples)


@pytest.fixture
def tok():
    return CharTokenizer(chars=TASK_CHARS, special_tokens=CHAT_SPECIALS)


@pytest.fixture
def template(tok):
    return ChatTemplate(tok)


# ------------------------------------------------------------------- tasks


@pytest.mark.parametrize("name", list(GENERATORS))
def test_every_task_generator_is_self_consistent(name):
    """The reference answer must actually be the answer, for 200 draws."""
    rng = random.Random(0)
    for _ in range(200):
        ex = GENERATORS[name](rng)
        assert ex.task == name
        assert reward_exact(ex, ex.answer) == 1.0
        assert ex.answer != ""


def test_task_charset_is_closed():
    """No example may use a character outside TASK_CHARS, or it becomes unencodable."""
    chars = set()
    for ex in sample_examples(4000, seed=1):
        chars |= set(ex.prompt) | set(ex.answer)
    assert chars <= set(TASK_CHARS)


def test_split_has_no_prompt_overlap():
    tr, va = split_examples(sample_examples(5000, seed=2), 0.1)
    assert len(tr) > 0 and len(va) > 0
    assert not (set(e.prompt for e in tr) & set(e.prompt for e in va))


def test_reward_exact_takes_the_first_line_only():
    ex = GENERATORS["add"](random.Random(0))
    assert reward_exact(ex, ex.answer) == 1.0
    assert reward_exact(ex, f"  {ex.answer}  \nmore text") == 1.0
    assert reward_exact(ex, ex.answer + "0") == 0.0
    assert reward_exact(ex, "") == 0.0


def test_reward_shaped_gives_partial_credit():
    ex = GENERATORS["reverse"](random.Random(3))
    assert reward_shaped(ex, ex.answer) == 1.0
    partial = reward_shaped(ex, ex.answer[:2])
    assert 0.0 < partial < 1.0
    assert reward_shaped(ex, ex.answer[:2]) >= reward_shaped(ex, ex.answer[:1])
    assert reward_shaped(ex, "zzzzz") < partial


def test_corruptions_are_always_wrong_and_plausible():
    rng = random.Random(4)
    for ex in sample_examples(2000, seed=5):
        bad = corrupt_answer(ex, rng)
        assert bad != ex.answer
        assert reward_exact(ex, bad) == 0.0
        assert bad != ""


def test_preference_pairs_and_accuracy_by_task():
    exs = sample_examples(80, seed=6)
    pairs = make_preference_pairs(exs, seed=7)
    assert len(pairs) == 80
    assert all(p["chosen"] != p["rejected"] for p in pairs)
    acc = accuracy_by_task(exs, [e.answer for e in exs])
    assert acc["overall"] == 1.0
    acc0 = accuracy_by_task(exs, ["" for _ in exs])
    assert acc0["overall"] == 0.0


# --------------------------------------------------------------- chat template


def test_template_renders_a_stable_header(template):
    assert template.prompt_text("hi") == "<|user|>\nhi<|end|>\n<|assistant|>\n"


def test_end_token_is_inside_the_loss_mask(template, tok):
    """The single most consequential SFT detail: the model must learn to stop."""
    e = template.encode("2 + 3 =", "5")
    trainable = [i for i, m in zip(e.ids, e.completion_mask) if m]
    assert tok.decode(trainable) == "5" + END


def test_prompt_tokens_are_not_in_the_loss_mask(template, tok):
    e = template.encode("2 + 3 =", "5")
    head = [i for i, m in zip(e.ids, e.completion_mask) if not m]
    assert tok.decode(head) == template.prompt_text("2 + 3 =")


def test_collate_shift_alignment(template):
    """Masked target positions must be exactly the completion tokens."""
    batch = [template.encode("2 + 3 =", "5"), template.encode("reverse cat =", "tac")]
    x, y, m = collate(batch, template.pad_id)
    assert x.shape == y.shape == m.shape
    for r, enc in enumerate(batch):
        got = [int(y[r, t]) for t in range(y.shape[1]) if m[r, t]]
        want = [i for i, mm in zip(enc.ids, enc.completion_mask) if mm]
        assert got == want


def test_collate_pads_to_the_batch_max(template):
    batch = [template.encode("a =", "1"), template.encode("a much longer prompt =", "12345")]
    x, _, m = collate(batch, template.pad_id)
    assert x.shape[1] == max(len(e.ids) for e in batch) - 1
    assert not bool(m[0, -1])                  # padding is never trained on


def test_truncation_keeps_the_completion(template, tok):
    e = template.encode("a very long prompt indeed yes =", "42", max_len=8)
    assert len(e.ids) == 8
    # the answer must survive; left-truncation is what guarantees that
    assert tok.decode([i for i, m in zip(e.ids, e.completion_mask) if m]).endswith(END)


def test_multi_turn_trains_only_assistant_turns(template, tok):
    e = template.encode_conversation([
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
        {"role": "user", "content": "ok"}, {"role": "assistant", "content": "sure"}])
    trainable = tok.decode([i for i, m in zip(e.ids, e.completion_mask) if m])
    assert "yo" in trainable and "sure" in trainable
    assert "hi" not in trainable and "ok" not in trainable


def test_template_requires_special_tokens():
    with pytest.raises(ValueError):
        ChatTemplate(CharTokenizer("abc"))


@pytest.mark.parametrize("tie", [True, False])
def test_resize_token_embeddings_preserves_old_logits(tie):
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=20, block_size=32, n_layer=2, n_head=2, n_embd=32,
                      tie_embeddings=tie)).eval()
    idx = torch.randint(0, 20, (2, 6))
    before, _ = m(idx)
    resize_token_embeddings(m, 26)
    after, _ = m(idx)
    assert m.cfg.vocab_size == 26
    assert after.shape[-1] == 26
    assert torch.allclose(before, after[:, :, :20], atol=1e-6)
    assert (m.lm_head.weight is m.tok_emb.weight) == tie


def test_resize_refuses_to_shrink():
    m = GPT(GPTConfig(vocab_size=20, n_layer=2, n_head=2, n_embd=32))
    with pytest.raises(ValueError):
        resize_token_embeddings(m, 10)


# ----------------------------------------------------------------------- DPO


def test_dpo_loss_is_log2_when_policy_equals_reference():
    z = torch.zeros(5)
    loss, m = dpo_loss(z + 1.5, z - 2.0, z + 1.5, z - 2.0, beta=0.1)
    assert abs(loss.item() - math.log(2)) < 1e-6
    assert abs(m["reward_margin"]) < 1e-9


def test_dpo_loss_closed_form():
    beta = 0.3
    pc, pr = torch.tensor([1.0]), torch.tensor([-1.0])
    rc, rr = torch.tensor([0.5]), torch.tensor([0.0])
    margin = (1.0 + 1.0) - (0.5 - 0.0)
    loss, m = dpo_loss(pc, pr, rc, rr, beta=beta)
    assert abs(loss.item() + math.log(1 / (1 + math.exp(-beta * margin)))) < 1e-6
    assert abs(m["reward_margin"] - beta * margin) < 1e-6


@pytest.mark.parametrize("margin", [-2.0, 0.0, 3.0])
def test_dpo_gradient_weight_is_sigmoid_of_negative_margin(margin):
    """The automatic difficulty weighting -- and why DPO stalls at large margins."""
    beta = 0.5
    p = torch.tensor([margin], requires_grad=True)
    z = torch.zeros(1)
    dpo_loss(p, z, z, z, beta=beta)[0].backward()
    want = -beta / (1 + math.exp(beta * margin))
    assert abs(float(p.grad) - want) < 1e-6


def test_dpo_reward_accuracy_metric():
    pc = torch.tensor([1.0, -1.0])
    pr = torch.tensor([0.0, 0.0])
    z = torch.zeros(2)
    _, m = dpo_loss(pc, pr, z, z, beta=0.1)
    assert m["reward_accuracy"] == 0.5


def test_dpo_label_smoothing_is_symmetric_and_minimised_at_zero():
    """cDPO with ls=0.5 has no preference direction: its minimum is margin 0."""
    z = torch.zeros(1)
    f = lambda x: dpo_loss(torch.tensor([x]), z, z, z, beta=0.5, label_smoothing=0.5)[0].item()
    assert abs(f(-3.0) - f(3.0)) < 1e-6
    assert f(0.0) < f(3.0)


def test_ipo_targets_a_specific_margin():
    """IPO's minimum is at margin = 1/(2*beta), not at infinity."""
    beta = 0.5
    z = torch.zeros(1)
    target = 1 / (2 * beta)
    f = lambda x: dpo_loss(torch.tensor([x]), z, z, z, beta=beta, variant="ipo")[0].item()
    assert f(target) < 1e-12
    assert f(target + 2) > f(target) and f(target - 2) > f(target)


def test_hinge_saturates():
    z = torch.zeros(1)
    f = lambda x: dpo_loss(torch.tensor([x]), z, z, z, beta=0.5, variant="hinge")[0].item()
    assert f(10.0) == 0.0
    assert f(-2.0) > f(0.0) > f(2.0)


def test_unknown_dpo_variant_raises():
    z = torch.zeros(1)
    with pytest.raises(ValueError):
        dpo_loss(z, z, z, z, variant="nope")


def test_simpo_normalises_by_length():
    """A long rejected answer must not win just by being long."""
    loss, m = simpo_loss(torch.tensor([-10.0]), torch.tensor([-20.0]),
                         torch.tensor([5.0]), torch.tensor([20.0]))
    # per-token: chosen -2.0, rejected -1.0 -> the rejected one is preferred
    assert m["reward_accuracy"] == 0.0
    assert loss.item() > math.log(2)


def test_masked_logprob_sum_ignores_unmasked_positions():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=30, block_size=32, n_layer=2, n_head=2, n_embd=32)).eval()
    x = torch.randint(0, 30, (2, 10))
    y = torch.randint(0, 30, (2, 10))
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, 2:5] = True
    a = masked_logprob_sum(m, x, y, mask)
    y2 = y.clone()
    y2[:, 7:] = 0                      # change positions outside the mask
    assert torch.allclose(a, masked_logprob_sum(m, x, y2, mask), atol=1e-6)
    assert torch.allclose(masked_logprob_sum(m, x, y, mask, average=True), a / 3, atol=1e-6)


def test_freeze_reference_is_detached():
    m = GPT(GPTConfig(vocab_size=20, n_layer=2, n_head=2, n_embd=32))
    ref = freeze_reference(m)
    assert not any(p.requires_grad for p in ref.parameters())
    assert not ref.training
    # mutating the policy must not change the reference
    with torch.no_grad():
        m.tok_emb.weight.add_(1.0)
    assert not torch.allclose(ref.tok_emb.weight, m.tok_emb.weight)


# ---------------------------------------------------------------------- GRPO


def test_group_advantages_are_normalised_per_group():
    r = torch.tensor([1., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0.])
    a = group_advantages(r, 4).view(-1, 4)
    assert torch.allclose(a.mean(1), torch.zeros(3), atol=1e-5)
    assert abs(float(a[0].std(unbiased=False)) - 1.0) < 1e-3
    # a group with no variance carries no signal at all
    assert torch.allclose(a[1], torch.zeros(4))
    assert torch.allclose(a[2], torch.zeros(4))
    assert abs(zero_variance_fraction(r, 4) - 2 / 3) < 1e-6


def test_group_advantages_without_std_normalisation():
    """Dr. GRPO: mean-centred only, so low-variance groups are not up-weighted."""
    r = torch.tensor([1., 0., 0., 1.])
    a = group_advantages(r, 4, std_normalize=False)
    assert torch.allclose(a, torch.tensor([0.5, -0.5, -0.5, 0.5]))


def test_group_advantages_rejects_ragged_input():
    with pytest.raises(ValueError):
        group_advantages(torch.zeros(7), 4)


def test_kl_k3_is_nonnegative_and_unbiased():
    torch.manual_seed(0)
    n = 200_000
    logits_p = torch.randn(6)
    logits_q = torch.randn(6)
    p = torch.log_softmax(logits_p, -1)
    q = torch.log_softmax(logits_q, -1)
    idx = torch.multinomial(p.exp().expand(n, 6), 1).squeeze(1)
    est = kl_k3(p[idx], q[idx])
    assert bool((est >= 0).all())                      # never negative, unlike -log r
    true = float((p.exp() * (p - q)).sum())
    assert abs(float(est.mean()) - true) < 0.02
    assert float(kl_k3(p, p).abs().max()) < 1e-12


def test_grpo_loss_value_and_gradient_at_ratio_one():
    """On-policy, the surrogate value is -mean(A) and the gradient is REINFORCE."""
    torch.manual_seed(0)
    old = torch.randn(4, 5)
    adv = torch.tensor([1.0, -1.0, 0.5, 0.0])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0],
                         [1, 1, 1, 1, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    logp = old.clone().requires_grad_(True)
    loss, m = grpo_loss(logp, old, adv, mask, loss_agg="token")
    assert abs(loss.item() + float((adv[:, None] * mask).sum() / mask.sum())) < 1e-6
    assert abs(m["ratio_mean"] - 1.0) < 1e-6
    loss.backward()
    want = -(adv[:, None] * mask.float()) / mask.sum()
    assert torch.allclose(logp.grad, want, atol=1e-7)


def test_grpo_sequence_aggregation_weights_short_sequences_more():
    adv = torch.ones(2)
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
    old = torch.zeros(2, 4)

    def grad(agg):
        lp = torch.zeros(2, 4, requires_grad=True)
        grpo_loss(lp, old, adv, mask, loss_agg=agg)[0].backward()
        return lp.grad

    gt, gs = grad("token"), grad("sequence")
    assert abs(float(gt[0, 0]) - float(gt[1, 0])) < 1e-8      # token: equal per token
    assert abs(float(gs[1, 0]) / float(gs[0, 0]) - 4.0) < 1e-6  # sequence: 4x heavier


def test_grpo_clipping_binds_in_both_directions():
    old = torch.zeros(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    # ratio e^1 with A=+1: the upper clip caps the objective at 1.2
    up, m_up = grpo_loss(old + 1.0, old, torch.ones(2), mask, clip_eps=0.2)
    assert abs(up.item() + 1.2) < 1e-5
    assert m_up["clip_frac"] == 1.0
    # ratio e^-1 with A=-1: the lower clip caps it at 0.8
    dn, _ = grpo_loss(old - 1.0, old, -torch.ones(2), mask, clip_eps=0.2)
    assert abs(dn.item() - 0.8) < 1e-5
    # and once clipped there is no gradient left, which is the whole point
    lp = (old + 1.0).clone().requires_grad_(True)
    grpo_loss(lp, old, torch.ones(2), mask, clip_eps=0.2)[0].backward()
    assert float(lp.grad.abs().max()) == 0.0


def test_grpo_kl_penalty_is_additive():
    old = torch.randn(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    adv = torch.tensor([1.0, -1.0])
    base, _ = grpo_loss(old, old, adv, mask, kl_coef=0.0)
    withkl, m = grpo_loss(old, old, adv, mask, ref_logp=old - 0.5, kl_coef=1.0)
    assert abs(withkl.item() - (base.item() + m["kl"])) < 1e-6
    assert m["kl"] > 0


def test_grpo_unknown_aggregation_raises():
    with pytest.raises(ValueError):
        grpo_loss(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1),
                  torch.ones(1, 2, dtype=torch.bool), loss_agg="nope")


def test_extract_completion_trims_at_end_token(tok):
    end = tok.special_tokens[END]
    ids = tok.encode("42") + [end] + tok.encode("garbage")
    kept, text = extract_completion(ids, tok, end)
    assert text == "42"
    assert kept[-1] == end                 # the terminator stays in the loss mask
    # no end token at all: keep everything
    kept2, text2 = extract_completion(tok.encode("42"), tok, end)
    assert text2 == "42" and end not in kept2


def test_sample_rollouts_shapes_and_rewards(template):
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=template.tok.vocab_size, block_size=64,
                          n_layer=2, n_head=2, n_embd=32)).eval()
    exs = sample_examples(3, seed=0)
    rollouts = sample_rollouts(model, exs, template, reward_exact, group_size=4,
                               max_new_tokens=6, device="cpu")
    assert len(rollouts) == 12
    for r in rollouts:
        assert 0.0 <= r.reward <= 1.0
        assert len(r.encoded.ids) == len(r.encoded.completion_mask)
        assert sum(r.encoded.completion_mask) >= 1     # at least the terminator
        # the prompt part must never be in the loss mask
        head_len = len(template.tok.encode(template.prompt_text(r.prompt)))
        assert sum(r.encoded.completion_mask[:head_len]) == 0


def test_per_token_logprobs_matches_log_softmax():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32)).eval()
    x = torch.randint(0, 20, (2, 6))
    y = torch.randint(0, 20, (2, 6))
    with torch.no_grad():
        logits, _ = m(x)
    want = torch.log_softmax(logits.float(), -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(per_token_logprobs(m, x, y), want, atol=1e-6)
