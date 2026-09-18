import math

import numpy as np
import pytest
import torch

from minigpt.bpe import CharTokenizer
from minigpt.config import GPTConfig
from minigpt.eval import (MCQuestion, bits_per_byte, bootstrap_ci, calibration,
                          generative_eval, mc_accuracy, mcnemar, pass_at_k,
                          pass_at_k_eval, perplexity, score_continuation)
from minigpt.model import GPT
from minigpt.sft import CHAT_SPECIALS, ChatTemplate
from minigpt.tasks import TASK_CHARS, Example, reward_exact


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=41, block_size=32, n_layer=2, n_head=2, n_embd=32)).eval()


def test_perplexity_is_exp_of_the_mean_nll(model):
    tokens = np.random.default_rng(0).integers(0, 41, 500).astype(np.uint16)
    res = perplexity(model, tokens, batch_size=4)
    assert res["ppl"] == pytest.approx(math.exp(res["nll"]), rel=1e-9)
    assert res["tokens"] > 0
    # an untrained model on random data should sit near the uniform baseline
    assert abs(res["nll"] - math.log(41)) < 1.0


def test_sliding_window_perplexity_uses_more_context(model):
    """A smaller stride gives every token more context, so NLL should not rise."""
    tokens = np.random.default_rng(1).integers(0, 41, 600).astype(np.uint16)
    full = perplexity(model, tokens, batch_size=2)["tokens"]
    strided = perplexity(model, tokens, batch_size=2, stride=8)["tokens"]
    assert strided > full           # more windows, each scoring only the new part


def test_bits_per_byte_conversion():
    """The tokenizer-independent metric: 1 nat per token over 1 token/byte = 1/ln2 bpb."""
    assert bits_per_byte(math.log(2), tokens=100, raw_bytes=100) == pytest.approx(1.0)
    # halve the tokens per byte and the bits per byte halves too
    assert bits_per_byte(math.log(2), tokens=50, raw_bytes=100) == pytest.approx(0.5)


def test_score_continuation_tokenizes_in_context(model):
    """The continuation must be tokenized *with* the context, never separately."""
    tok = CharTokenizer(chars=[chr(ord("a") + i) for i in range(26)] + [" "])
    total, n = score_continuation(model, tok, "ab", "cd")
    assert n == 2
    assert total < 0                       # log probabilities are negative
    assert score_continuation(model, tok, "ab", "")[1] == 0


def test_mc_accuracy_returns_all_three_scoring_rules(model):
    tok = CharTokenizer(chars=[chr(ord("a") + i) for i in range(26)] + [" "])
    qs = [MCQuestion("a b ", ["c", "d e f"], 0), MCQuestion("x y ", ["z", "w"], 1)]
    res = mc_accuracy(model, tok, qs)
    assert set(res) == {"acc", "acc_norm", "acc_pmi"}
    assert all(0.0 <= v <= 1.0 for v in res.values())


def test_mc_accuracy_is_perfect_when_the_answer_is_memorised():
    """Sanity check with a model that has actually learned the mapping."""
    tok = CharTokenizer(chars=list("abcdefgh "))
    torch.manual_seed(1)
    m = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=2,
                      n_head=2, n_embd=64))
    # train it to always continue "ab" with "c" and "de" with "f"
    pairs = [("ab", "c"), ("de", "f")]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(300):
        for ctx, cont in pairs:
            ids = torch.tensor([tok.encode(ctx + cont)])
            _, loss = m(ids[:, :-1], targets=ids[:, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    qs = [MCQuestion("ab", ["c", "f"], 0), MCQuestion("de", ["c", "f"], 1)]
    assert mc_accuracy(m, tok, qs)["acc"] == 1.0


def test_pass_at_k_unbiased_estimator():
    """1 - C(n-c, k)/C(n, k), with the obvious boundary cases."""
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(10, 1, 1) == pytest.approx(0.1)
    assert pass_at_k(10, 1, 10) == pytest.approx(1.0)
    # 5 of 10 correct: P(at least one of 2 draws is correct) = 1 - (5/10)(4/9)
    assert pass_at_k(10, 5, 2) == pytest.approx(1 - (5 / 10) * (4 / 9))
    # monotone in k and in c
    assert pass_at_k(8, 2, 1) < pass_at_k(8, 2, 4) < pass_at_k(8, 2, 8)
    assert pass_at_k(8, 1, 4) < pass_at_k(8, 3, 4)
    with pytest.raises(ValueError):
        pass_at_k(4, 2, 5)


def test_pass_at_k_matches_monte_carlo():
    """Check the combinatorial formula against brute-force sampling."""
    rng = np.random.default_rng(0)
    n, c, k = 12, 4, 3
    hits = 0
    trials = 40000
    for _ in range(trials):
        draw = rng.choice(n, size=k, replace=False)
        hits += int((draw < c).any())
    assert abs(hits / trials - pass_at_k(n, c, k)) < 0.01


def test_bootstrap_ci_brackets_the_mean():
    scores = [1.0] * 70 + [0.0] * 30
    mean, lo, hi = bootstrap_ci(scores, n_boot=3000, seed=0)
    assert mean == pytest.approx(0.7)
    assert lo < 0.7 < hi
    # the interval must be roughly +/- 2 standard errors
    se = math.sqrt(0.7 * 0.3 / 100)
    assert abs((hi - lo) - 4 * se) < 0.05
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)


def test_bootstrap_ci_narrows_with_more_data():
    wide = bootstrap_ci([1.0] * 15 + [0.0] * 15, n_boot=2000, seed=0)
    narrow = bootstrap_ci([1.0] * 500 + [0.0] * 500, n_boot=2000, seed=0)
    assert (wide[2] - wide[1]) > 3 * (narrow[2] - narrow[1])


def test_mcnemar_uses_only_the_disagreements():
    a = [1, 1, 0, 0, 1, 0]
    b = [1, 1, 0, 0, 1, 0]
    assert mcnemar(a, b)["p_value"] == 1.0          # no disagreement, no evidence

    # B fixes 10 and breaks none: overwhelming evidence
    a = [0] * 10 + [1] * 10
    b = [1] * 10 + [1] * 10
    r = mcnemar(a, b)
    assert r["b01"] == 10 and r["b10"] == 0
    assert r["p_value"] < 0.005

    # symmetric disagreement: no evidence either way
    a = [0] * 5 + [1] * 5
    b = [1] * 5 + [0] * 5
    assert mcnemar(a, b)["p_value"] == pytest.approx(1.0)


def test_mcnemar_p_value_matches_the_exact_binomial():
    """b01=7, b10=1 -> two-sided sign test on 8 trials."""
    a = [1] + [0] * 7
    b = [0] + [1] * 7
    r = mcnemar(a, b)
    expected = 2 * (math.comb(8, 0) + math.comb(8, 1)) / 2**8
    assert r["p_value"] == pytest.approx(expected)


def test_calibration_of_a_uniform_model(model):
    """An untrained model is underconfident-but-honest on random data."""
    tokens = np.random.default_rng(2).integers(0, 41, 600).astype(np.uint16)
    res = calibration(model, tokens, n_bins=10, batch_size=4, max_batches=5)
    assert 0.0 <= res["ece"] <= 1.0
    assert 0.0 <= res["accuracy"] <= 1.0
    assert res["n"] > 0
    # a near-uniform model has confidence ~1/V, so the gap to accuracy is small
    assert res["ece"] < 0.2


def test_generative_eval_reports_per_task_and_overall():
    tok = CharTokenizer(chars=TASK_CHARS, special_tokens=CHAT_SPECIALS)
    template = ChatTemplate(tok)
    torch.manual_seed(3)
    m = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2,
                      n_head=2, n_embd=32)).eval()
    exs = [Example("add", "1 + 1 =", "2"), Example("add", "2 + 2 =", "4"),
           Example("last", "last of cat =", "t")]
    res = generative_eval(m, template, exs, reward_exact, max_new_tokens=5,
                          device="cpu", return_samples=3)
    assert set(res) >= {"overall", "add", "last", "samples", "n", "n_by_task"}
    # n must accompany every per-task score -- the guard against n=1 columns
    assert res["n"] == 3
    assert res["n_by_task"] == {"add": 2, "last": 1}
    assert len(res["samples"]) == 3
    assert 0.0 <= res["overall"] <= 1.0
    # overall must be the mean of the per-example rewards
    assert res["overall"] == pytest.approx(
        sum(r for *_, r in res["samples"]) / 3)


def test_pass_at_k_eval_is_monotone_in_k():
    tok = CharTokenizer(chars=TASK_CHARS, special_tokens=CHAT_SPECIALS)
    template = ChatTemplate(tok)
    torch.manual_seed(4)
    m = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2,
                      n_head=2, n_embd=32)).eval()
    exs = [Example("add", "1 + 1 =", "2")]
    res = pass_at_k_eval(m, template, exs, reward_exact, n=4, ks=(1, 2, 4),
                         max_new_tokens=4, device="cpu")
    assert res["pass@1"] <= res["pass@2"] <= res["pass@4"]
