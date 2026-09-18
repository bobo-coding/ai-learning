"""Evaluation: the part that decides whether anything you did actually worked.

Four families, with the traps that make published numbers incomparable:

1. **Perplexity.**  Only comparable between models with the *same tokenizer*.
   A model with a bigger vocabulary spends fewer tokens per sentence, so its
   per-token loss is lower for free.  `bits_per_byte` fixes this and is what
   you should report when tokenizers differ.

2. **Multiple choice by likelihood.**  Three scoring rules in common use, which
   can rank models differently on the same benchmark:
   `acc` (sum log-prob), `acc_norm` (divided by length), and
   `acc_pmi` (divided out by the answer's unconditional likelihood).  Which one
   a paper used is often the difference between two "SOTA" claims.

3. **Generative + verifier.**  Sample, then check with a function.  Needs a
   decoding policy, and greedy vs temperature-1 changes the number by a lot --
   so it must be stated.

4. **pass@k.**  For "can it do it at all" with k tries.  The naive
   "sample k, did any pass" is a high-variance *biased* estimator; the
   unbiased one below samples n > k and computes a combinatorial expectation.

Plus the thing almost always missing: an error bar.  With 200 eval items, the
standard error on an accuracy near 0.5 is 3.5 points -- so a 2-point
"improvement" is noise.  `bootstrap_ci` and `mcnemar` are here to keep you
honest.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from .data import iter_eval_batches
from .generate import generate
from .model import GPT


# ---------------------------------------------------------------------------
# 1. Perplexity
# ---------------------------------------------------------------------------


@torch.no_grad()
def perplexity(model: GPT, tokens, batch_size: int = 8, stride: int | None = None,
               device=None, max_batches: int | None = None) -> dict[str, float]:
    """Token-level perplexity over a token stream.

    `stride < block_size` gives the sliding-window protocol: every token is
    scored with up to `block_size - stride` tokens of context instead of some
    tokens being scored with almost none.  It is strictly more favourable and
    strictly slower, and papers rarely say which they used.  We only score the
    *last* `stride` positions of each window so no token is counted twice.
    """
    device = device or next(model.parameters()).device
    model.eval()
    T = model.cfg.block_size
    stride = stride or T
    total_nll, total_tok = 0.0, 0
    for i, (x, y) in enumerate(iter_eval_batches(tokens, T, batch_size, device, stride=stride)):
        if max_batches is not None and i >= max_batches:
            break
        logits, _ = model(x)
        logp = torch.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)          # (B, T)
        if stride < T and i > 0:
            nll = nll[:, -stride:]                                    # avoid double counting
        total_nll += float(nll.sum())
        total_tok += nll.numel()
    mean = total_nll / max(1, total_tok)
    return {"nll": mean, "ppl": math.exp(min(mean, 20)), "tokens": total_tok}


def bits_per_byte(mean_nll_per_token: float, tokens: int, raw_bytes: int) -> float:
    """Convert per-token NLL (nats) to bits per byte -- tokenizer-independent.

        bpb = (total nats / ln 2) / n_bytes = nll_per_token * (tokens/bytes) / ln 2

    This is the only compression-style metric you can compare across models with
    different vocabularies, and it is what the scaling-law papers actually plot.
    """
    return mean_nll_per_token * tokens / raw_bytes / math.log(2)


# ---------------------------------------------------------------------------
# 2. Multiple choice by likelihood
# ---------------------------------------------------------------------------


@dataclass
class MCQuestion:
    context: str
    choices: list[str]
    answer: int


@torch.no_grad()
def score_continuation(model: GPT, tokenizer, context: str, continuation: str,
                       device=None) -> tuple[float, int]:
    """(sum log p(continuation | context), n_continuation_tokens).

    Detail that bites: the continuation must be tokenized *in context*, not
    separately, because BPE merges across the boundary (" Paris" is one token,
    "Paris" another).  Encoding the full string and slicing by length is the
    only reliable way.
    """
    device = device or next(model.parameters()).device
    ctx_ids = tokenizer.encode(context)
    full_ids = tokenizer.encode(context + continuation)
    n_cont = len(full_ids) - len(ctx_ids)
    if n_cont <= 0:
        return 0.0, 0
    ids = torch.tensor([full_ids[-model.cfg.block_size :]], dtype=torch.long, device=device)
    logits, _ = model(ids)
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    # position t predicts token t+1, so the continuation's log-probs live at
    # positions [len-n_cont-1, len-2].
    total = 0.0
    L = ids.size(1)
    for j in range(L - n_cont, L):
        total += float(logp[j - 1, ids[0, j]])
    return total, n_cont


@torch.no_grad()
def mc_accuracy(model: GPT, tokenizer, questions: list[MCQuestion], device=None) -> dict[str, float]:
    """All three likelihood-based multiple-choice scoring rules."""
    device = device or next(model.parameters()).device
    model.eval()
    hits = {"acc": 0, "acc_norm": 0, "acc_pmi": 0}
    for q in questions:
        raw, norm, pmi = [], [], []
        for ch in q.choices:
            s, n = score_continuation(model, tokenizer, q.context, ch, device)
            uncond, _ = score_continuation(model, tokenizer, "", ch, device)
            raw.append(s)
            norm.append(s / max(1, n))
            # PMI: log p(choice|context) - log p(choice).  Cancels the prior
            # preference for common strings, which is what makes plain `acc`
            # pick "the" over the right answer on some benchmarks.
            pmi.append(s - uncond)
        hits["acc"] += int(max(range(len(raw)), key=raw.__getitem__) == q.answer)
        hits["acc_norm"] += int(max(range(len(norm)), key=norm.__getitem__) == q.answer)
        hits["acc_pmi"] += int(max(range(len(pmi)), key=pmi.__getitem__) == q.answer)
    n = max(1, len(questions))
    return {k: v / n for k, v in hits.items()}


# ---------------------------------------------------------------------------
# 3. Generative evaluation against a verifier
# ---------------------------------------------------------------------------


@torch.no_grad()
def generative_eval(model: GPT, template, examples, reward_fn, max_new_tokens: int = 12,
                    greedy: bool = True, temperature: float = 1.0, batch_size: int = 32,
                    device=None, return_samples: int = 0):
    """Decode a completion per example and score it with `reward_fn`.

    Prompts are grouped by token length so a batch needs no padding -- padding a
    decoder prompt on the right would put the pad token *inside* the context and
    silently corrupt the generation.
    """
    from .grpo import extract_completion

    device = device or next(model.parameters()).device
    model.eval()
    by_len: dict[int, list[int]] = {}
    encoded = [template.tok.encode(template.prompt_text(e.prompt)) for e in examples]
    for i, ids in enumerate(encoded):
        by_len.setdefault(len(ids), []).append(i)

    completions: list[str] = [""] * len(examples)
    for _, idxs in sorted(by_len.items()):
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s : s + batch_size]
            x = torch.tensor([encoded[i] for i in chunk], dtype=torch.long, device=device)
            out = generate(model, x, max_new_tokens, greedy=greedy, temperature=temperature,
                           eos_id=template.end_id)
            for row, i in zip(out, chunk):
                new_ids = row[x.size(1):].tolist()
                _, text = extract_completion(new_ids, template.tok, template.end_id)
                completions[i] = text

    rewards = [float(reward_fn(e, c)) for e, c in zip(examples, completions)]
    res = {"overall": sum(rewards) / max(1, len(rewards))}
    by_task: dict[str, list[float]] = {}
    for e, r in zip(examples, rewards):
        by_task.setdefault(getattr(e, "task", "all"), []).append(r)
    for k, v in sorted(by_task.items()):
        res[k] = sum(v) / len(v)
    if return_samples:
        res["samples"] = [(e.prompt, c, e.answer, r) for e, c, r
                          in list(zip(examples, completions, rewards))[:return_samples]]
    return res


# ---------------------------------------------------------------------------
# 4. pass@k
# ---------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from `c` successes out of `n` samples (Codex paper).

        pass@k = 1 - C(n-c, k) / C(n, k)

    i.e. one minus the probability that a random size-k subset of the n samples
    contains no success.  The naive alternative -- sample exactly k and check --
    is unbiased only for that one k and has far higher variance, so you cannot
    read pass@1 and pass@10 off the same run.
    """
    if k > n:
        raise ValueError("k must be <= n")
    if n - c < k:
        return 1.0
    # Computed as a product to avoid overflow in the binomials.
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


@torch.no_grad()
def pass_at_k_eval(model: GPT, template, examples, reward_fn, n: int = 8, ks=(1, 4, 8),
                   temperature: float = 1.0, max_new_tokens: int = 12, device=None):
    """Sample `n` completions per example and report pass@k for each k in `ks`."""
    from .grpo import extract_completion

    device = device or next(model.parameters()).device
    model.eval()
    out = {f"pass@{k}": 0.0 for k in ks}
    for ex in examples:
        head = template.tok.encode(template.prompt_text(ex.prompt))
        x = torch.tensor([head] * n, dtype=torch.long, device=device)
        gen = generate(model, x, max_new_tokens, temperature=temperature,
                       eos_id=template.end_id)
        c = 0
        for row in gen:
            _, text = extract_completion(row[len(head):].tolist(), template.tok, template.end_id)
            c += int(reward_fn(ex, text) >= 1.0)
        for k in ks:
            out[f"pass@{k}"] += pass_at_k(n, c, k)
    return {k: v / max(1, len(examples)) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Statistics: the part that is usually skipped
# ---------------------------------------------------------------------------


def bootstrap_ci(scores: list[float], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap CI for a mean score.

    Use this before claiming an improvement.  For 0/1 scores it agrees with the
    normal approximation for n > ~50 and is more honest below that.
    """
    rng = random.Random(seed)
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        means.append(sum(scores[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return sum(scores) / n, lo, hi


def mcnemar(a_correct: list[int], b_correct: list[int]) -> dict[str, float]:
    """Paired test for "is model B better than model A" on the same items.

    Two independent confidence intervals overlapping does NOT mean there is no
    difference: models make correlated errors, and the paired test is far more
    sensitive.  Only the disagreements carry information:

        b01 = A wrong, B right      b10 = A right, B wrong

    Under the null those split 50/50, so this is a binomial sign test.
    """
    b01 = sum(1 for x, y in zip(a_correct, b_correct) if not x and y)
    b10 = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p_value": 1.0}
    # Exact two-sided binomial p-value.
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return {"b01": float(b01), "b10": float(b10), "p_value": min(1.0, 2 * tail)}


@torch.no_grad()
def calibration(model: GPT, tokens, n_bins: int = 10, batch_size: int = 8, device=None,
                max_batches: int = 20) -> dict[str, float]:
    """Expected calibration error of the next-token distribution.

    A language model's top-1 confidence should match its top-1 accuracy.  ECE
    is the average gap, bucketed by confidence.  Base models are usually well
    calibrated; RLHF reliably makes them overconfident, which is one of the few
    quantitative costs of alignment you can measure locally.
    """
    device = device or next(model.parameters()).device
    model.eval()
    conf_sum = [0.0] * n_bins
    acc_sum = [0.0] * n_bins
    count = [0] * n_bins
    for i, (x, y) in enumerate(iter_eval_batches(tokens, model.cfg.block_size, batch_size, device)):
        if i >= max_batches:
            break
        logits, _ = model(x)
        probs = torch.softmax(logits.float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        correct = (pred == y).float()
        b = (conf * n_bins).clamp(max=n_bins - 1e-6).long()
        for j in range(n_bins):
            sel = b == j
            k = int(sel.sum())
            if k:
                conf_sum[j] += float(conf[sel].sum())
                acc_sum[j] += float(correct[sel].sum())
                count[j] += k
    total = sum(count)
    ece = sum(count[j] / total * abs(acc_sum[j] / count[j] - conf_sum[j] / count[j])
              for j in range(n_bins) if count[j]) if total else 0.0
    return {
        "ece": ece,
        "accuracy": sum(acc_sum) / max(1, total),
        "confidence": sum(conf_sum) / max(1, total),
        "n": float(total),
    }
