# Lesson 9 — Evaluation

Read: `minigpt/eval.py` · Tests: `tests/test_eval.py`

The part that decides whether anything you did actually worked, and the part
where published numbers are least comparable. Four families of metric, each with
a trap.

## 1. Perplexity

```
ppl = exp(mean NLL per token)
```

`test_perplexity_is_exp_of_the_mean_nll` closes the identity, and
`test_cross_entropy_equals_negative_mean_logprob` in lesson 5 closes the other
half of it.

**The trap: perplexity is only comparable between models with the same
tokenizer.** A model with a bigger vocabulary spends fewer tokens on the same
sentence, so its per-token loss is lower *for free*. Comparing a 32k-vocab
model's perplexity to a 128k-vocab model's is meaningless.

The fix is **bits per byte**:

```
bpb = nll_per_token · (tokens / bytes) / ln 2
```

Tokenizer-independent, and what the scaling-law papers actually plot. For this
repo's regularized Shakespeare model: val loss 3.243 nats/token at 2.40
bytes/token gives **1.95 bits/byte**. A good byte-level compressor manages ~2
bits/byte on English text, so a 984k-parameter model is roughly at gzip level —
a sobering and useful calibration.

**The second trap: stride.** With `stride < block_size` every token is scored
with up to `block_size − stride` tokens of context, instead of some tokens
being scored with almost none. It is strictly more favourable and strictly
slower, and papers rarely say which they used. `perplexity(..., stride=...)`
supports both, and scores only the last `stride` positions of each window so no
token is double-counted.

## 2. Multiple choice by likelihood

Three scoring rules in common use, which can **rank models differently on the
same benchmark**:

| rule | score | fixes |
|---|---|---|
| `acc` | `Σ log p(choice \| context)` | — |
| `acc_norm` | `/ n_tokens` | bias against long answers |
| `acc_pmi` | `− log p(choice)` unconditionally | bias towards common strings |

Plain `acc` prefers short answers, because every extra token adds a negative
log-probability. `acc_norm` fixes that and introduces the opposite bias.
`acc_pmi` divides out the answer's prior, which is what stops a model picking
`"the"` over the right answer on some benchmarks. Which one a paper used is
often the difference between two "SOTA" claims.

**Implementation trap:** the continuation must be tokenized *in context*, not
separately, because BPE merges across the boundary (`" Paris"` is one token,
`"Paris"` another). `score_continuation` encodes the full string and slices by
length — the only reliable way.

## 3. Generative evaluation with a verifier

Sample, then check with a function. Requires a decoding policy, and the policy
changes the number by a lot. From this repo's pipeline, the *same* checkpoint:

| decoding | exact match |
|---|---|
| greedy | 0.963 |
| temperature 1.0 (pass@1) | 0.631 |

**A benchmark number without a stated decoding policy is not a number.**

`generative_eval` groups prompts by token length so a batch needs no padding.
That is not an optimisation — padding a decoder prompt on the *right* would put
the pad token inside the context and silently corrupt the generation.

## 4. pass@k

For "can it do this at all in `k` tries". The naive estimator — sample `k`, did
any pass — is unbiased only for that one `k` and has high variance, so you
cannot read pass@1 and pass@10 off the same run.

The unbiased estimator (Codex paper): sample `n > k`, count `c` successes, then

```
pass@k = 1 − C(n−c, k) / C(n, k)
```

i.e. one minus the probability that a random size-`k` subset contains no
success. `test_pass_at_k_matches_monte_carlo` checks the formula against 40,000
brute-force draws.

pass@k and pass@1 tell you different things. From the SFT checkpoint:

```
pass@1 = 0.535    pass@4 = 0.936    pass@8 = 0.983
```

The model *can* produce the right answer for 98% of prompts — it just does not
reliably put it first. That gap is the space RL operates in: RLVR's job is
largely to move probability mass towards answers that are already in the
distribution, which is exactly why lesson 8 concludes that RL amplifies existing
capabilities rather than installing new ones. A high pass@8 with a low pass@1 is
the signature of a model that is worth running RL on.

## The part that is almost always missing: error bars

With 200 evaluation items, the standard error on an accuracy near 0.5 is 3.5
points. So a 2-point "improvement" is noise. Two tools:

**Bootstrap CI** for a single number:

```python
mean, lo, hi = bootstrap_ci(scores)      # e.g. 0.547, [0.490, 0.603]
```

**McNemar's test** for "is B better than A" on the *same* items:

```
b01 = A wrong, B right       b10 = A right, B wrong
```

Under the null those split 50/50, so it is a binomial sign test.

This matters more than it sounds. Two overlapping confidence intervals do **not**
mean there is no difference: models make correlated errors, and the paired test
is far more sensitive. From lesson 7's table:

```
SFT 0.547  95% CI [0.490, 0.603]
SimPO 0.623  95% CI [0.567, 0.677]        ← intervals overlap
McNemar: fixed 78, broke 55, p = 0.0560   ← and indeed, not significant
```

versus

```
DPO+NLL 0.967 vs SFT 0.547
McNemar: fixed 134, broke 8, p < 0.0001   ← unambiguous
```

The `fixed`/`broke` counts are also diagnostically useful on their own: a change
that fixes 117 and breaks 24 is doing something different from one that fixes 78
and breaks 55, even at similar net accuracy.

## Run-to-run variance is part of your error bar

Honest caveat from this repo: MPS reductions are not bit-deterministic, so the
same SFT configuration produced **0.547** and **0.643** on two runs with
identical seeds and data. That is a 10-point swing, far more than the 2.8-point
sampling SE.

Why so large here? The training data is deliberately bimodal (45% systematically
biased demonstrations), so the model sits near a decision boundary between two
modes and small numerical differences tip many items at once. A model trained on
clean data is far more stable.

Two lessons:

- **Paired comparisons from a shared checkpoint are trustworthy; unpaired
  single-run comparisons across trainings are not.** This is why
  `exp_dpo_variants.py` starts every variant from the same `sft.pt` — that
  table's ordering is reliable even though the absolute SFT number is not.
- **If you cannot re-run, do not claim a few points.** Report the variance you
  measured, or run 3 seeds.

## Calibration

A language model's top-1 confidence should match its top-1 accuracy. Expected
calibration error is the average gap, bucketed by confidence.

Base models are usually well calibrated. RLHF reliably makes them
overconfident — one of the few quantitative costs of alignment you can measure
locally. `calibration()` is there to measure it before and after lessons 7–8.

## Exercises

1. **Compute bits-per-byte** for two models with different tokenizers (train
   BPE at vocab 512 and 2048) trained to the same token budget. Which looks
   better by perplexity? By bpb? Which is the honest comparison?

2. **Construct a benchmark where `acc` and `acc_norm` disagree.** You need
   choices of very different lengths. Then work out which rule you would prefer
   and why.

3. **Measure the stride effect.** Compute perplexity at `stride` = block_size,
   block_size/2, block_size/8. How much "improvement" can you buy with no model
   change at all?

4. **pass@k curve.** Sample n=32 per prompt and plot pass@k for k=1…32 for the
   SFT and GRPO checkpoints. Where do the curves cross, and what does that say
   about what RL did?

5. **Power analysis.** How many evaluation items do you need to detect a 2-point
   improvement at p<0.05 with 80% power, for an unpaired test and for McNemar,
   assuming 70% baseline accuracy and 85% error correlation? This is the
   calculation nobody does.

6. **Measure calibration before and after.** Run `calibration()` on the base,
   SFT, DPO and GRPO checkpoints. Does alignment make this model overconfident
   too?

---

**Previous:** [Lesson 8](08_rlvr_grpo.md) · **Next:** [Lesson 10 — Quantization](10_quantization.md)
