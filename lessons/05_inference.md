# Lesson 5 — Inference

Read: `minigpt/generate.py` · Tests: `tests/test_generate.py`

Three separable concerns that people constantly conflate:

1. **What distribution does the model give?** Fixed by the weights:
   `p(next | prefix) = softmax(logits)`.
2. **How do we turn it into a token?** Temperature, top-k, top-p, min-p *modify*
   that distribution. Every one trades diversity for reliability, and none makes
   the model smarter.
3. **How fast can we do it?** Decoding one token is **memory-bandwidth bound**:
   you read every weight to produce one token. That single fact explains the KV
   cache, speculative decoding, quantization, and why batching is free.

## The bandwidth argument, quantitatively

Llama-7B in fp16 is 13.5 GB of weights. One token needs all of them read once.
Take an A100's published figures — 800 GB/s HBM bandwidth and ~312 TFLOP/s
fp16 tensor-core peak; both are vendor specs, not measured here — and that is a
floor of ~17 ms/token ≈ 59 tok/s, *regardless of how fast the arithmetic is*:
14.3 GFLOP at 300 TFLOP/s would take 0.048 ms. You are **350× away** from
compute-bound.

(The 13.5 GB, 14.3 GFLOP, 17 ms and 350× all follow from `GPTConfig` and are
checked in `tests/test_model.py`; only the A100 numbers are quoted.)

Two consequences:

- **Batching is nearly free.** The weights are read once for the whole batch, so
  going from batch 1 to batch 32 costs almost nothing per token. Throughput
  scales, latency does not.
- **Halving the bytes nearly halves the latency.** That is the whole case for
  quantization at inference (lesson 10), and why it does almost nothing for
  training throughput.

## The KV cache

Without a cache, generating token `t` re-runs the model over the whole prefix:
`O(T²)` total work for `T` tokens. With one, each step attends to stored keys
and values and does `O(T)` total.

```bash
python -m scripts.sample --ckpt out/pretrain_reg/best.pt --bench
```

```
with KV cache            188.9 ms [188-191]    677.5 tok/s  +/-1%
no cache (recompute)     263.3 ms [262-267]    486.2 tok/s  +/-2%
-> cache speedup 1.39x at 128 tokens (range 1.37-1.42x)
```

**1.39×** for 128 new tokens on a tiny model, reproducible to ±2% over five
measurements. The gap grows with sequence length — the cached path is O(T) and
the uncached one O(T²) — so this is the *least* impressive setting for it;
exercise 3 asks you to fit the exponents.

(An earlier version of this lesson claimed 1.86× from a single run whose
uncached measurement had not warmed up. If you quote a speedup, quote its
spread.)

The cache is preallocated to `max_seq_len` and written in place, so generation
never reallocates. Its size —
`2 · n_layer · n_kv_head · head_dim · seq_len · batch · bytes` — is what
actually limits serving batch size, which is why lesson 2's GQA exists.

**The correctness property**: incremental decoding must reproduce a single full
forward pass *exactly*. `use_cache=False` exists so you can check it, and the
tests do:

```
test_greedy_cache_matches_no_cache       torch.equal — identical tokens
test_sampled_cache_matches_no_cache      same, with a seeded generator
```

If those pass, your cache, your RoPE offsets and your mask shift are all right.
If they do not, you have a bug that will look like "the model is a bit worse at
long context" and take a week to find.

## Sampling

### Temperature

Dividing logits by `T` is exactly raising probabilities to `1/T` and
renormalising. `T → 0` is argmax, `T = 1` is the model's own distribution,
`T > 1` flattens it.

### top-k

Keep the `k` highest logits. Crude but effective — it cuts the long tail that
causes most incoherence. Its weakness is that `k` is fixed: when the model is
confident, `k = 50` still admits 49 bad tokens.

Documented edge case: `top_k` cannot break exact ties, so a uniform
distribution survives `top_k=2` intact. That is the right call (dropping tied
tokens arbitrarily would bias sampling), but it means `k` is an upper bound on
selectivity. `test_top_k_with_ties_keeps_more_than_k` pins it.

### top-p (nucleus)

Keep the **smallest set whose cumulative mass reaches `p`**. Adapts to the
model's confidence, which is why it became the default.

The off-by-one everyone gets wrong: the token that *crosses* the threshold must
be kept. Otherwise a distribution with one token at 0.99 and `top_p = 0.9` would
have nothing left to sample.

```python
remove = cumulative - probs > p      # mass strictly *before* this token exceeded p
remove[..., 0] = False               # always keep the top token
```

This matches HuggingFace's `TopPLogitsWarper` exactly, and
`test_top_p_is_the_smallest_set_with_mass_at_least_p` checks all three boundary
cases.

### min-p

Keep tokens with `prob ≥ min_p · max_prob`. A cleaner formulation of the same
idea: the threshold is relative to the top token, so a confident step keeps
almost nothing and an uncertain one keeps a lot. No sort needed, and robust at
high temperature.

### What they actually do

Same 984k-parameter model, same seed, prompt `"\n"`:

```
greedy           \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n
T=0.5            To make my terms of my life, lamenting\nTo the city of a guilty
T=1.0            MERCUTIO:\nNay, service, or my mayorous queen: when it is now!
T=1.0 top_k=40   Provosit to the mind of Greece, if I\nbear the winter of this
T=1.0 top_p=0.9  He did throw o'er himself; and if it shall be yours.\n\nSICINIUS:
T=1.5            Remcentry be driel out joy you him our liege.\n\nProvost:\nves not
```

Note that **greedy decoding collapses into a repeat loop.** This is not a small
model artefact — it is what argmax does to any autoregressive model: the most
likely single token is often the one that continues the most likely pattern, and
the pattern is repetition. Greedy is right for verifiable tasks with one answer
(lesson 6 uses it) and wrong for open-ended text.

At `T = 1.5` the model invents non-words ("Remcentry", "driel"): the tail it is
now sampling from consists of tokens the model assigned almost no mass.

And a number from lesson 6's pipeline worth remembering: on the arithmetic
tasks, the same SFT checkpoint scores **0.963 greedy** and **0.631 at
temperature 1.0**. The decoding policy is not a detail — a benchmark number
without a stated decoding policy is not a number.

## Speculative decoding

Since decoding is bandwidth-bound, one target forward pass costs nearly the same
for 1 token as for `g+1` tokens. So: have a small draft model propose `g`
tokens, verify them all in one target pass, and keep the ones that survive a
statistical test.

Per round:

1. draft autoregressively proposes `x₁…x_g`, recording its own `q_i(x_i)`;
2. target scores all of them in **one** forward pass, giving `p₁…p_{g+1}`;
3. accept `x_i` with probability `min(1, p_i(x_i) / q_i(x_i))`;
4. at the first rejection, sample the replacement from the **residual**
   distribution `norm(max(0, p_i − q_i))` and discard the rest;
5. if all `g` were accepted, take a free bonus token from `p_{g+1}`.

**Step 4 is what makes this exact rather than an approximation.** A token `y` is
emitted at position `i` either by acceptance, with probability
`q(y)·min(1, p(y)/q(y)) = min(q(y), p(y))`, or by rejection followed by the
residual draw, which contributes the remaining `max(0, p(y) − q(y))`. The two
sum to `p(y)`. The draft model's quality affects only the *speed*, never the
output distribution.

That claim is worth testing properly, because a broken implementation still
produces plausible text. `test_speculative_decoding_is_distribution_preserving`
runs 6000 single-token generations with a deliberately *bad* draft model and
compares the empirical distribution to the target's exact distribution with a
chi-square test (7 d.o.f., threshold 24.3). And with `draft == target`,
acceptance is exactly 1.0 — `p/q = 1` everywhere.

Expected speedup:

```
accepted_per_round / (1 + g · cost_draft/cost_target)
```

So you want a draft that is both cheap *and* well-aligned. Commonly reported
practice — not measured here, since it needs two real models — is a draft 1–2
orders of magnitude smaller from the same family, `g` ≈ 4–8, and ~70% acceptance
for roughly 2× end to end. The variants (Medusa, EAGLE, n-gram lookup)
all attack the same ratio from different directions.

## Scoring

`token_logprobs` and `sequence_logprob` are the shared primitive behind
perplexity, multiple-choice evaluation, DPO and GRPO. Two details:

**The `(B, T, V)` logits tensor is the largest activation in an LLM.** At
`V = 128k, B = 8, T = 4096` it is 16 GB in fp32. The `chunk` argument exists
because computing the gather in vocabulary chunks is often what makes a
fine-tune fit — and lesson 13's fused cross-entropy kernel attacks the same
problem.

**Sum vs mean.** Sum gives the sequence log-likelihood that DPO uses; mean gives
the length-normalised score that multiple-choice evaluation uses. Choosing wrong
introduces a length bias that looks like a quality difference.

`test_cross_entropy_equals_negative_mean_logprob` closes the loop on the
identity that makes `perplexity = exp(loss)` true.

## Exercises

1. **Verify the bandwidth bound.** For the `small` preset, compute
   `weight_bytes / measured_bandwidth` and compare to the measured per-token
   decode latency. How close is it? What accounts for the gap?

2. **Batching is free.** Measure tok/s for batch = 1, 4, 16, 64 at fixed
   sequence length. Plot total throughput and per-sequence latency. Explain both
   curves with the bandwidth argument.

3. **KV-cache scaling.** Time cached vs uncached generation at 64, 256 and 1024
   new tokens. Fit the exponents — you should see ~1 and ~2.

4. **Build a real speculative setup.** Train a `nano` draft and a `small`
   target on the same data. Measure acceptance rate as a function of `lookahead`
   ∈ {2, 4, 8, 16} and compute the predicted vs actual speedup.

5. **Break speculative decoding** by replacing the residual distribution with
   `p_i` alone (a tempting simplification). Re-run the chi-square test. How large
   is the bias, and would you have noticed it by reading the output?

6. **Reproduce the greedy repeat loop** and then fix it three ways —
   `rep_penalty`, `top_p`, and `min_p`. Which preserves coherence best at equal
   diversity (measured as distinct-trigram rate)?

---

**Previous:** [Lesson 4](04_pretraining.md) · **Next:** [Lesson 6 — SFT](06_sft.md)
