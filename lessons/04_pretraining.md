# Lesson 4 — Pretraining

Read: `minigpt/train.py`, `minigpt/optim.py`, `minigpt/data.py` ·
Tests: `tests/test_optim_data.py`

Anyone can write `loss.backward(); opt.step()`. What separates a loop that
trains a model from one that wastes a week is everything around it — and each
piece in `train.py` exists for a specific failure it prevents.

## The data layout

The corpus is one long `uint16` array on disk. A training example is a random
window of `block_size + 1` tokens: the first `block_size` are the input, shifted
by one they are the targets. One sequence of length `T` therefore yields `T`
next-token prediction problems, which is why pretraining is so much more
sample-efficient than any supervised task.

Three decisions in `data.py` that are not arbitrary:

**`np.memmap`, not a list.** The OS pages in only the windows you touch, so a
300 GB corpus trains from a 16 GB machine with no code changes.

**`uint16` while `vocab < 65536`.** Covers GPT-2 (50257) and Llama (32000).
Halving the bytes halves the loader's I/O, which is the difference between a
saturated GPU and a starved one.

**Split by position, never at random.** Random token-level splitting leaks: a
validation window overlaps the training windows around it, and the model scores
well by memorising neighbours. `train_val_split` takes the tail of the corpus,
and `test_train_val_split_is_positional_not_random` pins it.

Windows are sampled independently at random rather than walked in order, so
consecutive batches are near-independent — which is what SGD's convergence
analysis assumes. The cost is that an "epoch" is only approximate.

## AdamW, written out

```
m_t = β₁ m_{t−1} + (1−β₁) g                    first moment  (momentum)
v_t = β₂ v_{t−1} + (1−β₂) g²                   second moment (per-param scale)
p  ← p · (1 − lr·wd)                           decoupled weight decay
p  ← p − lr · m̂_t / (√v̂_t + ε)                 with m̂ = m/(1−β₁ᵗ), v̂ = v/(1−β₂ᵗ)
```

Three things fall out of reading it:

**The update is scale-invariant in `g`.** Multiply every gradient by 100 and the
step barely changes, because `v` grows too.
`test_adamw_is_scale_invariant_in_the_gradient` verifies it to 1e-6. This is why
Adam works without per-layer LR tuning — and why gradient clipping has a much
weaker effect than you would expect.

**Bias correction only matters for the first ~1/(1−β₂) ≈ 1000 steps.** Without
it `v` starts near zero, `√v` is tiny, and the first steps are enormous. That is
a large part of why warmup exists even when you do have correction.

**`weight_decay` is decoupled** — a plain multiplicative shrink, not an L2 term
added to the gradient. Adding L2 to `g` would get divided by `√v`, making the
effective decay depend on gradient history. That was the bug "AdamW" fixed.

Our implementation is bitwise identical to `torch.optim.AdamW` over 30 steps for
the 2-D tensor and within 1 ULP for the 1-D one (torch's `_foreach_` path fuses
differently). `test_adamw_matches_torch` states exactly that, because "close
enough" is not a useful claim about an optimizer.

## Learning-rate schedules

```python
cosine_lr(step, base_lr, warmup, total, min_ratio)   # GPT-3 / Llama
wsd_lr(step, base_lr, warmup, total, decay_frac)     # warmup-stable-decay
```

Warmup exists because Adam's second moment is badly estimated at step 0; it
stops the first few steps from destroying the initialisation. Note
`(step + 1) / warmup`, so step 0 gets a nonzero LR rather than a wasted update.

WSD is increasingly the default for large runs, for an operational reason: the
constant middle phase gives you a checkpoint that is usable at *any* point, so
you decide the token budget afterwards and only then spend the decay. Cosine
bakes the total step count into the schedule, so extending a run means
restarting it.

## Gradient clipping

```python
total = ‖concat(all gradients)‖₂
scale = min(1, max_norm / (total + 1e-6))
```

One **global** norm over all parameters concatenated, not per-tensor — that is
what preserves the *direction* of the update. It must run after the last
`backward()` of an accumulation cycle and before `step()`, or you clip a partial
gradient.

Log the pre-clip norm. A spike is the earliest warning of a bad batch or a
divergent run, often several hundred steps before the loss curve shows anything.
In the healthy runs here it sits around 1.2–1.4 and drifts down.

(Note the post-clip norm lands just *below* `max_norm` because of the `+1e-6`
— `test_clip_grad_norm_matches_torch_and_is_global` asserts
`1 − 1e-6 < norm ≤ 1`, which is the honest bound.)

## Gradient accumulation

```python
for micro in range(grad_accum):
    _, loss = model(x, targets=y)
    (loss / grad_accum).backward()       # ← the division is mandatory
```

Decouples the batch that fits in memory from the batch the optimizer sees.
Forget the division and your effective learning rate silently scales with
`grad_accum`.

## Mixed precision, and a measured surprise

Rules: prefer bf16 over fp16 (same exponent range as fp32, so no loss scaler and
no `inf` surprises), and keep master weights and optimizer state in fp32.
`autocast` does exactly this — it casts the inputs of matmul-like ops and leaves
reductions and norms alone.

But measure your own hardware. On this M1 Max:

| precision | throughput |
|---|---|
| fp32 | **29.7k tok/s** |
| bf16 autocast | 22.8k tok/s |

bf16 is **23% slower**, because Apple's GPU has no bf16 matrix units — autocast
buys you casts and no faster math. On an A100 or H100 the same flag is a ~2×
win. This is the whole lesson: mixed precision is a hardware property, not a
software best practice.

## MFU: the second most useful metric

```python
mfu = 3 · flops_per_token · tokens_per_step / dt / peak_flops
```

(`3×` because backward is ~2× forward.) Vendor "peak TFLOPS" numbers assume
tensor-core utilisation you will never see, so `measure_peak_flops()` times the
best large square matmul this machine can actually do and uses *that* as the
denominator. MFU then means "fraction of achievable matmul throughput", and
40–50% is a genuinely well-tuned loop.

This repo's `small` preset reaches **32% MFU** on MPS at 29.7k tok/s. If yours
is at 5%, the problem is the data loader or launch overhead, not the model.

## A real overfitting curve

```bash
python -m minigpt.train --preset small --steps 2000 --batch-size 32 --lr 1e-3
```

10.8M parameters, 419k training tokens, no dropout:

| step | train loss | val loss |
|---|---|---|
| 250 | 2.813 | **3.360** ← best |
| 500 | 2.007 | 3.591 |
| 750 | 0.978 | 4.251 |
| 1000 | 0.465 | 4.931 |
| 1500 | 0.163 | 5.672 |
| 2000 | 0.102 | 5.868 |

Validation loss bottoms at step 250 and then *doubles* while training loss falls
to 0.10. Perplexity goes 28.8 → 353. The model has memorised the corpus.

This is not a bug, it is arithmetic. 2000 steps × 8192 tokens = 16.4M tokens
over a 419k-token corpus — **39 epochs**. And Chinchilla-optimal for 419k tokens
is roughly 20k parameters, so a 10.8M-parameter model is ~500× oversized for
this data.

The regularized run (984k parameters, dropout 0.15) behaves properly:

| step | train loss | val loss |
|---|---|---|
| 100 | 4.846 | 4.767 |
| 400 | 3.506 | 3.528 |
| 700 | 3.211 | 3.338 |
| 1000 | 3.026 | 3.290 |
| 1500 | 3.018 | **3.243** ← still improving |

11× fewer parameters, *better* validation loss, and no divergence. Then:

```bash
python -m scripts.sample --ckpt out/pretrain_reg/best.pt --prompt "ROMEO:"
```

```
ROMEO:
I'll make thee, as I am the villain,
And cousin I may change the cause;
And ere he could be sooner than I had
Aboss and promise to fight; the duke is not
In this poor valour pains, I have deserved.
```

From 984k parameters, iambic-ish, with correct speaker formatting and invented
but plausible words. `"Aboss"` is a nice reminder that it is modelling
character statistics, not English.

## Checkpointing

Save the **optimizer state**, not just the weights. Resuming without Adam's
moments is a fresh warmup in disguise and shows up as a loss bump that takes
hundreds of steps to recover.

## Exercises

1. **Find the compute-optimal size.** Train `nano`, `tiny`, `small` and
   `medium` on Tiny Shakespeare with matched token budgets. Plot best validation
   loss against parameter count. Where is the minimum, and how does it compare
   to the Chinchilla 20-tokens-per-parameter rule?

2. **Reproduce the overfitting curve, then fix it three ways** — dropout, weight
   decay, and early stopping. Which gives the best validation loss per unit of
   compute?

3. **SGD vs Adam.** Swap in `minigpt.optim.SGD` and find the best LR by sweep.
   How much worse is it, and which parameter group stalls first? (Hint:
   instrument the per-group gradient norms; the embedding layer's gradients are
   extremely sparse and badly scaled.)

4. **Break gradient accumulation.** Remove the `/ grad_accum` and compare the
   loss curve at `grad_accum` = 1, 4, 16. Show that the divergence is exactly
   equivalent to scaling the LR.

5. **Batch-size scaling.** At fixed total tokens, sweep
   `batch_size` ∈ {8, 16, 32, 64, 128} and record tok/s and MFU. Where does
   throughput saturate, and what does that tell you about the right
   `grad_accum`?

6. **Warmup ablation.** Train with `warmup_steps` = 0, 10, 100, 500. Plot the
   gradient norm over the first 200 steps for each. Explain the shape using the
   bias-correction argument above.

---

**Previous:** [Lesson 3](03_transformer.md) · **Next:** [Lesson 5 — Inference](05_inference.md)
