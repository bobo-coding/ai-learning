# Lesson 10 — Quantization

Read: `minigpt/quant.py` · Tests: `tests/test_quant_lora.py`

## Why it works at all

Decoding one token is memory-bound (lesson 5): you read every weight and do two
FLOPs with it. So

```
time per token ≈ bytes_of_weights / memory_bandwidth
```

Halve the bytes and you nearly halve the latency — even though you now spend
extra ALU cycles dequantizing, because those cycles were idle anyway.

This also explains why quantization does almost nothing for **training**
throughput: training runs at large batch and is compute-bound, so the weights
are read once and amortised over the whole batch.

## The core operation

```
scale  = max|w| / q_max          (symmetric, "absmax")
q      = round(w / scale)        in [−q_max, q_max]
ŵ      = q · scale
```

Everything else is a choice about *granularity* and *grid shape*.

### Granularity

One scale for the whole tensor is cheap and bad: a single outlier sets the scale
and crushes everything else. Measured on a 64×256 Gaussian matrix:

| scheme | rel L2 error | SNR |
|---|---|---|
| int8 per-tensor | 0.00984 | 40.1 dB |
| int8 per-row | 0.00707 | 43.0 dB |
| int8 group-64 | ~0.006 | ~44 dB |

Modest. Now add **one** 60-sigma outlier:

| scheme | rel L2 error |
|---|---|
| int8 per-tensor | 0.12299 |
| int8 per-row | 0.01629 |
| int8 group-64 | 0.00947 |

Per-tensor gets **13× worse** from one bad weight; group-64 barely notices.
`test_symmetric_per_tensor_vs_per_channel_vs_group` asserts both the ordering
and the outlier ratio.

Cost of finer granularity: at group-64 with an fp32 scale you pay
32 bits / 64 weights = 0.5 bits of overhead, so "int4" is really 4.5 bits. The
code reports this honestly (`memory_bytes()["effective_bits"]`), because
"4-bit quantization" that stores fp16 scales per 32 weights is 4.5 bits and
papers do not always say so. QLoRA's "double quantization" then quantizes the
scales themselves to recover ~0.37 of it.

### Symmetric vs affine

Symmetric has no zero-point, so the matmul stays a plain integer matmul.
Affine (`q = round(w/s) + z`) fits skewed distributions better but adds a
correction term: `(q − z)·s` expands to an integer matmul plus a rank-1 term
`z·Σx`, which is cheap but must not be forgotten. On weights, symmetric is
almost always the right choice — weights are roughly zero-centred. On post-ReLU
activations, affine wins clearly (the test uses `rand(...)·2 + 5` to show it).

### Grid shape: NF4

Uniform spacing is wrong for weights, which are roughly Gaussian — uniform
levels waste resolution in the tails where there is no mass. NF4 uses the
**quantiles of a standard normal**, rescaled to [−1, 1], with an exact zero.

`nf4_levels_from_normal()` re-derives the published constants from the inverse
normal CDF, matching to <1e-3 — so you can see where the magic numbers come
from rather than copying them. The exact zero matters: a pruned or masked weight
must stay exactly zero.

Measured on Gaussian weights at group-64:

| scheme | rel L2 |
|---|---|
| int4 uniform | 0.10780 |
| **NF4** | **0.09246** |

~14% lower error for the same 4 bits. That is roughly half a bit of free
accuracy from choosing the right grid.

## End to end: does it actually hurt?

Quantizing the trained Shakespeare model and re-measuring validation
perplexity on held-out text:

| scheme | val NLL | ppl | Δ NLL | weights (MB) |
|---|---|---|---|---|
| fp32 baseline | 3.1794 | 24.03 | — | 3.937 |
| int8 per-channel | 3.1797 | 24.04 | +0.0003 | 1.404 |
| int8 group-64 | 3.1795 | 24.04 | +0.0001 | 1.435 |
| int4 group-64 | 3.1883 | 24.25 | +0.0089 | 1.009 |
| **NF4 group-64** | **3.1852** | **24.17** | **+0.0058** | **1.009** |
| NF4 group-32 | 3.1878 | 24.24 | +0.0084 | 1.062 |
| NF4 g64 + quantized head | 3.1923 | 24.34 | +0.0129 | 1.082 |

Read three things off that:

- **int8 is free.** ΔNLL of 0.0003 is below run-to-run noise. If you are not
  already serving int8, you are leaving ~2× latency on the table for nothing.
- **NF4 beats uniform int4 end to end too** (+0.0058 vs +0.0089), consistent
  with the reconstruction error above.
- **Quantizing the lm_head is strictly worse here — on both axes.** It adds
  0.0071 NLL *and* **74 KB**, going from 1.009 MB to 1.082 MB.

  The extra bytes are not a rounding artefact, they are weight tying. This model
  shares one matrix between `tok_emb` and `lm_head`, so replacing the head with
  a `QuantizedLinear` does not compress anything — it *adds* a 4-bit copy while
  the fp32 embedding it was tied to stays exactly where it was:

  | | fp32 params | quantized buffers | total |
  |---|---|---|---|
  | head skipped | 529 KB | 479 KB | **1009 KB** |
  | head quantized | 529 KB | 553 KB | **1082 KB** |

  So with tied embeddings there is no argument for quantizing the head at all.
  Untied, you would save the bytes but still pay the accuracy: the head's output
  feeds straight into a softmax over the whole vocabulary, so its error is not
  averaged away by any later layer. Either way `quantize_model` skips it by
  default, and the embedding too.

(Note also that NF4 group-32 is *worse* than group-64 here. At ΔNLL differences
of 0.003 that is noise, not a finding — a good reminder to check whether your
measured difference exceeds your error bar before explaining it.)

Caveat: this is a 984k-parameter model. Small models with narrow matrices
quantize easily. The interesting failures appear at scale, where transformer
activations develop systematic outlier channels.

## Where the error actually goes

**Weight quantization error is benign** — roughly additive noise, and the
network was trained with dropout-scale perturbations anyway.

**Activation quantization error is not.** Transformer activations develop
systematic outlier *channels*: a handful of dimensions far larger than the rest,
in the same positions for every token. A per-tensor activation scale is then set
by the outliers and destroys everything else. The LLM.int8() paper reports this
emerging around the 6.7B-parameter scale and handles it by splitting those
channels into a separate fp16 matmul — that is their measurement, not one this
repo can reproduce at 984k parameters. What you *can* reproduce here is the same
mechanism on the KV cache:

```python
k[..., 7] *= 25        # one persistent outlier channel, as real keys have
```

| scale axis | rel L2 |
|---|---|
| per-token | 0.01717 |
| per-channel | 0.00643 |

2.7× better along the right axis. Keys have persistent outlier channels, so K
wants **per-channel** scales; values do not, so V is fine **per-token**. That
asymmetry is exactly what production KV-cache quantization does, and
`kv_cache_quant_error` lets you verify it.

This matters because the KV cache is often larger than the weights at long
context: 2.15 GB for one 4k Llama-7B sequence vs 13.5 GB of weights — so at
batch 8, the cache *is* your memory problem.

## Packing: making the saving real

Everything above stores 4-bit values in `uint8` tensors, which is convenient and
saves nothing. `pack_int4` puts two values per byte, which is what makes the
on-disk and in-VRAM size honest. Real kernels read packed bytes and unpack in
registers — and the unpacking is free, because you were waiting on memory.

## A checklist for quantizing a real model

1. **int8 weight-only first.** Nearly free, ~2× decode speedup, minutes of work.
2. **Skip the lm_head and embeddings.** Measured above.
3. **Group size 64 or 128 for 4-bit**, per-channel for 8-bit.
4. **NF4 over uniform int4** for weights; there is no reason not to.
5. **Measure perplexity on held-out data**, not reconstruction error. A small
   ΔNLL can still be a large behavioural change — also check a generative task.
6. **Quantize the KV cache before the weights** if you are long-context bound,
   and use per-channel scales for K.
7. **Activations last, and carefully.** Outlier channels are the whole
   difficulty.

## Exercises

1. **Find the breaking point.** Sweep `bits` ∈ {8, 6, 4, 3, 2} at group-64 and
   plot ΔNLL. Where does it become unusable? Does NF4 extend the usable range?

2. **Per-layer sensitivity.** Quantize one layer at a time to int4 and measure
   ΔNLL. Which layers are most sensitive? (Hypothesis worth testing: the first
   and last blocks.) Then build a mixed-precision scheme from the result.

3. **Implement GPTQ's core idea.** Instead of round-to-nearest, quantize columns
   left to right and propagate the error into the not-yet-quantized columns
   using a Hessian approximation from a calibration set. Compare against
   round-to-nearest at 4 bits.

4. **Verify the bandwidth story.** Measure actual decode tok/s for fp32 vs a
   truly-packed int4 model. Do you get the ~4× the byte count predicts? If not,
   where does it go? (Hint: the dequantize step in `QuantizedLinear.forward` is
   not fused.)

5. **Reproduce the activation-outlier problem.** Instrument the model to record
   per-channel activation magnitudes at each layer. Find the outlier channels.
   Then implement per-tensor int8 activation quantization and measure the damage.

6. **Double quantization.** Quantize the group scales to int8 with a second
   level of scales, and report the true effective bits per weight and the ΔNLL.

---

**Previous:** [Lesson 9](09_evaluation.md) · **Next:** [Lesson 11 — LoRA and QLoRA](11_lora.md)
