# Lesson 11 — LoRA and QLoRA

Read: `minigpt/lora.py` · Tests: `tests/test_quant_lora.py`

## The observation

A fine-tune moves the weights by a small amount, and that update turns out to be
close to low-rank. So instead of learning a full-rank `ΔW`, learn

```
W' = W + (α/r) · B A        A: (r, in),  B: (out, r),  r ≪ min(in, out)
```

and train only `A` and `B`. For a 4096×4096 matrix at `r = 8` that is 65k
trainable parameters instead of 16.7M — a **256×** reduction.

## Why it saves far more memory than it saves parameters

The weights were never the problem. AdamW keeps **two fp32 moments per trainable
parameter**, plus a gradient:

```
full fine-tune:  4 (weight) + 4 (grad) + 8 (moments) = 16 bytes/param
LoRA:            4 (frozen weight) + 12 bytes per *adapter* param
```

For a 7B model that is 112 GB of optimizer state versus about 0.3 GB. Freezing
the base removes gradients and moments for 99.9% of the parameters, and *that*
is the saving — not the 0.06% fewer trainable weights.

What you still pay is **activations**: backprop through the frozen `W` still
needs its input. So LoRA and gradient checkpointing are complementary, not
alternatives, and LoRA does not reduce activation memory at all.

Measured on this repo's small model:

| | total weights | trainable | Adam state |
|---|---|---|---|
| full fine-tune | 164k | 164k (100%) | 1.97 MB |
| LoRA r=8 | 195k | 30.7k (15.8%) | 0.37 MB |
| NF4 + LoRA r=8 | 195k | 30.7k (15.8%) | 0.37 MB |

and the weight *bytes*, which is where QLoRA shows up:

| | weight bytes |
|---|---|
| fp32 + LoRA | 780 KB |
| NF4 + LoRA | **231 KB** |

(The percentages look unimpressive because this model is tiny — `r=8` is a
large fraction of a 64-wide layer. At production widths the ratio is the 256×
above. `trainable_summary` counts quantized weight *buffers* as part of the
total, because counting only `model.parameters()` reports a QLoRA model as ~90%
trainable, which is nonsense.)

## The two initialisation rules

```python
nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
self.lora_B = nn.Parameter(torch.zeros(out_f, r))
```

- **`B = 0`** so the adapter output is *exactly* zero at step 0. The fine-tune
  starts from the base model's behaviour, not from noise.
  `test_lora_is_identity_at_init` asserts bit-level equality of the logits.
- **`A ≠ 0`** so that once `B` moves, the update has sensible scale. If both
  were zero the product's gradient would be identically zero and nothing would
  ever train — `∂(BA)/∂B = Aᵀ = 0` and `∂(BA)/∂A = Bᵀ = 0`.

## What `alpha` is for

The update is scaled by `α/r`, so **raising `r` does not change the initial
update magnitude**. That means you can sweep `r` without re-tuning the learning
rate, which is the entire reason the convention exists. Conventionally
`α = r` or `α = 2r` (this repo defaults to `2r`, so `scaling = 2.0` regardless
of `r`).

## Which layers to adapt

```python
targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down")
```

The original paper adapted only `q_proj` and `v_proj`. Later work — QLoRA
especially — found that adapting **every** linear layer is better at equal
trainable-parameter budget: with a fixed budget you are better off with a small
`r` everywhere than a large `r` in two places. So that is the default here.

`train_norms=True` additionally unfreezes the norm gains. They are ~0.01% of the
parameters and often help, because a fine-tune frequently wants to rescale
features rather than rotate them.

## Merging

```python
W ← W + (α/r) · B A
```

Zero inference overhead after merging: no extra matmuls, no extra memory, the
model is indistinguishable from a full fine-tune of the same weights.
`test_merge_is_exact_and_reversible` checks merge and unmerge both round-trip.

**Merging into a quantized base is refused**, and that is not a limitation to
work around — `W + BA` would have to be *re*-quantized, which changes the
result. So QLoRA adapters stay separate at inference. That is also why serving
many LoRAs off one base model is cheap: one copy of the 4-bit weights, one small
adapter per tenant, swapped per request.

## QLoRA

```python
quantize_model(model, bits=4, scheme="nf4", group_size=64)
apply_lora(model, LoRAConfig(r=8))
```

The base is NF4, the adapters are bf16/fp32, and gradients flow **through** the
dequantized weights to `A` and `B`. A 7B model then fine-tunes in ~6 GB instead
of ~60.

The reason it works as well as it does: the base is never updated, so its
quantization error is a **fixed bias**, not something that compounds over
training steps. The adapter can even learn to compensate for it.

`test_qlora_gradients_flow_through_a_4bit_base` confirms every trainable tensor
receives a gradient through the quantized path.

## When not to use LoRA

- **Teaching genuinely new knowledge.** LoRA is good at changing style, format
  and task behaviour; a rank-8 update cannot install a new domain vocabulary.
- **Very long training runs.** As the number of tokens grows, full fine-tuning
  pulls ahead — the low-rank assumption is an approximation, and it binds.
- **When you need the merged model to be quantized anyway.** Then quantize after
  a full fine-tune.

The practical rule: LoRA for adaptation, full fine-tune for capability.

## Exercises

1. **Rank sweep.** Fine-tune the Shakespeare model on a new style (all-caps, or
   a different play) at `r` ∈ {1, 2, 4, 8, 16, 64} and plot validation loss
   against trainable parameters. Where does it saturate? Compare to a full
   fine-tune.

2. **Verify the alpha claim.** For `r` ∈ {4, 8, 16, 32} at fixed LR, record the
   loss after 100 steps with `alpha = 2r` (scale-invariant) and with
   `alpha = 16` (fixed). Show that only the first is LR-transferable.

3. **Which layers matter?** Adapt only attention, only MLP, and both, at matched
   trainable-parameter counts. Which wins, and does the answer change with the
   task?

4. **Is the update actually low-rank?** Do a full fine-tune, compute
   `ΔW = W_after − W_before` for each layer, and plot its singular-value
   spectrum. What effective rank captures 90% of the Frobenius norm? Compare
   across layer types — this is the empirical basis for the whole method.

5. **QLoRA quality cost.** Fine-tune the same task with an fp32 base and an NF4
   base at the same `r`. Measure the final loss gap and the peak memory. Is the
   trade worth it here? At what model size would it become obviously worth it?

6. **Multi-adapter serving.** Train three adapters for three different tasks,
   then implement a forward pass that routes each row of a batch to a different
   adapter. Measure the throughput cost versus a single merged model.

---

**Previous:** [Lesson 10](10_quantization.md) · **Next:** [Lesson 12 — GPU kernels](12_gpu_kernels.md)
