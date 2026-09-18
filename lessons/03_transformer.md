# Lesson 3 — The transformer

Read: `minigpt/model.py`, `minigpt/config.py` · Tests: `tests/test_model.py`

## One block

```python
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

That is it. Everything else is a choice of `Norm`, a choice of `MLP`, and how
many times you repeat it.

**Pre-norm matters more than it looks.** The original transformer was
post-norm — `x = Norm(x + Attention(x))` — so the residual stream passes
*through* a LayerNorm every block and the identity path is not actually an
identity. Gradients reaching layer 0 are attenuated by every norm on the way,
which is why deep post-norm models need a long warmup and careful init to train
at all. With pre-norm the residual stream is a clean sum of block outputs,
gradients reach layer 0 undamped, and 100 layers is unremarkable.

## RMSNorm

```
y = x / √(mean(x²) + ε) · g
```

LayerNorm without the mean subtraction and without the bias. Dropping them
costs nothing measurable and removes a reduction plus a broadcast from the
kernel, which is why every model since Llama uses it.

The defining behavioural difference, and a good way to remember which is which:
LayerNorm is invariant to adding a constant, RMSNorm is not.
`test_rmsnorm_has_no_mean_subtraction` asserts exactly that.

One real implementation detail: the sum of squares is accumulated in fp32 even
in a bf16 model. A 4096-wide bf16 vector's `Σx²` overflows or underflows easily,
and this is one of the few places mixed precision needs a manual upcast. The
code upcasts fp16/bf16 and leaves fp32/fp64 alone — the version that
unconditionally called `.float()` silently broke float64 testing.

## SwiGLU, and where the 11008 comes from

```
GELU MLP:  W_down( gelu(W_up(x)) )                    2 matrices
SwiGLU:    W_down( silu(W_gate(x)) ⊙ W_up(x) )        3 matrices
```

The elementwise product makes the layer *quadratic* in `x`, so it can express
multiplicative interactions that one nonlinearity cannot. It measurably beats
GELU at equal parameter count — and to keep the count equal with three matrices
instead of two, the inner width is scaled by 2/3 and rounded up:

```python
h = int(2 * mlp_ratio * n_embd / 3)
hidden_dim = round_up(h, multiple_of)
```

For Llama-7B: `2·4·4096/3 = 10922.67` → `int` → `10922` → round up to a multiple
of 256 → **11008**. That is where the famous number comes from, and
`test_swiglu_hidden_dim_matches_llama` pins it.

Also note the GELU variant: GPT-2 shipped the `tanh` approximation. The exact
`erf` form differs by <1e-3 and either is fine to train with, but they are not
bit-equal, so a checkpoint must be run with whichever it was trained on.

## Weight tying

```python
self.lm_head.weight = self.tok_emb.weight
```

The output head *is* the input embedding matrix. It saves `vocab · n_embd`
parameters — 38.6M of GPT-2 small's 124M, so ~31% — and helps small models,
because every token's embedding now receives gradient both from being read and
from being predicted.

It also produces a trap worth knowing. Since logits = `residual @ Eᵀ` and the
residual stream at position `t` still contains `E[x_t]`, the largest logit at
init is the *current* token. So a model scored on **unshifted** targets appears
to have learned something before training:

```python
_, shifted   = m(idx[:, :-1], targets=idx[:, 1:])   # 6.91 ≈ ln(1000)   ✓
_, unshifted = m(idx, targets=idx)                  # 5.01              ✗
```

`test_tied_embeddings_bias_toward_copying_at_init` documents this. If your
initial loss is meaningfully below `ln(V)`, your targets are misaligned — that
one check catches most data-pipeline bugs on the first step.

## Initialisation

Two rules, both from GPT-2:

1. `normal(0, 0.02)` on every weight.
2. **Residual output projections get `0.02 / √(2·n_layer)`.**

Rule 2 is the one people skip. Each of the `2L` residual additions contributes
roughly unit variance, so without the downscale the variance of the residual
stream grows linearly with depth, and a deep model starts with saturated logits
and a huge initial loss. `test_residual_projections_are_downscaled` checks the
`o_proj` and `down` weights specifically.

Sanity check at init: loss should be `ln(V)` — 6.93 for a 1025-token
vocabulary. That is exactly what the training log shows at step 0.

## Optimizer parameter groups

```
decay matrices, do not decay anything one-dimensional
```

Norm gains and biases have no redundancy to regularise away, and decaying them
just biases the network towards smaller activations. Embeddings are 2-D and
*are* decayed here, matching GPT-2/nanoGPT.

`configure_optimizers` deduplicates by `id(p)`, which matters with weight
tying — the shared matrix would otherwise be decayed twice.
`test_optimizer_groups_split_by_dimension` asserts the group sizes sum to
`num_params()` exactly.

## The arithmetic you should be able to do on paper

`GPTConfig` implements it, and
`test_analytic_param_count_matches_module` checks the formula against the real
module for every preset — so the formula is not decoration.

**Parameters per block:**

```
attention: n_embd·(n_head·hd) + 2·n_embd·(n_kv_head·hd) + (n_head·hd)·n_embd
mlp:       3 · n_embd · hidden_dim          (SwiGLU)
norms:     2 · n_embd
```

**Forward FLOPs per token** (a multiply-add is 2 FLOPs):

```
dense:     2 · (all the above weights)                ≈ 2N
attention: 2 · 2 · T · n_head · hd / 2                 ← the quadratic term
```

For Llama-7B at 4k context:

```
params    6.74B      (mlp 4.33B, attention 2.15B, embeddings 2×131M)
FLOPs/tok 14.3B      of which attention is 7.5%
2N        13.5B      ← the rule of thumb, 6% low
KV cache  2.15 GB/seq @ 4k fp16   →  67 MB with MQA
```

The attention fraction is what tells you when the `6ND` rule breaks. It is 7.5%
at 4k and grows linearly in `T`:

```python
preset("llama7b").flops_per_token(4096)["fraction_attention"]    # 0.075
preset("llama7b").flops_per_token(32768)["fraction_attention"]   # 0.39
```

At 32k context, nearly 40% of the compute is attention and the rule of thumb is
useless. This is the quantitative case for linear-attention and sliding-window
variants.

## Switching architectures

Every choice is a config flag, so you can isolate what actually matters:

```python
GPTConfig(norm="layer", pos="learned", mlp="gelu", bias=True)   # GPT-2 exactly
GPTConfig(norm="rms",   pos="rope",    mlp="swiglu")            # Llama-style
preset("gpt2")      # the real 124M architecture: 124.44M params ✓
```

`test_kv_cache_equals_full_forward` runs over all twelve combinations, so any of
them can be trained and generated from.

## Exercises

1. **Derive the parameter count for `preset("gpt2")` on paper**, then check
   against `cfg.param_count()`. Account for the 38.6M saved by weight tying.

2. **Post-norm vs pre-norm.** Swap `Block.forward` to
   `x = self.norm1(x + self.attn(x))` and train a 12-layer model both ways for
   500 steps. Plot the gradient norm at layer 0 for each. This is the clearest
   demonstration of why the field switched.

3. **Turn off the residual downscale** (delete the `1/√(2L)` loop) and compare
   the initial loss and first-100-step curve at `n_layer` = 4, 12 and 24. At what
   depth does it start to matter?

4. **Ablate weight tying** at a fixed *parameter* budget: compare tied at
   `n_embd=384` against untied at the `n_embd` that gives the same total. Which
   wins on validation loss for Tiny Shakespeare, and would you expect the same
   at 100× the data?

5. **Find the crossover.** For a fixed 1B-parameter model, at what context
   length does `fraction_attention` exceed 0.5? Derive it algebraically, then
   confirm with `flops_per_token`.

6. **Why 2/3?** Show algebraically that SwiGLU with `hidden = 2/3 · 4 · n_embd`
   has the same parameter count as a GELU MLP with `hidden = 4 · n_embd`.

---

**Previous:** [Lesson 2](02_attention.md) · **Next:** [Lesson 4 — Pretraining](04_pretraining.md)
