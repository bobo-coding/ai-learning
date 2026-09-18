# Lesson 2 — Attention

Read: `minigpt/attention.py` · Tests: `tests/test_attention.py`

The module builds the same function five times, each one closer to what runs on
a real GPU. Every version is tested against `attention_reference`, a triple
Python loop that is unmistakably the definition.

## 1. The definition

For query position `i`:

```
a_ij = exp(q_i · k_j / √D) / Σ_j' exp(q_i · k_j' / √D)
o_i  = Σ_j a_ij v_j
```

Attention returns a *convex combination* of value vectors. With causal masking
the sum runs only over `j ≤ i`, which is what lets a decoder train on all `T`
positions at once: one forward pass yields `T` independent next-token problems.

Why `√D`? `q·k` is a sum of `D` products of roughly unit-variance terms, so its
standard deviation grows like `√D`. Without the scale, logits at `D = 128` have
sd ≈ 11, the softmax saturates, and the gradient through it vanishes. Try it:
set the scale to 1.0 and watch the initial loss and gradient norm.

```python
attention_reference(q, k, v, causal=True)     # the oracle
attention_batched(q, k, v, causal=True)       # softmax(QKᵀ/√D + mask) @ V
```

## 2. Why the batched version does not scale

`attention_batched` materialises the score matrix `S` of shape `(B, H, T, T)`.
At `B=8, H=12, T=4096` in fp32 that is **6.4 GB** — for one layer. Autograd then
keeps it for the backward pass. This is the entire reason FlashAttention exists:
not FLOPs, memory.

Note also where the `√D` goes. We divide the *scores*, but dividing `Q` once is
`D·T` multiplies against `T²` for the scores. At long context that matters, and
real kernels pre-scale `Q`.

## 3. Online softmax: the one trick

The stable softmax needs three passes: find the max, sum the exponentials,
divide. Online softmax does it in one, by carrying a running max `m` and running
denominator `l` and **rescaling whenever the max grows**:

```
m_new = max(m, max(x_block))
l_new = l · exp(m − m_new) + Σ exp(x_block − m_new)
```

That `exp(m − m_new)` correction factor is the whole idea. `online_softmax()`
isolates it in ten lines and the test checks it against `torch.softmax` and
`torch.logsumexp` to 1e-12 for block sizes 1, 3, 17 and 64.

## 4. FlashAttention

Apply the same rescaling to an *output accumulator* and the score matrix never
has to exist:

```python
for each query tile i:
    m, l, acc = -inf, 0, 0
    for each key tile j:
        s = q_i @ k_jᵀ · scale        (+ mask)
        m_new = max(m, rowmax(s))
        p     = exp(s − m_new)
        alpha = exp(m − m_new)
        l     = l · alpha + rowsum(p)
        acc   = acc · alpha + p @ v_j
        m     = m_new
    out_i = acc / l
    lse_i = m + log(l)
```

Memory drops from `O(T²)` to `O(T·D)`. Under causal masking, query tile `i` can
only see keys up to `(i+1)·BLOCK_M`, so later tiles are skipped entirely — the
2× saving that makes causal attention genuinely *cheaper*, not just smaller.

Watch the `-inf` guard in the code. A fully-masked row keeps `m = -inf`, and
`exp(-inf − -inf)` is `nan`, not 0. Every flash implementation needs this and
it is easy to miss because it only fires on edge tiles.

### The backward pass — the part that is usually skipped

We saved only `lse` (one float per query), so `S` and `P` must be
**recomputed** tile by tile in the backward. The softmax Jacobian collapses
neatly:

```
δ_i = Σ_d dO_id · O_id            (one row-dot, computed once up front)
dS  = P ⊙ (dO Vᵀ − δ)
dQ  = dS K · scale,  dK = dSᵀ Q · scale,  dV = Pᵀ dO
```

That `δ` is the same for every key tile, which is why it is hoisted out of the
loop. Recomputation costs one extra `QKᵀ`; it buys `O(T)` instead of `O(T²)`
memory, and that trade is why every transformer trained since 2022 uses it.

This is implemented as a real `torch.autograd.Function`, and the test suite
takes it seriously:

```
test_flash_forward_matches_definition[bq,bk]   exact, 8 tile-size combinations
test_flash_backward_matches_autograd            dQ/dK/dV vs autograd, float64
test_flash_gradcheck                            torch.autograd.gradcheck → True
test_flash_incremental_shapes                   Tq < Tk (the decoding case)
```

`gradcheck` is the strongest statement available for a custom autograd
function: it compares the analytic Jacobian against finite differences. If you
write your own kernel, this is the test to write first.

A note on precision: the accumulators `l` and `acc` are computed in fp32 even
for bf16 inputs, because a running sum in 16 bits loses far too much. The code
keeps fp64 when given fp64 so that `gradcheck` can be exact — an earlier version
hardcoded fp32 and "passed" with 1e-7 errors, which is exactly what a *wrong*
implementation looks like at float32 tolerances.

## 5. RoPE

Absolute position embeddings add a learned vector per position. RoPE instead
*rotates* each consecutive pair of channels by an angle proportional to the
position:

```
θ_i(pos) = pos · base^(−2i/D),   i = 0 … D/2 − 1
```

Channel pair 0 rotates once per token; the last pair once per ~`base` tokens. So
the pairs form a multi-resolution positional code, rather like the binary digits
of `pos`.

The payoff is a single identity: the dot product of two rotated vectors depends
on their positions **only through the difference**.

```python
def dot(m, n):
    return (apply_rope(q, cos, sin, offset=m) * apply_rope(k, cos, sin, offset=n)).sum()

dot(5, 2) == dot(10, 7) == dot(100, 97)      # verified to 1e-15 in the tests
```

So the model gets relative position for free, and can be evaluated at positions
it never saw — which is what "RoPE scaling" then extrapolates by stretching
`base`.

**Two conventions, and they are not compatible.** The interleaved form (this
repo, the RoFormer and Llama reference implementations) pairs channels
`(0,1), (2,3), …`. HuggingFace's `rotate_half` pairs channel `i` with `i + D/2`.
They differ by a fixed permutation of the Q/K output columns, so a checkpoint
works with exactly one of them — and mixing them does not error, it just
silently degrades quality. `apply_rope_complex` is a from-scratch reference via
complex multiplication, used to pin down which one this repo implements.

Build the tables in fp32 or better even for a bf16 model: `pos · inv_freq` for
the slowest channel at `pos = 100_000` needs more mantissa than bf16 has, and a
wrong angle is a silent quality bug.

## 6. GQA and the KV cache

At inference you keep every past key and value. Size:

```
2 · n_layer · n_kv_head · head_dim · seq_len · batch · bytes
```

For Llama-7B at 4k context in fp16 that is **2.15 GB per sequence** — more than
the entire activation memory of the forward pass, and the real limit on serving
batch size. `cfg.kv_cache_bytes()` computes it.

Grouped-query attention gives each group of `n_rep` query heads one shared K/V
head. Quality loss is small; the cache shrinks by `n_rep`:

```python
preset("llama7b").kv_cache_bytes(1)                 # 2.15 GB
preset("llama7b", n_kv_head=1).kv_cache_bytes(1)    # 67 MB  (MQA, 32×)
```

`repeat_kv` does the expansion with `expand` + `reshape`, not `repeat`, so no
memory is copied until the matmul reads it.

The correctness property that matters: **incremental decoding must reproduce a
single full forward pass, exactly.** Any bug in the cache, the RoPE offset, or
the mask shift breaks it. `test_kv_cache_equals_full_forward` checks it for
prefill lengths 1, 4 and 10, and `tests/test_model.py` re-checks it for all
twelve combinations of position encoding, norm and MLP.

The subtle part is the mask. With a cache, `T` queries attend to `offset + T`
keys, so the causal mask must be aligned **bottom-right**, not top-left.
`F.scaled_dot_product_attention(is_causal=True)` aligns top-left, which is wrong
whenever `Tq ≠ Tk` — so `CausalSelfAttention` builds the mask explicitly in that
case:

```python
m = torch.ones(Tq, Tk, dtype=torch.bool).tril(diagonal=Tk - Tq)
```

## Exercises

1. **Delete the scale.** Change `1/√D` to `1.0` in `attention_batched` and
   measure the entropy of the attention weights at `D = 16, 64, 256` on random
   inputs. At what `D` does the softmax effectively become an argmax?

2. **Count flash attention's memory.** Instrument `_FlashAttention.forward` to
   record peak tensor bytes, and compare against `attention_batched` for
   `T = 256, 1024, 4096`. Verify the `O(T²)` vs `O(T)` scaling empirically.

3. **Break the backward on purpose.** Drop the `− del_i` term from `ds` in the
   backward pass. `gradcheck` should fail — by how much, relative? Then check
   whether a float32 `allclose(atol=1e-4)` test would have caught it. This is
   why lesson 0 insists on float64.

4. **Implement the split-half RoPE convention** (`rotate_half`) and find the
   permutation matrix `P` such that `apply_rope(x) = rotate_half(x @ P) @ P⁻¹`.
   Confirm the relative-position property holds for both.

5. **Quantify the GQA quality cost.** Train two `small` models, one with
   `n_kv_head=6` and one with `n_kv_head=1`, for 1500 steps each. Compare
   validation loss and KV-cache size. Is the loss difference worth the 6×
   memory?

6. **Tile-size sweep.** Time `flash_attention` at `q_block, kv_block` ∈
   {16, 32, 64, 128} for `T = 1024`. The optimum is not the largest — explain
   why in terms of the accumulator's working-set size.

---

**Previous:** [Lesson 1](01_tokenization.md) · **Next:** [Lesson 3 — The transformer](03_transformer.md)
