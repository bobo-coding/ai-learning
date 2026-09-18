# Lesson 13 — Triton

Read: `kernels/triton/kernels.py`, `tritonsim/` ·
Run: `python -m kernels.triton.bench` · Tests: `tests/test_kernels.py`

## The bargain

In lesson 12 you assigned work to individual threads, staged tiles in shared
memory by hand, and placed every barrier yourself. Triton takes all of that.

You write code over **blocks** — small tensors whose shapes are known at compile
time — and the compiler handles thread assignment, shared-memory staging, vector
widths, and software pipelining. You keep the things that need a human: the
tiling *strategy*, the memory access *pattern*, and every mask.

| CUDA, by hand | Triton |
|---|---|
| `threadIdx`, per-thread scalars | blocks; **no thread index exists** |
| `__shared__` tile + 2 `__syncthreads()` | implicit in `tl.load` of a block |
| manual bounds check per thread | `mask=` on load/store |
| `__shfl_down_sync` reduction tree | `tl.sum(x, axis=1)` |
| hand-tuned `TILE`, manual unrolling | `BLOCK: tl.constexpr` + `@triton.autotune` |
| one output element per thread | a whole output tile per program |

In practice you get 80–95% of a hand-written CUDA kernel for a fraction of the
code — and for fused reductions like softmax and layernorm, Triton usually
*beats* what people write by hand, because its pipelining is better than theirs.

## Running Triton on a Mac

Triton ships no macOS wheels; it is an NVIDIA/AMD compiler. But the *programming
model* is what is worth learning, and it is small enough to interpret. So
`tritonsim/` implements `triton.jit` and the subset of `triton.language` these
kernels use, on top of plain PyTorch. The same source file runs here and,
unchanged, on a real GPU:

```python
try:
    import triton
    import triton.language as tl
except ImportError:
    import tritonsim as triton
    import tritonsim.language as tl
```

**What you get locally:** correct results, real block-level semantics, and
better error messages than a GPU gives you.

**What you do not get:** any performance signal at all. Every program instance
runs serially in Python, so this is ~1000× slower than a GPU and tells you
nothing about occupancy, warp scheduling, `num_stages`, or whether your tile
sizes are sane. Correctness here, performance there.

### Why the interpreter is genuinely useful

On a GPU, an unmasked out-of-bounds load returns whatever is adjacent in memory.
Silently. Your kernel produces subtly wrong numbers and nothing errors:

```python
@triton.jit
def buggy(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)                    # forgot the mask
    tl.store(out_ptr + offs, v, mask=offs < n)
```

Under `tritonsim`:

```
IndexError: tl.load on 'x_ptr' out of bounds: index 1000 not in [0, 1000).
An unmasked load past the end of a tensor is the classic Triton bug --
you almost certainly need mask=(offs < n).
```

It also catches a mask that disagrees with the offsets (`offs <= n` instead of
`offs < n`), out-of-bounds stores (which on hardware corrupt *other tensors*),
non-contiguous inputs, and non-power-of-two block sizes that would fail to
compile on real hardware. `tests/test_kernels.py` has a test for each.

**The one thing it cannot catch:** a missing `tl.atomic_add`. Program instances
run serially here, so a kernel that races on hardware looks perfectly correct.
That is the bug class you have to reason about rather than test for — and
`rmsnorm_bwd_kernel` is exactly where it arises.

## The whole programming model in six lines

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)     # a BLOCK-shaped array of offsets
    mask = offs < n_elements                     # not optional
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
```

Three things to internalise:

1. **There is no thread index.** `offs` is a *block* of offsets and every
   operation is on the whole block at once.
2. **Pointer arithmetic is in elements**, like C pointer arithmetic on a typed
   pointer — not bytes.
3. **Strides are yours.** Triton has no notion of shape. You pass strides in and
   index with them; contiguity is your problem.

## Where the masks bite: `other=`

```python
x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=-float("inf"))
x = x - tl.max(x, axis=0)
num = tl.exp(x)
out = num / tl.sum(num, axis=0)
```

`other=-inf` is what makes the masked lanes vanish: they contribute nothing to
the max and `exp(-inf) = 0` to the sum. With the default `other=0.0`, a row
shorter than `BLOCK` would get spurious probability mass at every padded
position — a bug that produces *plausible-looking wrong probabilities*, which is
the worst kind.

The test feeds a logit of 800 to confirm the stability, and rows of length 100
with `BLOCK = 128` to confirm the masking.

Note also this softmax does no looping, so `BLOCK` must cover the whole row —
the row has to fit in registers. Past ~64k columns you need a two-pass or online
variant, which is FlashAttention's algorithm applied to a single row.

## Backward passes and atomics

`rmsnorm_bwd_kernel` is where Triton starts paying off. The math:

```
r  = (mean(x²) + ε)^(−1/2)
y  = x·r·w
dx = r · ( dO·w − x·r²·mean(dO·w·x) )
```

That second term is the correction for `r` depending on *every* element of `x`.
Dropping it is the most common hand-written-norm-backward bug, and it produces
gradients that are *nearly* right — so a test with loose tolerances misses it.
`test_triton_rmsnorm_backward_matches_autograd` compares against autograd at
`atol=1e-5`.

`dw` sums over rows, so every program instance accumulates into the same vector:

```python
tl.atomic_add(dw_ptr + cols, dout * xhat, mask=mask)
```

Mandatory on hardware. Invisible in the interpreter. Reason about it.

Note also `rstd` (one float per row) is saved by the forward and consumed by the
backward, rather than recomputed. That is the classic forward/backward trade:
one float per row versus another full read of `x`.

## Matmul: the L2-friendly program ordering

```python
num_pid_in_group = GROUP_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_M
group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

This looks like obfuscation and is the opposite. The naive
`pid_m = pid // grid_n` walks a whole row of C before moving down, so
consecutive programs share a row of A but sweep *all* of B — and B falls out of
L2 before the next row could reuse it. Grouping `GROUP_M` rows into a
"super-row" traversed column-major means the concurrently-running programs touch
a `GROUP_M × BLOCK_N` corner of A and B, which fits in cache.

Worth 10–20% on large matmuls, for pure index arithmetic. It is the canonical
example of the kind of decision Triton leaves to you: the compiler schedules
*within* a program, it does not choose which tile each program gets.

Two other details in that kernel:

- **The accumulator is fp32** even for fp16 inputs. Accumulating in fp16 over a
  K=4096 reduction loses ~3 bits and is a real source of error.
- **Zero-padding handles the ragged tail.** Masked lanes load 0, and `0·b = 0`,
  so a single `tl.dot` handles a K that is not a multiple of `BLOCK_K`. Tested
  at 65×33×47.

## FlashAttention, properly tiled

The Metal and CUDA versions in lesson 12 gave each threadgroup one query, which
made them reduction-latency-bound and ~15× slower than PyTorch. The Triton
version tiles over **both** queries and keys:

```python
pid_m = tl.program_id(axis=0)        # which query tile
bh    = tl.program_id(axis=1)        # which (batch, head)
```

Because `BLOCK_M` queries share each loaded K/V tile, the inner loop is two
`tl.dot`s on tensor cores instead of a block-wide reduction — and the kernel
becomes compute-bound. **That is the entire performance story**, and it took a
restructuring rather than an optimisation.

The online-softmax update is lesson 2's, with the `-inf` guard:

```python
m_new  = tl.maximum(m_i, tl.max(s, axis=1))
m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)     # fully-masked row
p      = tl.exp(s - m_safe[:, None])
alpha  = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m_safe))
l_i    = l_i * alpha + tl.sum(p, axis=1)
acc    = acc * alpha[:, None] + tl.dot(p, v)
```

It also writes out `lse`, which is exactly what the backward pass needs to
recompute `P` without ever storing the `T×T` matrix. The test checks `lse`
against `torch.logsumexp` of the masked scores *separately* from the output,
because a wrong `lse` gives a correct forward and a wrong backward.

Causal masking skips whole K/V tiles: `hi = min(T, (pid_m+1) * BLOCK_M)`. Three
lines for the 2× saving.

## The kernel that actually matters: fused cross-entropy

Of everything here, this is the one most likely to save a real fine-tune.

The logits tensor is `(batch, seq, vocab)`. At `vocab=128k`, `batch·seq=8k`
that is **4 GB in fp32** — and eager PyTorch allocates it again for
`log_softmax` and again for the gradient. 12 GB for a loss.

The fused kernel computes the loss *and* the gradient in one pass, in place:

```python
m   = tl.max(x, axis=0)
z   = tl.sum(tl.exp(x - m), axis=0)
lse = m + tl.log(z)
loss = lse - x_label
grad = tl.exp(x - lse) - onehot(label)      # softmax(x) - onehot: the whole gradient
```

The gradient of mean-reduced cross-entropy w.r.t. the logits is just
`softmax(x) − onehot(label)`, so once you have `lse` you are one line away. The
`ignore_index` rows are zeroed, and the test checks that row's gradient is
**exactly** zero, not just small.

## Correctness under the interpreter

```
$ python -m kernels.triton.bench
backend: tritonsim interpreter (CPU)   device: cpu
OK   add (ragged tail)                  rel_l2=0.000e+00
OK   softmax                            rel_l2=1.353e-07
OK   softmax (logit 800)                rel_l2=0.000e+00
OK   rmsnorm forward                    rel_l2=0.000e+00
OK   rmsnorm saved rstd                 rel_l2=0.000e+00
OK   rmsnorm dx vs autograd             rel_l2=4.247e-08
OK   rmsnorm dw vs autograd             rel_l2=0.000e+00
OK   matmul 65x33x47 (ragged)           rel_l2=9.889e-08
OK   flash B1H2T40D16 causal=True       rel_l2=1.023e-07
OK   flash lse causal=True              rel_l2=2.188e-08
OK   cross_entropy per-row loss         rel_l2=0.000e+00
OK   cross_entropy gradient             rel_l2=7.660e-08
ALL CORRECTNESS CHECKS PASSED
```

Every mask, every offset, every piece of the online-softmax algebra and every
gradient formula, verified on a laptop. Then take the same file to a GPU and the
only remaining question is performance.

## Exercises

1. **Autotune for real.** On a GPU, add `@triton.autotune` over
   `BLOCK_M/N/K ∈ {32,64,128}`, `num_warps ∈ {2,4,8}`, `num_stages ∈ {2,3,4}` to
   the matmul and compare against cuBLAS across shapes. How close do you get?

2. **Remove the L2 grouping** (`pid_m = pid // num_pid_n`) and measure the
   difference at 4096³ on a real GPU. Explain the result in terms of L2 size.

3. **Write the FlashAttention backward** in Triton, using the math from lesson 2
   and the saved `lse`. Two kernels is the usual structure: one for `dK`/`dV`
   (loop over queries) and one for `dQ` (loop over keys), which avoids atomics.
   Verify with `gradcheck` under the interpreter first.

4. **Fuse RMSNorm into the QKV projection.** One kernel, no intermediate
   normalized tensor. Measure the memory saving analytically and the speedup
   empirically.

5. **Break a mask and watch the interpreter catch it.** Change `offs < n` to
   `offs <= n` in the softmax kernel. Then reason about what the same bug would
   have produced on a GPU, and how you would have found it.

6. **Extend `tritonsim`.** Add `tl.make_block_ptr` / `tl.advance` (the block
   -pointer API newer Triton kernels use) and port the matmul to it. This is the
   best way to understand what the abstraction is actually doing.

7. **Fused cross-entropy at scale.** On a GPU, compare peak memory and time
   against `F.cross_entropy` at `vocab` = 32k, 128k, 256k with
   `batch·seq = 8192`. This is the measurement that justifies the kernel.

---

**Previous:** [Lesson 12](12_gpu_kernels.md) · **Back to:** [the index](../README.md)
