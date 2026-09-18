# Lesson 12 — GPU kernels: Metal and CUDA

Read: `kernels/metal/kernels.py`, `kernels/cuda/kernels.cu` ·
Run: `python -m kernels.metal.bench` · Tests: `tests/test_kernels.py`

You do not have an NVIDIA GPU. You do have a GPU, and
`torch.mps.compile_shader` compiles Metal Shading Language at runtime and hands
back callables that take torch tensors directly. The concepts are identical to
CUDA; only the nouns change. So learn them here, then read the `.cu` file and
recognise everything.

## The dictionary

| CUDA | Metal | what it is |
|---|---|---|
| thread | thread | one lane of execution |
| warp (32) | SIMD-group (32) | lanes in lockstep |
| block / threadgroup | threadgroup | shares fast memory |
| `__shared__` | `threadgroup` | fast on-chip scratch, ~L1 latency (tens of KB; the exact budget is per-architecture) |
| `__syncthreads()` | `threadgroup_barrier(...)` | block-wide barrier |
| `__shfl_down_sync` | `simd_shuffle_down` / `simd_sum` | lane-to-lane register move |
| `threadIdx.x` | `thread_position_in_threadgroup` | |
| `blockIdx.x` | `threadgroup_position_in_grid` | |
| `blockDim.x` | `threads_per_threadgroup` | |
| `f<<<blocks, threads>>>` | `f(..., threads=TOTAL, group_size=...)` | **the launch shapes differ** |

That last row is the first bug everyone writes: CUDA's launch takes a number of
*blocks*, Metal's `dispatchThreads` takes a total number of *threads*. Off by a
factor of `group_size`.

Since Metal dispatches whole threadgroups, the grid is rounded up — which is why
every kernel here has a bounds check, and why the last threadgroup has idle
lanes.

## The six kernels, and what each one teaches

### 1. `vector_add` — indexing and occupancy

One thread per element, plus a bounds check. Then a grid-stride variant that
decouples the grid from the problem size. The measured result is the lesson:

All timings below are `median [min–max]` over 7 independent measurements
(`--repeats 7`). GPU clocks drift with thermal state, so a single run of a short
kernel is reproducible only to ~5–15% — and a difference smaller than that
interval is not a result. This matters: an earlier version of this lesson
reported an "11% improvement" from the SIMD reduction that turns out to be 1.2%
against ±4% noise.

| version | time (µs) | bandwidth |
|---|---|---|
| one thread per element | 356 [351–361] | **141 GB/s** |
| grid-stride, 4096 threads | 858 [762–881] | 59 GB/s |
| `torch a + b` | 341 [324–355] | 148 GB/s |

The grid-stride loop is **2.4× slower**. Not because of the loop, but because 4096
threads cannot keep enough memory requests in flight to saturate bandwidth. For
a memory-bound kernel, **occupancy is the whole game** — you need thousands of
concurrent outstanding loads to hide DRAM latency.

Also: hand-written matches PyTorch exactly here. For elementwise ops there is
nothing to win; the only reason to write one is fusion (kernel 4).

### 2. `row_sum` — reductions

Two implementations. The classic tree reduction in threadgroup memory:

```c
for (uint s = gsz / 2; s > 0; s >>= 1) {
    if (tid < s) sh[tid] += sh[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
```

Note the barrier is **outside** the `if`. A barrier inside a divergent branch is
a deadlock (or worse, silently stale data) because threads that skip the body
never arrive. This is the single most common shared-memory bug.

Then the SIMD version. `simd_sum(acc)` reduces across the 32 lanes of one
SIMD-group with **no shared memory and no barrier at all** — the lanes are
already in lockstep. Reduce the ≤8 per-SIMD partials at the end. `log2(TG)`
barriers become 1.

| version | time (µs) | bandwidth |
|---|---|---|
| tree reduction | 300 [297–305] | 55.9 GB/s |
| `simd_sum` | 296 [289–302] | 56.6 GB/s |
| `torch x.sum(-1)` | 268 [251–270] | 62.6 GB/s |

**`simd_sum` is 1.2% faster, against ±4% measurement noise — which is to say,
not measurably faster at all.** The benchmark prints exactly that verdict:

```
-> simd_sum is 1.2% faster; measurement noise is +/-4%  (within noise -- not a result)
```

That is worth sitting with. The barrier-free version *should* be faster, the
argument for it is sound, and at this size it makes no difference — because the
kernel is bound by reading 16 MB from DRAM, not by the handful of barriers.
Removing `log2(256) = 8` barriers from a kernel that spends 300 µs on memory
buys you nothing. Optimise the bound you are actually on.

(An earlier version of this lesson claimed 11% here, from a single run. It was
noise, and it is exactly the mistake the repeat-measurement harness exists to
prevent.)

### 3. `softmax_rows` — stable multi-pass reduction

Three passes over the row: max, sum of exp, write. The max subtraction is not
optional — `exp(800)` is `inf`, and one `inf` NaNs the whole row. The test feeds
a logit of 800 specifically.

Both reductions broadcast through threadgroup memory, and the second one must
wait on the first: the `bcast` variable is reused, so there is a barrier after
reading it as well as after writing it. Miss that second barrier and you get a
race that only shows up under load.

### 4. `rmsnorm` — fusion, and the actual reason to write kernels

```
y = x / √(mean(x²) + ε) · g
```

In eager PyTorch that is `pow`, `mean`, `rsqrt`, `mul`, `mul` — five kernels,
each reading and writing the whole tensor. Fused: read `x` twice, write once.

| | time (µs) | bandwidth |
|---|---|---|
| **Metal fused** | **392 [388–404]** | **171 GB/s** |
| torch eager (5 kernels) | 2911 [2897–2921] | 23.1 GB/s |

**7.4× faster (range 7.2–7.5× across runs)**, and the fused version runs at
171 GB/s — essentially the machine's measured peak. This is the case for hand-written kernels, in one
number: for memory-bound elementwise chains, eliminating round-trips to DRAM is
free performance that no amount of ALU tuning can match. It is also exactly what
`torch.compile` does automatically, which is why you should try that first.

### 5. `matmul` — arithmetic intensity

The naive version has each thread compute one output element, reading a full row
of A and column of B from device memory. Total traffic is `2·M·N·K` floats for
`2·M·N·K` FLOPs: **0.25 FLOP per byte**. This machine needs ~23 to be
compute-bound. So it runs at a few percent of peak no matter how good the ALUs
are.

Tiling fixes the ratio, not the FLOPs. The threadgroup cooperatively stages a
`TILE×TILE` block of A and of B into threadgroup memory; every value loaded is
then reused `TILE` times, so traffic drops by `TILE` and intensity rises to
~`TILE/2`.

```c
As[tid.y][tid.x] = ...;  Bs[tid.y][tid.x] = ...;
threadgroup_barrier(...);                            // publish the tile
for (uint k = 0; k < TILE; ++k) acc += As[tid.y][k] * Bs[k][tid.x];
threadgroup_barrier(...);                            // don't overwrite it yet
```

Both barriers are mandatory and for *different* reasons — the first publishes
the tile, the second stops a fast thread from overwriting it while a slow one is
still reading.

| 1024³ | time (µs) | TFLOP/s |
|---|---|---|
| naive | 4261 [4247–4287] | 0.50 |
| tiled (16×16) | 2750 [2715–2755] | **0.78** (1.55× naive) |
| `torch a @ b` | 619 [617–622] | **3.47** (4.44× tiled) |

1.55× from tiling — and still 4.4× behind the vendor library. Note these are the
*stable* measurements in the file (±1%): a 4.3 ms kernel averages over far more
thermal noise than a 300 µs one. That gap is the
honest lesson: the next steps are register blocking (several outputs per thread,
which is the single biggest remaining win), double-buffered tile loads, and
vectorised `float4` accesses. A production BLAS also picks tile sizes per shape
and uses matrix intrinsics (`simdgroup_matrix` / tensor cores). **You will not
beat cuBLAS in an afternoon, and you should not try.**

### 6. `flash_attn_fwd` — the online softmax on hardware

One threadgroup per `(batch·head, query)`. Each group stages its query vector in
threadgroup memory and streams the keys past it, keeping `(m, l, acc)` in
registers. Device traffic is `O(T·D)` per query and the `T×T` score matrix never
exists anywhere. This is lesson 2's algorithm, transcribed.

Correctness: matches `F.scaled_dot_product_attention` to 1e-6, causal and
non-causal, for several shapes.

Performance: **~17× slower than PyTorch's fused kernel** (6.89 ms vs 0.41 ms at
B1 H8 T512 D64), and the benchmark prints exactly that. The reason is structural, not a bug: each threadgroup walks
the keys *serially* and does a full block-wide reduction per key, so
reduction latency sets the runtime. Production kernels tile over **queries**
too, so one K/V tile in threadgroup memory feeds 32–128 queries and the inner
loop becomes a register-blocked matmul instead of a reduction. That restructuring
is the actual work of kernel engineering — and it is what lesson 13's Triton
version does.

## Correctness before performance

Every kernel in `kernels/metal/bench.py` is checked against a PyTorch reference
*before* it is timed. A fast wrong kernel is worth nothing, and GPU bugs are
uniquely nasty: an out-of-bounds read returns adjacent memory rather than
crashing, so you get plausible wrong numbers.

`tests/test_kernels.py` also checks the input guards — wrong device, wrong
dtype, non-contiguous, mismatched shapes — because a kernel that indexes raw
memory must validate, or a non-contiguous tensor silently produces garbage.

Note `rel_err < 1e-4` rather than exact equality for matmul: fp32 accumulation
in a different order gives ~1e-6 relative error for free. That is rounding, not
a bug; above ~1e-4 it is a bug.

## The CUDA file

`kernels/cuda/kernels.cu` is the same six kernels with CUDA spelling, built as a
`torch.utils.cpp_extension` module. **It has not been compiled on the machine
this was written on** — there is no `nvcc` here — which `kernels/cuda/README.md`
states plainly. It mirrors the Metal kernels that *are* verified, and
`kernels/cuda/bench.py` runs the identical correctness checks, so the first
thing to do on a real GPU is run it.

Read the two files side by side. The interesting differences:

- CUDA's `__shfl_down_sync(0xffffffff, v, offset)` takes an explicit lane mask,
  which is only correct if the whole warp reaches that line. That is why every
  kernel uses a block size that is a multiple of 32 and never reduces inside a
  divergent branch. Metal's `simd_sum` has the same requirement, implicitly.
- CUDA has dynamic shared memory (`extern __shared__`, sized at launch); the
  Metal path here uses static arrays.
- CUDA scalars pass by value; Metal needs `constant uint&` buffer bindings.

Then profile it, which is the part a Mac cannot teach you:

```bash
ncu --set full python -m kernels.cuda.bench      # per-kernel, occupancy, roofline
nsys profile python -m kernels.cuda.bench        # timeline: launch-bound? memcpy-bound?
```

The two numbers to read in `ncu`: **Memory Throughput %** vs **Compute (SM)
Throughput %** — whichever is higher is your bound — and **Achieved Occupancy**,
which tells you whether registers or shared memory are limiting you rather than
your algorithm.

## When to write a kernel at all

1. **Try `torch.compile` first.** It does fusion 4 automatically.
2. **Write one when you need fusion that the compiler will not do** — that is
   where the 7.4× lives.
3. **Write one when you need an algorithm the framework does not have** —
   FlashAttention, paged attention, fused MoE dispatch.
4. **Do not write matmul.** Ever.

## Exercises

1. **Register-block the matmul.** Make each thread compute a 4×4 micro-tile held
   in registers. This is the biggest single step toward the vendor library.
   Measure the TFLOP/s before and after.

2. **Double-buffer the tiles.** Prefetch tile `t+1` into a second threadgroup
   buffer while computing on tile `t`, so loads overlap the math and you need
   one barrier per iteration instead of two.

3. **Vectorise `vector_add`** with `float4` loads. Memory-bound kernels
   typically go from ~70% to ~90% of peak bandwidth. Then explain why it helps at
   all, given that the total bytes are unchanged. Use `--repeats 9` and check
   that your gain exceeds the printed interval before believing it.

4. **Sweep the threadgroup size** for `row_sum` at 64, 128, 256, 512, 1024 and
   for several row lengths. Plot the optimum. Why is the largest not always best?

5. **Tile flash attention over queries.** Give each threadgroup `BLOCK_M = 32`
   queries and stage the K/V tile in threadgroup memory. You should close most of
   the 17× gap. This is the hardest exercise here and the most instructive.

6. **Fuse a bigger chain.** Write one kernel for `RMSNorm → q/k/v projection`,
   avoiding the intermediate normalized tensor entirely. Measure against the
   eager sequence.

7. **Deliberately deadlock.** Move the `threadgroup_barrier` inside the
   `if (tid < s)` in `row_sum` and see what happens. Understanding the failure
   mode is worth more than avoiding it by rote.

---

**Previous:** [Lesson 11](11_lora.md) · **Next:** [Lesson 13 — Triton](13_triton.md)
