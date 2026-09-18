# Lesson 0 — Setup, and the ground rules

Read: `minigpt/utils.py`

## Install

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -e ".[dev]"
source .venv/bin/activate
python -m pytest -q
```

If the tests pass you have a working Metal GPU, a working PyTorch, and a
verified copy of everything in this repo.

## Your hardware

```python
import torch
torch.backends.mps.is_available()   # Apple GPU via the MPS backend
torch.mps.compile_shader            # runtime Metal compilation — lesson 12 needs this
```

On an M1 Max this repo measures:

- **~140 GB/s** achieved memory bandwidth (`kernels/metal/bench.py`, vector add)
- **~6.4 TFLOP/s** fp32 matmul at 2048³ (`measure_peak_flops`) — but only
  **~3.5 TFLOP/s** at 1024³ (`kernels/metal/bench.py`). Peak throughput is a
  function of problem size, so a single "peak FLOPS" number is meaningless
  without the shape it was measured at. This matters in lesson 4: MFU is a
  ratio, and quoting the wrong denominator moves it by 2x.
- **1024** threads per threadgroup, **32** lanes per SIMD group

Those three numbers explain most of what follows. A transformer forward pass
does `2N` FLOPs per token and reads `N` weights, so the *ratio* of compute to
bandwidth (here ~23 FLOP per byte) decides whether any given operation is
limited by arithmetic or by memory. Training, with large batches, is
compute-bound. Single-stream decoding is bandwidth-bound. Almost every
engineering decision in lessons 5, 10 and 12 follows from that split.

## Three ground rules

### 1. Never time without synchronizing

GPU work is queued asynchronously. `t0 = time(); y = x @ x; print(time()-t0)`
measures how long it took to *enqueue* a matmul — ~34 µs here, and essentially
independent of the matrix size. `minigpt/utils.py` has `sync()`, `timer()` and `benchmark()`; the last one
warms up, takes the median of several runs, and synchronizes around each.

```python
from minigpt.utils import benchmark
benchmark(lambda: a @ b, device="mps")     # seconds per call, honest
```

### 2. Check numerics in float64, run in float32

Almost every bug in this repo's history was invisible at float32 tolerances.
The FlashAttention backward was "correct to 1e-7" — which is exactly what a
*wrong* implementation looks like when your reference is also float32. In
float64 it was exactly right, and `torch.autograd.gradcheck` proved it.

So: tests use `dtype=torch.float64` and tolerances like `atol=1e-10`. Anything
that only agrees to 1e-6 is being compared in the wrong precision, and you
cannot tell a rounding difference from a real error.

`MINIGPT_DEVICE=cpu` forces CPU, because MPS has no float64 at all.

### 3. Relative error, not absolute

For kernels the standard metric is relative L2, `‖a−b‖ / ‖b‖`
(`minigpt.utils.rel_err`). An absolute tolerance is meaningless without knowing
the scale of the output. fp32 matmul reassociation gives ~1e-6 relative error
for free; anything above ~1e-4 is a bug, not rounding.

## The one formula to memorise

```
training FLOPs ≈ 6 · N · D
```

`N` parameters, `D` training tokens: 2·N per token forward, ~4·N backward. Use
it constantly:

```python
from minigpt.config import preset
cfg = preset("llama7b")
cfg.param_count()      # {'total': 6.74e9, 'attention': 2.15e9, 'mlp': 4.33e9, ...}
cfg.flops_per_token()  # {'dense': ..., 'attention': ..., 'fraction_attention': 0.075}
cfg.kv_cache_bytes(1)  # 2.15 GB for one 4k-token sequence in fp16
```

The formula ignores attention's quadratic term, which is fine while
`seq_len ≪ 12 · n_embd` and badly wrong past that — `flops_per_token` reports
it separately so you can see when. Lesson 3 derives all of it.

## How to read this repo

The library is meant to be read. Each module's docstring explains the *why*,
and inline comments flag the traps — the places where a plausible-looking
implementation is subtly wrong. The tests are the specification: when a lesson
claims something ("RoPE depends only on relative position", "speculative
decoding is exact"), there is a test with that name.

Suggested loop per lesson: read the lesson, read the module, run its tests with
`-v` to see the claims as a list, then do the exercises.

---

**Next:** [Lesson 1 — Tokenization](01_tokenization.md)
