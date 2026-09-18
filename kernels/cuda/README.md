# CUDA kernels

`kernels.cu` implements the same six kernels as `kernels/metal/kernels.py`.
Read them side by side: the algorithms are identical, only the spelling changes.
The concept table at the top of each file is the translation dictionary.

**This code has not been compiled on the machine it was written on** — this repo
was built on an Apple Silicon Mac, which has no `nvcc`. It is written to the
same structure as the Metal kernels that *are* verified here (`python -m
kernels.metal.bench` passes), and `bench.py` runs the identical correctness
checks, so the first thing to do on a real GPU is run it.

## Running it

Any NVIDIA GPU with a CUDA-enabled PyTorch build:

```bash
pip install torch          # a CUDA build, not the CPU/MPS wheel
export TORCH_CUDA_ARCH_LIST=8.9        # your arch: 8.0 A100, 8.9 L4/4090, 9.0 H100
python -m kernels.cuda.bench
```

The first run JIT-compiles with `nvcc` (~1 min) and caches the result in
`~/.cache/torch_extensions`. Set `VERBOSE=1` to see the compiler command line
— worth doing once, because it shows exactly which flags `cpp_extension` adds.

In Colab, pick a GPU runtime and run:

```python
!git clone <this repo> && cd ai-learning && python -m kernels.cuda.bench
```

## Profiling — the part that actually teaches you something

Correctness first, then find out *why* a kernel is slow:

```bash
# Per-kernel timings and the achieved occupancy
ncu --set full -o profile python -m kernels.cuda.bench

# Timeline: are you launch-bound? memcpy-bound? serialised on the host?
nsys profile -o timeline python -m kernels.cuda.bench
```

Two numbers to look for in `ncu`:

- **Memory Throughput %** vs **Compute (SM) Throughput %.** Whichever is higher
  is your bound. `vector_add` and `rmsnorm` should be >80% memory; `matmul_tiled`
  should be neither, which is the signal that it is latency-bound and needs
  register blocking (more than one output per thread).
- **Achieved Occupancy.** If it is far below theoretical, you are limited by
  registers or shared memory per block, not by your algorithm.

## Exercises

1. `matmul_tiled` computes one output per thread. Make each thread compute a
   4×4 micro-tile held in registers. This is the single biggest step from ~1
   TFLOP/s toward cuBLAS, because it raises arithmetic intensity without needing
   more shared memory.
2. Double-buffer the tile loads in `matmul_tiled`: prefetch tile `t+1` into a
   second shared buffer while computing on tile `t`, so the loads overlap the
   math and you need only one `__syncthreads()` per iteration.
3. Add `float4` vectorised loads to `vector_add`. One `LDG.E.128` instead of
   four `LDG.E.32` typically gets a memory-bound kernel from ~70% to ~90% of
   peak bandwidth.
4. Tile `flash_attn_fwd` over queries (e.g. 32 queries per block) and stage the
   K/V tile in shared memory. The inner loop becomes a small matmul and the
   kernel stops being reduction-latency-bound. Compare against
   `torch.nn.functional.scaled_dot_product_attention`.
5. Use `nvcc --ptxas-options=-v` to print register and shared-memory usage per
   kernel, then work out the occupancy limit by hand and check it against `ncu`.
