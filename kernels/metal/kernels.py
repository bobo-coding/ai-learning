"""GPU kernels written in Metal Shading Language, runnable on this Mac.

`torch.mps.compile_shader` compiles MSL at runtime and hands back callables that
take torch tensors directly.  That makes an Apple Silicon Mac a perfectly good
place to learn GPU programming: the concepts are identical to CUDA, only the
nouns change.

    CUDA                        Metal                       what it is
    ------------------------    ------------------------    --------------------
    thread                      thread                      one lane of execution
    warp (32)                   SIMD-group (32)              lockstep group
    block / threadgroup         threadgroup                 shares fast memory
    grid                        grid                        all threads
    __shared__                  threadgroup                 ~64 KB, ~L1 latency
    __syncthreads()             threadgroup_barrier(...)    block-wide barrier
    __shfl_down_sync            simd_shuffle_down            lane-to-lane move
    threadIdx.x                 thread_position_in_threadgroup
    blockIdx.x                  threadgroup_position_in_grid
    blockDim.x                  threads_per_threadgroup
    gridDim.x * blockDim.x      threads_per_grid
    cudaDeviceSynchronize()     torch.mps.synchronize()

One real difference in the launch API: CUDA's `<<<blocks, threads>>>` takes the
number of *blocks*, while Metal's `dispatchThreads` (what `threads=` is here)
takes the total number of *threads*.  Off-by-a-factor-of-group_size is the first
bug everyone writes.

The lessons, in order:
  1. `vector_add`     -- indexing and bounds checks
  2. `row_sum`        -- threadgroup memory + tree reduction + SIMD shuffles
  3. `softmax_rows`   -- a numerically stable two-pass reduction
  4. `rmsnorm`        -- kernel fusion: 4 memory passes become 1
  5. `matmul_*`       -- naive vs tiled: why arithmetic intensity is everything
  6. `flash_attention`-- online softmax in registers, no T x T matrix
"""

from __future__ import annotations

import torch

# Every kernel below is bounds-checked, so the grid can be rounded up to a
# multiple of the threadgroup size -- which it must be, since Metal dispatches
# whole threadgroups.
METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// 1. Elementwise: the "hello world" of GPU programming.
// ---------------------------------------------------------------------------
// One thread per element.  The bounds check exists because the grid is rounded
// up to a whole number of threadgroups, so the last group has idle lanes.
kernel void vector_add(device const float* a   [[buffer(0)]],
                       device const float* b   [[buffer(1)]],
                       device float*       out [[buffer(2)]],
                       constant uint&      n   [[buffer(3)]],
                       uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = a[gid] + b[gid];
}

// A grid-stride loop: decouples the grid size from the problem size, so one
// launch configuration works for any n.  Idiomatic and portable -- but the grid
// still has to be big enough to saturate memory, which the benchmark shows.
kernel void vector_add_stride(device const float* a   [[buffer(0)]],
                              device const float* b   [[buffer(1)]],
                              device float*       out [[buffer(2)]],
                              constant uint&      n   [[buffer(3)]],
                              uint gid    [[thread_position_in_grid]],
                              uint stride [[threads_per_grid]])
{
    for (uint i = gid; i < n; i += stride) out[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// 2. Reduction: one threadgroup per row.
// ---------------------------------------------------------------------------
#define TG 256

// Classic tree reduction in threadgroup memory.  Three things to notice:
//   * the serial grid-stride prologue means TG threads can reduce any N;
//   * the barrier must be OUTSIDE the `if (tid < s)`, or threads that skip the
//     body never arrive and the group deadlocks (or worse, reads stale data);
//   * log2(TG) barriers is the cost -- which is why the SIMD version below,
//     with no barriers inside a 32-lane group, is faster.
kernel void row_sum(device const float* x   [[buffer(0)]],
                    device float*       out [[buffer(1)]],
                    constant uint&      N   [[buffer(2)]],
                    uint grp [[threadgroup_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint gsz [[threads_per_threadgroup]])
{
    threadgroup float sh[TG];
    float acc = 0.0f;
    for (uint i = tid; i < N; i += gsz) acc += x[grp * N + i];
    sh[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = gsz / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[grp] = sh[0];
}

// The same reduction using SIMD-group intrinsics.  `simd_sum` reduces across
// the 32 lanes of one SIMD-group with no shared memory and no barrier at all
// (the lanes are already in lockstep).  We then reduce the <=8 per-SIMD partial
// sums.  This is the standard shape of a fast reduction on any modern GPU.
kernel void row_sum_simd(device const float* x   [[buffer(0)]],
                         device float*       out [[buffer(1)]],
                         constant uint&      N   [[buffer(2)]],
                         uint grp      [[threadgroup_position_in_grid]],
                         uint tid      [[thread_position_in_threadgroup]],
                         uint gsz      [[threads_per_threadgroup]],
                         uint simd_lane [[thread_index_in_simdgroup]],
                         uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[32];
    float acc = 0.0f;
    for (uint i = tid; i < N; i += gsz) acc += x[grp * N + i];

    acc = simd_sum(acc);                       // 32 lanes -> lane 0
    if (simd_lane == 0) partial[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint n_simd = (gsz + 31) / 32;
        float total = 0.0f;
        for (uint i = 0; i < n_simd; ++i) total += partial[i];
        out[grp] = total;
    }
}

// ---------------------------------------------------------------------------
// 3. Row softmax: two reductions, then a write.
// ---------------------------------------------------------------------------
// Numerically stable form: subtract the row max before exponentiating.  Without
// it, a logit of 100 overflows fp32 (exp(100) = 2.7e43 is fine, exp(800) is
// inf) and the whole row becomes NaN.  We broadcast both reductions through
// threadgroup memory.
kernel void softmax_rows(device const float* x   [[buffer(0)]],
                         device float*       out [[buffer(1)]],
                         constant uint&      N   [[buffer(2)]],
                         uint grp [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint gsz [[threads_per_threadgroup]],
                         uint simd_lane [[thread_index_in_simdgroup]],
                         uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[32];
    threadgroup float shared_val;
    const uint base = grp * N;
    const uint n_simd = (gsz + 31) / 32;

    // pass 1: row max
    float m = -INFINITY;
    for (uint i = tid; i < N; i += gsz) m = max(m, x[base + i]);
    m = simd_max(m);
    if (simd_lane == 0) partial[simd_id] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = -INFINITY;
        for (uint i = 0; i < n_simd; ++i) t = max(t, partial[i]);
        shared_val = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float row_max = shared_val;

    // pass 2: sum of exp
    float s = 0.0f;
    for (uint i = tid; i < N; i += gsz) s += exp(x[base + i] - row_max);
    s = simd_sum(s);
    if (simd_lane == 0) partial[simd_id] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (uint i = 0; i < n_simd; ++i) t += partial[i];
        shared_val = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / shared_val;

    // pass 3: write
    for (uint i = tid; i < N; i += gsz) out[base + i] = exp(x[base + i] - row_max) * inv;
}

// ---------------------------------------------------------------------------
// 4. Fused RMSNorm: y = x / sqrt(mean(x^2) + eps) * g
// ---------------------------------------------------------------------------
// In eager PyTorch this is pow, mean, rsqrt, mul, mul -- five kernels, each
// reading and writing the whole tensor.  Fused, it reads x twice and writes
// once.  For a memory-bound op that is a ~2.5x speedup, and it is the single
// most common reason to write a custom kernel at all.
kernel void rmsnorm(device const float* x   [[buffer(0)]],
                    device const float* g   [[buffer(1)]],
                    device float*       out [[buffer(2)]],
                    constant uint&      N   [[buffer(3)]],
                    constant float&     eps [[buffer(4)]],
                    uint grp [[threadgroup_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint gsz [[threads_per_threadgroup]],
                    uint simd_lane [[thread_index_in_simdgroup]],
                    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[32];
    threadgroup float shared_scale;
    const uint base = grp * N;

    float ss = 0.0f;
    for (uint i = tid; i < N; i += gsz) { float v = x[base + i]; ss += v * v; }
    ss = simd_sum(ss);
    if (simd_lane == 0) partial[simd_id] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint n_simd = (gsz + 31) / 32;
        float t = 0.0f;
        for (uint i = 0; i < n_simd; ++i) t += partial[i];
        shared_scale = rsqrt(t / float(N) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = shared_scale;

    for (uint i = tid; i < N; i += gsz) out[base + i] = x[base + i] * scale * g[i];
}

// ---------------------------------------------------------------------------
// 5. Matmul: C = A @ B,  A (M,K), B (K,N), C (M,N)
// ---------------------------------------------------------------------------
// Naive version: each thread computes one output element and reads a full row
// of A and column of B from device memory.  Total device traffic is
// 2*M*N*K floats for 2*M*N*K FLOPs -- an arithmetic intensity of 0.25 FLOP per
// byte, when the hardware needs ~50 to be compute-bound.  So this runs at a few
// percent of peak no matter how good the ALUs are.
kernel void matmul_naive(device const float* A [[buffer(0)]],
                         device const float* B [[buffer(1)]],
                         device float*       C [[buffer(2)]],
                         constant uint&      M [[buffer(3)]],
                         constant uint&      N [[buffer(4)]],
                         constant uint&      K [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x, row = gid.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

#define TILE 16
// Tiled version: the threadgroup cooperatively stages a TILE x TILE block of A
// and of B into threadgroup memory, then every thread reads those TILE values
// from fast memory instead of device memory.  Each loaded value is now reused
// TILE times, so device traffic drops by a factor of TILE and arithmetic
// intensity rises to ~TILE/2 FLOP per byte.
//
// The two barriers are both mandatory and for different reasons: the first
// publishes the tile before anyone reads it, the second stops a fast thread
// from overwriting the tile while a slow one is still reading it.
kernel void matmul_tiled(device const float* A [[buffer(0)]],
                         device const float* B [[buffer(1)]],
                         device float*       C [[buffer(2)]],
                         constant uint&      M [[buffer(3)]],
                         constant uint&      N [[buffer(4)]],
                         constant uint&      K [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]],
                         uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];

    const uint row = gid.y, col = gid.x;
    float acc = 0.0f;

    for (uint t = 0; t < (K + TILE - 1) / TILE; ++t) {
        const uint a_col = t * TILE + tid.x;
        const uint b_row = t * TILE + tid.y;
        As[tid.y][tid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[tid.y][tid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; ++k) acc += As[tid.y][k] * Bs[k][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// 6. FlashAttention forward, one threadgroup per (batch*head, query)
// ---------------------------------------------------------------------------
// The online-softmax recurrence, now in registers.  Each threadgroup owns one
// query vector and streams the keys/values past it, keeping only (m, l, acc).
// Device memory traffic is O(T*D) per query instead of O(T^2), and the T x T
// score matrix is never written anywhere.
//
// This is the "one query per group" shape: simple, and already the right
// algorithm.  It is also, measurably, ~15x slower than PyTorch's fused kernel,
// and the benchmark prints exactly that -- because each threadgroup walks the
// keys *serially* and reduces a dot product across the whole group per key, so
// the reduction latency, not memory or math, sets the runtime.  Production
// kernels tile over queries too, so one K/V tile in threadgroup memory feeds
// many queries and the inner loop becomes a register-blocked matmul.  Getting
// from here to there is the actual work of kernel engineering.
#define MAXD 128
kernel void flash_attn_fwd(device const float* Q [[buffer(0)]],
                           device const float* K [[buffer(1)]],
                           device const float* V [[buffer(2)]],
                           device float*       O [[buffer(3)]],
                           constant uint&      T [[buffer(4)]],
                           constant uint&      D [[buffer(5)]],
                           constant float&  scale [[buffer(6)]],
                           constant uint&  causal [[buffer(7)]],
                           uint grp [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint gsz [[threads_per_threadgroup]],
                           uint simd_lane [[thread_index_in_simdgroup]],
                           uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[32];
    threadgroup float q_sh[MAXD];
    threadgroup float acc_sh[MAXD];
    threadgroup float bcast[2];

    const uint bh = grp / T;              // which (batch, head)
    const uint qi = grp % T;              // which query position
    const uint qk_base = bh * T * D;
    const uint n_simd = (gsz + 31) / 32;

    // Stage this query vector in threadgroup memory: every key iteration reads
    // all D of its components, so loading it once is a large win.
    for (uint d = tid; d < D; d += gsz) {
        q_sh[d] = Q[qk_base + qi * D + d] * scale;
        acc_sh[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_run = -INFINITY;              // running max
    float l_run = 0.0f;                   // running denominator
    const uint k_limit = (causal != 0) ? (qi + 1) : T;

    for (uint kj = 0; kj < k_limit; ++kj) {
        // dot(q, k_j), reduced across the threadgroup
        float dot = 0.0f;
        for (uint d = tid; d < D; d += gsz) dot += q_sh[d] * K[qk_base + kj * D + d];
        dot = simd_sum(dot);
        if (simd_lane == 0) partial[simd_id] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float s = 0.0f;
            for (uint i = 0; i < n_simd; ++i) s += partial[i];
            bcast[0] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float score = bcast[0];

        // online softmax update -- identical algebra to minigpt.attention
        const float m_new = max(m_run, score);
        const float correction = (m_run == -INFINITY) ? 0.0f : exp(m_run - m_new);
        const float p = exp(score - m_new);
        l_run = l_run * correction + p;
        m_run = m_new;

        // rescale the accumulator, then add p * v_j
        for (uint d = tid; d < D; d += gsz)
            acc_sh[d] = acc_sh[d] * correction + p * V[qk_base + kj * D + d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    for (uint d = tid; d < D; d += gsz) O[qk_base + qi * D + d] = acc_sh[d] * inv_l;
}
"""

_LIB = None


def library():
    """Compile (once) and return the Metal shader library."""
    global _LIB
    if _LIB is None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Metal kernels need an Apple Silicon GPU (torch MPS backend)")
        _LIB = torch.mps.compile_shader(METAL_SOURCE)
    return _LIB


def _check(*tensors):
    for t in tensors:
        if t.device.type != "mps":
            raise ValueError("Metal kernels require tensors on device='mps'")
        if t.dtype != torch.float32:
            raise ValueError("these kernels are written for float32")
        if not t.is_contiguous():
            raise ValueError("pass contiguous tensors (a kernel indexes raw memory)")


def ceil_to(n: int, m: int) -> int:
    """Round `n` up to a multiple of `m` -- Metal dispatches whole threadgroups."""
    return ((n + m - 1) // m) * m


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def vector_add(a, b, group_size: int = 256, stride_loop: bool = False):
    _check(a, b)
    out = torch.empty_like(a)
    n = a.numel()
    lib = library()
    if stride_loop:
        # A deliberately small fixed grid (4096 threads), each looping.  This
        # is the portable form, but measure it: on this GPU it is ~2x SLOWER
        # than one-thread-per-element, because 4096 threads cannot keep enough
        # memory requests in flight to saturate bandwidth.  Occupancy, not
        # launch overhead, is what matters for a memory-bound kernel.
        lib.vector_add_stride(a, b, out, n, threads=4096, group_size=group_size)
    else:
        lib.vector_add(a, b, out, n, threads=ceil_to(n, group_size), group_size=group_size)
    return out


def row_sum(x, group_size: int = 256, simd: bool = True):
    _check(x)
    x2 = x.reshape(-1, x.shape[-1])
    rows, n = x2.shape
    out = torch.empty(rows, device=x.device, dtype=x.dtype)
    kern = library().row_sum_simd if simd else library().row_sum
    kern(x2, out, n, threads=rows * group_size, group_size=group_size)
    return out.reshape(x.shape[:-1])


def softmax_rows(x, group_size: int = 256):
    _check(x)
    x2 = x.reshape(-1, x.shape[-1])
    rows, n = x2.shape
    out = torch.empty_like(x2)
    library().softmax_rows(x2, out, n, threads=rows * group_size, group_size=group_size)
    return out.reshape(x.shape)


def rmsnorm(x, weight, eps: float = 1e-5, group_size: int = 256):
    _check(x, weight)
    x2 = x.reshape(-1, x.shape[-1])
    rows, n = x2.shape
    if weight.numel() != n:
        raise ValueError("weight must have the same size as the last dim of x")
    out = torch.empty_like(x2)
    library().rmsnorm(x2, weight, out, n, float(eps),
                      threads=rows * group_size, group_size=group_size)
    return out.reshape(x.shape)


def matmul(a, b, tiled: bool = True, tile: int = 16):
    _check(a, b)
    if a.dim() != 2 or b.dim() != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"bad shapes for matmul: {tuple(a.shape)} @ {tuple(b.shape)}")
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    kern = library().matmul_tiled if tiled else library().matmul_naive
    # x is the fastest-varying axis, so it maps to the output column: adjacent
    # threads then write adjacent addresses (a coalesced store).
    kern(a, b, c, M, N, K,
         threads=(ceil_to(N, tile), ceil_to(M, tile)), group_size=(tile, tile))
    return c


def flash_attention(q, k, v, causal: bool = True, group_size: int = 128):
    """Attention for (B, H, T, D) float32 MPS tensors, D <= 128."""
    _check(q, k, v)
    B, H, T, D = q.shape
    if D > 128:
        raise ValueError("this kernel stores the query in threadgroup memory: D <= 128")
    import math

    out = torch.empty_like(q)
    qf, kf, vf = (t.reshape(B * H, T, D).contiguous() for t in (q, k, v))
    of = out.reshape(B * H, T, D)
    library().flash_attn_fwd(qf, kf, vf, of, T, D, 1.0 / math.sqrt(D), 1 if causal else 0,
                             threads=B * H * T * group_size, group_size=group_size)
    return out
