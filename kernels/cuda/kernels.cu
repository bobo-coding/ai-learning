// The same six kernels as kernels/metal/kernels.py, in CUDA.
//
// Read the two files side by side: the algorithms are identical and only the
// vocabulary changes.  This file cannot run on a Mac (no NVIDIA GPU), so it is
// here to be *read* and to be run on a rented GPU or Colab -- see README.md.
//
//   Metal                              CUDA
//   --------------------------------   -----------------------------------
//   kernel void f(...)                 __global__ void f(...)
//   device const float*                const float* __restrict__
//   constant uint&                     uint (by value)
//   threadgroup float sh[N]            __shared__ float sh[N]
//   threadgroup_barrier(...)           __syncthreads()
//   thread_position_in_threadgroup     threadIdx.x
//   threadgroup_position_in_grid       blockIdx.x
//   threads_per_threadgroup            blockDim.x
//   thread_position_in_grid            blockIdx.x * blockDim.x + threadIdx.x
//   simd_sum(v)                        __shfl_down_sync reduction (below)
//   thread_index_in_simdgroup          threadIdx.x % warpSize
//   dispatchThreads(TOTAL_THREADS)     f<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>
//
// The launch-shape difference is the one that actually bites: Metal takes a
// total thread count, CUDA takes a *block* count.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#define CHECK_CUDA(x)  TORCH_CHECK((x).is_cuda(),  #x " must be a CUDA tensor")
#define CHECK_CONT(x)  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_F32(x)   TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_IN(x)    CHECK_CUDA(x); CHECK_CONT(x); CHECK_F32(x)

namespace {

constexpr int WARP = 32;

// ---------------------------------------------------------------------------
// Warp- and block-level reductions.
// ---------------------------------------------------------------------------
// `__shfl_down_sync` moves a register value between lanes of the same warp with
// no memory traffic and no barrier -- the CUDA equivalent of Metal's simd_sum.
// The 0xffffffff mask says "all 32 lanes participate", which is only correct
// when the whole warp reaches this line; that is why every kernel below uses a
// block size that is a multiple of 32 and never reduces inside a divergent if.
__inline__ __device__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int off = WARP / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

__inline__ __device__ float warp_reduce_max(float v) {
  #pragma unroll
  for (int off = WARP / 2; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  return v;
}

// Two-stage block reduction: reduce within each warp, write one value per warp
// to shared memory, then have warp 0 reduce those.  `__syncthreads()` is needed
// only once, between the stages.
__inline__ __device__ float block_reduce_sum(float v, float* shared) {
  const int lane = threadIdx.x % WARP;
  const int wid  = threadIdx.x / WARP;
  v = warp_reduce_sum(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  const int n_warps = (blockDim.x + WARP - 1) / WARP;
  v = (threadIdx.x < n_warps) ? shared[threadIdx.x] : 0.0f;
  if (wid == 0) v = warp_reduce_sum(v);
  return v;  // valid in thread 0
}

__inline__ __device__ float block_reduce_max(float v, float* shared) {
  const int lane = threadIdx.x % WARP;
  const int wid  = threadIdx.x / WARP;
  v = warp_reduce_max(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();
  const int n_warps = (blockDim.x + WARP - 1) / WARP;
  v = (threadIdx.x < n_warps) ? shared[threadIdx.x] : -FLT_MAX;
  if (wid == 0) v = warp_reduce_max(v);
  return v;  // valid in thread 0
}

// ---------------------------------------------------------------------------
// 1. Elementwise
// ---------------------------------------------------------------------------
__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

__global__ void vector_add_stride_kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ out, int n) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    out[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// 2. Row reduction
// ---------------------------------------------------------------------------
__global__ void row_sum_kernel(const float* __restrict__ x,
                               float* __restrict__ out, int N) {
  __shared__ float shared[WARP];
  const long base = (long)blockIdx.x * N;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) acc += x[base + i];
  acc = block_reduce_sum(acc, shared);
  if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

// ---------------------------------------------------------------------------
// 3. Row softmax (numerically stable, two reductions)
// ---------------------------------------------------------------------------
__global__ void softmax_rows_kernel(const float* __restrict__ x,
                                    float* __restrict__ out, int N) {
  __shared__ float shared[WARP];
  __shared__ float bcast;
  const long base = (long)blockIdx.x * N;

  float m = -FLT_MAX;
  for (int i = threadIdx.x; i < N; i += blockDim.x) m = fmaxf(m, x[base + i]);
  m = block_reduce_max(m, shared);
  if (threadIdx.x == 0) bcast = m;
  __syncthreads();
  const float row_max = bcast;
  __syncthreads();  // protect `bcast` before it is reused below

  float s = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) s += __expf(x[base + i] - row_max);
  s = block_reduce_sum(s, shared);
  if (threadIdx.x == 0) bcast = s;
  __syncthreads();
  const float inv = 1.0f / bcast;

  for (int i = threadIdx.x; i < N; i += blockDim.x)
    out[base + i] = __expf(x[base + i] - row_max) * inv;
}

// ---------------------------------------------------------------------------
// 4. Fused RMSNorm
// ---------------------------------------------------------------------------
__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ g,
                               float* __restrict__ out, int N, float eps) {
  __shared__ float shared[WARP];
  __shared__ float bcast;
  const long base = (long)blockIdx.x * N;

  float ss = 0.0f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    const float v = x[base + i];
    ss += v * v;
  }
  ss = block_reduce_sum(ss, shared);
  if (threadIdx.x == 0) bcast = rsqrtf(ss / (float)N + eps);
  __syncthreads();
  const float scale = bcast;

  for (int i = threadIdx.x; i < N; i += blockDim.x)
    out[base + i] = x[base + i] * scale * g[i];
}

// ---------------------------------------------------------------------------
// 5. Matmul
// ---------------------------------------------------------------------------
__global__ void matmul_naive_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C, int M, int N, int K) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += A[(long)row * K + k] * B[(long)k * N + col];
  C[(long)row * N + col] = acc;
}

#define TILE 16
__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;

  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    const int a_col = t * TILE + threadIdx.x;
    const int b_row = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? A[(long)row * K + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? B[(long)b_row * N + col] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) C[(long)row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// 6. FlashAttention forward -- one block per (batch*head, query)
// ---------------------------------------------------------------------------
__global__ void flash_attn_fwd_kernel(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      float* __restrict__ O,
                                      int T, int D, float scale, int causal) {
  extern __shared__ float smem[];      // [D] query + [D] accumulator
  float* q_sh   = smem;
  float* acc_sh = smem + D;
  __shared__ float shared[WARP];
  __shared__ float bcast;

  const int bh = blockIdx.x / T;
  const int qi = blockIdx.x % T;
  const long base = (long)bh * T * D;

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    q_sh[d]   = Q[base + (long)qi * D + d] * scale;
    acc_sh[d] = 0.0f;
  }
  __syncthreads();

  float m_run = -INFINITY;
  float l_run = 0.0f;
  const int k_limit = causal ? (qi + 1) : T;

  for (int kj = 0; kj < k_limit; ++kj) {
    float dot = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
      dot += q_sh[d] * K[base + (long)kj * D + d];
    dot = block_reduce_sum(dot, shared);
    if (threadIdx.x == 0) bcast = dot;
    __syncthreads();
    const float score = bcast;

    const float m_new = fmaxf(m_run, score);
    const float corr  = (m_run == -INFINITY) ? 0.0f : __expf(m_run - m_new);
    const float p     = __expf(score - m_new);
    l_run = l_run * corr + p;
    m_run = m_new;

    for (int d = threadIdx.x; d < D; d += blockDim.x)
      acc_sh[d] = acc_sh[d] * corr + p * V[base + (long)kj * D + d];
    __syncthreads();
  }

  const float inv_l = (l_run > 0.0f) ? 1.0f / l_run : 0.0f;
  for (int d = threadIdx.x; d < D; d += blockDim.x)
    O[base + (long)qi * D + d] = acc_sh[d] * inv_l;
}

}  // namespace

// ---------------------------------------------------------------------------
// Host-side launchers (the torch-facing API)
// ---------------------------------------------------------------------------

at::Tensor vector_add(at::Tensor a, at::Tensor b, bool stride_loop) {
  CHECK_IN(a); CHECK_IN(b);
  TORCH_CHECK(a.numel() == b.numel(), "size mismatch");
  const at::cuda::OptionalCUDAGuard guard(at::device_of(a));
  auto out = at::empty_like(a);
  const int n = a.numel();
  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (stride_loop) {
    vector_add_stride_kernel<<<16, threads, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
  } else {
    const int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
  }
  return out;
}

at::Tensor row_sum(at::Tensor x) {
  CHECK_IN(x);
  const at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  auto x2 = x.reshape({-1, x.size(-1)});
  const int rows = x2.size(0), N = x2.size(1);
  auto out = at::empty({rows}, x.options());
  row_sum_kernel<<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      x2.data_ptr<float>(), out.data_ptr<float>(), N);
  auto shape = x.sizes().vec();
  shape.pop_back();
  return out.reshape(shape);
}

at::Tensor softmax_rows(at::Tensor x) {
  CHECK_IN(x);
  const at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  auto x2 = x.reshape({-1, x.size(-1)});
  const int rows = x2.size(0), N = x2.size(1);
  auto out = at::empty_like(x2);
  softmax_rows_kernel<<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      x2.data_ptr<float>(), out.data_ptr<float>(), N);
  return out.reshape(x.sizes());
}

at::Tensor rmsnorm(at::Tensor x, at::Tensor g, double eps) {
  CHECK_IN(x); CHECK_IN(g);
  const at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  auto x2 = x.reshape({-1, x.size(-1)});
  const int rows = x2.size(0), N = x2.size(1);
  TORCH_CHECK(g.numel() == N, "weight size must equal last dim of x");
  auto out = at::empty_like(x2);
  rmsnorm_kernel<<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      x2.data_ptr<float>(), g.data_ptr<float>(), out.data_ptr<float>(), N, (float)eps);
  return out.reshape(x.sizes());
}

at::Tensor matmul(at::Tensor a, at::Tensor b, bool tiled) {
  CHECK_IN(a); CHECK_IN(b);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0), "bad matmul shapes");
  const at::cuda::OptionalCUDAGuard guard(at::device_of(a));
  const int M = a.size(0), K = a.size(1), N = b.size(1);
  auto c = at::empty({M, N}, a.options());
  const dim3 block(TILE, TILE);
  const dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (tiled) {
    matmul_tiled_kernel<<<grid, block, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
  } else {
    matmul_naive_kernel<<<grid, block, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
  }
  return c;
}

at::Tensor flash_attention(at::Tensor q, at::Tensor k, at::Tensor v, bool causal) {
  CHECK_IN(q); CHECK_IN(k); CHECK_IN(v);
  TORCH_CHECK(q.dim() == 4, "expected (B, H, T, D)");
  const at::cuda::OptionalCUDAGuard guard(at::device_of(q));
  const int B = q.size(0), H = q.size(1), T = q.size(2), D = q.size(3);
  auto out = at::empty_like(q);
  auto qf = q.reshape({B * H, T, D});
  auto kf = k.reshape({B * H, T, D});
  auto vf = v.reshape({B * H, T, D});
  auto of = out.reshape({B * H, T, D});
  const int threads = 128;
  const size_t smem = 2 * (size_t)D * sizeof(float);   // query + accumulator
  flash_attn_fwd_kernel<<<B * H * T, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
      qf.data_ptr<float>(), kf.data_ptr<float>(), vf.data_ptr<float>(), of.data_ptr<float>(),
      T, D, 1.0f / std::sqrt((float)D), causal ? 1 : 0);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add", &vector_add, "a + b", py::arg("a"), py::arg("b"),
        py::arg("stride_loop") = false);
  m.def("row_sum", &row_sum, "sum over the last dim");
  m.def("softmax_rows", &softmax_rows, "softmax over the last dim");
  m.def("rmsnorm", &rmsnorm, "fused RMSNorm", py::arg("x"), py::arg("g"), py::arg("eps") = 1e-5);
  m.def("matmul", &matmul, "C = A @ B", py::arg("a"), py::arg("b"), py::arg("tiled") = true);
  m.def("flash_attention", &flash_attention, "causal attention",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = true);
}
