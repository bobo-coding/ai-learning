"""`triton.language` (`tl`) implemented on PyTorch tensors.

Mental model for everything here: a value in a Triton kernel is a **block** --
a small tensor whose shape is known at compile time -- and the only way data
enters or leaves a block is `load`/`store` through a pointer expression with an
optional mask. Everything else is elementwise math and reductions over blocks.

The three things a Triton beginner gets wrong, all of which this module will
catch or warn about:

1. **Forgetting the mask.** `tl.arange(0, BLOCK)` produces `BLOCK` offsets even
   when only `n < BLOCK` are valid. On hardware, the extra loads read whatever
   is adjacent in memory (garbage, silently); `load` here raises IndexError
   instead.
2. **Confusing element offsets with byte offsets.** Triton pointer arithmetic is
   in *elements*, like C pointer arithmetic on a typed pointer.
3. **Forgetting that strides are yours to compute.** Triton has no notion of
   shape. You pass strides in and index with them; a tensor's contiguity is
   your problem.
"""

from __future__ import annotations

import math as _math
import warnings

import torch

from . import Ptr, program_context

# --- dtypes ---------------------------------------------------------------
# These are literally torch dtypes, so `x.to(tl.float32)` works unchanged.
float64 = torch.float64
float32 = torch.float32
float16 = torch.float16
bfloat16 = torch.bfloat16
int64 = torch.int64
int32 = torch.int32
int16 = torch.int16
int8 = torch.int8
int1 = torch.bool
uint8 = torch.uint8

pi = _math.pi


class constexpr:
    """`tl.constexpr`: a compile-time constant.

    Usable as an annotation (`BLOCK: tl.constexpr`) and as a wrapper. In real
    Triton this marks a value the compiler specialises on -- change it and the
    kernel is recompiled. Here it is transparent.
    """

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return int(self.value)

    def __int__(self):
        return int(self.value)

    def __repr__(self):
        return f"constexpr({self.value!r})"


def _v(x):
    return x.value if isinstance(x, constexpr) else x


# --- program/grid ---------------------------------------------------------


def program_id(axis: int = 0) -> int:
    """Index of this program instance along `axis` -- Triton's `blockIdx`."""
    pid, _ = program_context()
    return pid[int(_v(axis))]


def num_programs(axis: int = 0) -> int:
    _, grid = program_context()
    return grid[int(_v(axis))]


# --- block construction ---------------------------------------------------


def arange(start, end, dtype=int32) -> torch.Tensor:
    """`[start, end)` as an int32 block -- the basic index generator."""
    start, end = int(_v(start)), int(_v(end))
    n = end - start
    if n <= 0:
        raise ValueError(f"tl.arange({start}, {end}) is empty")
    if n & (n - 1):
        warnings.warn(
            f"tl.arange(0, {n}): real Triton requires the length to be a power of two. "
            "This runs here but will not compile on a GPU.",
            stacklevel=2,
        )
    return torch.arange(start, end, dtype=dtype)


def zeros(shape, dtype=float32) -> torch.Tensor:
    return torch.zeros([int(_v(s)) for s in shape], dtype=_v(dtype))


def full(shape, value, dtype=float32) -> torch.Tensor:
    return torch.full([int(_v(s)) for s in shape], float(_v(value)), dtype=_v(dtype))


def zeros_like(x) -> torch.Tensor:
    return torch.zeros_like(x)


def broadcast_to(x, shape) -> torch.Tensor:
    return x.broadcast_to([int(_v(s)) for s in shape])


def reshape(x, shape) -> torch.Tensor:
    return x.reshape([int(_v(s)) for s in shape])


def trans(x, *dims) -> torch.Tensor:
    if dims:
        return x.permute([int(_v(d)) for d in dims])
    return x.transpose(-1, -2)


def expand_dims(x, axis) -> torch.Tensor:
    return x.unsqueeze(int(_v(axis)))


def cat(a, b, can_reorder=False) -> torch.Tensor:
    return torch.cat([a, b], dim=-1)


# --- memory ---------------------------------------------------------------


def _index_of(ptr: Ptr):
    if not isinstance(ptr, Ptr):
        raise TypeError(f"tl.load/store expects a pointer, got {type(ptr).__name__}. "
                        "Did you pass a value where a `x_ptr + offs` expression belongs?")
    idx = ptr.offs
    if not torch.is_tensor(idx):
        idx = torch.tensor(idx)
    return idx.long(), ptr


def load(ptr, mask=None, other=0.0, **_ignored):
    """Gather from `ptr`, using `other` wherever `mask` is false.

    The bounds check is the reason to develop here: on a GPU an unmasked
    out-of-range load returns adjacent memory and your kernel produces subtly
    wrong numbers with no error. This raises.
    """
    idx, p = _index_of(ptr)
    n = p.flat.numel()
    if mask is None:
        bad = (idx < 0) | (idx >= n)
        if bool(bad.any()):
            first = int(idx.reshape(-1)[bad.reshape(-1).nonzero()[0]])
            raise IndexError(
                f"tl.load on '{p._name}' out of bounds: index {first} not in [0, {n}). "
                "An unmasked load past the end of a tensor is the classic Triton bug -- "
                "you almost certainly need mask=(offs < n)."
            )
        return p.flat[idx]

    mask = mask if torch.is_tensor(mask) else torch.tensor(mask)
    mask = mask.broadcast_to(idx.shape).bool()
    live = idx[mask]
    if live.numel():
        bad = (live < 0) | (live >= n)
        if bool(bad.any()):
            raise IndexError(
                f"tl.load on '{p._name}': index {int(live[bad][0])} not in [0, {n}) "
                "even though mask is true -- the mask does not match the offsets."
            )
    safe = torch.where(mask, idx, torch.zeros_like(idx))
    vals = p.flat[safe]
    return torch.where(mask, vals, torch.full_like(vals, _v(other)))


def store(ptr, value, mask=None, **_ignored):
    """Scatter `value` to `ptr` where `mask` is true."""
    idx, p = _index_of(ptr)
    n = p.flat.numel()
    value = value if torch.is_tensor(value) else torch.tensor(value)
    value = value.broadcast_to(idx.shape).to(p.flat.dtype)
    if mask is None:
        bad = (idx < 0) | (idx >= n)
        if bool(bad.any()):
            first = int(idx.reshape(-1)[bad.reshape(-1).nonzero()[0]])
            raise IndexError(
                f"tl.store on '{p._name}' out of bounds: index {first} not in [0, {n}). "
                "An unmasked store past the end corrupts other tensors on a GPU."
            )
        p.flat[idx.reshape(-1)] = value.reshape(-1)
        return
    mask = (mask if torch.is_tensor(mask) else torch.tensor(mask)).broadcast_to(idx.shape).bool()
    sel = idx[mask]
    if sel.numel():
        bad = (sel < 0) | (sel >= n)
        if bool(bad.any()):
            raise IndexError(f"tl.store on '{p._name}': masked index {int(sel[bad][0])} "
                             f"not in [0, {n})")
        p.flat[sel] = value[mask]


def atomic_add(ptr, value, mask=None, **_ignored):
    """Atomic read-modify-write.

    Needed whenever two program instances accumulate into the same address --
    e.g. dK/dV in a flash-attention backward tiled over queries. In an
    interpreter the "atomic" part is free (execution is serial), which means
    **this simulator cannot catch a missing atomic**: a kernel that races on
    hardware will look correct here. It is the one class of bug you must reason
    about rather than test for.
    """
    idx, p = _index_of(ptr)
    value = (value if torch.is_tensor(value) else torch.tensor(value)).broadcast_to(idx.shape)
    value = value.to(p.flat.dtype)
    if mask is not None:
        mask = (mask if torch.is_tensor(mask) else torch.tensor(mask)).broadcast_to(idx.shape).bool()
        idx, value = idx[mask], value[mask]
    old = p.flat[idx.reshape(-1)].clone()
    p.flat.index_put_((idx.reshape(-1),), value.reshape(-1), accumulate=True)
    return old


# --- math -----------------------------------------------------------------


def _t(x, dtype=None):
    if torch.is_tensor(x):
        return x
    return torch.tensor(_v(x), dtype=dtype)


def exp(x):
    return torch.exp(_t(x))


def exp2(x):
    return torch.exp2(_t(x))


def log(x):
    return torch.log(_t(x))


def log2(x):
    return torch.log2(_t(x))


def sqrt(x):
    return torch.sqrt(_t(x))


def rsqrt(x):
    return torch.rsqrt(_t(x))


def sin(x):
    return torch.sin(_t(x))


def cos(x):
    return torch.cos(_t(x))


def abs(x):  # noqa: A001 - mirrors tl.abs
    return torch.abs(_t(x))


def sigmoid(x):
    return torch.sigmoid(_t(x))


def tanh(x):
    return torch.tanh(_t(x))


def erf(x):
    return torch.erf(_t(x))


def floor(x):
    return torch.floor(_t(x))


def maximum(a, b):
    return torch.maximum(_t(a), _t(b).to(_t(a).dtype) if torch.is_tensor(a) else _t(b))


def minimum(a, b):
    return torch.minimum(_t(a), _t(b).to(_t(a).dtype) if torch.is_tensor(a) else _t(b))


def where(cond, a, b):
    cond = cond if torch.is_tensor(cond) else torch.tensor(cond)
    a, b = _t(a), _t(b)
    if a.dtype != b.dtype:
        b = b.to(a.dtype)
    return torch.where(cond.bool(), a, b)


def fdiv(a, b, ieee_rounding=False):
    return _t(a) / _t(b)


def cdiv(a, b):
    return -(-int(_v(a)) // int(_v(b)))


# --- reductions -----------------------------------------------------------


def sum(x, axis=None, keep_dims=False):  # noqa: A001 - mirrors tl.sum
    if axis is None:
        return x.sum()
    return x.sum(dim=int(_v(axis)), keepdim=bool(keep_dims))


def max(x, axis=None, keep_dims=False, return_indices=False):  # noqa: A001
    if axis is None:
        return x.amax()
    a = int(_v(axis))
    if return_indices:
        v, i = x.max(dim=a, keepdim=bool(keep_dims))
        return v, i
    return x.amax(dim=a, keepdim=bool(keep_dims))


def min(x, axis=None, keep_dims=False, return_indices=False):  # noqa: A001
    if axis is None:
        return x.amin()
    a = int(_v(axis))
    if return_indices:
        v, i = x.min(dim=a, keepdim=bool(keep_dims))
        return v, i
    return x.amin(dim=a, keepdim=bool(keep_dims))


def argmax(x, axis, keep_dims=False):
    return x.argmax(dim=int(_v(axis)), keepdim=bool(keep_dims))


def cumsum(x, axis=0, reverse=False):
    a = int(_v(axis))
    if reverse:
        return x.flip(a).cumsum(dim=a).flip(a)
    return x.cumsum(dim=a)


# --- the tensor-core op ---------------------------------------------------


def dot(a, b, acc=None, allow_tf32=True, out_dtype=float32, **_ignored):
    """Block matrix multiply -- the op that maps to tensor cores.

    On hardware `tl.dot` has real constraints the interpreter does not enforce:
    both operands must be 2-D blocks with power-of-two dims, and each dim must
    be at least 16. The accumulator is fp32 even for fp16 inputs, which is why
    `out_dtype` defaults to fp32 and why you should not cast the accumulator
    down inside the loop.
    """
    out = a.to(_v(out_dtype)) @ b.to(_v(out_dtype))
    return out if acc is None else acc + out


# --- no-ops and debugging -------------------------------------------------


def static_assert(cond, msg=""):
    if not bool(_v(cond)):
        raise AssertionError(f"tl.static_assert failed: {msg}")


def static_print(*args):
    print("tl.static_print:", *args)


def device_print(prefix, *args):
    print(prefix, *[a if not torch.is_tensor(a) else a.tolist() for a in args])


def debug_barrier():
    pass


def multiple_of(x, values):
    return x


def max_contiguous(x, values):
    return x


def assume(cond):
    return None


class _Math:
    """`tl.math.*` -- the same functions, under the namespace some kernels use."""

    exp = staticmethod(exp)
    exp2 = staticmethod(exp2)
    log = staticmethod(log)
    log2 = staticmethod(log2)
    sqrt = staticmethod(sqrt)
    rsqrt = staticmethod(rsqrt)
    sin = staticmethod(sin)
    cos = staticmethod(cos)
    tanh = staticmethod(tanh)
    erf = staticmethod(erf)
    abs = staticmethod(abs)
    floor = staticmethod(floor)
    max = staticmethod(maximum)
    min = staticmethod(minimum)
    fdiv = staticmethod(fdiv)


math = _Math()
extra = _Math()
