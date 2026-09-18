"""A Triton interpreter, so Triton kernels can be written and debugged on a Mac.

Triton ships no macOS wheels -- it is an NVIDIA/AMD compiler. But Triton's
*programming model* is the part worth learning, and that model is small enough
to interpret: a kernel is a Python function over **blocks** of data, launched
once per point of a grid, where every memory access is an explicit
``load``/``store`` through a pointer arithmetic expression plus a mask.

So this package implements `triton.jit` and the subset of `triton.language`
those kernels use, on top of plain PyTorch. The *same source file* then runs
here and, unchanged, on a real GPU:

    try:
        import triton
        import triton.language as tl
    except ImportError:
        import tritonsim as triton
        import tritonsim.language as tl

What you get locally: correct results, real block-level semantics, and
**better error messages than a GPU gives you** -- an out-of-bounds access that
would be silent memory corruption on hardware raises an `IndexError` naming the
offending index here (see `language.load`).

What you do not get: any performance signal at all. Every program instance runs
serially in Python, so this is 1000x slower than the GPU and tells you nothing
about occupancy, warp scheduling, `num_stages`, or whether your tile sizes are
sane. Correctness here, performance there.

Faithfulness notes -- places where real Triton is stricter:
  * `BLOCK` sizes and `tl.arange` bounds must be powers of two on hardware.
    `language.arange` warns when they are not, so a kernel that works here
    still compiles there.
  * `tl.arange` yields int32, as on hardware, so a stride computation that
    overflows int32 overflows here too rather than silently working.
  * tensors must be contiguous, because the interpreter indexes their flat
    storage the same way a real kernel indexes a raw pointer.
"""

from __future__ import annotations

import functools
import threading

import torch

__all__ = ["jit", "cdiv", "next_power_of_2", "Config", "autotune", "heuristics",
           "Ptr", "language", "program_context"]


# ---------------------------------------------------------------------------
# Launch context: which program instance is currently executing
# ---------------------------------------------------------------------------


class _Context(threading.local):
    def __init__(self):
        self.pid: tuple[int, int, int] | None = None
        self.grid: tuple[int, int, int] | None = None


_ctx = _Context()


def program_context():
    if _ctx.pid is None:
        raise RuntimeError("tl.program_id() called outside a kernel launch")
    return _ctx.pid, _ctx.grid


# ---------------------------------------------------------------------------
# Pointers
# ---------------------------------------------------------------------------


class Ptr:
    """A pointer into a tensor's flat storage, plus a (possibly block-shaped) offset.

    This is the object that makes Triton's model click: in a Triton kernel a
    "pointer" is not a single address but an *array* of addresses of whatever
    block shape you built with `tl.arange` and broadcasting. `p + offs` produces
    another Ptr; `tl.load(p)` turns the addresses into values.
    """

    __slots__ = ("flat", "offs", "_name")

    def __init__(self, tensor=None, flat=None, offs=0, name: str = "ptr"):
        if tensor is not None:
            if not tensor.is_contiguous():
                raise ValueError(
                    "tritonsim needs contiguous tensors (it indexes flat storage, exactly as "
                    "a real kernel indexes a raw pointer). Pass x.contiguous()."
                )
            self.flat = tensor.reshape(-1)
        else:
            self.flat = flat
        self.offs = offs
        self._name = name

    def _with(self, offs):
        return Ptr(flat=self.flat, offs=offs, name=self._name)

    def __add__(self, other):
        return self._with(self.offs + _unwrap(other))

    __radd__ = __add__

    def __sub__(self, other):
        return self._with(self.offs - _unwrap(other))

    def __repr__(self):
        shape = tuple(self.offs.shape) if torch.is_tensor(self.offs) else ()
        return f"Ptr({self._name}, n={self.flat.numel()}, block_shape={shape})"


def _unwrap(x):
    return x.offs if isinstance(x, Ptr) else x


# ---------------------------------------------------------------------------
# @triton.jit and the kernel[grid](...) launch syntax
# ---------------------------------------------------------------------------


def _wrap_arg(a, name="arg"):
    if torch.is_tensor(a):
        return Ptr(a, name=name)
    from .language import constexpr

    if isinstance(a, constexpr):
        return a.value
    return a


class _Launcher:
    def __init__(self, fn, grid):
        self.fn = fn
        self.grid = grid

    def __call__(self, *args, **kwargs):
        # A callable grid receives the meta-parameters, which is how real Triton
        # kernels compute `triton.cdiv(N, META['BLOCK'])`.
        meta = dict(kwargs)
        grid = self.grid(meta) if callable(self.grid) else self.grid
        if isinstance(grid, int):
            grid = (grid,)
        grid = tuple(int(g) for g in grid) + (1,) * (3 - len(grid))
        if any(g < 0 for g in grid):
            raise ValueError(f"negative grid dimension: {grid}")

        names = self.fn.__code__.co_varnames[: self.fn.__code__.co_argcount]
        cargs = [_wrap_arg(a, names[i] if i < len(names) else f"arg{i}")
                 for i, a in enumerate(args)]
        ckw = {k: _wrap_arg(v, k) for k, v in kwargs.items()}

        saved = (_ctx.pid, _ctx.grid)
        try:
            _ctx.grid = grid
            # Serial execution in x-fastest order. Real Triton makes no ordering
            # guarantee between program instances, so a kernel whose result
            # depends on this order is wrong -- and will be wrong on hardware.
            for z in range(grid[2]):
                for y in range(grid[1]):
                    for x in range(grid[0]):
                        _ctx.pid = (x, y, z)
                        self.fn(*cargs, **ckw)
        finally:
            _ctx.pid, _ctx.grid = saved
        return None


class JITFunction:
    def __init__(self, fn):
        self.fn = fn
        functools.update_wrapper(self, fn)

    def __getitem__(self, grid):
        return _Launcher(self.fn, grid)

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self.fn.__name__} is a Triton kernel: launch it as "
            f"{self.fn.__name__}[grid](...), not by calling it directly."
        )


def jit(fn=None, **_kwargs):
    """Stand-in for `triton.jit`."""
    if fn is None:
        return lambda f: JITFunction(f)
    return JITFunction(fn)


# ---------------------------------------------------------------------------
# Autotuning: accepted, and reduced to "use the first config"
# ---------------------------------------------------------------------------


class Config:
    def __init__(self, kwargs: dict, num_warps: int = 4, num_stages: int = 2, **extra):
        self.kwargs = dict(kwargs)
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.extra = extra

    def __repr__(self):
        return f"Config({self.kwargs}, num_warps={self.num_warps}, num_stages={self.num_stages})"


def autotune(configs, key=None, **_kw):
    """Accepts `@triton.autotune(...)` and always picks `configs[0]`.

    Real autotuning times every config on the target GPU and caches the winner
    per distinct `key`. There is nothing to time here, so the decorator exists
    only so that a kernel written for hardware still runs. `num_warps` and
    `num_stages` have no meaning in an interpreter and are ignored.
    """
    chosen = configs[0] if configs else Config({})

    def deco(kernel):
        inner = kernel.fn if isinstance(kernel, JITFunction) else kernel

        @functools.wraps(inner)
        def fn(*args, **kwargs):
            merged = {**chosen.kwargs, **kwargs}
            return inner(*args, **merged)

        return JITFunction(fn)

    return deco


def heuristics(values: dict):
    """Accepts `@triton.heuristics({'BLOCK': lambda args: ...})`."""

    def deco(kernel):
        inner = kernel.fn if isinstance(kernel, JITFunction) else kernel

        @functools.wraps(inner)
        def fn(*args, **kwargs):
            for name, f in values.items():
                if name not in kwargs:
                    kwargs[name] = f(kwargs)
            return inner(*args, **kwargs)

        return JITFunction(fn)

    return deco


# ---------------------------------------------------------------------------
# Helpers that live on the `triton` namespace
# ---------------------------------------------------------------------------


def cdiv(a, b) -> int:
    """Ceiling division -- the standard way to size a Triton grid."""
    return -(-int(a) // int(b))


def next_power_of_2(n: int) -> int:
    n = int(n)
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


from . import language  # noqa: E402  (circular-safe: language imports only _ctx helpers)
