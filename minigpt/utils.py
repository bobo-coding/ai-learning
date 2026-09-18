"""Small shared helpers: device selection, seeding, parameter counting, timing."""

from __future__ import annotations

import contextlib
import math
import os
import random
import time

import numpy as np
import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available device.

    Order of preference: explicit `prefer` -> MPS (Apple GPU) -> CUDA -> CPU.
    `MINIGPT_DEVICE=cpu` in the environment overrides everything, which is handy
    for debugging numerics (MPS has a few lower-precision fast paths).
    """
    env = os.environ.get("MINIGPT_DEVICE")
    if env:
        return torch.device(env)
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(module: torch.nn.Module, trainable_only: bool = False) -> int:
    ps = module.parameters()
    if trainable_only:
        ps = (p for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in ps)


def human(n: float) -> str:
    """Format a number with a magnitude suffix: 1234567 -> '1.23M'."""
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def sync(device: torch.device | str) -> None:
    """Block until queued GPU work finishes -- required before any timing."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@contextlib.contextmanager
def timer(label: str = "", device: torch.device | str = "cpu", quiet: bool = False):
    """Wall-clock timer that synchronizes the device on entry and exit.

    Yields a one-element list; after the block, `out[0]` holds elapsed seconds.
    """
    out = [0.0]
    sync(device)
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        sync(device)
        out[0] = time.perf_counter() - t0
        if not quiet:
            print(f"{label}: {out[0] * 1e3:.2f} ms")


def benchmark(fn, device="cpu", warmup: int = 3, iters: int = 20) -> float:
    """Median seconds per call of `fn`, with warmup and device sync."""
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error ||a-b|| / ||b||, the standard kernel-correctness metric."""
    a = a.detach().float()
    b = b.detach().float()
    denom = b.norm().item()
    if denom == 0.0:
        return a.norm().item()
    return (a - b).norm().item() / denom


def allclose_report(a: torch.Tensor, b: torch.Tensor, name: str = "", atol=1e-5, rtol=1e-4) -> bool:
    a32, b32 = a.detach().float().cpu(), b.detach().float().cpu()
    ok = torch.allclose(a32, b32, atol=atol, rtol=rtol)
    max_abs = (a32 - b32).abs().max().item() if a32.numel() else 0.0
    print(f"{'OK  ' if ok else 'FAIL'} {name:<34} max_abs={max_abs:.3e} rel_l2={rel_err(a32, b32):.3e}")
    return ok


def cosine_lr(step: int, *, base_lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    """Linear warmup then cosine decay to `min_ratio * base_lr`.

    This is the schedule used by GPT-3/Llama-style pretraining runs.
    """
    if step < warmup:
        # +1 so step 0 gets a nonzero LR rather than a wasted update.
        return base_lr * (step + 1) / max(1, warmup)
    if step >= total:
        return base_lr * min_ratio
    progress = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * coeff)
