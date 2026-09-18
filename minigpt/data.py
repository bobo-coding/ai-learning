"""Token streams on disk, and how batches are cut out of them.

For pretraining, the entire corpus is one long list of token ids.  A training
example is just a random window of `block_size + 1` tokens: the first
`block_size` are the input, shifted by one they are the target.  So one sequence
of length T yields T next-token prediction problems, and *that* is why
pretraining is so sample-efficient compared to any supervised task.

Storing tokens in a `np.memmap` rather than a Python list matters once the
corpus is bigger than RAM: the OS pages in only the windows you touch, so a
300 GB corpus trains from a 16 GB machine with no code changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def token_dtype(vocab_size: int):
    """Smallest integer type that can hold the vocabulary.

    uint16 covers vocabularies up to 65535, which includes GPT-2 (50257) and
    Llama (32000).  Halving the bytes halves the disk I/O of the data loader,
    which is the difference between a saturated GPU and a starved one.
    """
    return np.uint16 if vocab_size < 2**16 else np.uint32


def write_tokens(ids, path: str | Path, vocab_size: int) -> Path:
    """Write token ids to `path` as a flat binary array."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(ids, dtype=token_dtype(vocab_size))
    if arr.size and int(arr.max()) >= vocab_size:
        raise ValueError(f"token id {int(arr.max())} >= vocab_size {vocab_size}")
    arr.tofile(path)
    return path


def read_tokens(path: str | Path, vocab_size: int) -> np.ndarray:
    """Memory-map a token file written by `write_tokens`."""
    return np.memmap(path, dtype=token_dtype(vocab_size), mode="r")


class TokenBatcher:
    """Draws random `(x, y)` windows from a flat token array.

    Sampling windows independently at random (rather than walking the corpus in
    order) is deliberate: consecutive batches are then near-independent, which
    is what SGD's convergence analysis assumes.  The cost is that a given token
    appears in a varying number of windows -- an "epoch" is only approximate.
    """

    def __init__(self, tokens: np.ndarray, block_size: int, batch_size: int,
                 device="cpu", seed: int = 1337):
        if len(tokens) < block_size + 1:
            raise ValueError(f"need > block_size+1={block_size + 1} tokens, got {len(tokens)}")
        self.tokens = tokens
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        bs, T = self.batch_size, self.block_size
        ix = self.rng.integers(0, len(self.tokens) - T - 1, size=bs)
        # Build the batch in numpy (cheap, contiguous) then move it once.
        xs = np.stack([self.tokens[i : i + T] for i in ix]).astype(np.int64)
        ys = np.stack([self.tokens[i + 1 : i + 1 + T] for i in ix]).astype(np.int64)
        x = torch.from_numpy(xs)
        y = torch.from_numpy(ys)
        if self.device.type != "cpu":
            # non_blocking only helps with pinned memory on CUDA; harmless elsewhere.
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        return x, y


def iter_eval_batches(tokens: np.ndarray, block_size: int, batch_size: int, device="cpu",
                      stride: int | None = None):
    """Deterministic, non-overlapping sweep over `tokens` -- use this for eval.

    Random windows are fine for training but make validation loss noisy and
    non-comparable across runs.  With `stride < block_size` you get the sliding
    -window perplexity protocol used in papers, where every token is scored with
    the maximum available context.
    """
    stride = stride or block_size
    starts = list(range(0, len(tokens) - block_size - 1, stride))
    for i in range(0, len(starts), batch_size):
        chunk = starts[i : i + batch_size]
        xs = np.stack([tokens[s : s + block_size] for s in chunk]).astype(np.int64)
        ys = np.stack([tokens[s + 1 : s + 1 + block_size] for s in chunk]).astype(np.int64)
        yield torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)


def pack_documents(docs: list[list[int]], eos_id: int, block_size: int,
                   drop_last: bool = True) -> np.ndarray:
    """Concatenate documents separated by EOS and cut into fixed-length rows.

    "Packing" wastes no compute on padding, at the price of letting attention
    look across a document boundary.  Every major pretraining run does it
    anyway: the model learns that EOS means "forget what came before", and the
    throughput gain is large.  When you cannot accept the leakage (short SFT
    examples, contrastive objectives) you need either padding or a block
    -diagonal attention mask.
    """
    flat: list[int] = []
    for d in docs:
        flat.extend(d)
        flat.append(eos_id)
    n_rows = len(flat) // block_size
    if n_rows == 0:
        if drop_last:
            raise ValueError("not enough tokens to fill one row")
        n_rows = 1
        flat = flat + [eos_id] * (block_size - len(flat))
    return np.array(flat[: n_rows * block_size], dtype=np.int64).reshape(n_rows, block_size)


def train_val_split(tokens: np.ndarray, val_fraction: float = 0.1):
    """Split a token stream by position, never at random.

    Random token-level splitting leaks: a validation window would overlap the
    training windows around it, and the model can score well by memorising
    neighbours.  Splitting by position (and ideally by document) is the only
    honest option for language modelling.
    """
    n_val = int(len(tokens) * val_fraction)
    if n_val < 1:
        raise ValueError("val_fraction too small for this corpus")
    return tokens[:-n_val], tokens[-n_val:]
