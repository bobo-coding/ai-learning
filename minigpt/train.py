"""The pretraining loop, with the pieces that make it a *real* one.

Anyone can write `loss.backward(); opt.step()`.  What separates a loop that
trains a model from one that wastes a week is the surrounding machinery, and
each piece here exists for a specific failure it prevents:

* **gradient accumulation** -- decouples the batch size that fits in memory
  from the batch size the optimizer sees.  Loss must be divided by the number
  of micro-steps, or the effective LR silently scales with accumulation.
* **LR warmup + decay** -- Adam's second moment is badly estimated at step 0;
  warmup stops the first few steps from destroying the initialisation.
* **global gradient clipping** -- one pathological batch can otherwise move the
  weights far enough to never recover. Log the pre-clip norm: it spikes long
  before the loss does.
* **deterministic validation** -- random eval windows make val loss noisy
  enough to hide a real regression.
* **checkpointing optimizer state, not just weights** -- resuming without
  Adam's moments is a fresh warmup in disguise and shows up as a loss bump.
* **throughput + MFU logging** -- tells you whether a slow run is the model or
  the data loader.  Usually the data loader.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .data import TokenBatcher, iter_eval_batches
from .model import GPT
from .optim import clip_grad_norm, wsd_lr
from .utils import cosine_lr, human, pick_device, seed_everything, sync


@dataclass
class TrainConfig:
    # data
    train_bin: str = "data/shakespeare_train.bin"
    val_bin: str = "data/shakespeare_val.bin"
    # optimization
    batch_size: int = 16             # micro-batch that actually runs on device
    grad_accum: int = 1              # micro-batches per optimizer step
    max_steps: int = 2000
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    schedule: str = "cosine"         # "cosine" | "wsd" | "constant"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # runtime
    device: str | None = None
    amp_dtype: str = "none"          # "none" | "bf16" | "fp16"
    compile: bool = False
    seed: int = 1337
    # logging / checkpointing
    eval_interval: int = 250
    eval_batches: int = 20
    log_interval: int = 10
    out_dir: str = "out/pretrain"
    peak_flops: float | None = None  # device peak, for MFU; None -> measured
    resume: str | None = None

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum

    def lr_at(self, step: int, block_size: int) -> float:
        if self.schedule == "constant":
            return self.lr if step >= self.warmup_steps else self.lr * (step + 1) / max(1, self.warmup_steps)
        if self.schedule == "wsd":
            return wsd_lr(step, base_lr=self.lr, warmup=self.warmup_steps,
                          total=self.max_steps, min_ratio=self.min_lr_ratio)
        return cosine_lr(step, base_lr=self.lr, warmup=self.warmup_steps,
                         total=self.max_steps, min_ratio=self.min_lr_ratio)


def measure_peak_flops(device, dtype=torch.float32, n: int = 2048) -> float:
    """Measured large-matmul throughput, used as the MFU denominator.

    Vendor "peak TFLOPS" numbers assume perfect tensor-core utilisation you will
    never see.  Measuring the best square matmul this machine can actually do is
    a denominator you can trust: MFU then means "fraction of achievable matmul
    throughput", and 40-50% is a genuinely well-tuned training loop.
    """
    device = torch.device(device)
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(3):
        a @ b
    sync(device)
    t0 = time.perf_counter()
    iters = 10
    for _ in range(iters):
        a @ b
    sync(device)
    dt = (time.perf_counter() - t0) / iters
    return 2.0 * n**3 / dt


def amp_context(device: torch.device, amp_dtype: str):
    """autocast context, or a no-op if disabled/unsupported.

    Mixed precision is the single biggest free speedup on modern hardware.  Two
    rules: (1) prefer bf16 over fp16 -- same exponent range as fp32, so no loss
    scaler and no `inf` surprises; (2) the master weights and the optimizer
    state stay fp32.  autocast does exactly this: it casts the *inputs of
    matmul-like ops* and leaves reductions and norms in fp32.
    """
    if amp_dtype == "none":
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp_dtype]
    try:
        return torch.autocast(device_type=device.type, dtype=dtype)
    except (RuntimeError, ValueError) as e:  # pragma: no cover - device dependent
        print(f"[warn] autocast({device.type}, {amp_dtype}) unavailable ({e}); running fp32")
        return nullcontext()


@torch.no_grad()
def evaluate(model: GPT, tokens: np.ndarray, batch_size: int, device, max_batches: int = 20,
             amp_dtype: str = "none") -> dict[str, float]:
    """Deterministic validation loss and perplexity over the first windows."""
    model.eval()
    total_loss, total_tok = 0.0, 0
    for i, (x, y) in enumerate(iter_eval_batches(tokens, model.cfg.block_size, batch_size, device)):
        if i >= max_batches:
            break
        with amp_context(torch.device(device), amp_dtype):
            _, loss = model(x, targets=y)
        # Weight by token count so the average is exact even on a ragged last batch.
        n = y.numel()
        total_loss += loss.item() * n
        total_tok += n
    model.train()
    mean = total_loss / max(1, total_tok)
    return {"loss": mean, "ppl": math.exp(min(mean, 20.0)), "tokens": total_tok}


class Trainer:
    def __init__(self, model: GPT, tcfg: TrainConfig, train_tokens: np.ndarray,
                 val_tokens: np.ndarray | None = None):
        self.cfg = tcfg
        self.device = torch.device(tcfg.device) if tcfg.device else pick_device()
        seed_everything(tcfg.seed)

        self.model = model.to(self.device)
        self.raw_model = self.model           # keep a handle to the un-compiled module
        if tcfg.compile:
            self.model = torch.compile(self.model)

        self.opt = self.raw_model.configure_optimizers(
            lr=tcfg.lr, weight_decay=tcfg.weight_decay, betas=(tcfg.beta1, tcfg.beta2)
        )
        self.train_tokens = train_tokens
        self.val_tokens = val_tokens
        self.batcher = TokenBatcher(train_tokens, model.cfg.block_size, tcfg.batch_size,
                                    device=self.device, seed=tcfg.seed)
        # fp16 needs a loss scaler (its 5-bit exponent underflows on small
        # gradients); bf16 does not.  This is the main practical reason to prefer bf16.
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=(tcfg.amp_dtype == "fp16"))
        self.step = 0
        self.best_val = float("inf")
        self.history: list[dict] = []
        self.out_dir = Path(tcfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.peak_flops = tcfg.peak_flops
        if tcfg.resume:
            self.load_checkpoint(tcfg.resume)

    # ------------------------------------------------------------------ one step

    def train_step(self) -> dict[str, float]:
        cfg = self.cfg
        lr = cfg.lr_at(self.step, self.raw_model.cfg.block_size)
        for g in self.opt.param_groups:
            g["lr"] = lr

        t0 = time.perf_counter()
        self.opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(cfg.grad_accum):
            x, y = self.batcher()
            with amp_context(self.device, cfg.amp_dtype):
                _, loss = self.model(x, targets=y)
                # Each micro-batch contributes 1/grad_accum of the gradient, so
                # the accumulated gradient equals that of the full batch.
                scaled = loss / cfg.grad_accum
            self.scaler.scale(scaled).backward()
            loss_sum += loss.item()

        # Unscale before clipping, or you clip the scaled gradient.
        self.scaler.unscale_(self.opt)
        gnorm = clip_grad_norm(
            [p for gp in self.opt.param_groups for p in gp["params"]], cfg.grad_clip
        )
        self.scaler.step(self.opt)
        self.scaler.update()

        sync(self.device)
        dt = time.perf_counter() - t0
        n_tok = cfg.batch_size * cfg.grad_accum * self.raw_model.cfg.block_size
        rec = {
            "step": self.step,
            "loss": loss_sum / cfg.grad_accum,
            "lr": lr,
            "grad_norm": gnorm,
            "dt": dt,
            "tok_per_s": n_tok / dt,
        }
        if self.peak_flops:
            rec["mfu"] = self.raw_model.estimate_mfu(
                cfg.batch_size * cfg.grad_accum * self.raw_model.cfg.block_size, dt, self.peak_flops
            )
        self.step += 1
        return rec

    # --------------------------------------------------------------------- loop

    def fit(self, verbose: bool = True) -> list[dict]:
        cfg = self.cfg
        if self.peak_flops is None:
            self.peak_flops = measure_peak_flops(self.device)
            if verbose:
                print(f"measured matmul throughput: {human(self.peak_flops)}FLOP/s")
        if verbose:
            n = self.raw_model.num_params()
            print(f"model: {human(n)} params ({human(self.raw_model.num_params(True))} non-embedding) "
                  f"| device {self.device} | amp {cfg.amp_dtype}")
            print(f"batch {cfg.batch_size} x accum {cfg.grad_accum} x block "
                  f"{self.raw_model.cfg.block_size} = "
                  f"{human(cfg.tokens_per_step * self.raw_model.cfg.block_size)} tokens/step")

        self.model.train()
        start = self.step
        for _ in range(start, cfg.max_steps):
            rec = self.train_step()
            self.history.append(rec)
            if verbose and (self.step % cfg.log_interval == 0 or self.step == 1):
                msg = (f"step {rec['step']:5d} | loss {rec['loss']:.4f} | lr {rec['lr']:.2e} "
                       f"| gnorm {rec['grad_norm']:6.3f} | {rec['tok_per_s'] / 1e3:7.1f}k tok/s")
                if "mfu" in rec:
                    msg += f" | mfu {rec['mfu'] * 100:4.1f}%"
                print(msg)
            if self.val_tokens is not None and self.step % cfg.eval_interval == 0:
                ev = evaluate(self.model, self.val_tokens, cfg.batch_size, self.device,
                              cfg.eval_batches, cfg.amp_dtype)
                rec["val_loss"], rec["val_ppl"] = ev["loss"], ev["ppl"]
                if verbose:
                    print(f"  eval @ {self.step}: val_loss {ev['loss']:.4f} ppl {ev['ppl']:.2f}")
                if ev["loss"] < self.best_val:
                    self.best_val = ev["loss"]
                    self.save_checkpoint("best.pt")
        self.save_checkpoint("final.pt")
        (self.out_dir / "history.json").write_text(json.dumps(self.history))
        return self.history

    # ------------------------------------------------------------- checkpointing

    def save_checkpoint(self, name: str):
        torch.save({
            "model": self.raw_model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "model_cfg": self.raw_model.cfg.to_dict(),
            "train_cfg": asdict(self.cfg),
            "step": self.step,
            "best_val": self.best_val,
        }, self.out_dir / name)

    def load_checkpoint(self, path: str):
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(blob["model"])
        self.opt.load_state_dict(blob["optimizer"])
        self.step = blob["step"]
        self.best_val = blob.get("best_val", float("inf"))
        print(f"resumed from {path} at step {self.step}")


def main(argv: list[str] | None = None):
    import argparse

    from .data import read_tokens

    ap = argparse.ArgumentParser(description="pretrain a minigpt")
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--train-bin", default="data/shakespeare_train.bin")
    ap.add_argument("--val-bin", default="data/shakespeare_val.bin")
    ap.add_argument("--meta", default="data/shakespeare_meta.json")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--schedule", default="cosine")
    ap.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="out/pretrain")
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args(argv)

    from .config import preset as make_preset

    meta = json.loads(Path(args.meta).read_text())
    overrides = {"vocab_size": meta["vocab_size"]}
    if args.block_size:
        overrides["block_size"] = args.block_size
    cfg = make_preset(args.preset, **overrides)

    train_tokens = read_tokens(args.train_bin, cfg.vocab_size)
    val_tokens = read_tokens(args.val_bin, cfg.vocab_size)
    tcfg = TrainConfig(
        train_bin=args.train_bin, val_bin=args.val_bin, batch_size=args.batch_size,
        grad_accum=args.grad_accum, max_steps=args.steps, lr=args.lr, schedule=args.schedule,
        amp_dtype=args.amp, device=args.device, out_dir=args.out_dir,
        eval_interval=args.eval_interval, compile=args.compile, resume=args.resume,
    )
    trainer = Trainer(GPT(cfg), tcfg, train_tokens, val_tokens)
    trainer.fit()
    print(f"best val loss {trainer.best_val:.4f} -> {trainer.out_dir}/best.pt")


if __name__ == "__main__":
    main()
