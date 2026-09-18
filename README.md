# Transformers from scratch, on a Mac

A hands-on curriculum that builds a working LLM stack from the tokenizer up:
architecture, pretraining, inference, fine-tuning (SFT / DPO / RLVR),
evaluation, quantization, and GPU kernels in Metal, CUDA and Triton.

Everything runs on an Apple Silicon Mac. Everything is tested against a
reference implementation — `pytest` is the contract, not the prose.

Written for someone with a maths and programming background who wants to
understand *why* each piece is the way it is, not just how to call it.

**Read the lessons online: <https://bobo-coding.github.io/ai-learning/>**

---

## Setup

```bash
uv venv --python 3.12 .venv            # or: python3.12 -m venv .venv
VIRTUAL_ENV=.venv uv pip install -e ".[dev]"
source .venv/bin/activate

python -m pytest -q                     # 324 tests, ~15 seconds
```

`MINIGPT_DEVICE=cpu` forces CPU everywhere, which is useful when you want exact
numerics (MPS has a few lower-precision fast paths).

## The 30-second tour

```bash
python -m scripts.prepare_shakespeare --vocab-size 1024   # tokenize a corpus
python -m minigpt.train --preset small --steps 2000        # pretrain
python -m scripts.sample --ckpt out/pretrain/best.pt       # generate text
python -m scripts.run_alignment                            # SFT -> DPO -> RLVR
python -m kernels.metal.bench                              # GPU kernels, verified
python -m kernels.triton.bench                             # Triton, via interpreter
```

---

## Curriculum

Each lesson is a markdown file in `lessons/` pairing the theory with the code
that implements it and the experiment that verifies it. Read them in order; the
code is meant to be read alongside — either here on GitHub or on the
[website](https://bobo-coding.github.io/ai-learning/), where every source path
in a lesson is a link.

| # | Lesson | Code | What you build |
|---|--------|------|----------------|
| 0 | [Setup and ground rules](lessons/00_setup.md) | `minigpt/utils.py` | Device selection, timing that isn't a lie, the numerics discipline the rest depends on |
| 1 | [Tokenization](lessons/01_tokenization.md) | `minigpt/bpe.py` | Byte-level BPE with GPT-2/GPT-4 pre-tokenization, from scratch |
| 2 | [Attention](lessons/02_attention.md) | `minigpt/attention.py` | The definition → batched → online softmax → FlashAttention forward *and* backward, plus RoPE, GQA, KV cache |
| 3 | [The transformer](lessons/03_transformer.md) | `minigpt/model.py`, `minigpt/config.py` | RMSNorm, SwiGLU, pre-norm blocks, weight tying, and the parameter/FLOP/KV-cache arithmetic |
| 4 | [Pretraining](lessons/04_pretraining.md) | `minigpt/train.py`, `minigpt/optim.py`, `minigpt/data.py` | AdamW from scratch, LR schedules, gradient accumulation, clipping, MFU — and a real overfitting curve |
| 5 | [Inference](lessons/05_inference.md) | `minigpt/generate.py` | Sampling (top-k/top-p/min-p), KV-cache decoding, and provably-exact speculative decoding |
| 6 | [SFT](lessons/06_sft.md) | `minigpt/sft.py`, `minigpt/tasks.py` | Chat templates, completion-only loss, and why the stop token must be in the mask |
| 7 | [Preference optimization](lessons/07_dpo.md) | `minigpt/dpo.py` | DPO derived from the RLHF objective — plus a measured demonstration of how it fails |
| 8 | [RLVR with GRPO](lessons/08_rlvr_grpo.md) | `minigpt/grpo.py` | Group-relative advantages, PPO clipping, K3 KL, entropy collapse |
| 9 | [Evaluation](lessons/09_evaluation.md) | `minigpt/eval.py` | Perplexity vs bits-per-byte, three MC scoring rules, unbiased pass@k, and error bars |
| 10 | [Quantization](lessons/10_quantization.md) | `minigpt/quant.py` | int8/int4/NF4 from first principles, where the error goes, honest byte accounting |
| 11 | [LoRA and QLoRA](lessons/11_lora.md) | `minigpt/lora.py` | Low-rank adapters, why they save more memory than parameters, merging |
| 12 | [GPU kernels](lessons/12_gpu_kernels.md) | `kernels/metal/`, `kernels/cuda/` | Six kernels in Metal (running here) and CUDA (for a real GPU), with the concept dictionary |
| 13 | [Triton](lessons/13_triton.md) | `kernels/triton/`, `tritonsim/` | Block-level kernels, plus an interpreter so they run and debug on a Mac |

## What's in the box

```
minigpt/          the library: 4,375 lines, every piece explained in its docstring
kernels/metal/    Metal Shading Language kernels, compiled at runtime by torch.mps
kernels/cuda/     the same six kernels in CUDA, to read here and run on a GPU
kernels/triton/   Triton kernels that run under real Triton or the interpreter
tritonsim/        a Triton interpreter on PyTorch — better error messages than a GPU
                  (kernels + tritonsim: 1,960 lines)
lessons/          the curriculum (~3,000 lines of prose, with the numbers)
scripts/          data prep and the end-to-end experiments
tests/            the correctness contract: 324 tests, 2,574 lines
results/          the logs behind every number quoted in the lessons
```

## The website

`lessons/` is the single source of truth. A small generator adapts it for the
web rather than duplicating it:

```bash
pip install -e ".[docs]"
python -m scripts.build_docs      # lessons/ + README.md -> docs/
mkdocs serve                      # preview at localhost:8000
```

`scripts/build_docs.py` rewrites links that point out of the published set to
absolute GitHub URLs, and turns inline code spans that name a real tracked file
into links — so on the site, `minigpt/bpe.py` in a lesson header is clickable.
It refuses to build if a link points at a file that is not in the repo, which is
the check that keeps 404s off the site.

`.github/workflows/pages.yml` runs that on every push to `main` that touches a
lesson, then `mkdocs build --strict`, then `scripts/check_site_links.py` over
the built HTML, and only then deploys. All three steps are failures, not
warnings.

## The correctness discipline

Teaching code that is subtly wrong is worse than no teaching code, so every
non-trivial claim here is checked against an independent reference:

- **FlashAttention** (forward *and* the hand-written backward) passes
  `torch.autograd.gradcheck` in float64, and matches a triple-loop reference
  exactly for every tile size.
- **AdamW** and **SGD** are bitwise identical to `torch.optim` over 30 steps
  (one tensor differs by 1 ULP — `tests/test_optim_data.py` says which and why).
- **RoPE** matches a complex-arithmetic reference, and the relative-position
  property `⟨RoPE(q,m), RoPE(k,n)⟩ = f(m−n)` is verified to 1e-15.
- **KV-cache decoding** reproduces a single full forward pass exactly, for every
  combination of position encoding, norm and MLP.
- **Speculative decoding** is checked distribution-preserving with a chi-square
  test against the target model's exact next-token distribution.
- **The analytic parameter count** in `GPTConfig.param_count()` matches the real
  module for every preset, and the Llama-7B formula lands within 0.01% of 6.74B.
- **The NF4 grid** is re-derived from normal quantiles and matches the published
  constants to 1e-3.
- **Every GPU kernel** is checked against a PyTorch reference before it is timed.

## Measured results

Real numbers from this repo on an M1 Max, not quoted from papers.

**Fused kernels beat eager PyTorch by a lot** (`python -m kernels.metal.bench`):

| kernel | Metal | PyTorch eager | speedup |
|---|---|---|---|
| RMSNorm (8192×1024) | 405 µs / 166 GB/s | 2929 µs / 23 GB/s | **7.2×** |
| matmul 1024³ tiled | 2754 µs / 0.78 TFLOP/s | 604 µs / 3.56 TFLOP/s | 0.22× |

The first row is the case for writing kernels (five memory passes become one);
the second is the case against (you will not beat a vendor BLAS in an
afternoon). The tiled matmul is 1.55× the naive one, which is the lesson about
arithmetic intensity, in numbers.

**bf16 autocast is *slower* than fp32 on MPS** — 22.8k vs 29.7k tok/s — because
Apple's GPU has no bf16 matrix units, so autocast buys casts and no math. On
CUDA the same flag is a ~2× win. Measure your own hardware.

**A 10.8M-parameter model on a 419k-token corpus overfits spectacularly**
(`results/pretrain_overfit_curve.json`): validation loss bottoms at **3.360 at
step 250** and climbs to **5.868 by step 2000** while training loss falls to
0.102 — 39 epochs over a corpus that Chinchilla says wants a ~20k-parameter
model. The regularized 984k run reaches **3.243** and is still improving.
Lesson 4 walks through both curves.

**int8 weight quantization is free** and NF4 beats uniform int4, measured
end-to-end on held-out perplexity (lesson 10):

| scheme | val NLL | Δ | weights |
|---|---|---|---|
| fp32 | 3.1794 | — | 3.94 MB |
| int8 per-channel | 3.1797 | +0.0003 | 1.40 MB |
| NF4 group-64 | 3.1852 | +0.0058 | 1.01 MB |
| int4 group-64 | 3.1883 | +0.0089 | 1.01 MB |

**The post-training pipeline** (`python -m scripts.run_alignment`) trains one
1.8M-parameter model through four stages on verifiable tasks, ~8 minutes total:

| stage | held-out exact match (95% CI) | McNemar vs previous |
|---|---|---|
| base (pretrained) | 0.000 | |
| SFT (45% systematically-wrong demos) | 0.643 [0.590, 0.697] | |
| DPO + NLL | 0.950 [0.923, 0.973] | fixed 104, broke 12, p<0.0001 |
| GRPO (RLVR) | 0.967 [0.947, 0.987] | fixed 7, broke 2, p=0.18 |

**And the same pipeline reproduces a real DPO failure.** With the NLL term
removed, `python -m scripts.exp_dpo_variants` runs six preference-optimization
configs from one shared SFT checkpoint:

| config | exact match | Δ vs SFT | final reward accuracy |
|---|---|---|---|
| plain DPO β=0.1 | **0.203** | −0.343 | 1.000 |
| DPO β=0.5 | 0.433 | −0.113 | 1.000 |
| **DPO + NLL (RPO)** | **0.967** | **+0.420** | 1.000 |
| IPO β=0.5 | 0.963 | +0.417 | 1.000 |
| SimPO | 0.623 | +0.077 | 1.000 |
| DPO, mixed rejected samples | 0.857 | +0.310 | 0.900 |

Reward accuracy is ~1.0 in every row — the preference objective is solved
perfectly while task accuracy ranges from 0.20 to 0.97. Plain DPO learned
"smaller is better" from systematically-biased rejected samples and overshot
past the correct answer. Lesson 7 walks through the mechanism and the fixes; the
raw logs are in `results/`.

## Hardware notes

Written and verified on an M1 Max (32 GB). Everything runs on CPU too, more
slowly. The CUDA kernels are the one thing that cannot be verified here; they
are written to mirror the Metal kernels that are, and
`kernels/cuda/README.md` says exactly what to run first on a real GPU.
