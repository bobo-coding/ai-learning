# Measured results

The logs and reports the lessons quote, so the numbers can be checked rather
than taken on trust. Everything here was produced on an Apple M1 Max (32 GB) by
the commands named below. Checkpoints are not committed (132 MB each); re-run the
commands to regenerate them.

| file | produced by | what it shows |
|---|---|---|
| `pretrain_overfit.log`, `pretrain_overfit_curve.json` | `python -m minigpt.train --preset small --steps 2000 --batch-size 32 --lr 1e-3` | A 10.8M-parameter model on a 419k-token corpus. Val loss bottoms at 3.360 (step 250) and rises to 5.868 while train loss falls to 0.102. Lesson 4. |
| `pretrain_regularized.log`, `pretrain_regularized_curve.json` | `tiny` preset, dropout 0.15, 1500 steps | 984k parameters reaching val loss 3.243 and still improving — 11× smaller, better result. Lesson 4. |
| `alignment_rpo.log`, `alignment_report.json` | `python -m scripts.run_alignment` | The default pipeline: base 0.000 → SFT 0.578 → DPO+NLL 0.892 → GRPO 0.895, on a **stratified** 400-prompt eval set (50 per task), with CIs, McNemar tests and per-task n. Lessons 6–8. |
| `alignment_plain_dpo.log`, `alignment_plain_dpo_report.json` | same, `--dpo-sft-weight 0 --out-dir out/align_plaindpo` | Plain DPO collapsing 0.583 → 0.380, and GRPO recovering to 0.627 (fixed 101, broke 2) — but **not** recovering `add`/`sub`, which stay at 0.06/0.04. Lessons 7–8. |
| `zero_variance_probe.log` | `python -m scripts.probe_zero_variance --ckpt out/align_plaindpo/dpo.pt` | Why: 100% of `add` groups are all-wrong on the collapsed checkpoint, so its advantage is identically zero and RLVR has no signal at all. Lesson 8. |
| `alignment_random_noise.log`, `alignment_random_noise_report.json` | same, `--random-noise --out-dir out/align_randnoise` | Random rather than systematic label noise: SFT reaches 0.777 vs 0.578, from the same 45% corruption rate. Lesson 6. |
| `dpo_variants.log`, `dpo_variants.json` | `python -m scripts.exp_dpo_variants` | Six preference-optimization configs from one SFT checkpoint: 0.415 to 0.892, all with reward accuracy 0.98–1.00. Lesson 7. |

Kernel benchmarks are not archived because they are fast to reproduce and
hardware-specific:

```bash
python -m kernels.metal.bench      # correctness + timings, Apple GPU
python -m kernels.triton.bench     # correctness via the tritonsim interpreter
```

## Two caveats that changed how these are reported

**Per-task numbers need their n.** The evaluation set was originally
`val_ex[:300]` — a prefix of a shuffled list, which yields the *training
mixture*, not a balanced benchmark: 92 `sort` prompts, 1 `mul` prompt, and two
tasks absent entirely. Per-task accuracies were therefore quoted to two decimals
on samples of 1 and 3. **Every log in this directory has now been regenerated**
with `stratified_sample` (50 per task), and `generative_eval` returns
`n_by_task` so every score prints with its n.

Re-running changed three headline numbers materially, which is the point:
`mul` went from 0.00 (n=1) to 0.32 (n=50); the random-noise SFT result fell from
0.963 to 0.777 once the eval stopped over-weighting the tasks it was good at;
and GRPO's recovery of a collapsed checkpoint turned out to be partial rather
than complete, with `add` unrecoverable for a reason that is now measured.

**Timings need their spread.** Short GPU kernels are reproducible only to
~5–15% on a laptop, so single-run figures quoted to three significant figures
were not measurements. `benchmark_repeat` now reports `median [min–max]`, and
`kernels/metal/bench.py` prints `within noise -- not a result` when a difference
is smaller than the interval. One previously reported "11% improvement" is 1.2%
against ±4% noise.

## A caveat on reproducibility

MPS reductions are not bit-deterministic, so re-running these produces slightly
different numbers. The SFT stage in particular has produced 0.547, 0.578 and
0.643 across runs with identical seeds (on two different eval sets), because
that training data is deliberately bimodal. Comparisons that start from a *shared* checkpoint (the DPO variants
table) are paired and reliable; absolute numbers across separate training runs
are not. See lesson 9.
