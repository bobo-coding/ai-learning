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
| `alignment_plain_dpo.log` | same, `--dpo-sft-weight 0` | Plain DPO collapsing 0.547 → 0.203, and GRPO recovering it to 0.603 (fixed 120, broke 0). Lessons 7–8. **Predates the stratified eval set** — its overall numbers rest on n=300 and its McNemar counts are paired, so those stand, but ignore its per-task columns. |
| `alignment_random_noise.log` | same, `--random-noise` | Random rather than systematic label noise: SFT reaches 0.963 from the same 45% corruption rate. Lesson 6. Also predates the stratified eval set. |
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
on samples of 1 and 3. The current runs use `stratified_sample` (50 per task)
and `generative_eval` returns `n_by_task`, so every score prints with its n.
Logs marked above as predating that change have untrustworthy per-task columns.

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
