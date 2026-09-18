# Measured results

The logs and reports the lessons quote, so the numbers can be checked rather
than taken on trust. Everything here was produced on an Apple M1 Max (32 GB) by
the commands named below. Checkpoints are not committed (132 MB each); re-run the
commands to regenerate them.

| file | produced by | what it shows |
|---|---|---|
| `pretrain_overfit.log`, `pretrain_overfit_curve.json` | `python -m minigpt.train --preset small --steps 2000 --batch-size 32 --lr 1e-3` | A 10.8M-parameter model on a 419k-token corpus. Val loss bottoms at 3.360 (step 250) and rises to 5.868 while train loss falls to 0.102. Lesson 4. |
| `pretrain_regularized.log`, `pretrain_regularized_curve.json` | `tiny` preset, dropout 0.15, 1500 steps | 984k parameters reaching val loss 3.243 and still improving — 11× smaller, better result. Lesson 4. |
| `alignment_rpo.log`, `alignment_report.json` | `python -m scripts.run_alignment` | The default pipeline: base 0.000 → SFT 0.643 → DPO+NLL 0.950 → GRPO 0.967, with CIs and McNemar tests. Lessons 6–8. |
| `alignment_plain_dpo.log` | same, `--dpo-sft-weight 0` | Plain DPO collapsing 0.547 → 0.203, and GRPO recovering it to 0.603 (fixed 120, broke 0). Lessons 7–8. |
| `alignment_random_noise.log` | same, `--random-noise` | Random rather than systematic label noise: SFT reaches 0.963 from the same 45% corruption rate. Lesson 6. |
| `dpo_variants.log`, `dpo_variants.json` | `python -m scripts.exp_dpo_variants` | Six preference-optimization configs from one SFT checkpoint: 0.203 to 0.967, all with reward accuracy ≈ 1.0. Lesson 7. |

Kernel benchmarks are not archived because they are fast to reproduce and
hardware-specific:

```bash
python -m kernels.metal.bench      # correctness + timings, Apple GPU
python -m kernels.triton.bench     # correctness via the tritonsim interpreter
```

## A caveat on reproducibility

MPS reductions are not bit-deterministic, so re-running these produces slightly
different numbers. The SFT stage in particular varied between 0.547 and 0.643
across two runs with identical seeds, because that training data is deliberately
bimodal. Comparisons that start from a *shared* checkpoint (the DPO variants
table) are paired and reliable; absolute numbers across separate training runs
are not. See lesson 9.
