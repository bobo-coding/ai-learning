# Lesson 8 — RLVR with GRPO

Read: `minigpt/grpo.py` · Tests: `tests/test_alignment.py`

RLVR — reinforcement learning from **verifiable** rewards — is the setup that
made reasoning models work. If you can check an answer programmatically
(arithmetic, unit tests, a proof checker, a compiler), you do not need a learned
reward model at all. Sample completions, score them with the checker, and push
up the log-probability of the ones that passed.

That removes the single biggest failure mode of RLHF: a learned reward model is
a function the policy can find adversarial inputs to, and it *will*. A verifier
cannot be hacked, only satisfied.

## GRPO's one idea

PPO needs a value network to estimate the baseline `V(s)`. For LLMs that means a
second model the size of the policy, with its own optimizer state and its own
training instability.

GRPO replaces it with an **empirical baseline**: draw `G` completions for the
*same* prompt and use the group's own mean reward.

```
A_i = (r_i − mean(r₁…r_G)) / (std(r₁…r_G) + ε)
```

No value head, no GAE, no bootstrapping. The variance reduction is real, because
all `G` samples share the prompt, so the prompt's intrinsic difficulty cancels
out — which is exactly what the value function was estimating.

`group_advantages` implements it, and `test_group_advantages_are_normalised_per_group`
checks mean 0 / std 1 per group.

**Dr. GRPO variant:** `std_normalize=False` keeps only the mean subtraction.
Dividing by the group std up-weights groups that happen to be low-variance (7
of 8 correct), which is a difficulty bias nobody asked for.

## The objective

```
L = −E[ min( ρ_t·A, clip(ρ_t, 1−ε, 1+ε)·A ) ] + β·KL(π ‖ π_ref)
ρ_t = π_θ(y_t | ·) / π_old(y_t | ·)
```

### The clipping, concretely

`min()` of the clipped and unclipped terms is a *pessimistic* bound: it only
ever reduces the objective, which is what stops a large ratio taking a huge step.
The direction that binds depends on the sign of `A`:

| ratio | A | objective | which clip binds |
|---|---|---|---|
| e ≈ 2.72 | +1 | −1.2 | upper: caps how far a *good* action can be boosted |
| e⁻¹ ≈ 0.37 | −1 | +0.8 | lower: caps how far a *bad* action can be suppressed |

`test_grpo_clipping_binds_in_both_directions` checks both numbers exactly, and
that the gradient is *identically zero* once a clip binds. That last part is the
point: PPO does not merely shrink the step, it removes the incentive entirely.

With `inner_epochs=1` the first inner step has `ρ = 1` exactly, so the loss
value is `−mean(A)` and the gradient is plain REINFORCE, `−A·∇log π`.
`test_grpo_loss_value_and_gradient_at_ratio_one` verifies both. Worth knowing:
on-policy GRPO *is* REINFORCE with a group baseline, and the clipping only
starts mattering when you take several gradient steps per batch of rollouts.

### The KL estimator

```python
kl_k3 = exp(r) − r − 1,    r = log π_ref − log π
```

Schulman's K3 estimator. It is unbiased for the KL *and* always non-negative,
unlike the naive `−r`, which is unbiased but frequently negative on a single
sample. A negative KL penalty actively pushes the policy *away* from the
reference — the opposite of the intent.
`test_kl_k3_is_nonnegative_and_unbiased` checks both properties against the
exact KL over 200k samples.

### Token vs sequence aggregation

```python
loss_agg="token"      # sum over all tokens / total token count   (DAPO)
loss_agg="sequence"   # mean within each sequence, then mean      (original GRPO)
```

Sequence aggregation weights a short sequence's tokens more heavily — the test
measures exactly 4× for a length-1 vs length-4 sequence. That means a long wrong
answer dilutes its own penalty, which is a mild incentive towards length. Token
aggregation is the DAPO fix.

## The three failure modes, all visible in the metrics

Running GRPO on the *broken* checkpoint above, where there is actually
something to learn, the per-iteration log looks like:

```
grpo it   0 | reward 0.609 | zero-var 0.75 | kl 0.0000 | clip 0.000 | H 0.495 | eval 0.175
grpo it  24 | reward 0.328 | zero-var 0.88 | kl 0.0155 | clip 0.000 | H 0.286 | eval 0.425
grpo it  48 | reward 0.750 | zero-var 1.00 | kl 0.2683 | clip 0.000 | H 0.091 | eval 0.562
grpo it 119 | reward 0.328 | zero-var 0.50 | kl 1.2744 | clip 0.000 | H 0.168 | eval 0.625
```

**1. Zero-variance groups (`zero_var_frac`).** If all `G` completions are right —
or all wrong — the advantage is exactly 0 and the group contributes *no
gradient*. Above, it is 0.75–1.00 for much of the run: three quarters of the
compute produced nothing. Early in training most groups are all-wrong; late,
most are all-right. This is the metric that tells you your task difficulty is
mismatched to your model, and the fix is curriculum (drop solved prompts, raise
temperature, or sample harder prompts) — which is exactly what DAPO's dynamic
sampling does.

**2. Entropy collapse (`H`).** 0.495 → 0.091 over 48 iterations. Reward-only
training sharpens the policy until it emits one answer per prompt and
exploration stops. Once `H` is near zero, every group is zero-variance and
learning halts. Countermeasures: an entropy bonus, a higher sampling
temperature, or a KL penalty strong enough to hold the policy back.

**3. KL drift.** 0.0 → 1.27. `kl_coef = 0.02` here is loose enough to let the
policy move a long way from the reference. That is *sometimes* what you want in
RLVR (the verifier is trustworthy, so drift is fine), and sometimes how you lose
general capability while the one measured task improves.

Notice `reward` bounces around (0.609 → 0.328 → 0.750 → 0.328) while `eval`
climbs monotonically (0.175 → 0.425 → 0.562 → 0.625). Training reward is
computed on 8 random prompts at temperature 1.0; held-out eval is greedy on a
fixed set. **Never judge an RL run by its training reward** — that is what
`eval_fn` is for.

## Two measured results

120 iterations × 8 prompts × 8 samples = 7,680 rollouts, about 30 seconds on an
M1 Max.

### Starting from a good checkpoint

The default pipeline (`python -m scripts.run_alignment`), scored on 400
stratified held-out prompts, 50 per task:

| stage | held-out exact match (95% CI) | McNemar vs previous |
|---|---|---|
| base | 0.000 | |
| SFT | 0.578 [0.527, 0.625] | |
| DPO (RPO, β=0.1 + NLL) | 0.892 [0.860, 0.922] | fixed 138, broke 12, p<0.0001 |
| **GRPO** | **0.895** [0.863, 0.925] | fixed 6, broke 5, **p = 1.00** |

GRPO adds 0.3 points and — honestly — **that is nothing at all**. Six items
fixed, five broken, p = 1.00. At 0.89 there is almost nothing left to reach, and
the metrics say so up front: mean training reward starts at **0.828** on
iteration 0 and ends at 0.969, with `zero-var` high throughout. When the model
already solves the task, RLVR has no gradient to give you.

Per task, the one place it moved at all is `mul` (0.52 → 0.54) and `sub`
(0.88 → 0.90), both within noise at n=50. It also lost 4 points on `add`.

This is worth stating plainly because the literature is full of RLVR gains: the
gains are real *when there is headroom*. Measuring on a near-saturated checkpoint
and reporting the delta is how you manufacture a result that will not replicate.

### Starting from a broken checkpoint

The interesting case. Take a plain-DPO run that collapsed (lesson 7) and give
GRPO a verifier:

| stage | held-out exact match | McNemar |
|---|---|---|
| SFT | 0.547 | |
| DPO (plain, β=0.1) | 0.203 | fixed 24, broke 127 |
| **GRPO after that** | **0.603** | **fixed 120, broke 0**, p<0.0001 |

GRPO **recovered everything plain DPO broke** — 120 items fixed, zero broken.
That is what a trustworthy reward signal buys you: the verifier does not care
which direction the policy drifted in, it only rewards correct answers. A
preference model trained on the same data would have been just as confused as
DPO was.

(Those three figures come from an earlier run on the *unstratified* eval set —
see `results/alignment_plain_dpo.log`. The per-task columns from that run are
not trustworthy for the reason lesson 9 explains, but the overall numbers rest
on n=300 and the McNemar counts are paired, so the conclusion stands.)

**RL amplifies capabilities the model already has; it does not install new
ones.** In the first table `mul` moves 0.52 → 0.54 and nothing else improves,
because with `zero_var_frac` high, GRPO almost never saw a group where some
samples got a hard prompt right and others wrong. There is no gradient without
disagreement.

## Implementation notes

`old_logp` must come from the policy *as it was when it sampled*. With
`inner_epochs=1` we recompute it under `no_grad` so the first step has `ρ = 1`
exactly. With more inner epochs you must snapshot it before the first update.

Sampling temperature is a genuine hyperparameter here, not a cosmetic one: it
**is** the exploration policy. Too low and every group is zero-variance; too
high and the advantages are dominated by noise.

The terminator stays in the loss mask (`extract_completion` keeps it), because
the model must be credited or blamed for choosing to stop — drop it and you get
a policy that never terminates.

## Exercises

1. **Fix the zero-variance problem.** Implement DAPO-style dynamic sampling:
   over-sample prompts, discard groups with zero variance, and keep sampling
   until the batch is full of informative groups. Measure the change in
   `eval_overall` per rollout.

2. **Add an entropy bonus** (`loss -= ent_coef * entropy`) and sweep
   `ent_coef` ∈ {0, 0.001, 0.01}. Plot `H` and held-out accuracy. Is there a
   setting that keeps exploration alive without hurting the reward?

3. **Turn off the KL penalty** (`kl_coef=0`) and run 300 iterations. Does the
   held-out accuracy keep rising? Now also evaluate on a *different* task the
   reward never covered — this is the capability-loss measurement.

4. **Verifier vs reward model.** Train a small reward model on the preference
   pairs from lesson 7, use it in place of `reward_exact`, and run GRPO. Watch
   for reward hacking: reward going up while true accuracy goes down. Report both
   curves on the same axes.

5. **Reward shaping and its cost.** Swap in `reward_shaped` (partial credit for
   a correct prefix). Does it converge faster? Does the model learn to emit
   truncated answers to farm partial credit? (Look at mean output length.)

6. **Token vs sequence aggregation.** Run both with a task whose answers have
   very different lengths (mix `add` and `sort`). Measure mean output length
   under each — this is the length-bias effect DAPO identified, in miniature.

7. **On-policy vs off-policy.** Set `inner_epochs` ∈ {1, 2, 4} and watch
   `clip_frac`. At what point does clipping start to bind, and what does it do to
   sample efficiency per rollout?

---

**Previous:** [Lesson 7](07_dpo.md) · **Next:** [Lesson 9 — Evaluation](09_evaluation.md)
