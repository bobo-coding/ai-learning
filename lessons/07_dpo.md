# Lesson 7 — Preference optimization

Read: `minigpt/dpo.py` · Tests: `tests/test_alignment.py` ·
Experiment: `scripts/exp_dpo_variants.py`

## Where DPO comes from

RLHF's objective is

```
max_π  E_{y~π(·|x)}[ r(x,y) ]  −  β · KL( π(·|x) ‖ π_ref(·|x) )
```

This has a closed-form solution — the Gibbs distribution:

```
π*(y|x) = π_ref(y|x) · exp(r(x,y)/β) / Z(x)
```

Invert it, and the reward is recovered from the optimal policy:

```
r(x,y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
```

Now substitute that into the Bradley–Terry preference model
`P(y_w ≻ y_l) = σ(r(x,y_w) − r(x,y_l))`. The intractable `log Z(x)` **cancels**,
because it depends only on `x`. What remains is a loss you can minimise directly
on preference pairs — no reward model, no sampling, no RL:

```
L = −log σ( β · [ (log π(y_w|x) − log π_ref(y_w|x))
                − (log π(y_l|x) − log π_ref(y_l|x)) ] )
```

That cancellation is the entire contribution of the DPO paper, and it is why a
method that replaces a whole RL pipeline fits in eight lines of code.

## What the gradient does

```
∇L = −β · σ(−β·margin) · [ ∇log π(y_w) − ∇log π(y_l) ]
```

The scalar `σ(−β·margin)` is an automatic difficulty weight: pairs the model
already orders correctly contribute almost nothing.
`test_dpo_gradient_weight_is_sigmoid_of_negative_margin` checks it at three
margins.

Two consequences:

- It is **stable without a value function**, which is why DPO is so much easier
  to run than PPO.
- It **quietly stops learning** once the margin is large. So the metric to watch
  is `reward_margin`, not the loss.

Sanity check you can do in your head: when `π = π_ref`, the margin is 0 and the
loss is exactly `log 2 = 0.6931`. `test_dpo_loss_is_log2_when_policy_equals_reference`
asserts it, and every DPO training log in this repo starts at 0.6931.

## What DPO is *not*

**DPO optimises a relative objective.** Both log-probabilities may fall as long
as the rejected one falls faster — and in practice they do. In the plain-DPO run
here, `logp_chosen` ends at **−4.60** while reward accuracy reaches **0.978**.

Nothing in the objective says the probability mass freed from the rejected
answer has to land on the chosen one. It lands wherever the model finds it
easiest to put.

## The experiment

This is the part worth running yourself:

```bash
python -m scripts.run_alignment              # produces out/alignment/sft.pt
python -m scripts.exp_dpo_variants
```

Six configurations, all starting from the *same* SFT checkpoint (held-out exact
match 0.578), all with the same preference data — pairs of (correct answer,
systematically-wrong answer), scored on 400 stratified held-out prompts:

| config | exact match (95% CI) | Δ vs SFT | fixed / broke | p | reward acc | margin | logp_chosen |
|---|---|---|---|---|---|---|---|
| SFT (start) | 0.578 [0.527, 0.625] | — | — | — | — | — | — |
| DPO β=0.1 | **0.415** [0.367, 0.463] | −0.163 | 49 / 114 | 0.0000 | 0.978 | +1.60 | −4.60 |
| DPO β=0.5 | 0.412 [0.365, 0.460] | −0.165 | 49 / 115 | 0.0000 | 1.000 | +4.93 | −3.53 |
| **DPO β=0.1 + NLL (RPO)** | **0.892** [0.860, 0.922] | **+0.315** | 138 / 12 | 0.0000 | 1.000 | +1.02 | −0.71 |
| IPO β=0.5 | 0.890 [0.858, 0.920] | +0.312 | 131 / 6 | 0.0000 | 1.000 | +0.50 | −0.41 |
| SimPO β=2, γ=0.5 | 0.555 [0.505, 0.603] | −0.022 | 79 / 88 | 0.5360 | 1.000 | +3.87 | −1.42 |
| DPO β=0.1, mixed rejected | 0.750 [0.708, 0.790] | +0.172 | 118 / 49 | 0.0000 | 0.889 | +0.91 | −1.05 |

**Read the reward-accuracy column.** It is 0.98–1.00 in every row. The
preference objective is solved essentially perfectly in all six cases, while
held-out task accuracy ranges from **0.41 to 0.89**. *The preference objective
is not the task.* If you monitor DPO by its loss and its reward accuracy — which
is what most tooling shows you — all six of these runs look like unqualified
successes.

The `logp_chosen` column is the leading indicator that reward accuracy hides:
the two runs that lost accuracy are exactly the two where the chosen answer's
own log-probability collapsed (−4.60, −3.53), while the two that worked kept it
near zero (−0.71, −0.41).

### Why plain DPO lost 16 points

The SFT model had learned the systematic bias `answer + 1`, and **every**
rejected sample in the preference data is `answer + 1`. So "smaller" is a
*direction* the model can follow to satisfy every single comparison — without
ever having to identify the correct answer.

It followed it. Here is the distribution of `prediction − truth` over the
numeric answers after plain DPO:

```
   +0 :  81      <- correct
  -11 :  23
  -12 :  22
   -1 :  22
   -2 :  11
  -10 :  11
  -20 :   6
```

Essentially every error is **negative**. And the clustering at −1, −10, −11, −12
says what "smaller" means to a character-level model: it decrements *digits*,
often more than one.

```
'11 + 81 ='  ->  '81'    (want 92)     both digits decremented
'68 + 38 ='  ->  '94'    (want 106)
'19 + 24 ='  ->  '42'    (want 43)     the plain off-by-one
'81 + 57 ='  ->  '1277'  (want 138)    some outputs degenerate entirely
```

The per-task breakdown shows the same thing, and it is not uniform damage:

| task | SFT | plain DPO | |
|---|---|---|---|
| add | 0.56 | **0.00** | destroyed |
| sub | 0.46 | **0.02** | destroyed |
| mul | 0.32 | 0.16 | halved |
| reverse | 0.62 | 0.28 | halved |
| sort | 0.92 | 0.62 | damaged |
| last | 1.00 | 0.80 | damaged |
| max | 0.34 | **0.54** | *improved* |
| count | 0.40 | **0.90** | *improved* |

`add` and `sub` have wide numeric ranges, so a downward shift sails straight
past the right answer and lands on another wrong one. `count` answers are tiny
integers (0–3): there, shifting down by one from the biased `answer + 1` lands
exactly on the truth, so the same drift that destroys `add` *fixes* `count`.

That is the sharpest version of the lesson. DPO did not learn the task; it
learned a direction. Whether that direction helps or hurts is an accident of
each task's answer space — and the aggregate number (0.415) averages the two
into something that looks like mediocre performance rather than the two opposite
failures it actually is.

This is the documented "DPO pushes probability mass to unintended regions"
failure, reproduced on a laptop in 60 seconds of training. It is not exotic: any
time your rejected samples share a systematic structure, DPO can learn the
structure instead of the target.

### Why the fixes work

**RPO / DPO+NLL (0.892).** Add the ordinary cross-entropy of the chosen answer:

```python
loss = dpo_loss(...) + sft_weight * nll_of_chosen
```

This anchors the *absolute* likelihood of the right answer instead of only the
ratio. Note `logp_chosen` ends at **−0.71** instead of −4.60 — the model is
still confident in the correct answer. One extra term, +0.32 accuracy.

**IPO (0.890).** Replace the sigmoid with a squared loss targeting a *finite*
margin `1/(2β)`:

```python
loss = (margin - 1/(2*beta))**2
```

There is no incentive to push past the target, so no runaway. Note its final
margin is **+0.50** — exactly `1/(2·0.5)` — while DPO β=0.5 ran to **+4.93**.
IPO also broke the fewest items of any variant (6).
`test_ipo_targets_a_specific_margin` verifies the minimum is where the algebra
says.

**Mixed rejected samples (0.750).** Half systematic, half random corruptions.
Now there is no single direction that satisfies every comparison, so the model
has to actually identify the correct answer. This is a *data* fix rather than a
loss fix, and it is usually the cheaper one: it corresponds to collecting
preference data from a diverse set of policies rather than one.

**Larger β (0.412).** A stronger pull towards the reference does **not** save
it here — β=0.5 is statistically indistinguishable from β=0.1 (0.412 vs 0.415).
β limits how far the policy drifts per unit of margin, but it does not remove
the incentive to follow the "smaller" direction, so the policy simply buys the
same drift with a larger margin (+4.93 vs +1.60).

**SimPO (0.555).** Reference-free and length-normalised:

```
L = −log σ( β·(logp_w/|y_w| − logp_l/|y_l|) − γ )
```

Half the memory (no reference model) and no `π_ref` hyperparameter, and the
length normalisation removes DPO's bias towards short answers. But nothing
anchors the policy to its starting point, so it drifts more. Here it is a wash —
79 items fixed against 88 broken, p = 0.54.

## Variants and what each one is for

| variant | loss | use it when |
|---|---|---|
| `sigmoid` | `−log σ(β·margin)` | the default; pair it with `sft_weight > 0` |
| `sigmoid` + `label_smoothing` (cDPO) | mix in the flipped-label loss | you believe a fraction of your labels are wrong |
| `ipo` | `(margin − 1/2β)²` | margins are running away; log-probs collapsing |
| `hinge` | `relu(1 − β·margin)` | you want pairs beyond the margin fully ignored |
| `simpo` | reference-free, length-normalised | memory-bound, or fighting a length bias |

`label_smoothing = 0.5` makes the loss symmetric with its minimum at margin 0 —
i.e. it expresses no preference at all.
`test_dpo_label_smoothing_is_symmetric_and_minimised_at_zero` pins that, which
is the right way to understand what the parameter interpolates between.

## Practical guidance

- **LR 5e-6 to 2e-5** — 20–100× below SFT. DPO with an SFT learning rate
  destroys the model in a few hundred steps.
- **1 epoch.** DPO overfits preference data fast, and the damage is invisible in
  the loss. Every run in the table above is a single epoch and the two plain-DPO
  variants still lost 16 points.
- **β = 0.1 is the usual default.** Do not expect a larger β to rescue a
  mis-specified preference set — measured above, it did not.
- **The reference must be the SFT checkpoint you started from.** The base model
  lets DPO undo the SFT formatting; an EMA of the policy makes the KL term
  meaningless.
- **Always keep an SFT term** (`sft_weight ≈ 1.0`) unless you have measured that
  you do not need it. This repo's pipeline defaults to it because of the table
  above.
- **Evaluate on the task, not on the objective**, with a paired test. McNemar
  (`minigpt.eval.mcnemar`) is the right one: it looks only at the items the two
  models disagree on, which is far more sensitive than comparing two
  independent confidence intervals.

## Exercises

1. **Reproduce the collapse.** Run `--dpo-sft-weight 0` and confirm accuracy
   falls. Then print the model's top-5 tokens for the first answer digit before
   and after. Where did the probability mass go?

2. **Sweep `sft_weight`** ∈ {0, 0.1, 0.5, 1, 5}. Plot held-out accuracy and
   `logp_chosen`. Is there a regime where the DPO term stops contributing
   anything?

3. **Derive the DPO gradient** from the loss by hand, and confirm the
   `σ(−β·margin)` weight numerically with `torch.autograd.grad`.

4. **Implement KTO** (which needs only a binary good/bad label per sample, not
   pairs) and compare against DPO on the same data. Which needs less data for
   the same result?

5. **Make DPO succeed with a systematic bias** *without* the NLL term, purely by
   changing the data. How diverse do the rejected samples need to be? Measure
   accuracy as a function of the systematic fraction.

6. **Build the length bias.** Make the chosen answers systematically longer than
   the rejected ones and measure the change in mean output length under DPO vs
   SimPO. This is the real-world effect that made "verbose assistant" a known
   RLHF artefact.

---

**Previous:** [Lesson 6](06_sft.md) · **Next:** [Lesson 8 — RLVR with GRPO](08_rlvr_grpo.md)
