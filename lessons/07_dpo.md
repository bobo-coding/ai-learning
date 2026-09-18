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
here, `logp_chosen` went from −0.47 to −4.52 while reward accuracy rose to
0.983.

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
match 0.547), all with the same preference data — pairs of (correct answer,
systematically-wrong answer):

| config | held-out exact match | Δ vs SFT | p (McNemar) | final reward acc | final margin |
|---|---|---|---|---|---|
| SFT (start) | 0.547 | — | — | — | — |
| DPO β=0.1 | **0.203** | −0.343 | 0.0000 | 1.000 | +2.41 |
| DPO β=0.5 | 0.433 | −0.113 | 0.0048 | 1.000 | +5.67 |
| **DPO β=0.1 + NLL (RPO)** | **0.967** | **+0.420** | 0.0000 | 1.000 | +1.28 |
| IPO β=0.5 | 0.963 | +0.417 | 0.0000 | 1.000 | +0.50 |
| SimPO β=2, γ=0.5 | 0.623 | +0.077 | 0.0560 | 1.000 | +3.32 |
| DPO β=0.1, mixed rejected | 0.857 | +0.310 | 0.0000 | 0.900 | +1.02 |

**Read the reward-accuracy column.** It is ~1.0 in every row. The preference
objective is solved perfectly in all six cases, while held-out task accuracy
ranges from 0.20 to 0.97. *The preference objective is not the task.* If you
monitor DPO by its loss and its reward accuracy — which is what most tooling
shows you — all six of these runs look like unqualified successes.

### Why plain DPO collapsed to 0.203

Look at what it produces:

```
'68 - 38 ='  ->  '29'    (want 30; SFT said 31)
'max 82 91 68 54 ='  ->  '71'    (want 91)
```

The SFT model had learned the systematic bias `answer + 1`. Every rejected
sample in the preference data is `answer + 1`. So "smaller" is a *direction* the
model can follow to satisfy every comparison — and it followed it straight past
the correct answer to `answer − 1`, which is equally wrong and equally preferred
over `answer + 1`.

This is the documented "DPO pushes probability mass to unintended regions"
failure, reproduced on a laptop in 60 seconds of training. It is not exotic: any
time your rejected samples share a systematic structure, DPO can learn the
structure instead of the target.

### Why the fixes work

**RPO / DPO+NLL (0.967).** Add the ordinary cross-entropy of the chosen answer:

```python
loss = dpo_loss(...) + sft_weight * nll_of_chosen
```

This anchors the *absolute* likelihood of the right answer instead of only the
ratio. Note `logp_chosen` ends at **−0.11** instead of −4.52 — the model is
still confident in the correct answer. One extra term, +0.42 accuracy.

**IPO (0.963).** Replace the sigmoid with a squared loss targeting a *finite*
margin `1/(2β)`:

```python
loss = (margin - 1/(2*beta))**2
```

There is no incentive to push past the target, so no runaway. Note its final
margin is +0.50 — exactly `1/(2·0.5)` — while DPO's ran to +5.67.
`test_ipo_targets_a_specific_margin` verifies the minimum is where the algebra
says.

**Mixed rejected samples (0.857).** Half systematic, half random corruptions.
Now there is no single direction that satisfies every comparison, so the model
has to actually identify the correct answer. This is a *data* fix rather than a
loss fix, and it is usually the cheaper one: it corresponds to collecting
preference data from a diverse set of policies rather than one.

**Larger β (0.433).** A stronger pull towards the reference limits the damage
but does not prevent it — it slows the drift rather than removing the incentive.

**SimPO (0.623).** Reference-free and length-normalised:

```
L = −log σ( β·(logp_w/|y_w| − logp_l/|y_l|) − γ )
```

Half the memory (no reference model) and no `π_ref` hyperparameter, and the
length normalisation removes DPO's bias towards short answers. But nothing
anchors the policy to its starting point, so it drifts more. Here it helped, but
not significantly (p = 0.056).

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
  the loss. The `--dpo-epochs 2` run in this repo's first attempt reached reward
  accuracy 1.0 and margin +2.3 while task accuracy fell 23 points.
- **β = 0.1 is the usual default**, 0.5 if you see log-probs collapsing.
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
