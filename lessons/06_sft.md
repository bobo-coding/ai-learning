# Lesson 6 — Supervised fine-tuning

Read: `minigpt/sft.py`, `minigpt/tasks.py` · Tests: `tests/test_alignment.py`

A pretrained model continues text. It does not answer questions, because nothing
in pretraining said *"when you see a question, an answer follows, and then you
stop"*. Watch the base model from this repo's pipeline try:

```
'68 - 38 ='             -> '68 - 38 = 38\n6'
'max 82 91 68 54 ='     -> 'max 68 24 = 68'
'sort 15 32 49 41 7 ='  -> 'sort 21 17 1 ='
```

Held-out exact match: **0.000**. It has learned the *surface form* of the corpus
perfectly — it produces well-formed arithmetic-looking lines — and it has no
concept of being asked something.

SFT is plain cross-entropy training that teaches exactly three things.

## 1. A format

```
<|user|>\n{content}<|end|>\n<|assistant|>\n{content}<|end|>
```

Special tokens delimit turns so the model can distinguish "text I should
continue" from "text I should respond to". At generation time you stop right
after the `<|assistant|>\n` header, so the model's first predicted token is the
first token of its reply.

The header must be **byte-identical** between training and inference. A stray
newline or a trailing space is a silent quality regression, not an error — which
is why `prompt_text()` is a single function used by training, evaluation and
rollouts alike, and `test_template_renders_a_stable_header` pins its exact
string.

## 2. Loss on the completion only

```python
head_ids = tok.encode(prompt_text(prompt))       # mask 0
tail_ids = tok.encode(response + END)            # mask 1
```

Training on the prompt tokens too is not catastrophic, but it spends capacity
modelling the *user's* distribution and measurably hurts small models. The mask
is the whole trick.

The shift happens in `collate`, and it is worth staring at:

```python
xs.append(ids[:-1])                  # inputs
ys.append(ids[1:])                   # targets
ms.append(cm[1:])                    # the mask shifts too
```

Position `t` predicts token `t+1`, so the mask must shift with the targets. Off
by one here and the model answers the *previous* question — a bug that produces
fluent, confident, wrong output.
`test_collate_shift_alignment` asserts that the set of masked target positions
is *exactly* the completion tokens, for every example in a batch.

## 3. When to stop

**The end-of-turn token must be inside the loss mask.** This is the single most
consequential line in the module:

```python
tail_ids = tok.encode(response + END)     # END is trainable
```

Omit it — the most common SFT bug — and the model produces a perfect answer
followed by an endless hallucinated conversation, because it was never taught
that stopping is an action. `test_end_token_is_inside_the_loss_mask` checks the
decoded trainable span is `"5<|end|>"`, not `"5"`.

The same logic applies to truncation. If an example is too long we truncate from
the **left**, keeping the completion intact — a truncated answer teaches the
model to stop mid-sentence.

## Adding chat tokens to a pretrained model

`resize_token_embeddings` grows the embedding (and the untied head). Two
details:

- New rows must be initialised at the *same scale* as the old ones. Zeros make
  the new tokens unreachable until their gradient wakes up; large values make
  them dominate early. The code uses the existing matrix's own std.
- With tied embeddings, resizing the input resizes the head for free. Untied,
  both must be resized or you get a shape error at the first loss.

`test_resize_token_embeddings_preserves_old_logits` verifies that logits for
pre-existing tokens are unchanged, for both tied and untied models.

## Verifiable tasks

`minigpt/tasks.py` defines eight tasks (`add`, `sub`, `mul`, `reverse`, `sort`,
`count`, `last`, `max`), each with a prompt generator, an exact answer, and a
verifier function. Synthetic, deliberately — SFT, DPO and RLVR are only legible
when "is this correct?" is a function rather than a vibe.

Two design decisions that are really about honest evaluation:

**The split is by prompt, not by row.** `split_examples` deduplicates prompts
first and holds out 12% of the *unique* prompts, so no evaluation prompt appears
in training. Random row splitting would let `12 + 7 =` appear in both halves and
turn the benchmark into a memorisation check. A surprising amount of published
arithmetic accuracy is obtained this way by accident.

**The evaluation set is stratified, and every per-task score is reported with
its n.** This is not fussiness — it is the fix for a real bug that was in this
repo. The eval set used to be `val_ex[:300]`, a prefix of a shuffled list, which
gives you the *training mixture* rather than a balanced benchmark:

```
before (val[:300]):  sort 92, max 89, add 70, sub 45, count 3, mul 1
                     reverse 0, last 0        <- absent entirely
after  (stratified): 50 of every task
```

So `mul = 0.00` was **one example**, quoted to two decimal places, and two tasks
were missing from every table without anyone noticing. `stratified_sample` and
`generative_eval`'s `n_by_task` exist so that cannot recur. Lesson 9 has the
general version of this argument.

Two of the tasks also had prompt spaces too small to hold out from at all:
`reverse` and `last` were capped by a 37-word list, leaving 4 held-out prompts
each. `WORDS` is now 437 entries (deterministically generated pseudo-words) and
`mul` runs to 19×19 rather than 12×12.

## The measured result

```bash
python -m scripts.run_alignment
```

45% of the demonstrations are deliberately wrong, in a *consistent* way
(`systematic_corrupt`: always off-by-one upward, always drop the last list
element). After 3 epochs on 52,717 demonstrations, scored on 400 held-out
prompts, 50 per task:

| | exact match (95% CI) |
|---|---|
| base (pretrained) | 0.000 |
| **SFT** | **0.578** [0.527, 0.625] |

| task | n | SFT |
|---|---|---|
| last | 50 | 1.00 |
| sort | 50 | 0.92 |
| reverse | 50 | 0.62 |
| add | 50 | 0.56 |
| sub | 50 | 0.46 |
| count | 50 | 0.40 |
| max | 50 | 0.34 |
| mul | 50 | 0.32 |

Three things to read off that:

**The format was learned completely.** Every output is now a bare answer with a
stop token. That part of SFT is easy and works on the first epoch.

**The systematic bias was learned too.** `'68 - 38 =' → '31'` (want 30);
`'sort 15 32 49 41 7 =' → '7 15 32 41'` (dropped the last element). The model
faithfully reproduced the demonstrators' mistake, because cross-entropy fits
whatever distribution you show it.

**The per-task spread is the interesting part, and it tracks task structure.**
`last` is trivial (copy one character) and is solved. `sort` is nearly solved
because the systematic corruption there — dropping the final element — is easy
to distinguish from the majority pattern. `max` and `count` sit lowest because
the off-by-one corruption produces an answer that is *itself plausible* for
those tasks, so the two modes are hardest to separate. `mul` at 0.32 is a
genuine generalization limit: 19×19 products, 12% held out, and the model has to
interpolate a multiplication table it only partly saw.

### The noise matters more than its amount

Running the same pipeline with `--random-noise` (each wrong demonstration wrong
in a *different* way) gives **0.963** instead of 0.578 — from the same 45%
corruption rate. (That figure predates the stratified eval set, so treat it as
indicative rather than directly comparable; the direction and the size of the
gap are the point.)

Unbiased label noise is averaged away by cross-entropy: the correct answer
remains the single most likely continuation, and the noise only flattens the
distribution around it. A *systematic* bias puts a competing mode in the data,
and the model learns that mode. When you read "our data is 5% noisy", the
question to ask is not how much, but whether the errors correlate.

### Greedy vs sampled

Same checkpoint: **0.578 greedy** vs **0.477 pass@1 at temperature 1.0**. And
`pass@8 = 0.953` — the model can produce the right answer for almost every
prompt, it just does not reliably put it first. That gap is the room lessons 7
and 8 work in.

A benchmark number without a stated decoding policy is not a number.

### One honest caveat

MPS reductions are not bit-deterministic, and this SFT configuration has
produced **0.547**, **0.578** and **0.643** across runs with identical seeds
(the first two on different eval sets, so compare only the last two loosely).
The swing is larger than the ~2.5-point sampling error, because the training
data is deliberately bimodal (45% consistently biased) and the model sits near a
decision boundary between the two modes, so small numerical differences tip many
items at once.

Consequence for the next two lessons: comparisons that start from the *same*
checkpoint are trustworthy (they are paired); comparing absolute numbers across
separate training runs is not. Lesson 9 covers how to tell the difference.

## Practical hyperparameters

```python
SFTConfig(epochs=3, lr=3e-4, weight_decay=0.0)
```

- **LR ~10× below pretraining.** The model is already good; large steps destroy
  what pretraining bought.
- **No weight decay.** On a fine-tune, decay mostly just forgets.
- **2–4 epochs.** More and the model memorises the demonstrations; a rising
  validation loss with a still-falling training loss is the signal.
- **Watch `pad_frac`.** The run above reports `mask waste 0.91` — 91% of the
  tensor positions are prompt or padding and contribute nothing. Sorting each
  batch by length, or packing several short examples per row with a
  block-diagonal mask, is where the throughput is on real SFT data.

## Exercises

1. **Delete the stop token from the mask** (`tok.encode(response)` instead of
   `response + END`), retrain, and generate. Describe exactly what the output
   looks like and why.

2. **Break the shift.** Use `ms.append(cm[:-1])` instead of `cm[1:]`. Does the
   loss still go down? What does the model produce? This is the bug worth being
   able to recognise by its symptom.

3. **Train on the prompt too** (all-ones mask) and compare held-out accuracy at
   equal steps. Measure how much of the loss is being spent on prompt tokens.

4. **Sweep the noise.** Run `--sft-noise` ∈ {0, 0.15, 0.3, 0.45, 0.6} with and
   without `--random-noise`. Plot accuracy against noise for both. The two curves
   have different shapes — explain the difference in terms of what cross-entropy
   fits.

5. **Fix `mul`.** Raise `--n-examples` until held-out `mul` accuracy leaves
   zero, or add a chain-of-thought format that decomposes the multiplication.
   Which is cheaper?

6. **Reduce the mask waste.** Implement length-sorted batching, measure the
   change in `pad_frac` and in tokens/second, and confirm the final accuracy is
   unchanged.

---

**Previous:** [Lesson 5](05_inference.md) · **Next:** [Lesson 7 — Preference optimization](07_dpo.md)
