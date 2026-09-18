"""A suite of small, *verifiable* tasks -- the spine of the alignment lessons.

Why synthetic tasks instead of a real instruction dataset?  Because SFT, DPO and
especially RLVR are only legible when you can answer "is this output correct?"
with a function instead of a vibe.  Every task here has:

  * a prompt generator,
  * an exact reference answer,
  * a verifier `reward(prompt, completion) -> float in [0, 1]`.

That makes it possible to (a) evaluate honestly, (b) build preference pairs with
known ground truth instead of human labels, and (c) run reinforcement learning
with a real reward signal on a laptop.  The tasks are deliberately easy enough
that a ~1M-parameter character-level model can learn them in minutes, so you can
see each training stage actually change behaviour.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Example:
    task: str
    prompt: str
    answer: str

    def as_pair(self) -> tuple[str, str]:
        return self.prompt, self.answer


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

WORDS = [
    "cat", "dog", "bird", "fish", "lion", "bear", "wolf", "frog", "mouse", "horse",
    "apple", "bread", "cheese", "grape", "lemon", "melon", "onion", "peach", "plum",
    "river", "stone", "cloud", "storm", "flame", "grass", "light", "night", "ocean",
    "table", "chair", "plate", "spoon", "knife", "glass", "brush", "paper", "pencil",
]


def _gen_add(rng: random.Random) -> Example:
    a, b = rng.randint(0, 99), rng.randint(0, 99)
    return Example("add", f"{a} + {b} =", str(a + b))


def _gen_sub(rng: random.Random) -> Example:
    a, b = rng.randint(0, 99), rng.randint(0, 99)
    a, b = max(a, b), min(a, b)          # keep answers non-negative
    return Example("sub", f"{a} - {b} =", str(a - b))


def _gen_mul(rng: random.Random) -> Example:
    a, b = rng.randint(0, 12), rng.randint(0, 12)
    return Example("mul", f"{a} * {b} =", str(a * b))


def _gen_reverse(rng: random.Random) -> Example:
    w = rng.choice(WORDS)
    return Example("reverse", f"reverse {w} =", w[::-1])


def _gen_sort(rng: random.Random) -> Example:
    n = rng.randint(3, 5)
    nums = [rng.randint(0, 49) for _ in range(n)]
    return Example("sort", "sort " + " ".join(map(str, nums)) + " =",
                   " ".join(map(str, sorted(nums))))


def _gen_count(rng: random.Random) -> Example:
    w = rng.choice(WORDS)
    ch = rng.choice(sorted(set(w)))
    return Example("count", f"count {ch} in {w} =", str(w.count(ch)))


def _gen_last(rng: random.Random) -> Example:
    w = rng.choice(WORDS)
    return Example("last", f"last of {w} =", w[-1])


def _gen_max(rng: random.Random) -> Example:
    nums = [rng.randint(0, 99) for _ in range(rng.randint(2, 4))]
    return Example("max", "max " + " ".join(map(str, nums)) + " =", str(max(nums)))


GENERATORS = {
    "add": _gen_add,
    "sub": _gen_sub,
    "mul": _gen_mul,
    "reverse": _gen_reverse,
    "sort": _gen_sort,
    "count": _gen_count,
    "last": _gen_last,
    "max": _gen_max,
}

# The character set the tasks use, fixed so the tokenizer is reproducible.
TASK_CHARS = sorted(set("0123456789 +-*=abcdefghijklmnopqrstuvwxyz\n"))


# ---------------------------------------------------------------------------
# Sampling and verification
# ---------------------------------------------------------------------------


def sample_examples(n: int, tasks: list[str] | None = None, seed: int = 0) -> list[Example]:
    """Draw `n` examples, cycling through `tasks` for a balanced mixture.

    Deduplication is deliberately *not* done: the natural duplicate rate of
    `add` is high (10^4 possible prompts) and leaving it alone means train and
    test overlap, exactly as they do in real arithmetic benchmarks.  The
    held-out split below is what you use for honest numbers.
    """
    rng = random.Random(seed)
    tasks = tasks or list(GENERATORS)
    return [GENERATORS[tasks[i % len(tasks)]](rng) for i in range(n)]


def split_examples(examples: list[Example], val_fraction: float = 0.1):
    """Split with *prompt-level* deduplication, so no val prompt appears in train.

    This is the split that matters.  Random row splitting leaks: `12 + 7 =`
    appearing in both halves turns the eval into a memorisation check, which is
    how a lot of published arithmetic accuracy is obtained by accident.
    """
    seen: dict[str, list[Example]] = {}
    for ex in examples:
        seen.setdefault(ex.prompt, []).append(ex)
    prompts = sorted(seen)
    rng = random.Random(12345)
    rng.shuffle(prompts)
    n_val = max(1, int(len(prompts) * val_fraction))
    val_prompts = set(prompts[:n_val])
    train = [e for p in prompts if p not in val_prompts for e in seen[p]]
    val = [seen[p][0] for p in prompts if p in val_prompts]
    return train, val


def normalise(text: str) -> str:
    """Strip whitespace and stop at the first newline -- a lenient answer parse."""
    return text.strip().split("\n")[0].strip()


def reward_exact(example: Example, completion: str) -> float:
    """1.0 if the completion's first line equals the reference answer.

    Exact match is the right default for a verifiable task: it cannot be gamed
    by verbosity, and it gives RL a clean 0/1 signal.  Its weakness -- that
    "59." scores 0 -- is why `reward_shaped` exists below.
    """
    return 1.0 if normalise(completion) == example.answer else 0.0


def reward_shaped(example: Example, completion: str) -> float:
    """Exact match, plus partial credit for a prefix of the right answer.

    Shaped rewards make RL converge faster on long answers, and they are also
    the classic way to accidentally teach reward hacking: here, a model could
    farm 0.4 by emitting the first digit and stopping.  The `format` term
    rewards terminating cleanly, which pushes back on exactly that.
    """
    pred = normalise(completion)
    if pred == example.answer:
        return 1.0
    ref = example.answer
    common = 0
    for a, b in zip(pred, ref):
        if a != b:
            break
        common += 1
    prefix = 0.5 * common / max(1, len(ref))
    fmt = 0.1 if pred and len(pred) <= len(ref) + 2 else 0.0
    return min(0.9, prefix + fmt)


def corrupt_answer(example: Example, rng: random.Random) -> str:
    """Produce a *plausibly wrong* answer -- the `rejected` side of a DPO pair.

    Plausible matters.  If rejected samples were random noise, the preference
    model would only learn "be well-formed", and DPO would teach formatting
    rather than correctness.  These corruptions keep length and character class
    and change the content, so the only way to win the comparison is to be right.
    """
    ans = example.answer
    kinds = ["offby", "digit", "shuffle", "truncate"]
    for _ in range(8):
        kind = rng.choice(kinds)
        if kind == "offby" and ans.lstrip("-").isdigit():
            out = str(int(ans) + rng.choice([-3, -2, -1, 1, 2, 3]))
        elif kind == "digit":
            i = rng.randrange(len(ans))
            pool = "0123456789" if ans[i].isdigit() else "abcdefghijklmnopqrstuvwxyz"
            out = ans[:i] + rng.choice(pool) + ans[i + 1 :]
        elif kind == "shuffle" and len(ans) > 1:
            chars = list(ans)
            rng.shuffle(chars)
            out = "".join(chars)
        else:
            out = ans[: max(1, len(ans) - 1)]
        if out != ans:
            return out
    return ans + "0"


def systematic_corrupt(example: Example) -> str:
    """A *consistent* wrong answer for this example -- always the same mistake.

    This is the difference between label noise you can ignore and label noise
    you cannot.  `corrupt_answer` is random, so 30% random-wrong demonstrations
    still leave the correct answer as the single most likely continuation and
    cross-entropy averages the noise away -- SFT barely notices.  A *systematic*
    error (always off-by-one, always dropping the last character) puts a
    competing mode in the data, and the model learns that mode too.

    Which is exactly the situation preference optimization and RLVR exist for:
    the demonstrations are biased, but the *comparisons* (or the verifier) are
    not.
    """
    ans = example.answer
    if ans.lstrip("-").isdigit():
        return str(int(ans) + 1)                 # always off by one, upward
    if " " in ans:
        parts = ans.split(" ")                   # always drop the last element
        return " ".join(parts[:-1]) if len(parts) > 1 else ans + "x"
    if len(ans) > 1:
        return ans[:-1]                          # always truncate by one char
    return ans + "x"


def make_preference_pairs(examples: list[Example], seed: int = 0,
                          systematic: bool = False) -> list[dict]:
    """Turn verified examples into (prompt, chosen, rejected) triples.

    `systematic=True` uses `systematic_corrupt` for the rejected side, which is
    the realistic setting: preference data is collected on the *model's own*
    outputs, so the rejected sample is whatever mistake the model actually
    makes -- not a random string.
    """
    rng = random.Random(seed)
    out = []
    for e in examples:
        bad = systematic_corrupt(e) if systematic else corrupt_answer(e, rng)
        if bad == e.answer:
            bad = corrupt_answer(e, rng)
        out.append({"task": e.task, "prompt": e.prompt, "chosen": e.answer, "rejected": bad})
    return out


def accuracy_by_task(examples: list[Example], completions: list[str],
                     reward=reward_exact) -> dict[str, float]:
    """Per-task mean reward plus the overall mean.

    Always report per-task: an aggregate hides that the model nailed `add` and
    learned nothing about `sort`, which is the single most common way an
    alignment experiment looks like it worked when it did not.
    """
    buckets: dict[str, list[float]] = {}
    for ex, comp in zip(examples, completions):
        buckets.setdefault(ex.task, []).append(reward(ex, comp))
    out = {k: sum(v) / len(v) for k, v in sorted(buckets.items())}
    allv = [x for v in buckets.values() for x in v]
    out["overall"] = sum(allv) / max(1, len(allv))
    return out
