"""Inference: how a trained model actually produces text.

Three separable concerns, and people conflate them constantly:

1. **What distribution does the model give?**  That is fixed by the weights:
   `p(next | prefix) = softmax(logits)`.
2. **How do we turn it into a token?**  Temperature, top-k, top-p, min-p --
   these *modify* the distribution.  Every one of them trades diversity for
   reliability, and none of them makes the model smarter.
3. **How fast can we do it?**  Decoding one token at a time is memory-bandwidth
   bound, not compute bound: you read every weight to produce one token.  The
   KV cache removes the redundant O(T^2) recompute; speculative decoding
   attacks the bandwidth bound directly by verifying several tokens per pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import GPT


# ---------------------------------------------------------------------------
# Logit processors
# ---------------------------------------------------------------------------


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by `temperature`.

    T -> 0 is argmax; T = 1 is the model's own distribution; T > 1 flattens it.
    Because it acts on logits (pre-softmax), temperature is exactly a rescaling
    of the *log*-probabilities: p_i^(1/T) renormalised.
    """
    if temperature <= 0:
        raise ValueError("use greedy decoding instead of temperature=0")
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k highest logits, set the rest to -inf.

    Crude but effective: it cuts the long tail that causes most incoherence.
    The weakness is that k is fixed -- when the model is confident, k=50 still
    admits 49 bad tokens; when it is uncertain, k=50 may cut good ones.
    """
    if k <= 0 or k >= logits.size(-1):
        return logits
    kth = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens with cumulative prob >= p.

    Adapts to the model's confidence, which is why it became the default.  Note
    the off-by-one that everybody gets wrong: the token that *crosses* the
    threshold must be kept, otherwise a distribution with one token at p=0.99
    and top_p=0.9 would have nothing left to sample.
    """
    if not (0.0 < p < 1.0):
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    # Remove tokens once the cumulative mass *before* them already reached p.
    remove = cumulative - probs > p
    remove[..., 0] = False                      # always keep the top token
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Keep tokens with prob >= min_p * max_prob.

    A cleaner formulation of the same idea as top-p: the threshold is relative
    to the top token, so a confident step keeps almost nothing and an uncertain
    step keeps a lot.  Cheap (no sort) and robust at high temperature.
    """
    if min_p <= 0.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    threshold = min_p * probs.amax(dim=-1, keepdim=True)
    return logits.masked_fill(probs < threshold, float("-inf"))


def repetition_penalty(logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float):
    """Divide (or multiply, if negative) the logits of already-seen tokens.

    A blunt instrument: it also penalises tokens that *should* repeat ("the",
    a variable name, a name).  Useful for small under-trained models, which
    fall into loops; unnecessary for well-trained ones.
    """
    if penalty == 1.0:
        return logits
    for b in range(logits.size(0)):
        seen = torch.unique(prev_tokens[b])
        vals = logits[b, seen]
        # Dividing a negative logit makes it larger, so branch on the sign.
        logits[b, seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def sample_from_logits(logits, temperature=1.0, top_k=0, top_p=0.0, min_p=0.0, greedy=False,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    """Turn (B, V) logits into (B, 1) sampled token ids."""
    if greedy or temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = apply_temperature(logits.float(), temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    logits = min_p_filter(logits, min_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
             top_k: int = 0, top_p: float = 0.0, min_p: float = 0.0, greedy: bool = False,
             eos_id: int | None = None, use_cache: bool = True,
             rep_penalty: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    """Autoregressive decoding, with or without a KV cache.

    `use_cache=False` recomputes the whole prefix each step -- O(T^2) total work
    instead of O(T).  It is here so you can check that the cache is correct
    (they must produce identical tokens) and time the difference.

    Sequences that emit `eos_id` are frozen: we keep stepping the batch but
    overwrite finished rows with EOS, which is simpler than ragged batching and
    is what most serving stacks do below a certain scale.
    """
    was_training = model.training
    model.eval()
    B = idx.size(0)
    device = idx.device
    caches = model.make_caches(B, model.cfg.block_size) if use_cache else None
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    out = idx

    # Prefill: one forward over the whole prompt, filling the cache.
    if use_cache:
        logits, _ = model(idx, caches=caches)
        logits = logits[:, -1]
    else:
        logits, _ = model(idx[:, -model.cfg.block_size :])
        logits = logits[:, -1]

    for _ in range(max_new_tokens):
        step_logits = logits.clone()
        if rep_penalty != 1.0:
            step_logits = repetition_penalty(step_logits, out, rep_penalty)
        nxt = sample_from_logits(step_logits, temperature, top_k, top_p, min_p, greedy, generator)
        if eos_id is not None:
            nxt = torch.where(finished.unsqueeze(1), torch.full_like(nxt, eos_id), nxt)
            finished = finished | (nxt.squeeze(1) == eos_id)
        out = torch.cat([out, nxt], dim=1)
        if eos_id is not None and bool(finished.all()):
            break
        if out.size(1) >= model.cfg.block_size and use_cache:
            break  # cache is full; a real server would evict or slide here
        if use_cache:
            logits, _ = model(nxt, caches=caches)
            logits = logits[:, -1]
        else:
            logits, _ = model(out[:, -model.cfg.block_size :])
            logits = logits[:, -1]

    if was_training:
        model.train()
    return out


@torch.no_grad()
def generate_text(model: GPT, tokenizer, prompt: str, max_new_tokens: int = 100, **kw) -> str:
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    if ids.numel() == 0:
        ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = generate(model, ids, max_new_tokens, **kw)
    return tokenizer.decode(out[0])


# ---------------------------------------------------------------------------
# Speculative decoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def speculative_decode(target: GPT, draft: GPT, idx: torch.Tensor, max_new_tokens: int,
                       lookahead: int = 4, temperature: float = 1.0,
                       generator: torch.Generator | None = None,
                       eos_id: int | None = None) -> tuple[torch.Tensor, dict]:
    """Draft-and-verify decoding that provably samples from the target model.

    Per round:
      1. the small `draft` model autoregressively proposes `lookahead` tokens
         x_1..x_g, recording its own probabilities q_i(x_i);
      2. the big `target` model scores all of them in ONE forward pass, giving
         p_1..p_{g+1};
      3. each proposal is accepted with probability min(1, p_i(x_i)/q_i(x_i));
      4. at the first rejection we sample the replacement token from the
         *residual* distribution norm(max(0, p_i - q_i)) and discard the rest;
      5. if all g were accepted we get a free bonus token from p_{g+1}.

    Step 4 is what makes this exact rather than an approximation -- the accepted
    -or-residual mixture has exactly distribution p.  (Proof sketch: a token y
    is emitted at position i either by acceptance, with probability
    q(y)*min(1,p(y)/q(y)) = min(q(y), p(y)), or by rejection followed by the
    residual draw, which contributes the remaining max(0, p(y)-q(y)).  The two
    sum to p(y).)

    The win is purely about *bandwidth*: one target forward pass costs nearly
    the same for 1 or g+1 tokens, because decoding is bound by reading weights.
    Speedup ~ accepted_per_round / (1 + g * cost_draft/cost_target).
    """
    target.eval()
    draft.eval()
    if idx.size(0) != 1:
        raise ValueError("speculative_decode is written for batch size 1")
    device = idx.device
    out = idx
    stats = {"rounds": 0, "proposed": 0, "accepted": 0, "bonus": 0}

    def probs_of(model, seq, n_last):
        logits, _ = model(seq[:, -model.cfg.block_size :])
        return F.softmax(logits[0, -n_last:].float() / temperature, dim=-1)

    while out.size(1) < idx.size(1) + max_new_tokens:
        g = min(lookahead, idx.size(1) + max_new_tokens - out.size(1))
        # ---- 1. draft proposes g tokens (no cache here: clarity over speed)
        proposal = out
        q_rows = []
        for _ in range(g):
            q = probs_of(draft, proposal, 1)[0]                     # (V,)
            tok = torch.multinomial(q, 1, generator=generator)
            q_rows.append(q)
            proposal = torch.cat([proposal, tok.view(1, 1)], dim=1)
        draft_tokens = proposal[0, -g:]
        q_probs = torch.stack(q_rows)                               # (g, V)

        # ---- 2. target scores the g proposals plus one extra position
        # Positions -g-1..-1 of the logits predict tokens at -g..end, i.e. the
        # g proposals and one more.  Off-by-one here is the classic bug.
        p_probs = probs_of(target, proposal, g + 1)                 # (g+1, V)

        # ---- 3/4. sequential accept/reject
        n_accepted = 0
        rejected = False
        for i in range(g):
            x = int(draft_tokens[i])
            p_x, q_x = float(p_probs[i, x]), float(q_probs[i, x])
            r = torch.rand((), generator=generator, device=device).item()
            if q_x > 0 and r < min(1.0, p_x / q_x):
                n_accepted += 1
                continue
            residual = (p_probs[i] - q_probs[i]).clamp_min(0)
            total = residual.sum()
            # If p is entirely dominated by q, fall back to p (measure-zero in
            # theory, reachable in fp32).
            residual = residual / total if total > 0 else p_probs[i]
            new_tok = torch.multinomial(residual, 1, generator=generator)
            out = torch.cat([proposal[:, : out.size(1) + n_accepted], new_tok.view(1, 1)], dim=1)
            rejected = True
            break

        stats["rounds"] += 1
        stats["proposed"] += g
        stats["accepted"] += n_accepted

        if not rejected:
            # ---- 5. all accepted: take the bonus token from p_{g+1}
            bonus = torch.multinomial(p_probs[g], 1, generator=generator)
            out = torch.cat([proposal, bonus.view(1, 1)], dim=1)
            stats["bonus"] += 1

        if eos_id is not None and (out[0, idx.size(1) :] == eos_id).any():
            first = int((out[0, idx.size(1):] == eos_id).nonzero()[0]) + idx.size(1)
            out = out[:, : first + 1]
            break
        if out.size(1) >= target.cfg.block_size:
            break

    stats["acceptance_rate"] = stats["accepted"] / max(1, stats["proposed"])
    stats["tokens_per_round"] = (out.size(1) - idx.size(1)) / max(1, stats["rounds"])
    return out, stats


# ---------------------------------------------------------------------------
# Scoring (used by eval, SFT, DPO and GRPO)
# ---------------------------------------------------------------------------


def token_logprobs(model: GPT, idx: torch.Tensor, targets: torch.Tensor | None = None,
                   chunk: int = 0) -> torch.Tensor:
    """Log p(target_t | prefix) for every position: shape (B, T).

    Defaults to next-token scoring of `idx` itself, so column t is
    log p(idx[t+1] | idx[:t+1]) and the last column is dropped by the caller.

    `chunk > 0` computes the gather in vocabulary chunks.  The (B, T, V) logits
    tensor is the single largest activation in an LLM -- at V=128k, B=8, T=4096
    it is 16 GB in fp32 -- so chunking here is often what makes a fine-tune fit.
    """
    logits, _ = model(idx)
    if targets is None:
        logits = logits[:, :-1]
        targets = idx[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def sequence_logprob(model: GPT, idx: torch.Tensor, mask: torch.Tensor | None = None,
                     average: bool = False) -> torch.Tensor:
    """Sum (or mean) of token log-probs over the positions selected by `mask`.

    `mask` is aligned with the *targets*, i.e. shape (B, T-1).  Sum gives the
    sequence log-likelihood used by DPO; mean gives the length-normalised score
    used by multiple-choice evaluation and by length-debiased ranking.
    """
    lp = token_logprobs(model, idx)
    if mask is None:
        mask = torch.ones_like(lp)
    mask = mask.to(lp.dtype)
    total = (lp * mask).sum(dim=-1)
    return total / mask.sum(dim=-1).clamp_min(1) if average else total
