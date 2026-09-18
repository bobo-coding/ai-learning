"""Supervised fine-tuning: turning a next-token predictor into an assistant.

A pretrained model continues text.  It does not answer questions, because
nothing in the pretraining data said "when you see a question, an answer
follows and then you stop".  SFT is plain cross-entropy training that teaches
exactly three things:

1. **A format.**  Special tokens delimit turns, so the model can tell "text I
   should continue" from "text I should respond to".
2. **Loss on the completion only.**  Training on the prompt tokens too is not
   catastrophic, but it spends capacity modelling the *user's* distribution and
   measurably hurts small models.  The mask is the whole trick.
3. **When to stop.**  The end-of-turn token must be inside the loss mask.  Omit
   it -- the single most common SFT bug -- and the model produces a perfect
   answer followed by an endless hallucinated conversation.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from .model import GPT
from .optim import clip_grad_norm
from .utils import cosine_lr, pick_device, seed_everything

# The chat control tokens.  Any unique strings work; what matters is that they
# are single tokens (so the model cannot "half-emit" a role marker) and that
# they never occur in natural text.
PAD = "<|pad|>"
EOT = "<|endoftext|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"
CHAT_SPECIALS = [PAD, EOT, USER, ASSISTANT, END]


@dataclass
class Encoded:
    ids: list[int]
    completion_mask: list[int]   # 1 on tokens the model must learn to produce


class ChatTemplate:
    """Renders conversations to text and to (ids, completion_mask).

    The rendered form is

        <|user|>\\n{content}<|end|>\\n<|assistant|>\\n{content}<|end|>

    and generation-time prompts stop right after the `<|assistant|>\\n` header,
    so the model's first predicted token is the first token of its reply.  Note
    that the header must be *identical* between training and inference down to
    the whitespace; a stray newline is a silent quality regression.
    """

    def __init__(self, tokenizer):
        self.tok = tokenizer
        missing = [t for t in CHAT_SPECIALS if t not in getattr(tokenizer, "special_tokens", {})]
        if missing:
            raise ValueError(f"tokenizer is missing chat special tokens: {missing}. "
                             f"Call tokenizer.add_special_tokens(CHAT_SPECIALS) first.")
        self.pad_id = tokenizer.special_tokens[PAD]
        self.end_id = tokenizer.special_tokens[END]
        self.eot_id = tokenizer.special_tokens[EOT]

    # ---------------------------------------------------------------- rendering

    def render(self, messages: list[dict], add_generation_prompt: bool = False) -> str:
        parts = []
        for m in messages:
            tag = USER if m["role"] == "user" else ASSISTANT
            parts.append(f"{tag}\n{m['content']}{END}\n")
        text = "".join(parts)
        if add_generation_prompt:
            text += f"{ASSISTANT}\n"
        return text

    def prompt_text(self, prompt: str) -> str:
        return self.render([{"role": "user", "content": prompt}], add_generation_prompt=True)

    # ---------------------------------------------------------------- encoding

    def encode(self, prompt: str, response: str, max_len: int | None = None) -> Encoded:
        """Encode one turn, masking everything except the assistant's reply."""
        head = self.prompt_text(prompt)
        head_ids = self.tok.encode(head)
        # END is included in the completion so the model learns to terminate.
        tail_ids = self.tok.encode(response + END)
        ids = head_ids + tail_ids
        mask = [0] * len(head_ids) + [1] * len(tail_ids)
        if max_len is not None and len(ids) > max_len:
            # Truncate from the *left*, keeping the completion intact: a
            # truncated answer teaches the model to stop mid-sentence.
            ids, mask = ids[-max_len:], mask[-max_len:]
        return Encoded(ids, mask)

    def encode_conversation(self, messages: list[dict], max_len: int | None = None) -> Encoded:
        """Multi-turn: every assistant turn contributes to the loss."""
        ids: list[int] = []
        mask: list[int] = []
        for m in messages:
            tag = USER if m["role"] == "user" else ASSISTANT
            header = self.tok.encode(f"{tag}\n")
            body = self.tok.encode(f"{m['content']}{END}\n")
            ids += header + body
            trainable = 1 if m["role"] == "assistant" else 0
            mask += [0] * len(header) + [trainable] * len(body)
        if max_len is not None and len(ids) > max_len:
            ids, mask = ids[:max_len], mask[:max_len]
        return Encoded(ids, mask)


def collate(batch: list[Encoded], pad_id: int, device="cpu"):
    """Pad a list of encoded examples into (x, y, loss_mask) tensors.

    The shift happens here: `x = ids[:-1]`, `y = ids[1:]`, and the loss mask is
    also shifted by one, because position t predicts token t+1.  Getting that
    shift wrong by one position is the second most common SFT bug and shows up
    as a model that answers the *previous* question.
    """
    max_len = max(len(e.ids) for e in batch)
    xs, ys, ms = [], [], []
    for e in batch:
        pad = max_len - len(e.ids)
        ids = e.ids + [pad_id] * pad
        cm = e.completion_mask + [0] * pad
        xs.append(ids[:-1])
        ys.append(ids[1:])
        ms.append(cm[1:])
    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    m = torch.tensor(ms, dtype=torch.bool, device=device)
    return x, y, m


def resize_token_embeddings(model: GPT, new_vocab_size: int, init_std: float | None = None) -> GPT:
    """Grow the embedding (and untied head) to make room for new special tokens.

    Adding chat tokens after pretraining is routine.  Two details:
      * new rows must be initialised to the *same scale* as the old ones -- zeros
        make the new tokens unreachable until their gradient wakes up, and
        large values make them dominate early;
      * if embeddings are tied, resizing the input embedding resizes the head
        for free.  If they are not, both must be resized or you get a shape
        error at the first loss.
    """
    old = model.tok_emb.weight.data
    old_n, dim = old.shape
    if new_vocab_size == old_n:
        return model
    if new_vocab_size < old_n:
        raise ValueError("shrinking the vocabulary would drop trained embeddings")
    std = init_std if init_std is not None else float(old.std())
    new_emb = nn.Embedding(new_vocab_size, dim).to(old.device, old.dtype)
    nn.init.normal_(new_emb.weight, mean=0.0, std=std)
    new_emb.weight.data[:old_n] = old
    model.tok_emb = new_emb
    if model.cfg.tie_embeddings:
        model.lm_head = nn.Linear(dim, new_vocab_size, bias=False).to(old.device, old.dtype)
        model.lm_head.weight = model.tok_emb.weight
    else:
        head = nn.Linear(dim, new_vocab_size, bias=False).to(old.device, old.dtype)
        nn.init.normal_(head.weight, mean=0.0, std=std)
        head.weight.data[:old_n] = model.lm_head.weight.data
        model.lm_head = head
    model.cfg.vocab_size = new_vocab_size
    return model


@dataclass
class SFTConfig:
    epochs: int = 3
    batch_size: int = 32
    lr: float = 3e-4               # 10x below pretraining: the model is already good
    weight_decay: float = 0.0      # decay on a fine-tune mostly just forgets
    warmup_frac: float = 0.05
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    log_interval: int = 20
    seed: int = 0


def sft_loss(model: GPT, x, y, mask) -> torch.Tensor:
    """Cross-entropy over masked positions only."""
    _, loss = model(x, targets=y, loss_mask=mask)
    return loss


def train_sft(model: GPT, examples: list[Encoded], cfg: SFTConfig, pad_id: int,
              val_examples: list[Encoded] | None = None, device=None, verbose: bool = True):
    """Fine-tune `model` on encoded examples.  Returns the loss history."""
    device = torch.device(device) if device else pick_device()
    model = model.to(device)
    seed_everything(cfg.seed)
    opt = model.configure_optimizers(lr=cfg.lr, weight_decay=cfg.weight_decay)

    order = list(range(len(examples)))
    steps_per_epoch = math.ceil(len(order) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total * cfg.warmup_frac))
    rng = random.Random(cfg.seed)
    history: list[dict] = []
    step = 0
    model.train()

    for epoch in range(cfg.epochs):
        rng.shuffle(order)
        for i in range(0, len(order), cfg.batch_size):
            # Sorting each batch by length would reduce padding waste; we keep
            # it simple and just pad to the batch max.
            batch = [examples[j] for j in order[i : i + cfg.batch_size]]
            x, y, m = collate(batch, pad_id, device)
            lr = cosine_lr(step, base_lr=cfg.lr, warmup=warmup, total=total,
                           min_ratio=cfg.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr
            t0 = time.perf_counter()
            loss = sft_loss(model, x, y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = clip_grad_norm([p for gp in opt.param_groups for p in gp["params"]],
                                   cfg.grad_clip)
            opt.step()
            rec = {"step": step, "epoch": epoch, "loss": loss.item(), "lr": lr,
                   "grad_norm": gnorm, "dt": time.perf_counter() - t0,
                   "pad_frac": 1.0 - float(m.sum()) / m.numel()}
            history.append(rec)
            if verbose and step % cfg.log_interval == 0:
                print(f"sft step {step:4d}/{total} ep{epoch} | loss {rec['loss']:.4f} "
                      f"| lr {lr:.2e} | gnorm {gnorm:5.2f} | mask waste {rec['pad_frac']:.2f}")
            step += 1
        if val_examples is not None:
            vl = eval_sft(model, val_examples, pad_id, cfg.batch_size, device)
            if verbose:
                print(f"  epoch {epoch}: val loss {vl:.4f}")
            history[-1]["val_loss"] = vl
    return history


@torch.no_grad()
def eval_sft(model: GPT, examples: list[Encoded], pad_id: int, batch_size: int, device) -> float:
    """Token-weighted mean loss over the completion positions."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(examples), batch_size):
        x, y, m = collate(examples[i : i + batch_size], pad_id, device)
        _, loss = model(x, targets=y, loss_mask=m)
        k = int(m.sum())
        tot += loss.item() * k
        n += k
    model.train()
    return tot / max(1, n)
