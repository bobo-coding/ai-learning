"""A decoder-only transformer, assembled from parts you can read in one sitting.

Layout of one block (pre-norm, the arrangement every modern LLM uses):

    x = x + Attention(Norm(x))
    x = x + MLP(Norm(x))

Pre-norm matters more than it looks.  In the original post-norm transformer the
residual stream passes *through* a LayerNorm every block, so the identity path
is not actually an identity and deep models need a warmup-heavy schedule to
train at all.  With pre-norm the residual stream is a clean sum of block
outputs -- gradients reach layer 0 undamped, and you can stack 100 layers.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention, KVCache, rope_tables
from .config import GPTConfig


class RMSNorm(nn.Module):
    """Root-mean-square layer norm: LayerNorm without the mean subtraction.

        y = x / sqrt(mean(x^2) + eps) * g

    Dropping the mean (and the bias) costs nothing measurable in quality and
    removes a reduction plus a broadcast from the kernel, which is why every
    model since Llama uses it.  Note the cast to fp32: the sum of squares of a
    4096-wide bf16 vector overflows/underflows easily, and this is one of the
    few places where mixed precision genuinely needs a manual upcast.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        acc = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        xa = x.to(acc)
        xa = xa * torch.rsqrt(xa.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return xa.to(dtype) * self.weight


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (nn.LayerNorm cannot switch its bias off)."""

    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


def make_norm(cfg: GPTConfig, dim: int) -> nn.Module:
    if cfg.norm == "rms":
        return RMSNorm(dim, eps=cfg.norm_eps)
    return LayerNorm(dim, bias=cfg.bias, eps=cfg.norm_eps)


class MLP(nn.Module):
    """Position-wise feed-forward network: `W_down(act(W_up(x)))`.

    This is where most of the parameters live (2/3 of a block at ratio 4), and
    the usual reading is that attention moves information between positions
    while the MLP does the per-position computation on it.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.up = nn.Linear(cfg.n_embd, cfg.hidden_dim, bias=cfg.bias)
        self.down = nn.Linear(cfg.hidden_dim, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # tanh-approximate GELU: what GPT-2 actually shipped.  The exact erf
        # form differs by <1e-3 and either is fine, but they are not bit-equal,
        # so you must match whichever the checkpoint was trained with.
        return self.dropout(self.down(F.gelu(self.up(x), approximate="tanh")))


class SwiGLU(nn.Module):
    """Gated feed-forward: `W_down( silu(W_gate(x)) * W_up(x) )`.

    Three matrices instead of two.  The elementwise product makes the layer
    quadratic in x, so it can express multiplicative interactions a single
    nonlinearity cannot, and it measurably beats GELU at equal parameter count
    (which is why `hidden_dim` shrinks by 2/3 -- see GPTConfig.hidden_dim).
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.n_embd, cfg.hidden_dim, bias=cfg.bias)
        self.up = nn.Linear(cfg.n_embd, cfg.hidden_dim, bias=cfg.bias)
        self.down = nn.Linear(cfg.hidden_dim, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    """One pre-norm transformer block."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = make_norm(cfg, cfg.n_embd)
        self.attn = CausalSelfAttention(
            cfg.n_embd, cfg.n_head, cfg.n_kv_head, dropout=cfg.dropout,
            bias=cfg.bias, rope=(cfg.pos == "rope"), impl=cfg.attn_impl,
        )
        self.norm2 = make_norm(cfg, cfg.n_embd)
        self.mlp = SwiGLU(cfg) if cfg.mlp == "swiglu" else MLP(cfg)

    def forward(self, x, cos=None, sin=None, cache: KVCache | None = None):
        x = x + self.attn(self.norm1(x), cos, sin, cache=cache)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = make_norm(cfg, cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # Weight tying: the output head *is* the input embedding matrix.
            # Saves vocab*n_embd parameters (40% of GPT-2 small!) and helps small
            # models, because every token's embedding now gets gradient from both
            # being read and being predicted.
            self.lm_head.weight = self.tok_emb.weight

        if cfg.pos == "rope":
            cos, sin = rope_tables(cfg.head_dim, cfg.block_size, base=cfg.rope_base)
            # Buffers, not parameters: they move with `.to(device)` but are not
            # trained and (persistent=False) are not written to checkpoints.
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = self.rope_sin = None

        self.apply(self._init_weights)
        # GPT-2's trick: scale down the *residual output* projection of every
        # block by 1/sqrt(2*n_layer).  Each of the 2L residual additions has
        # roughly unit variance, so without this the variance of the residual
        # stream grows linearly with depth and deep models start with saturated
        # logits.
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down.weight")):
                torch.nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    # ------------------------------------------------------------------ forward

    def forward(self, idx, targets=None, loss_mask=None, caches: list[KVCache] | None = None):
        """Run the model.

        `idx`     : (B, T) int64 token ids
        `targets` : (B, T) int64 next-token labels, or -100 to ignore a position
        `loss_mask`: (B, T) bool/float, 1 where the position should count.  This
                    is how SFT trains on completions only.
        `caches`  : one KVCache per layer, for incremental decoding.

        Returns (logits, loss).  `loss` is None when `targets` is None.
        """
        B, T = idx.shape
        offset = caches[0].length if caches is not None else 0
        if offset + T > self.cfg.block_size:
            raise ValueError(f"sequence length {offset + T} exceeds block_size {self.cfg.block_size}")

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(offset, offset + T, device=idx.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        for i, block in enumerate(self.blocks):
            x = block(x, self.rope_cos, self.rope_sin, cache=caches[i] if caches else None)
        x = self.norm_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten to (B*T, V) vs (B*T,).  ignore_index=-100 matches the
            # HuggingFace convention so masked datasets are interchangeable.
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1).clone()
            if loss_mask is not None:
                flat_targets[~loss_mask.reshape(-1).bool()] = -100
            loss = F.cross_entropy(flat_logits.float(), flat_targets, ignore_index=-100)
        return logits, loss

    # ------------------------------------------------------------------ helpers

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def make_caches(self, batch: int, max_seq_len: int | None = None, dtype=None) -> list[KVCache]:
        cfg = self.cfg
        max_seq_len = max_seq_len or cfg.block_size
        device = next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        return [
            KVCache(batch, cfg.n_kv_head, max_seq_len, cfg.head_dim, device, dtype)
            for _ in range(cfg.n_layer)
        ]

    def configure_optimizers(self, lr: float, weight_decay: float = 0.1, betas=(0.9, 0.95),
                             eps: float = 1e-8, fused: bool | None = None):
        """AdamW with the standard two parameter groups.

        Rule of thumb that has survived a decade: decay matrices, do not decay
        anything one-dimensional.  Norm gains and biases have no redundancy to
        regularise away, and decaying them just biases the network towards
        smaller activations.  (Embeddings are 2-D and *are* decayed here, which
        is what GPT-2/nanoGPT do.)
        """
        decay, no_decay = [], []
        seen: set[int] = set()
        for _, p in self.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        kwargs = dict(lr=lr, betas=betas, eps=eps)
        if fused:
            kwargs["fused"] = True
        return torch.optim.AdamW(groups, **kwargs)

    def estimate_mfu(self, tokens_per_step: int, dt: float, peak_flops: float) -> float:
        """Model FLOPs utilisation: what fraction of the chip you are actually using.

        The single most useful training metric after loss.  If MFU is 5% you have
        a data or launch-overhead problem, not a model problem.
        """
        fwd = self.cfg.flops_per_token(self.cfg.block_size)["total"]
        # backward is ~2x forward, so training is ~3x forward FLOPs
        flops_per_step = 3.0 * fwd * tokens_per_step
        return flops_per_step / dt / peak_flops

    @torch.no_grad()
    def crop_block_size(self, block_size: int):
        """Shrink the context window of a trained model (e.g. to fine-tune cheaply)."""
        assert block_size <= self.cfg.block_size
        self.cfg.block_size = block_size
        if self.pos_emb is not None:
            self.pos_emb.weight = nn.Parameter(self.pos_emb.weight[:block_size])
        if self.rope_cos is not None:
            self.rope_cos = self.rope_cos[:block_size]
            self.rope_sin = self.rope_sin[:block_size]

    def save(self, path: str):
        torch.save({"cfg": self.cfg.to_dict(), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu", **cfg_overrides) -> GPT:
        blob = torch.load(path, map_location=map_location, weights_only=False)
        cfg = GPTConfig(**{**blob["cfg"], **cfg_overrides})
        model = cls(cfg)
        model.load_state_dict(blob["state_dict"])
        return model
