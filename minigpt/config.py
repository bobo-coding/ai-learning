"""Model configuration, plus the parameter/FLOP accounting that goes with it."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class GPTConfig:
    """Every architectural choice in one place.

    Defaults follow a modern (Llama-style) decoder: RMSNorm, RoPE, SwiGLU, no
    biases, tied embeddings.  Flip `norm`, `pos` and `mlp` to get GPT-2 exactly,
    which is useful for isolating which of those choices actually matter.
    """

    vocab_size: int = 512
    block_size: int = 256            # maximum context length
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int | None = None     # None -> = n_head (plain MHA); 1 -> MQA
    n_embd: int = 256
    mlp_ratio: float = 4.0           # hidden width of the MLP, as a multiple of n_embd
    mlp_multiple_of: int = 64        # round SwiGLU hidden width up to this (Llama uses 256)
    dropout: float = 0.0
    bias: bool = False               # biases in Linear/Norm layers (GPT-2 has them)
    norm: str = "rms"                # "rms" | "layer"
    pos: str = "rope"                # "rope" | "learned" | "none"
    mlp: str = "swiglu"              # "swiglu" | "gelu"
    rope_base: float = 10000.0
    tie_embeddings: bool = True      # share input embedding with the output head
    attn_impl: str = "sdpa"          # "sdpa" | "math" | "flash"
    init_std: float = 0.02
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd={self.n_embd} not divisible by n_head={self.n_head}")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head={self.n_head} not divisible by n_kv_head={self.n_kv_head}")
        if self.norm not in ("rms", "layer"):
            raise ValueError(f"unknown norm {self.norm!r}")
        if self.pos not in ("rope", "learned", "none"):
            raise ValueError(f"unknown pos {self.pos!r}")
        if self.mlp not in ("swiglu", "gelu"):
            raise ValueError(f"unknown mlp {self.mlp!r}")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def hidden_dim(self) -> int:
        """MLP inner width.

        SwiGLU uses *three* matrices instead of two, so to keep the parameter
        count equal to a 4x GELU MLP the inner width is scaled by 2/3 (and then
        rounded to a multiple of 64 for kernel efficiency) -- this is exactly
        what Llama does, and why you see 11008 rather than 16384 for a 4096-wide
        model.
        """
        if self.mlp == "swiglu":
            h = int(2 * self.mlp_ratio * self.n_embd / 3)
            m = self.mlp_multiple_of
            return m * ((h + m - 1) // m)
        return int(self.mlp_ratio * self.n_embd)

    def to_dict(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------- accounting

    def param_count(self) -> dict[str, int]:
        """Analytic parameter count, broken down by component.

        Worth being able to do on paper: it tells you where your memory goes and
        it is the first sanity check that your model is wired the way you think.
        """
        C, L, H = self.n_embd, self.n_layer, self.n_head
        hd, kv = self.head_dim, self.n_kv_head
        b = 1 if self.bias else 0

        embed = self.vocab_size * C
        pos = self.block_size * C if self.pos == "learned" else 0
        # q: C x (H*hd), k/v: C x (kv*hd) each, o: (H*hd) x C
        attn = (C * H * hd + 2 * C * kv * hd + H * hd * C) + b * (H * hd + 2 * kv * hd + C)
        n_mlp_mat = 3 if self.mlp == "swiglu" else 2
        mlp = n_mlp_mat * C * self.hidden_dim + b * (
            (2 * self.hidden_dim + C) if self.mlp == "swiglu" else (self.hidden_dim + C)
        )
        # 2 norms per block + 1 final norm; RMSNorm has a weight only.
        per_norm = C * (2 if (self.norm == "layer" and self.bias) else 1)
        norms = (2 * L + 1) * per_norm
        head = 0 if self.tie_embeddings else self.vocab_size * C

        return {
            "embedding": embed + pos,
            "attention": L * attn,
            "mlp": L * mlp,
            "norms": norms,
            "lm_head": head,
            "total": embed + pos + L * (attn + mlp) + norms + head,
            "non_embedding": L * (attn + mlp) + norms,
        }

    def flops_per_token(self, seq_len: int | None = None) -> dict[str, float]:
        """Forward FLOPs per token, counting a multiply-add as 2 FLOPs.

        The famous rule of thumb is ``C ~= 6 * N * D`` for training (2N forward,
        4N backward, per token).  It ignores attention's quadratic term, which
        is fine while ``seq_len << 12 * n_embd`` and badly wrong past that --
        so we report it separately.
        """
        T = seq_len if seq_len is not None else self.block_size
        C, L, H = self.n_embd, self.n_layer, self.n_head
        hd, kv = self.head_dim, self.n_kv_head

        proj = 2 * (C * H * hd + 2 * C * kv * hd + H * hd * C)
        n_mlp_mat = 3 if self.mlp == "swiglu" else 2
        mlp = 2 * n_mlp_mat * C * self.hidden_dim
        # QK^T and PV: each is 2 * T * (H*hd) FLOPs per token over the full
        # sequence; averaged over a causal sequence only half the keys are
        # visible, hence the T (not 2T) here.
        attn_quadratic = 2 * 2 * T * H * hd / 2
        head = 2 * C * self.vocab_size

        dense = L * (proj + mlp) + head
        return {
            "dense": float(dense),
            "attention": float(L * attn_quadratic),
            "total": float(dense + L * attn_quadratic),
            "fraction_attention": L * attn_quadratic / (dense + L * attn_quadratic),
        }

    def kv_cache_bytes(self, batch: int = 1, seq_len: int | None = None, bytes_per: int = 2) -> int:
        """Bytes of KV cache for `batch` sequences of `seq_len` tokens."""
        T = seq_len if seq_len is not None else self.block_size
        return 2 * self.n_layer * self.n_kv_head * self.head_dim * T * batch * bytes_per


# A few ready-made sizes.  "tiny" trains to something readable on a Mac in a
# couple of minutes; "gpt2" is the real 124M architecture for comparison.
PRESETS: dict[str, dict] = {
    "nano": dict(n_layer=3, n_head=3, n_embd=48, block_size=64, mlp="gelu", norm="layer", pos="learned"),
    "tiny": dict(n_layer=4, n_head=4, n_embd=128, block_size=128),
    "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=256),
    "medium": dict(n_layer=8, n_head=8, n_embd=512, n_kv_head=2, block_size=512),
    "gpt2": dict(
        vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768,
        bias=True, norm="layer", pos="learned", mlp="gelu", tie_embeddings=True,
    ),
    "llama7b": dict(
        vocab_size=32000, block_size=4096, n_layer=32, n_head=32, n_embd=4096,
        mlp_ratio=4.0, mlp_multiple_of=256, bias=False, norm="rms", pos="rope",
        mlp="swiglu", tie_embeddings=False,
    ),
}


def preset(name: str, **overrides) -> GPTConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; have {sorted(PRESETS)}")
    return GPTConfig(**{**PRESETS[name], **overrides})
