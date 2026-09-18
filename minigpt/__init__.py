"""minigpt -- a small, correct transformer stack built for learning.

The modules are meant to be read in curriculum order:

    bpe        tokenization (byte-level BPE from scratch)
    attention  attention, online softmax, FlashAttention fwd+bwd, RoPE, KV cache
    config     architecture config + parameter/FLOP/KV-cache accounting
    model      RMSNorm, SwiGLU, pre-norm blocks, the GPT module
    data       token streams on disk, batching, packing, splits
    optim      AdamW and gradient clipping from scratch, LR schedules
    train      the pretraining loop
    generate   sampling, KV-cache decoding, speculative decoding
    sft        supervised fine-tuning (chat templates, completion-only loss)
    dpo        direct preference optimization
    grpo       RLVR with group-relative policy optimization
    lora       LoRA / QLoRA adapters
    quant      int8 / int4 / NF4 quantization and quantized linear layers
    eval       perplexity, multiple-choice, generative and pass@k evaluation
"""

from .config import GPTConfig, preset
from .model import GPT

__all__ = ["GPT", "GPTConfig", "preset"]
__version__ = "0.1.0"
