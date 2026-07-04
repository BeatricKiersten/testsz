"""
Model configuration for EmpathyTransformer.
Lightweight, fast, emotion-aware causal language model.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int = 16384
    d_model: int = 512
    n_layers: int = 12          # 8 → 12 (deeper = better understanding)
    n_heads: int = 8
    n_kv_heads: int = 4         # GQA: 8 query, 4 key/value (hemat memori 25%)
    d_ff: int = 2048            # FFN inner dim (SwiGLU → actual hidden = 2/3 * d_ff)
    max_seq_len: int = 512
    dropout: float = 0.1
    attn_dropout: float = 0.0

    # Efficiency
    use_rotary: bool = True
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    use_flash_attn: bool = True  # PyTorch SDPA (Flash Attention on CUDA)
    bias: bool = False
    tie_weights: bool = True

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    beta1: float = 0.9
    beta2: float = 0.95
    batch_size: int = 16
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 4

    # Advanced
    thinking_steps: int = 0       # 0 = off, 4 = depth-thinking loop 4x
    gradient_checkpointing: bool = True  # hemat VRAM, lambat 20%

    # Saving
    save_every: int = 500
    log_every: int = 10

    @property
    def total_params_est(self) -> int:
        """Rough parameter count estimate."""
        head_dim = self.d_model // self.n_heads
        kv_dim = self.n_kv_heads * head_dim

        # embedding: vocab * d_model
        emb = self.vocab_size * self.d_model
        # per layer: Q, K, V, O + FFN gate+up+down
        q = self.d_model * self.d_model
        k = self.d_model * kv_dim
        v = self.d_model * kv_dim
        o = self.d_model * self.d_model
        ffn_hidden = self.d_ff * 2 // 3
        ffn = 3 * self.d_model * ffn_hidden  # gate, up, down
        per_layer = q + k + v + o + ffn
        total = emb + per_layer * self.n_layers
        if self.tie_weights:
            total -= emb
        return total
