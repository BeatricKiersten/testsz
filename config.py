"""
Model configuration for EmpathyTransformer V4.
Lightweight, fast, emotion-aware causal language model.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int = 16384
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 4         # GQA: 8 query, 4 key/value
    d_ff: int = 2048            # SwiGLU → actual hidden = 2/3 * d_ff
    max_seq_len: int = 512
    dropout: float = 0.1
    attn_dropout: float = 0.0

    # V4 upgrades — Peri-LN, QKV-norm, sliding window
    use_rotary: bool = True
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    use_flash_attn: bool = True
    bias: bool = False
    tie_weights: bool = True

    # V4: Peri-LN — norm before AND after each sublayer (Gemma 2 style)
    peri_ln: bool = True

    # V4: QKV-norm — normalize Q, K, V projections before attention
    qkv_norm: bool = True

    # V4: Sliding window alternating — global every N layers
    sliding_window_size: int = 256    # half of max_seq_len
    sliding_window_every: int = 4     # global every 4 layers, window for rest

    # V4: Scaled embedding — emb * sqrt(d_model)
    scale_embedding: bool = True

    # V4: Pre-LN head — RMSNorm before lm_head
    pre_ln_head: bool = True

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 0.03       # Power Lines paper: 0.01-0.05
    warmup_steps: int = 200          # WSD: 5% of total
    beta1: float = 0.9
    beta2: float = 0.95             # Chinchilla + MiniCPM
    batch_size: int = 16             # Increased from 8 — T4 16GB bisa
    grad_clip: float = 1.0          # Wajib

    # Advanced
    thinking_steps: int = 0
    gradient_checkpointing: bool = True

    # Saving
    save_every: int = 500
    log_every: int = 5

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
