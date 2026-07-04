"""
EmpathyTransformer V3 — dengan KV Cache, Fused QKV, depth-thinking loop.
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._cos_cached = None
        self._sin_cached = None

    def _build_cache(self, seq_len: int, device: torch.device):
        if self._cos_cached is not None and self._cos_cached.shape[0] >= seq_len:
            return
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos().to(device)
        self._sin_cached = emb.sin().to(device)

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[1]
        self._build_cache(seq_len, x.device)
        cos = self._cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        hidden = int(d_ff * 2 / 3)
        self.gate = nn.Linear(d_model, hidden, bias=bias)
        self.up = nn.Linear(d_model, hidden, bias=bias)
        self.down = nn.Linear(hidden, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# GQA Attention — Fused QKV + KV Cache + Flash Attention
# ---------------------------------------------------------------------------

class GQAAttention(nn.Module):
    """Grouped Query Attention dengan fused QKV proj, KV cache, Flash Attn."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        # Fused QKV: satu matriks besar, di-slice jadi Q, K, V
        q_dim = config.d_model
        kv_dim = self.n_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(config.d_model, q_dim + kv_dim + kv_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = config.attn_dropout
        self.use_flash = config.use_flash_attn and torch.cuda.is_available()
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len)

        # Kv cache offset tracker (buat generate)
        self._cache_offset = 0

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        # Fused QKV: satu matriks → slice
        qkv = self.qkv_proj(x)
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        # RoPE
        cos, sin = self.rotary(x)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV Cache
        if past_key_value is not None:
            k_cached, v_cached = past_key_value
            k = torch.cat([k_cached, k], dim=1)
            v = torch.cat([v_cached, v], dim=1)

        present_kv = (k, v) if use_cache else None
        T_full = k.shape[1]  # length after cache concat

        # Expand KV → Query heads (GQA repeat)
        if self.n_rep > 1:
            # (B, T, n_kv, d) → (B, T, n_heads, d)
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1).reshape(B, T_full, self.n_heads, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1).reshape(B, T_full, self.n_heads, self.head_dim)

        # Flash Attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        is_causal = (attention_mask is None and T == T_full)
        if self.use_flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None if is_causal else attention_mask.unsqueeze(1).unsqueeze(1) if attention_mask is not None else None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale
            if is_causal:
                causal = torch.triu(torch.full((T, T_full), float('-inf'), device=x.device), diagonal=T_full - T + 1)
                attn = attn + causal.unsqueeze(0).unsqueeze(0)
            if attention_mask is not None:
                attn = attn + attention_mask.unsqueeze(1).unsqueeze(1)
            attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
            if self.attn_dropout > 0 and self.training:
                attn = F.dropout(attn, p=self.attn_dropout)
            y = (attn @ v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y), present_kv


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model) if config.use_rmsnorm else nn.LayerNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model) if config.use_rmsnorm else nn.LayerNorm(config.d_model)
        self.attn = GQAAttention(config)
        self.ffn = SwiGLU(config.d_model, config.d_ff, config.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        x, present_kv = self.attn(self.attn_norm(x), attention_mask, past_key_value, use_cache)
        x = residual + x

        x = x + self.ffn(self.ffn_norm(x))
        return x, present_kv


# ---------------------------------------------------------------------------
# Depth-Thinking Loop
# ---------------------------------------------------------------------------

class ThinkingBlock(nn.Module):
    """
    Recurrent depth-thinking loop di latent space.
    Loop hidden state melalui attention+ffn N kali dengan gated residual.

    Gate init: sigmoid(2.0) ≈ 0.88 → pertahankan 88% identitas per loop.
    """
    def __init__(self, block: TransformerBlock, d_model: int, thinking_steps: int = 4):
        super().__init__()
        self.block = block
        self.thinking_steps = thinking_steps
        # Gate: sigmoid(weight) → 0.88 saat init
        self.gate = nn.Parameter(torch.full((1, 1, d_model), 2.0))

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        identity = x
        for _ in range(self.thinking_steps):
            x_out, _ = self.block(x, attention_mask)  # no cache inside thinking loop
            gate = torch.sigmoid(self.gate)
            x = gate * x_out + (1 - gate) * x  # gated residual
        return x


# ---------------------------------------------------------------------------
# EmpathyTransformer V3
# ---------------------------------------------------------------------------

class EmpathyTransformer(nn.Module):
    """
    Transformer causal + Fused QKV + KV Cache + depth-thinking loop.

    Usage:
        cfg = ModelConfig(thinking_steps=4)
        model = EmpathyTransformer(cfg)
        logits, past_kv = model(input_ids, use_cache=True)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.n_layers)
        ])

        # Depth-thinking loop (optional)
        self.thinking_steps = getattr(config, 'thinking_steps', 0)
        if self.thinking_steps > 0:
            # Loop di layer tengah: layer [n_layers//4, n_layers//2)
            self.think_start = config.n_layers // 4
            self.think_end = min(config.n_layers // 2, config.n_layers)
            # Ganti block di range itu jadi ThinkingBlock
            for i in range(self.think_start, self.think_end):
                tw = ThinkingBlock(self.blocks[i], config.d_model, self.thinking_steps)
                self.blocks[i] = tw

        # Final norm
        self.final_norm = RMSNorm(config.d_model) if config.use_rmsnorm else nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        self._init_weights()
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'gate' in name:
                nn.init.constant_(p, 2.0)  # sigmoid(2) = 0.88
            elif p.ndim >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ):
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Seq {T} > max {self.config.max_seq_len}"

        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)

        x = self.token_emb(input_ids)
        x = self.dropout(x)

        new_past_key_values = [] if use_cache else None
        require_grad = self.training and x.requires_grad

        for i, block in enumerate(self.blocks):
            pkv = past_key_values[i] if past_key_values else None

            if isinstance(block, ThinkingBlock):
                # Thinking loop — no cache inside
                x = block(x, attention_mask)
                kv = None
            elif require_grad and getattr(self.config, 'gradient_checkpointing', False):
                x, kv = torch.utils.checkpoint.checkpoint(
                    block, x, attention_mask, pkv, use_cache, use_reentrant=False
                )
            else:
                x, kv = block(x, attention_mask, pkv, use_cache)

            if use_cache:
                new_past_key_values.append(kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_past_key_values if use_cache else None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate dengan KV cache — O(1) per token baru."""
        past_kv = None
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.config.max_seq_len:
                # Pindah sliding window
                input_ids = input_ids[:, -self.config.max_seq_len:]
                past_kv = None

            logits, past_kv = self(
                input_ids[:, -1:] if past_kv is not None else input_ids,
                use_cache=True,
                past_key_values=past_kv,
            )
            logits = logits[:, -1, :]

            # Sampling
            if temperature > 0:
                logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float('-inf')
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                mask = cum_probs > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False
                sorted_logits[mask] = float('-inf')
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).any():
                break

        return input_ids


# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'size_mb': total * 4 / 1024 / 1024}


if __name__ == '__main__':
    cfg = ModelConfig(vocab_size=4096, d_model=256, n_layers=6, n_heads=8,
                      n_kv_heads=4, d_ff=1024, max_seq_len=128,
                      thinking_steps=4)
    # Enable gradient checkpointing
    cfg.gradient_checkpointing = True
    m = EmpathyTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, kv = m(x, use_cache=True)
    print(f"Forward: {list(logits.shape)}, cache: {len(kv)} blocks")

    # Test generate
    gen = m.generate(x[:, :5], max_new_tokens=10)
    print(f"Generate: {list(gen.shape)}")

    info = count_params(m)
    print(f"Params: {info['total']:,} ({info['size_mb']:.1f}MB)")
    print(f"Thinking steps: {cfg.thinking_steps}")
