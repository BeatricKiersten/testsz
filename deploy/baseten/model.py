"""
Baseten (Truss) deployment handler for EmpathyTransformer.

Baseten uses Truss — a standard format for deploying ML models.
This file defines how the model loads and serves predictions.

Deploy:
    pip install truss
    truss push

Or via Baseten UI: upload this directory.
"""

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import ModelConfig
from model import EmpathyTransformer
from tokenizer import load_tokenizer, encode, decode


class BasetenModel:
    """
    Truss-compatible model class for Baseten deployment.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        self._model = None
        self._tokenizer = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load(self):
        """Load model weights — called once at cold start."""
        base = Path(os.path.dirname(__file__))
        model_path = base / 'model.pt'
        tokenizer_path = base / 'tokenizer.json'
        config_path = base / 'config.json'

        # Config
        if config_path.exists():
            with open(config_path) as f:
                cfg_dict = json.load(f)
            cfg = ModelConfig(**{k: v for k, v in cfg_dict.items()
                                if k in ModelConfig.__dataclass_fields__})
        else:
            cfg = ModelConfig()

        # Model
        self._model = EmpathyTransformer(cfg).to(self._device)
        if model_path.exists():
            ckpt = torch.load(str(model_path), map_location=self._device, weights_only=True)
            if 'model_state_dict' in ckpt:
                self._model.load_state_dict(ckpt['model_state_dict'])
            else:
                self._model.load_state_dict(ckpt)
        self._model.eval()

        # Tokenizer
        self._tokenizer = load_tokenizer(str(tokenizer_path))

        n_params = sum(p.numel() for p in self._model.parameters())
        print(f"Model loaded: {n_params:,} params on {self._device}")

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle prediction request.

        Expected request:
            {"prompt": "I feel lonely today", ...optional gen params}
        Returns:
            {"response": "...", "tokens_generated": N, "time_ms": ...}
        """
        if self._model is None:
            self.load()

        prompt = request.get('prompt', '')
        max_new_tokens = request.get('max_new_tokens', 100)
        temperature = request.get('temperature', 0.7)
        top_k = request.get('top_k', 30)
        top_p = request.get('top_p', 0.85)

        if not prompt:
            return {"error": "Missing 'prompt' field"}

        import time
        start = time.time()

        # Encode
        enc = encode(self._tokenizer, prompt, max_length=self._model.config.max_seq_len)
        input_ids = torch.tensor([enc['input_ids']], dtype=torch.long, device=self._device)

        # Trim to actual prompt length
        prompt_len = len(self._tokenizer.encode(prompt).ids)
        input_ids = input_ids[:, :prompt_len]

        # Generate
        output_ids = self._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=2,
        )

        # Decode
        response = decode(self._tokenizer, output_ids[0].tolist(), skip_special=True)

        elapsed_ms = (time.time() - start) * 1000
        tokens_generated = len(output_ids[0]) - prompt_len

        return {
            'response': response,
            'tokens_generated': tokens_generated,
            'time_ms': round(elapsed_ms, 2),
            'tokens_per_sec': round(tokens_generated / (elapsed_ms / 1000), 2) if elapsed_ms > 0 else 0,
        }
