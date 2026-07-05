"""
Lightweight inference for EmpathyTransformer.

Usage:
    python inference.py --model ./checkpoints/best.pt --tokenizer ./tokenizer.json \
                        --prompt "I feel lonely today"

Interactive mode:
    python inference.py --model ./checkpoints/best.pt --tokenizer ./tokenizer.json
"""

import os
import sys
import time
import json
import argparse
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig
from model import EmpathyTransformer
from tokenizer import load_tokenizer, encode, decode


def load_model(model_path: str, device: str = 'cpu') -> EmpathyTransformer:
    """Load trained model from checkpoint."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Try to load config from checkpoint dir
    ckpt_dir = os.path.dirname(model_path)
    config_path = os.path.join(ckpt_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg_dict = json.load(f)
        cfg = ModelConfig(**{k: v for k, v in cfg_dict.items()
                            if k in ModelConfig.__dataclass_fields__})
    else:
        # Default — must match training config
        cfg = ModelConfig()

    model = EmpathyTransformer(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    info = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {info:,} params ({info * 4 / 1024 / 1024:.2f}MB)")

    return model


def generate_text(model: EmpathyTransformer, tokenizer, prompt: str,
                  max_new_tokens: int = 100,
                  temperature: float = 0.8,
                  top_k: int = 40,
                  top_p: float = 0.9,
                  device: str = 'cpu') -> str:
    """Generate text from prompt."""
    # Encode prompt
    enc = encode(tokenizer, prompt, max_length=model.config.max_seq_len)
    input_ids = torch.tensor([enc['input_ids']], dtype=torch.long, device=device)

    # Remove padding for generation
    prompt_len = len(tokenizer.encode(prompt).ids)
    input_ids = input_ids[:, :prompt_len]

    # Generate
    start = time.time()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=2,  # </s>
    )
    elapsed = time.time() - start

    # Decode
    generated = output_ids[0].tolist()
    response = decode(tokenizer, generated, skip_special=True)

    # Stats
    new_tokens = len(output_ids[0]) - prompt_len
    tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0

    print(f"  Generated {new_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")

    return response


def interactive_mode(model, tokenizer, device):
    """Interactive chat loop."""
    print("\nEntering interactive mode. Type 'quit' to exit.\n")
    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ('quit', 'exit', 'q'):
            break

        response = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=128,
            temperature=0.7,
            top_k=30,
            top_p=0.85,
            device=device,
        )
        print(f"AI: {response}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to checkpoint .pt')
    parser.add_argument('--tokenizer', required=True, help='Path to tokenizer.json')
    parser.add_argument('--prompt', help='Single prompt (omit for interactive mode)')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--top-p', type=float, default=0.85)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    model = load_model(args.model, device)
    tokenizer = load_tokenizer(args.tokenizer)

    if args.prompt:
        response = generate_text(
            model, tokenizer, args.prompt,
            args.max_new_tokens, args.temperature,
            args.top_k, args.top_p, device
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Response: {response}")
    else:
        interactive_mode(model, tokenizer, device)
