"""
BPE tokenizer for EmpathyTransformer.

Train on your dataset with:
    python tokenizer.py --data your_data.txt --vocab-size 16384
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.processors import TemplateProcessing


def create_tokenizer(vocab_size: int = 16384) -> Tokenizer:
    """Create a BPE tokenizer with pre-norm and post-processor."""
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Replace(r'\s+', ' '),
    ])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    tok.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:3 </s>",
        special_tokens=[
            ("<s>", 1),
            ("<pad>", 0),
            ("</s>", 2),
            ("<unk>", 3),
        ],
    )
    return tok


def train_tokenizer(data_path: str, vocab_size: int = 16384, save_path: Optional[str] = None):
    """
    Train BPE tokenizer on text file(s).

    Args:
        data_path: path to .txt file or directory with .txt files
        vocab_size: vocabulary size (default: 16384 — small for speed)
        save_path: path to save tokenizer.json
    """
    tok = create_tokenizer(vocab_size)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        show_progress=True,
        min_frequency=2,
    )

    # Collect files
    path = Path(data_path)
    if path.is_file():
        files = [str(path)]
    elif path.is_dir():
        files = [str(f) for f in path.glob("**/*.txt")]
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")

    print(f"Training tokenizer on {len(files)} file(s)...")
    tok.train(files, trainer)

    save = save_path or os.path.join(os.path.dirname(data_path), "tokenizer.json")
    tok.save(str(save))
    print(f"Tokenizer saved to {save}")
    print(f"Vocabulary size: {tok.get_vocab_size()}")
    return tok


def load_tokenizer(path: str) -> Tokenizer:
    """Load trained tokenizer."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def encode(tok: Tokenizer, text: str, max_length: int = 512) -> dict:
    """Encode text to input_ids + attention_mask."""
    enc = tok.encode(text)
    ids = enc.ids[:max_length]
    mask = [1] * len(ids)

    # Pad
    pad_len = max_length - len(ids)
    if pad_len > 0:
        ids = ids + [0] * pad_len
        mask = mask + [0] * pad_len

    return {'input_ids': ids, 'attention_mask': mask}


def decode(tok: Tokenizer, ids: List[int], skip_special: bool = True) -> str:
    """Decode token ids back to text."""
    return tok.decode(ids, skip_special_tokens=skip_special)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Text file or dir for training')
    parser.add_argument('--vocab-size', type=int, default=16384, help='Vocabulary size')
    parser.add_argument('--save', type=str, default=None, help='Output path for tokenizer.json')
    args = parser.parse_args()

    train_tokenizer(args.data, args.vocab_size, args.save)
