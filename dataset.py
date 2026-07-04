"""
Flexible dataset loader for EmpathyTransformer.

Supports multiple formats:
  - Plain .txt: one paragraph per line
  - JSONL: {"text": "..."} per line
  - CSV: with 'text' column
  - Emotion-labeled JSONL: {"text": "...", "emotion": "joy"}

The emotional labels are optional for basic LM training.
For empathy fine-tuning, use the format:
  {"input": "I feel sad today", "response": "I hear you...", "emotion": "sadness"}
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Union

import torch
from torch.utils.data import Dataset, IterableDataset

from tokenizer import load_tokenizer, encode


class TextDataset(Dataset):
    """Memory-mapped text dataset for tokenized data.

    Args:
        data_path: path to tokenized .pt file (pre-processed)
        or raw text path (.txt, .jsonl, .csv)
    """

    def __init__(self, data_path: str, tokenizer_path: str,
                 max_length: int = 512, raw: bool = False):
        self.max_length = max_length
        self.tokenizer = load_tokenizer(tokenizer_path)

        if data_path.endswith('.pt'):
            # Pre-tokenized
            self.data = torch.load(data_path)
            self.raw = False
        else:
            # Raw text — load & tokenize
            self.raw = True
            self.texts = self._load_texts(data_path)

    def _load_texts(self, path: str) -> List[str]:
        path = Path(path)
        texts = []
        if path.suffix == '.txt':
            lines = path.read_text(encoding='utf-8').splitlines()
            texts = [l.strip() for l in lines if l.strip()]
        elif path.suffix == '.jsonl':
            with open(path) as f:
                for line in f:
                    item = json.loads(line)
                    text = item.get('text') or item.get('input', '') + ' ' + item.get('response', '')
                    if text:
                        texts.append(text)
        elif path.suffix == '.csv':
            import csv
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'text' in row:
                        texts.append(row['text'])
        return texts

    def __len__(self) -> int:
        if not self.raw:
            return len(self.data)
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.raw:
            item = self.data[idx]
            return {
                'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(item.get('attention_mask', [1]*self.max_length), dtype=torch.long),
            }

        text = self.texts[idx]
        enc = encode(self.tokenizer, text, self.max_length)
        return {
            'input_ids': torch.tensor(enc['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(enc['attention_mask'], dtype=torch.long),
        }


class EmpathyDataset(Dataset):
    """
    Dataset for empathy/emotion-aware training.

    Expected JSONL format:
      {"input": "user text", "response": "empathetic response", "emotion": "sadness"}
      or just {"text": "some conversation or emotional content"}
    """

    def __init__(self, data_path: str, tokenizer_path: str, max_length: int = 512):
        self.max_length = max_length
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.items = self._load(data_path)

    def _load(self, path: str) -> List[dict]:
        items = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                # Construct training text from available fields
                if 'input' in item and 'response' in item:
                    text = f"[{item.get('emotion', 'neutral')}] {item['input']} [/EMOTION] {item['response']}"
                elif 'text' in item:
                    text = item['text']
                else:
                    continue
                items.append({'text': text, **item})
        print(f"Loaded {len(items)} empathy examples from {path}")
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.items[idx]['text']
        enc = encode(self.tokenizer, text, self.max_length)

        # For causal LM, input_ids == labels (shifted inside train loop)
        return {
            'input_ids': torch.tensor(enc['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(enc['attention_mask'], dtype=torch.long),
        }


def preprocess_and_save(data_path: str, tokenizer_path: str,
                        output_path: str, max_length: int = 512):
    """Pre-tokenize entire dataset and save as .pt for faster loading."""
    from tqdm import tqdm

    print(f"Preprocessing {data_path} -> {output_path}")
    dataset = TextDataset(data_path, tokenizer_path, max_length, raw=True)
    tokenized = []
    for i in tqdm(range(len(dataset))):
        tokenized.append(dataset[i])
    torch.save(tokenized, output_path)
    print(f"Saved {len(tokenized)} tokenized examples to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--tokenizer', required=True, help='Path to tokenizer.json')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--save', help='Path to save pre-tokenized .pt (optional)')
    args = parser.parse_args()

    if args.save:
        preprocess_and_save(args.data, args.tokenizer, args.save, args.max_length)
    else:
        ds = TextDataset(args.data, args.tokenizer, args.max_length, raw=True)
        print(f"Dataset size: {len(ds)} examples")
        print(f"Sample: {ds[0]['input_ids'].tolist()[:20]}...")
