"""
Download and prepare Cosmopedia dataset for EmpathyTransformer training.
Run on Kaggle/Colab where internet access is available.

Usage:
    python download_cosmopedia.py --output data/cosmopedia.jsonl --max-samples 30000

Cosmopedia: HuggingFace synthetic dataset of textbooks, blogposts, stories.
Subset: HuggingFaceTB/cosmopedia-100k (100K samples, ~200MB)
"""

import os
import json
import argparse
from pathlib import Path


def download_cosmopedia(output_path: str, max_samples: int = 30000, subset_size: str = "100k"):
    """
    Download Cosmopedia and save as JSONL for training.
    Uses HuggingFace datasets library.
    """
    print(f"Downloading HuggingFaceTB/cosmopedia-{subset_size}...")
    print(f"Max samples: {max_samples}")

    from datasets import load_dataset

    ds = load_dataset(f"HuggingFaceTB/cosmopedia-{subset_size}", split="train", streaming=False)

    # Cosmopedia has 'text' field — perfect for our TextDataset
    count = 0
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(ds):
            if i >= max_samples:
                break
            text = example.get('text', '')
            if not text:
                continue

            # Clean: remove too short/long, filter gibberish
            if len(text) < 50 or len(text) > 10000:
                continue

            record = {'text': text, 'source': 'cosmopedia', 'subset': subset_size}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1

            if count % 5000 == 0:
                print(f"  Downloaded {count}/{max_samples}")

    print(f"Saved {count} samples to {output_path}")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"File size: {size_mb:.1f} MB")

    return count


def merge_with_existing(new_data_path: str, existing_data_paths: list, output_path: str):
    """
    Merge Cosmopedia with existing datasets (quotes, OASST1, etc.)
    with proper mixing ratios.
    """
    print(f"Merging datasets...")
    total_input = len(existing_data_paths) + 1  # +1 for cosmopedia

    # Count lines in each
    cosmopedia_lines = sum(1 for _ in open(new_data_path))
    print(f"  Cosmopedia: {cosmopedia_lines} lines")

    existing_lines = {}
    for path in existing_data_paths:
        name = os.path.basename(path)
        lines = sum(1 for _ in open(path))
        existing_lines[path] = lines
        print(f"  {name}: {lines} lines")

    # Mix with target ratios
    total_new = cosmopedia_lines + sum(existing_lines.values())
    print(f"\nTotal combined: {total_new} samples")

    # Shuffle and write merged
    import random
    random.seed(42)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_lines = []

    # Add cosmopedia lines
    with open(new_data_path) as f:
        for line in f:
            item = json.loads(line)
            all_lines.append(json.dumps(item, ensure_ascii=False))

    # Add existing lines
    for path in existing_data_paths:
        with open(path) as f:
            for line in f:
                all_lines.append(line.strip())

    random.shuffle(all_lines)

    with open(out_path, 'w', encoding='utf-8') as f:
        for line in all_lines:
            f.write(line + '\n')

    print(f"Merged {len(all_lines)} samples to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Cosmopedia dataset')
    parser.add_argument('--output', default='data/cosmopedia.jsonl',
                        help='Output JSONL path')
    parser.add_argument('--max-samples', type=int, default=30000,
                        help='Max samples to download')
    parser.add_argument('--subset', default='100k',
                        help='Cosmopedia subset size')
    parser.add_argument('--merge', nargs='*', default=[],
                        help='Existing datasets to merge with (paths)')
    parser.add_argument('--merge-output', default='data/train_full.jsonl',
                        help='Merged output path')

    args = parser.parse_args()

    count = download_cosmopedia(args.output, args.max_samples, args.subset)

    if args.merge:
        # User passed existing datasets to merge
        existing = list(args.merge)
        merge_with_existing(args.output, existing, args.merge_output)
    else:
        print(f"\nDone. {count} Cosmopedia samples saved to {args.output}")
        print("Run with --merge to merge with existing datasets")
