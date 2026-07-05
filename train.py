"""
Train EmpathyTransformer from scratch.

Usage:
    # Train on GPU (FP16 AMP — 2-3x faster)
    python train.py --data ./data/train_mixed.jsonl --tokenizer ./tokenizer.json \\
                    --epochs 10 --batch-size 16 --lr 1e-3 --amp --grad-accum 8

    # Resume from checkpoint
    python train.py --resume ./checkpoints/step_1000.pt

    # Quick test on CPU
    python train.py --data ./data/sample.jsonl --tokenizer ./tokenizer.json \\
                    --epochs 1 --batch-size 2 --device cpu
"""

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path
from typing import Optional

# Unbuffer output supaya log muncul real-time di Colab
print = __import__('functools').partial(print, flush=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig
from model import EmpathyTransformer, count_params
from dataset import EmpathyDataset, TextDataset


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup + cosine decay in one LambdaLR (resume-safe)."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return 0.01 + (1.0 - 0.01) * step / max(1, warmup_steps)
        # Cosine decay from 1.0 → 0.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.001, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, device, loss_fn):
    """Single training step."""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    logits, _ = model(input_ids, attention_mask)

    # Shift for causal LM: predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                   shift_labels.view(-1))
    # Mask padding tokens
    loss = loss.view(shift_logits.shape[:2]) * shift_mask
    loss = loss.sum() / shift_mask.sum()

    return loss


@torch.no_grad()
def validate(model, val_loader, device, loss_fn, use_amp=False):
    """Compute validation perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, _ = model(input_ids, attention_mask)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                       shift_labels.view(-1))
        loss = loss.view(shift_logits.shape[:2]) * shift_mask
        loss_sum = loss.sum().item()
        n_tokens = shift_mask.sum().item()

        total_loss += loss_sum
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    model.train()
    return avg_loss, perplexity


def save_checkpoint(model, raw_model, optimizer, scheduler, step, epoch, config, path, amp_scaler=None):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_sd = raw_model.state_dict() if raw_model is not None else model.state_dict()
    ckpt = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model_sd,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': config,
        'loss': getattr(save_checkpoint, 'last_loss', None),
    }
    if amp_scaler is not None:
        ckpt['amp_scaler'] = amp_scaler.state_dict()
    torch.save(ckpt, path)
    print(f"  Checkpoint saved: {path}")


def train(args):
    # Device
    has_cuda = torch.cuda.is_available()
    device = torch.device(args.device if has_cuda and args.device == 'cuda' else 'cpu')
    print(f"Device: {device}")

    # Auto DataParallel untuk multi-GPU (Kaggle 2×T4, dll)
    use_dp = has_cuda and torch.cuda.device_count() > 1 and not args.no_dp
    if use_dp:
        print(f"Multi-GPU: DataParallel on {torch.cuda.device_count()} GPUs")

    if args.amp and not has_cuda:
        print("  --amp disabled (no CUDA)")
        args.amp = False

    # Config
    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_length,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.grad_accum,
        thinking_steps=args.thinking_steps,
        gradient_checkpointing=args.grad_checkpoint,
    )
    print(f"Model config: {cfg.total_params_est:,} estimated params")

    # Model
    model = EmpathyTransformer(cfg).to(device)

    if use_dp:
        model = nn.DataParallel(model)
        raw_model = model.module
    else:
        raw_model = model

    info = count_params(raw_model)
    print(f"Actual params: {info['total']:,} ({info['size_mb']:.2f}MB)")

    if args.amp:
        # FP16 AMP on T4, BF16 on Ampere+
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"AMP enabled: {amp_dtype}")
    else:
        amp_dtype = None

    # Optimizer
    optimizer = AdamW(
        raw_model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    # Dataset
    if args.empathy:
        dataset = EmpathyDataset(args.data, args.tokenizer, args.max_length)
    else:
        dataset = TextDataset(args.data, args.tokenizer, args.max_length,
                              raw=not args.data.endswith('.pt'))

    # Split train/val
    val_size = min(int(len(dataset) * 0.05), 500)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Loaded {len(dataset)} examples — Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # LR scheduler
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    # Loss
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    # Resume
    start_step = 0
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'amp_scaler' in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt['amp_scaler'])
        start_step = ckpt['step']
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from step {start_step}, epoch {start_epoch}")

    # Create checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(ckpt_dir / 'config.json', 'w') as f:
        json.dump(vars(cfg), f, default=str, indent=2)

    # Training loop
    effective_batch = args.batch_size * args.grad_accum
    print(f"\n{'='*60}")
    print(f"Training — {args.epochs} epochs, {total_steps} steps")
    print(f"Batch {args.batch_size} × accum {args.grad_accum} = effective {effective_batch}")
    if args.thinking_steps:
        print(f"Depth-thinking: {args.thinking_steps} loops")
    print(f"{'='*60}")

    model.train()
    global_step = start_step
    best_val_loss = float('inf')
    accum_steps = cfg.gradient_accumulation_steps
    total_tokens_processed = 0
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for i, batch in enumerate(train_loader):
            # Forward + backward with AMP
            with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                loss = train_step(model, batch, device, loss_fn)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                tokens_in_batch = (batch['attention_mask'] > 0).sum().item()
                total_tokens_processed += tokens_in_batch
                epoch_loss += loss.item() * accum_steps
                epoch_steps += 1

                if global_step % args.log_every == 0:
                    elapsed = time.time() - t0
                    tokens_per_sec = total_tokens_processed / elapsed if elapsed > 0 else 0
                    curr_lr = scheduler.get_last_lr()[0]
                    print(
                        f"  Step {global_step:>6d} | "
                        f"Loss: {loss.item() * accum_steps:.4f} | "
                        f"LR: {curr_lr:.2e} | "
                        f"Tok/s: {tokens_per_sec:.0f} | "
                        f"Epoch: {epoch+1}/{args.epochs}"
                    )

                # Validation
                if global_step % args.val_every == 0:
                    val_loss, ppl = validate(model, val_loader, device, loss_fn,
                                            use_amp=(amp_dtype is not None))
                    print(f"  >>> Validation — Loss: {val_loss:.4f}, Perplexity: {ppl:.2f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, raw_model, optimizer, scheduler,
                                        global_step, epoch, cfg,
                                        ckpt_dir / 'best.pt', scaler)
                    model.train()

                # Save checkpoint
                if global_step % args.save_every == 0:
                    save_checkpoint(model, raw_model, optimizer, scheduler,
                                    global_step, epoch, cfg,
                                    ckpt_dir / f'step_{global_step}.pt', scaler)

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"  Epoch {epoch+1} done — Avg loss: {avg_epoch_loss:.4f}")

    # Final save
    save_checkpoint(model, raw_model, optimizer, scheduler,
                    global_step, args.epochs, cfg,
                    ckpt_dir / 'final.pt', scaler)
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    total_time = time.time() - t0
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EmpathyTransformer V3')
    parser.add_argument('--data', required=True, help='Dataset path')
    parser.add_argument('--tokenizer', required=True, help='Tokenizer path')
    parser.add_argument('--resume', help='Resume from checkpoint path')

    # Model
    parser.add_argument('--vocab-size', type=int, default=16384)
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=12)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-kv-heads', type=int, default=4)
    parser.add_argument('--d-ff', type=int, default=2048)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--thinking-steps', type=int, default=0,
                        help='Depth-thinking latent loop steps (0=off)')

    # Training
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Peak LR (Pythia 70M scale)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--grad-accum', type=int, default=8,
                        help='Gradient accumulation steps (effective batch = batch × accum)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision (FP16/BF16)')
    parser.add_argument('--no-dp', action='store_true',
                        help='Disable DataParallel (multi-GPU)')
    parser.add_argument('--grad-checkpoint', action='store_true',
                        help='Enable gradient checkpointing (hemat VRAM)')

    # Logging/saving
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--val-every', type=int, default=250)
    parser.add_argument('--checkpoint-dir', default='./checkpoints')

    # Data
    parser.add_argument('--empathy', action='store_true',
                        help='Use EmpathyDataset (input/response/emotion format)')

    args = parser.parse_args()
    train(args)
