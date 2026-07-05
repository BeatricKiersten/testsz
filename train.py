"""
Train EmpathyTransformer V4.

Usage:
    # Full training on Kaggle (2×T4)
    python train.py --data ./data/train_mixed.jsonl --tokenizer ./tokenizer.json \
                    --epochs 10 --batch-size 16 --lr 1e-3 --amp --grad-accum 4

    # Resume from checkpoint
    python train.py --resume ./checkpoints/best.pt

    # CPU quick test
    python train.py --data ./data/sample.jsonl --tokenizer ./tokenizer.json \
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig
from model import EmpathyTransformer, count_params
from dataset import EmpathyDataset, TextDataset


def get_wsd_scheduler(optimizer, warmup_steps: int, total_steps: int, decay_frac: float = 0.1):
    """
    WSD: Warmup-Stable-Decay scheduler (MiniCPM paper).
    - 5% steps: warmup 0 → 1
    - 85% steps: constant 1
    - 10% steps: exponential decay 1 → 0.01
    """
    stable_steps = total_steps - warmup_steps
    decay_steps = int(total_steps * decay_frac)
    stable_end = total_steps - decay_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        elif step < stable_end:
            return 1.0
        else:
            # Exponential decay: halve every ~10% of decay period
            decay_progress = (step - stable_end) / max(1, decay_steps)
            return max(0.01, 0.5 ** (decay_progress * 5))
    return LambdaLR(optimizer, lr_lambda)


def get_dropout_rate(global_step: int, total_steps: int,
                     start_dropout: float = 0.2,
                     end_dropout: float = 0.1) -> float:
    """Linear decay dropout selama training."""
    progress = min(1.0, global_step / max(1, total_steps))
    return start_dropout + (end_dropout - start_dropout) * progress


def train_step(model, batch, device, loss_fn):
    """Single training step."""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    logits, _ = model(input_ids, attention_mask)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                   shift_labels.view(-1))
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
    print(f"  Checkpoint saved: {path}", flush=True)


def train(args):
    # Device
    has_cuda = torch.cuda.is_available()
    device = torch.device(args.device if has_cuda and args.device == 'cuda' else 'cpu')
    print(f"Device: {device}")

    # Auto DataParallel
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
        thinking_steps=args.thinking_steps,
        gradient_checkpointing=args.grad_checkpoint,
        dropout=args.dropout,
        scale_embedding=not args.no_scale_emb,
        peri_ln=not args.no_peri_ln,
    )
    print(f"Model config: {cfg.total_params_est:,} estimated params")
    print(f" V4 features: Peri-LN={cfg.peri_ln}, QKV-norm={cfg.qkv_norm}, "
          f"ScaledEmb={cfg.scale_embedding}, PreLNHead={cfg.pre_ln_head}, "
          f"SlidingWindow={cfg.sliding_window_size}")

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
        # T4 (SM 7.5) gak punya bf16 tensor cores — pake fp16
        amp_dtype = torch.float16 if torch.cuda.get_device_capability(0) < (8, 0) else torch.bfloat16
        print(f"AMP enabled: {amp_dtype}")
    else:
        amp_dtype = None

    # Optimizer with separate embedding LR (V4: emb LR ×3)
    embed_params = []
    non_embed_params = []
    for name, p in raw_model.named_parameters():
        if 'token_emb' in name or 'lm_head' in name:
            embed_params.append(p)
        else:
            non_embed_params.append(p)

    optimizer = AdamW([
        {'params': non_embed_params, 'lr': cfg.learning_rate, 'weight_decay': cfg.weight_decay},
        {'params': embed_params, 'lr': cfg.learning_rate * args.emb_lr_mult, 'weight_decay': cfg.weight_decay},
    ], betas=(cfg.beta1, cfg.beta2), eps=1e-8)

    # Dataset
    if args.empathy:
        dataset = EmpathyDataset(args.data, args.tokenizer, args.max_length)
    else:
        dataset = TextDataset(args.data, args.tokenizer, args.max_length,
                              raw=not args.data.endswith('.pt'))

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

    # V4: WSD scheduler
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_wsd_scheduler(optimizer, args.warmup_steps, total_steps, decay_frac=0.1)
    print(f"WSD scheduler: {total_steps} total steps, {args.warmup_steps} warmup, "
          f"{int(total_steps*0.1)} decay steps")

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
        if scheduler is not None and 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict']:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_step = ckpt['step']
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from step {start_step}, epoch {start_epoch}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(ckpt_dir / 'config.json', 'w') as f:
        json.dump(vars(cfg), f, default=str, indent=2)

    # Training loop
    effective_batch = args.batch_size * args.grad_accum * (torch.cuda.device_count() if use_dp else 1)
    print(f"\n{'='*60}")
    print(f"Training — {args.epochs} epochs, ~{total_steps} steps")
    print(f"Batch {args.batch_size} × accum {args.grad_accum} × {torch.cuda.device_count() if use_dp else 1} GPU"
          f" = effective {effective_batch}")
    if args.thinking_steps:
        print(f"Depth-thinking: {args.thinking_steps} loops")
    print(f"Peak LR: {cfg.learning_rate}, Emb LR mult: {args.emb_lr_mult}x")
    print(f"Betas: ({cfg.beta1}, {cfg.beta2}), WD: {cfg.weight_decay}, Clip: {cfg.grad_clip}")
    print(f"{'='*60}")

    model.train()
    global_step = start_step
    best_val_loss = float('inf')
    accum_steps = args.grad_accum
    total_tokens_processed = 0
    t0 = time.time()
    stuck_loss_counter = 0  # early stopping helper
    stuck_loss_threshold = 3

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for i, batch in enumerate(train_loader):
            # V4: Dynamic dropout decay
            current_dropout = get_dropout_rate(global_step, total_steps, args.dropout, args.dropout * 0.5)
            for module in model.modules():
                if hasattr(module, 'p') and hasattr(module, 'training'):
                    if isinstance(module, nn.Dropout):
                        module.p = current_dropout

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

                tokens_in_batch = (batch['attention_mask'] > 0).sum().item()
                total_tokens_processed += tokens_in_batch
                epoch_loss += loss.item() * accum_steps
                epoch_steps += 1

                if global_step % args.log_every == 0:
                    elapsed = time.time() - t0
                    tokens_per_sec = total_tokens_processed / elapsed if elapsed > 0 else 0
                    curr_lr = scheduler.get_last_lr()[0] if scheduler else cfg.learning_rate
                    print(
                        f"  Step {global_step:>6d} | "
                        f"Loss: {loss.item() * accum_steps:.4f} | "
                        f"LR: {curr_lr:.2e} | "
                        f"Tok/s: {tokens_per_sec:.0f} | "
                        f"Drop: {current_dropout:.2f} | "
                        f"Epoch: {epoch+1}/{args.epochs}",
                        flush=True,
                    )
                    save_checkpoint.last_loss = loss.item() * accum_steps

                # Validation
                if global_step % args.val_every == 0:
                    val_loss, ppl = validate(model, val_loader, device, loss_fn,
                                            use_amp=(amp_dtype is not None))
                    print(f"  >>> Validation — Loss: {val_loss:.4f}, Perplexity: {ppl:.2f}", flush=True)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, raw_model, optimizer, scheduler,
                                        global_step, epoch, cfg,
                                        ckpt_dir / 'best.pt', scaler)
                    model.train()

                    # Early stopping check
                    if val_loss > getattr(save_checkpoint, 'last_val_loss', float('inf')):
                        stuck_loss_counter += 1
                        print(f"  >>> Val loss increased ({stuck_loss_counter}/{stuck_loss_threshold})", flush=True)
                        if stuck_loss_counter >= stuck_loss_threshold:
                            print("  >>> Early stopping triggered")
                            break
                    else:
                        stuck_loss_counter = 0
                    save_checkpoint.last_val_loss = val_loss

                # Save checkpoint
                if global_step % args.save_every == 0:
                    save_checkpoint(model, raw_model, optimizer, scheduler,
                                    global_step, epoch, cfg,
                                    ckpt_dir / f'step_{global_step}.pt', scaler)

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"  Epoch {epoch+1} done — Avg loss: {avg_epoch_loss:.4f}", flush=True)

        # Save epoch checkpoint
        save_checkpoint(model, raw_model, optimizer, scheduler,
                        global_step, epoch, cfg,
                        ckpt_dir / f'epoch_{epoch+1}.pt', scaler)

        # Check early stopping
        if stuck_loss_counter >= stuck_loss_threshold:
            print("  Early stopping at epoch end")
            break

    # Final save
    save_checkpoint(model, raw_model, optimizer, scheduler,
                    global_step, args.epochs, cfg,
                    ckpt_dir / 'final.pt', scaler)
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    total_time = time.time() - t0
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EmpathyTransformer V4')
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
    parser.add_argument('--thinking-steps', type=int, default=0)

    # Training
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Per-GPU batch (T4 16GB → 16 is fine)')
    parser.add_argument('--grad-accum', type=int, default=4,
                        help='Grad accum (effective BS = batch × accum × GPU)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Starting dropout rate (decays to half)')
    parser.add_argument('--emb-lr-mult', type=float, default=3.0,
                        help='Embedding LR multiplier vs body')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-dp', action='store_true')
    parser.add_argument('--no-scale-emb', action='store_true',
                        help='Disable scaled embedding')
    parser.add_argument('--no-peri-ln', action='store_true',
                        help='Disable Peri-LN (post-norm) — fallback to standard Pre-LN')
    parser.add_argument('--grad-checkpoint', action='store_true')

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
