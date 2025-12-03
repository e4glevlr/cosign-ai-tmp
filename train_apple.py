"""
Fine-tune UniSign ST-GCN encoder on Apple Vision extracted poses.

This script implements domain adaptation by:
1. Loading the pretrained checkpoint (RTMPose trained)
2. Freezing the mT5 language model (preserve Vietnamese knowledge)
3. Fine-tuning only the ST-GCN encoder layers (learn Apple Vision topology)

Training uses warmup + cosine decay learning rate schedule for optimal convergence.

Usage:
    python train_apple.py \
        --dataset VSL \
        --use_apple_pose \
        --finetune pretrained_weight/best_checkpoint.pth \
        --output_dir output/apple_finetuned \
        --epochs 50 \
        --lr 1e-4 \
        --batch_size 16
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models import Uni_Sign
from datasets import S2T_Dataset
from config import train_label_paths, dev_label_paths
import utils


def get_args():
    parser = argparse.ArgumentParser('UniSign Apple Vision Fine-tuning')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='VSL', 
                        choices=['VSL', 'CSL_Daily', 'CSL_News', 'WLASL'],
                        help='Dataset to use for training')
    parser.add_argument('--use_apple_pose', action='store_true',
                        help='Use Apple Vision extracted poses instead of RTMPose')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum pose sequence length')
    
    # Model
    parser.add_argument('--finetune', type=str, required=True,
                        help='Path to pretrained checkpoint')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension for ST-GCN')
    parser.add_argument('--rgb_support', action='store_true',
                        help='Use RGB support (not recommended for Apple Vision training)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Peak learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for optimizer')
    parser.add_argument('--label_smoothing', type=float, default=0.2,
                        help='Label smoothing factor')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='output/apple_finetuned',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log metrics every N steps')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def freeze_mt5(model):
    """Freeze mT5 language model weights."""
    frozen_params = 0
    for name, param in model.named_parameters():
        if 'mt5_model' in name:
            param.requires_grad = False
            frozen_params += param.numel()
    return frozen_params


def get_trainable_params(model):
    """Get trainable parameters (ST-GCN encoder only)."""
    trainable = []
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append((name, param))
            trainable_params += param.numel()
    return trainable, trainable_params


def create_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """Create warmup + cosine decay scheduler."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.01,  # Start at 1% of lr
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-6  # Minimum LR
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    return scheduler


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, args):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()
    
    for step, batch in enumerate(dataloader):
        # Move data to device
        src_input, tgt_input = batch
        for key in src_input:
            if isinstance(src_input[key], torch.Tensor):
                # Convert to float32 first, then move to device (MPS requirement)
                src_input[key] = src_input[key].float().to(device)
        
        # Forward pass
        outputs = model(src_input, tgt_input)
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Logging
        if step % args.log_every == 0:
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            samples_per_sec = (step + 1) * args.batch_size / elapsed
            print(f"  Step {step}/{len(dataloader)} | Loss: {loss.item():.4f} | "
                  f"LR: {current_lr:.2e} | Speed: {samples_per_sec:.1f} samples/s")
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device, num_beams=4, max_new_tokens=100):
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    predictions = []
    references = []
    
    for batch in dataloader:
        src_input, tgt_input = batch
        for key in src_input:
            if isinstance(src_input[key], torch.Tensor):
                # Convert to float32 first, then move to device (MPS requirement)
                src_input[key] = src_input[key].float().to(device)
        
        # Forward for loss
        outputs = model(src_input, tgt_input)
        total_loss += outputs['loss'].item()
        num_batches += 1
        
        # Generate predictions
        generated = model.generate(
            outputs, 
            max_new_tokens=max_new_tokens,
            num_beams=num_beams
        )
        
        # Decode predictions
        pred_texts = model.mt5_tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend(pred_texts)
        references.extend(tgt_input['gt_sentence'])
    
    avg_loss = total_loss / num_batches
    
    # Calculate simple accuracy (exact match)
    correct = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    accuracy = correct / len(predictions) if predictions else 0.0
    
    return avg_loss, accuracy, predictions, references


def main():
    args = get_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 60)
    print("UniSign Apple Vision Fine-tuning")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Use Apple Vision poses: {args.use_apple_pose}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = S2T_Dataset(train_label_paths[args.dataset], args, 'train')
    val_dataset = S2T_Dataset(dev_label_paths[args.dataset], args, 'dev')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
    
    # Create model
    print("\nInitializing model...")
    model = Uni_Sign(args)
    
    # Load pretrained checkpoint
    if args.finetune and os.path.exists(args.finetune):
        print(f"Loading checkpoint from {args.finetune}...")
        state_dict = torch.load(args.finetune, map_location='cpu', weights_only=True)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully!")
    else:
        print("WARNING: No checkpoint loaded, training from scratch!")
    
    # Freeze mT5
    frozen_params = freeze_mt5(model)
    print(f"\nFrozen mT5 parameters: {frozen_params:,}")
    
    # Get trainable params
    trainable, trainable_params = get_trainable_params(model)
    print(f"Trainable ST-GCN parameters: {trainable_params:,}")
    print(f"Trainable layers: {[name for name, _ in trainable[:10]]}...")
    
    # Move to device
    device = torch.device(args.device)
    model = model.to(device)
    
    # Create optimizer (only for trainable params)
    optimizer = AdamW(
        [p for _, p in trainable],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    steps_per_epoch = len(train_loader)
    scheduler = create_scheduler(
        optimizer, 
        args.warmup_epochs, 
        args.epochs, 
        steps_per_epoch
    )
    
    print(f"\nSteps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {steps_per_epoch * args.epochs}")
    print(f"Warmup steps: {args.warmup_epochs * steps_per_epoch}")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    best_accuracy = 0.0
    training_log = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args
        )
        
        # Evaluate
        val_loss, accuracy, _, _ = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'lr': scheduler.get_last_lr()[0],
            'time': epoch_time
        }
        training_log.append(log_entry)
        
        print(f"\n  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
                'accuracy': accuracy,
            }, output_dir / 'best_checkpoint_apple.pth')
            print(f"  ★ New best model saved!")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
                'accuracy': accuracy,
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # Save training log
        with open(output_dir / 'training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, output_dir / 'final_checkpoint.pth')
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best accuracy: {best_accuracy:.2%}")
    print(f"Checkpoints saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  1. Test: python scripts/run_json_inference.py --pose_json <test.json> --finetune {output_dir}/best_checkpoint_apple.pth")
    print(f"  2. Compare with RTMPose baseline to verify accuracy improvement")


if __name__ == '__main__':
    main()
