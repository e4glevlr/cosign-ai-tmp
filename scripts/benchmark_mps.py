#!/usr/bin/env python3
"""
Benchmark training speed on M3 Max with MPS
Estimates total training time for VSL dataset
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import time
import torch
from torch.utils.data import DataLoader
from datasets import S2T_Dataset
from models import Uni_Sign
from config import train_label_paths
from argparse import Namespace

def main():
    print("=" * 60)
    print("Benchmarking Training Speed on M3 Max")
    print("=" * 60)
    
    # Check MPS
    mps_available = torch.backends.mps.is_available()
    print(f"\nMPS available: {mps_available}")
    device = torch.device("mps" if mps_available else "cpu")
    print(f"Using device: {device}")
    
    # Settings
    batch_size = 8  # Reduced for MPS memory
    epochs = 50
    
    print(f"\nSettings: batch_size={batch_size}, epochs={epochs}")
    
    # Create args for dataset and model
    args = Namespace(
        hidden_dim=256,
        dataset="VSL",
        rgb_support=False,
        max_length=256,
        use_apple_pose=False,  # Use RTMPose for benchmark
        label_smoothing=0.0,
    )
    
    # Load dataset
    print("\nLoading dataset...")
    
    train_dataset = S2T_Dataset(
        path=train_label_paths["VSL"],
        args=args,
        phase='train'
    )
    print(f"Train samples: {len(train_dataset)}")
    
    # DataLoader - MPS compatible settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # MPS requires 0
        pin_memory=False,  # MPS không support
        collate_fn=train_dataset.collate_fn
    )
    
    batches_per_epoch = len(train_loader)
    print(f"Batches per epoch: {batches_per_epoch}")
    
    # Load model
    print("\nLoading model...")
    model = Uni_Sign(args)
    model = model.to(device)
    
    # Freeze mT5
    for param in model.mt5_model.parameters():
        param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    
    # Benchmark
    print("\nBenchmarking (3 batches)...")
    model.train()
    
    times = []
    for i, (src_input, tgt_input) in enumerate(train_loader):
        if i >= 3:
            break
            
        # Move data to device - explicit float32 conversion
        for key in ['body', 'left', 'right', 'face_all']:
            src_input[key] = src_input[key].float().to(device)
        src_input['attention_mask'] = src_input['attention_mask'].to(device)
        
        # Forward + backward
        start = time.time()
        
        optimizer.zero_grad()
        outputs = model(src_input, tgt_input)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        
        # Sync MPS
        if device.type == "mps":
            torch.mps.synchronize()
        
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Batch {i+1}: {elapsed:.2f}s (loss: {loss.item():.4f})")
    
    # Calculate estimates
    avg_time = sum(times) / len(times)
    epoch_time = avg_time * batches_per_epoch
    total_time = epoch_time * epochs
    
    print("\n" + "=" * 60)
    print("TRAINING TIME ESTIMATE")
    print("=" * 60)
    print(f"Average batch time: {avg_time:.2f}s")
    print(f"Estimated time per epoch: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
    print(f"\nTotal for {epochs} epochs:")
    print(f"  - {total_time:.0f} seconds")
    print(f"  - {total_time/60:.1f} minutes")
    print(f"  - {total_time/3600:.2f} hours")
    print("=" * 60)

if __name__ == "__main__":
    main()
