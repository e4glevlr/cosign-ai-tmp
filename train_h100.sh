#!/bin/bash
# ============================================================
# Training script for H100
# Run inside tmux for background execution
# ============================================================

source venv/bin/activate

python train_apple.py \
    --dataset VSL \
    --use_apple_pose \
    --finetune pretrained_weight/best_checkpoint.pth \
    --output_dir output/apple_vsl_h100 \
    --epochs 50 \
    --batch_size 64 \
    --lr 5e-5 \
    --warmup_epochs 10 \
    --device cuda \
    2>&1 | tee training_log.txt
