#!/bin/bash
# ============================================================
# Setup script for H100 training
# Run this on the H100 server after copying files
# ============================================================

set -e

echo "============================================================"
echo "Setting up UniSign Apple Vision Training on H100"
echo "============================================================"

# Create virtual environment
echo "[1/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
echo "[2/4] Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo "[3/4] Installing dependencies..."
pip install transformers==4.36.0
pip install sentencepiece
pip install einops
pip install tqdm
pip install numpy
pip install sacrebleu

# Verify GPU
echo "[4/4] Verifying GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "============================================================"
echo "Setup complete! Run training with:"
echo "============================================================"
echo ""
echo "source venv/bin/activate"
echo "python train_apple.py \\"
echo "    --dataset VSL \\"
echo "    --use_apple_pose \\"
echo "    --finetune pretrained_weight/best_checkpoint.pth \\"
echo "    --output_dir output/apple_vsl_h100 \\"
echo "    --epochs 50 \\"
echo "    --batch_size 64 \\"
echo "    --lr 5e-5 \\"
echo "    --warmup_epochs 10 \\"
echo "    --device cuda"
echo ""
