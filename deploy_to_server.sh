#!/bin/bash
# ============================================================
# Deploy và Train trên GPU Server
# Chạy script này trên SERVER sau khi upload files
# ============================================================

# Setup paths
export PATH="/opt/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib64

echo "=== Checking GPU ==="
nvidia-smi

echo ""
echo "=== Setting up Python environment ==="
cd /root/cosign-ai
python3 -m venv venv
source venv/bin/activate

echo ""
echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.36.0 sentencepiece einops tqdm numpy sacrebleu

echo ""
echo "=== Verifying CUDA ==="
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "=== Starting Training ==="
python train_apple.py \
    --dataset VSL \
    --use_apple_pose \
    --finetune pretrained_weight/best_checkpoint.pth \
    --output_dir output/apple_vsl_gpu \
    --epochs 50 \
    --batch_size 32 \
    --lr 5e-5 \
    --warmup_epochs 10 \
    --device cuda \
    2>&1 | tee training_log.txt
