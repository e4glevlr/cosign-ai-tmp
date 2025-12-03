#!/usr/bin/env python3
"""
Inference script for Apple Vision fine-tuned UniSign model.

Usage:
    python inference_apple.py --video path/to/video.mp4
    python inference_apple.py --video path/to/video.mp4 --checkpoint output/best_checkpoint_apple.pth
"""

import os
import sys
import argparse
import subprocess
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np
import pickle
from argparse import Namespace

from models import Uni_Sign
from datasets import load_part_kp


def extract_pose_apple_vision(video_path: str, output_json: str) -> bool:
    """Extract pose using Apple Vision PoseExtractor."""
    extractor = Path(__file__).parent / "bin" / "PoseExtractor"
    
    if not extractor.exists():
        print(f"Error: PoseExtractor not found at {extractor}")
        return False
    
    # Create temp video output (required by PoseExtractor)
    with tempfile.NamedTemporaryFile(suffix='.mov', delete=False) as tmp:
        tmp_video = tmp.name
    
    try:
        result = subprocess.run(
            [str(extractor), video_path, tmp_video, output_json],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"PoseExtractor error: {result.stderr}")
            return False
            
        return Path(output_json).exists()
        
    finally:
        # Cleanup temp video
        if os.path.exists(tmp_video):
            os.remove(tmp_video)


def convert_apple_json_to_pose(json_path: str) -> dict:
    """Convert Apple Vision JSON to pose format for model."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames = data.get('frames', [])
    if not frames:
        raise ValueError("No frames in JSON")
    
    num_frames = len(frames)
    num_keypoints = 133  # COCO-WholeBody format
    
    # Initialize arrays
    skeletons = np.zeros((num_frames, 1, num_keypoints, 2), dtype=np.float32)
    scores = np.zeros((num_frames, 1, num_keypoints), dtype=np.float32)
    
    for i, frame in enumerate(frames):
        keypoints = frame.get('keypoints', [])
        frame_scores = frame.get('scores', [])
        
        # keypoints is list of [x, y] pairs
        for idx, kp in enumerate(keypoints):
            if idx < num_keypoints and len(kp) >= 2:
                skeletons[i, 0, idx, 0] = kp[0]
                skeletons[i, 0, idx, 1] = kp[1]
                if idx < len(frame_scores):
                    scores[i, 0, idx] = frame_scores[idx]
    
    # Load part keypoints (body, left, right, face_all)
    kps = load_part_kp(skeletons, scores, force_ok=True)
    
    return kps


def load_model(checkpoint_path: str, device: torch.device) -> Uni_Sign:
    """Load the fine-tuned model."""
    args = Namespace(
        hidden_dim=256,
        dataset="VSL",
        rgb_support=False,
        label_smoothing=0.0,
    )
    
    model = Uni_Sign(args)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


def inference(model: Uni_Sign, kps: dict, device: torch.device, 
              max_new_tokens: int = 50, num_beams: int = 4) -> str:
    """Run inference on pose data."""
    
    # Prepare input
    src_input = {}
    for key in ['body', 'left', 'right', 'face_all']:
        # Add batch dimension and move to device
        tensor = kps[key].unsqueeze(0).float().to(device)
        src_input[key] = tensor
    
    # Create attention mask
    seq_len = src_input['body'].shape[1]
    src_input['attention_mask'] = torch.ones(1, seq_len, dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Forward pass to get embeddings
        tgt_input = {'gt_sentence': [''], 'gt_gloss': ['']}
        outputs = model(src_input, tgt_input)
        
        # Generate text
        generated_ids = model.generate(
            outputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams
        )
        
        # Decode
        text = model.mt5_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return text


def main():
    parser = argparse.ArgumentParser(description='UniSign Apple Vision Inference')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--checkpoint', type=str, 
                        default='output/best_checkpoint_apple.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use')
    parser.add_argument('--max_tokens', type=int, default=50,
                        help='Maximum tokens to generate')
    parser.add_argument('--num_beams', type=int, default=4,
                        help='Number of beams for beam search')
    
    args = parser.parse_args()
    
    # Check video exists
    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Select device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 60)
    print("UniSign Apple Vision Inference")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Step 1: Extract pose
    print("\n[1/3] Extracting pose with Apple Vision...")
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        json_path = tmp.name
    
    try:
        success = extract_pose_apple_vision(args.video, json_path)
        if not success:
            print("Failed to extract pose")
            sys.exit(1)
        print("✓ Pose extracted")
        
        # Step 2: Convert to model format
        print("\n[2/3] Converting pose data...")
        kps = convert_apple_json_to_pose(json_path)
        print(f"✓ Frames: {kps['body'].shape[0]}")
        
        # Step 3: Load model and run inference
        print("\n[3/3] Running inference...")
        model = load_model(args.checkpoint, device)
        print("✓ Model loaded")
        
        result = inference(model, kps, device, args.max_tokens, args.num_beams)
        
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"\n🤟 Vietnamese Sign Language Translation:\n")
        print(f"   \"{result}\"\n")
        print("=" * 60)
        
    finally:
        # Cleanup
        if os.path.exists(json_path):
            os.remove(json_path)


if __name__ == "__main__":
    main()
