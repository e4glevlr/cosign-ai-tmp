import json
import numpy as np
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from optimize_pose import load_json

def save_json(kps, scores, path):
    data = {"keypoints": kps.tolist(), "scores": scores.tolist()}
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"Saved: {path}")

def run_experiments(apple_path, truth_path):
    print("Loading data...")
    a_kps, a_scores = load_json(apple_path)
    t_kps, t_scores = load_json(truth_path)
    
    # Align length
    min_len = min(len(a_kps), len(t_kps))
    a_kps = a_kps[:min_len]
    t_kps = t_kps[:min_len]
    
    # Base is Apple (optimized, no smooth)
    # We will inject Truth parts into this base
    
    # Indices
    idx_body = slice(0, 17)
    idx_feet = slice(17, 23)
    idx_face = slice(23, 91)
    idx_left = slice(91, 112)
    idx_right = slice(112, 133)
    
    # 1. Test Body (Shoulders, Arms positions)
    # Truth Body + Apple Hands + Apple Face
    hybrid_body = a_kps.copy()
    hybrid_body[:, :, idx_body, :] = t_kps[:, :, idx_body, :]
    hybrid_body[:, :, idx_feet, :] = t_kps[:, :, idx_feet, :] # Copy feet with body
    save_json(hybrid_body, a_scores, "json/exp_body_truth.json")
    
    # 2. Test Hands (Both Fingers)
    # Apple Body + Truth Hands + Apple Face
    hybrid_hands = a_kps.copy()
    hybrid_hands[:, :, idx_left, :] = t_kps[:, :, idx_left, :]
    hybrid_hands[:, :, idx_right, :] = t_kps[:, :, idx_right, :]
    save_json(hybrid_hands, a_scores, "json/exp_hands_truth.json")
    
    # 3. Test Left Hand Only
    hybrid_left = a_kps.copy()
    hybrid_left[:, :, idx_left, :] = t_kps[:, :, idx_left, :]
    save_json(hybrid_left, a_scores, "json/exp_left_truth.json")

    # 4. Test Right Hand Only
    hybrid_right = a_kps.copy()
    hybrid_right[:, :, idx_right, :] = t_kps[:, :, idx_right, :]
    save_json(hybrid_right, a_scores, "json/exp_right_truth.json")

if __name__ == "__main__":
    run_experiments("json/BG1_S002_pose_optimized_nosmooth.json", "json/BG1_S002_pose_truth.json")
