import json
import numpy as np
import argparse
import sys
import os
# Ensure scripts dir is in path to import optimize_pose
sys.path.insert(0, os.path.dirname(__file__))
from optimize_pose import load_json

def create_hybrid(apple_path, truth_path, output_path):
    # Load
    a_kps, a_scores = load_json(apple_path)
    t_kps, t_scores = load_json(truth_path)
    
    # Align lengths
    min_len = min(len(a_kps), len(t_kps))
    a_kps = a_kps[:min_len]
    t_kps = t_kps[:min_len]
    a_scores = a_scores[:min_len]
    
    # Hybrid Construction: Apple Body + Truth Face
    hybrid_kps = a_kps.copy()
    
    # Replace Face (23-90) with Truth
    hybrid_kps[:, :, 23:91, :] = t_kps[:, :, 23:91, :]
    
    # Save
    output_data = {
        "keypoints": hybrid_kps.tolist(),
        "scores": a_scores.tolist() # Use Apple scores
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f)
    print(f"Hybrid saved to {output_path}")

def create_reverse_hybrid(apple_path, truth_path, output_path):
    # Truth Body + Apple Face
    # This tests if Apple Face 'poisons' the good Truth Body.
    
    a_kps, a_scores = load_json(apple_path)
    t_kps, t_scores = load_json(truth_path)
    
    min_len = min(len(a_kps), len(t_kps))
    a_kps = a_kps[:min_len]
    t_kps = t_kps[:min_len]
    t_scores = t_scores[:min_len]
    
    hybrid_kps = t_kps.copy()
    
    # Replace Face (23-90) with Apple
    hybrid_kps[:, :, 23:91, :] = a_kps[:, :, 23:91, :]
    
    output_data = {
        "keypoints": hybrid_kps.tolist(),
        "scores": t_scores.tolist()
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f)
    print(f"Reverse Hybrid saved to {output_path}")

if __name__ == "__main__":
    # 1. Apple Body (No Smooth) + Truth Face
    create_hybrid("json/BG1_S002_pose_optimized_nosmooth.json", "json/BG1_S002_pose_truth.json", "json/BG1_S002_hybrid_truth_face_nosmooth.json")
    
    # 2. Truth Body + Apple Face (New Test -> Expect "Hôm nay"?)
    create_reverse_hybrid("json/BG1_S002_pose_optimized.json", "json/BG1_S002_pose_truth.json", "json/BG1_S002_hybrid_apple_face.json")
