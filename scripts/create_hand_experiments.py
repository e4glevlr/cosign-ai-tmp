"""
Create targeted experiments to isolate which hand component is causing failures
"""
import json
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from optimize_pose import load_json

def save_json(kps, scores, path):
    output_data = {
        "keypoints": kps.tolist(),
        "scores": scores.tolist()
    }
    with open(path, 'w') as f:
        json.dump(output_data, f)
    print(f"Saved: {path}")

def create_hand_experiments(apple_path, truth_path):
    """Create experiments to test hand components"""
    a_kps, a_scores = load_json(apple_path)
    t_kps, t_scores = load_json(truth_path)

    min_len = min(len(a_kps), len(t_kps))
    a_kps = a_kps[:min_len]
    t_kps = t_kps[:min_len]
    a_scores = a_scores[:min_len]
    t_scores = t_scores[:min_len]

    # Experiment 1: Apple Everything + Truth Left Hand
    exp1 = a_kps.copy()
    exp1[:, :, 91:112, :] = t_kps[:, :, 91:112, :]
    save_json(exp1, a_scores, "json/exp_truth_left_hand.json")

    # Experiment 2: Apple Everything + Truth Right Hand
    exp2 = a_kps.copy()
    exp2[:, :, 112:133, :] = t_kps[:, :, 112:133, :]
    save_json(exp2, a_scores, "json/exp_truth_right_hand.json")

    # Experiment 3: Apple Everything + Truth Both Hands
    exp3 = a_kps.copy()
    exp3[:, :, 91:112, :] = t_kps[:, :, 91:112, :]
    exp3[:, :, 112:133, :] = t_kps[:, :, 112:133, :]
    save_json(exp3, a_scores, "json/exp_truth_both_hands.json")

    # Experiment 4: Apple Body+Face + Truth Hands (Already done in exp3)

    # Experiment 5: Truth Body+Face + Apple Hands
    exp5 = t_kps.copy()
    exp5[:, :, 91:112, :] = a_kps[:, :, 91:112, :]
    exp5[:, :, 112:133, :] = a_kps[:, :, 112:133, :]
    save_json(exp5, t_scores, "json/exp_truth_body_apple_hands.json")

    print("\nExperiments created! Run inference with:")
    print("  ./.venv/bin/python scripts/run_json_inference.py --pose_json json/exp_truth_left_hand.json --finetune pretrained_weight/best_checkpoint.pth")
    print("  ./.venv/bin/python scripts/run_json_inference.py --pose_json json/exp_truth_right_hand.json --finetune pretrained_weight/best_checkpoint.pth")
    print("  ./.venv/bin/python scripts/run_json_inference.py --pose_json json/exp_truth_both_hands.json --finetune pretrained_weight/best_checkpoint.pth")
    print("  ./.venv/bin/python scripts/run_json_inference.py --pose_json json/exp_truth_body_apple_hands.json --finetune pretrained_weight/best_checkpoint.pth")

if __name__ == "__main__":
    create_hand_experiments("json/BG1_S002_pose_optimized_v2.json", "json/BG1_S002_pose_truth.json")

