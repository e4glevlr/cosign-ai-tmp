"""
Create a smart hybrid that uses truth body but properly aligns Apple hands
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

def create_smart_hybrid(apple_path, truth_path, output_path):
    """
    Use truth body but align Apple hands to truth wrist positions
    This ensures spatial consistency between body and hands
    """
    a_kps, a_scores = load_json(apple_path)
    t_kps, t_scores = load_json(truth_path)

    min_len = min(len(a_kps), len(t_kps))
    a_kps = a_kps[:min_len]
    t_kps = t_kps[:min_len]
    a_scores = a_scores[:min_len]
    t_scores = t_scores[:min_len]

    # Start with truth
    hybrid_kps = t_kps.copy()

    # For each frame, align Apple hands to truth wrists
    for t in range(min_len):
        # Left Hand Alignment
        # Truth left wrist position (index 9)
        truth_left_wrist = t_kps[t, 0, 9, :]
        # Apple left hand wrist (index 91)
        apple_left_wrist = a_kps[t, 0, 91, :]

        if apple_left_wrist[0] > 0 and truth_left_wrist[0] > 0:
            # Calculate offset needed to align Apple hand wrist to truth wrist
            offset_lh = truth_left_wrist - apple_left_wrist

            # Apply offset to all left hand points (91-111)
            for idx in range(91, 112):
                if a_kps[t, 0, idx, 0] > 0:
                    hybrid_kps[t, 0, idx, :] = a_kps[t, 0, idx, :] + offset_lh

        # Right Hand Alignment
        truth_right_wrist = t_kps[t, 0, 10, :]
        apple_right_wrist = a_kps[t, 0, 112, :]

        if apple_right_wrist[0] > 0 and truth_right_wrist[0] > 0:
            offset_rh = truth_right_wrist - apple_right_wrist

            for idx in range(112, 133):
                if a_kps[t, 0, idx, 0] > 0:
                    hybrid_kps[t, 0, idx, :] = a_kps[t, 0, idx, :] + offset_rh

    save_json(hybrid_kps, t_scores, output_path)
    print(f"Smart hybrid created: Truth body with aligned Apple hands")

if __name__ == "__main__":
    # Use raw Apple pose for hand topology, truth for body
    create_smart_hybrid(
        "json/BG1_S002_pose.json",
        "json/BG1_S002_pose_truth.json",
        "json/BG1_S002_smart_hybrid.json"
    )

