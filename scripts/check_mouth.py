import json
import numpy as np
import argparse

def check_mouth(path, label):
    with open(path, 'r') as f:
        data = json.load(f)
    
    if 'frames' in data:
        frames = sorted(data['frames'], key=lambda x: x['frame_index'])
        kps = np.array([f['keypoints'] for f in frames])
    elif 'keypoints' in data:
        kps = np.array(data['keypoints'])[:, 0, :, :]

    # Inner Lips: 83-90
    # Calculate average height (Openness)
    # Top lip indices: 83, 84, 85
    # Bottom lip indices: 87, 88, 89 (approx)
    
    lips = kps[:, 83:91, :]
    # Height = Max Y - Min Y
    heights = np.max(lips[:, :, 1], axis=1) - np.min(lips[:, :, 1], axis=1)
    
    # Filter valid
    valid = heights > 0
    avg_height = np.mean(heights[valid])
    
    print(f"{label} Average Mouth Height: {avg_height:.2f}")

if __name__ == "__main__":
    check_mouth("json/BG1_S002_pose_optimized.json", "Apple")
    check_mouth("json/BG1_S002_pose_truth.json", "Truth")
