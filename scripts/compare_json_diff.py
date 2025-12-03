import json
import numpy as np
import argparse

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if 'frames' in data:
        # Convert Apple format to tensor for comparison
        frames = sorted(data['frames'], key=lambda x: x['frame_index'])
        kps = [f['keypoints'] for f in frames]
        return np.array(kps)
    elif 'keypoints' in data:
        return np.array(data['keypoints'])[:, 0, :, :] # (T, 133, 2)

def compare(path1, path2):
    k1 = load_json(path1)
    k2 = load_json(path2)
    
    print(f"Shape 1: {k1.shape}")
    print(f"Shape 2: {k2.shape}")
    
    min_len = min(len(k1), len(k2))
    k1 = k1[:min_len]
    k2 = k2[:min_len]
    
    # Calculate Euclidean distance
    diff = np.linalg.norm(k1 - k2, axis=-1) # (T, 133)
    
    # Parts
    parts = {
        "Body (0-16)": (0, 17),
        "Face (23-90)": (23, 91),
        "Left Hand (91-111)": (91, 112),
        "Right Hand (112-132)": (112, 133)
    }
    
    print("\nMean Error per Part (Pixels):")
    for name, (start, end) in parts.items():
        part_diff = diff[:, start:end]
        # Filter out zeros (missing data)
        valid_mask = (k1[:, start:end, 0] > 0) & (k2[:, start:end, 0] > 0)
        if np.sum(valid_mask) > 0:
            mean_err = np.mean(part_diff[valid_mask])
            print(f"{name}: {mean_err:.2f}")
        else:
            print(f"{name}: No valid overlap")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path1')
    parser.add_argument('path2')
    args = parser.parse_args()
    compare(args.path1, args.path2)
