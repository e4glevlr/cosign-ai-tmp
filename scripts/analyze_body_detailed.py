import json
import numpy as np
import argparse

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if 'frames' in data:
        frames = sorted(data['frames'], key=lambda x: x['frame_index'])
        kps = [f['keypoints'] for f in frames]
        return np.array(kps)
    elif 'keypoints' in data:
        kps = np.array(data['keypoints'])
        if len(kps.shape) == 4:
            return kps[:, 0, :, :]
        return kps

def analyze_body_detailed(apple_path, truth_path):
    """Analyze per-joint body differences"""
    apple_kps = load_json(apple_path)
    truth_kps = load_json(truth_path)

    min_len = min(len(apple_kps), len(truth_kps))
    apple_kps = apple_kps[:min_len]
    truth_kps = truth_kps[:min_len]

    # Body keypoint names (COCO format)
    body_parts = {
        0: "Nose",
        1: "Left Eye",
        2: "Right Eye",
        3: "Left Ear",
        4: "Right Ear",
        5: "Left Shoulder",
        6: "Right Shoulder",
        7: "Left Elbow",
        8: "Right Elbow",
        9: "Left Wrist",
        10: "Right Wrist",
        11: "Left Hip",
        12: "Right Hip",
        13: "Left Knee",
        14: "Right Knee",
        15: "Left Ankle",
        16: "Right Ankle",
    }

    print("=" * 80)
    print("DETAILED BODY JOINT ANALYSIS")
    print("=" * 80)

    # Calculate per-joint statistics
    for idx in range(17):
        diff = np.linalg.norm(apple_kps[:, idx, :] - truth_kps[:, idx, :], axis=-1)
        valid_mask = (apple_kps[:, idx, 0] > 0) & (truth_kps[:, idx, 0] > 0)

        if np.sum(valid_mask) > 0:
            dx = truth_kps[valid_mask, idx, 0] - apple_kps[valid_mask, idx, 0]
            dy = truth_kps[valid_mask, idx, 1] - apple_kps[valid_mask, idx, 1]

            mean_error = np.mean(diff[valid_mask])
            median_dx = np.median(dx)
            median_dy = np.median(dy)
            std_dx = np.std(dx)
            std_dy = np.std(dy)

            print(f"{idx:2d} {body_parts[idx]:15s}: Error={mean_error:6.2f}px  "
                  f"Offset=({median_dx:+6.2f}, {median_dy:+6.2f})  "
                  f"StdDev=({std_dx:5.2f}, {std_dy:5.2f})")
        else:
            print(f"{idx:2d} {body_parts[idx]:15s}: No valid data")

    # Group analysis
    print("\n" + "=" * 80)
    print("BODY REGION ANALYSIS")
    print("=" * 80)

    regions = {
        "Head (0-4)": list(range(5)),
        "Shoulders (5-6)": [5, 6],
        "Elbows (7-8)": [7, 8],
        "Wrists (9-10)": [9, 10],
        "Hips (11-12)": [11, 12],
        "Lower Body (13-16)": list(range(13, 17)),
    }

    for region_name, indices in regions.items():
        all_dx = []
        all_dy = []
        all_errors = []

        for idx in indices:
            valid = (apple_kps[:, idx, 0] > 0) & (truth_kps[:, idx, 0] > 0)
            if np.sum(valid) > 0:
                dx = truth_kps[valid, idx, 0] - apple_kps[valid, idx, 0]
                dy = truth_kps[valid, idx, 1] - apple_kps[valid, idx, 1]
                err = np.linalg.norm(apple_kps[valid, idx, :] - truth_kps[valid, idx, :], axis=-1)

                all_dx.extend(dx)
                all_dy.extend(dy)
                all_errors.extend(err)

        if all_dx:
            print(f"\n{region_name}")
            print(f"  Mean Error:   {np.mean(all_errors):.2f}px")
            print(f"  Median Offset: ({np.median(all_dx):+.2f}, {np.median(all_dy):+.2f})")
            print(f"  Std Dev:       ({np.std(all_dx):.2f}, {np.std(all_dy):.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--apple', required=True)
    parser.add_argument('--truth', required=True)
    args = parser.parse_args()

    analyze_body_detailed(args.apple, args.truth)

