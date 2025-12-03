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

def analyze_hand_topology(apple_path, truth_path):
    """Analyze per-finger differences between Apple Vision and RTMPose"""
    apple_kps = load_json(apple_path)
    truth_kps = load_json(truth_path)

    min_len = min(len(apple_kps), len(truth_kps))
    apple_kps = apple_kps[:min_len]
    truth_kps = truth_kps[:min_len]

    # Define hand structure
    hands = {
        "Left Hand": {
            "wrist": 91,
            "thumb": list(range(92, 96)),      # CMC, MP, IP, Tip
            "index": list(range(96, 100)),     # MCP, PIP, DIP, Tip
            "middle": list(range(100, 104)),   # MCP, PIP, DIP, Tip
            "ring": list(range(104, 108)),     # MCP, PIP, DIP, Tip
            "pinky": list(range(108, 112)),    # MCP, PIP, DIP, Tip
        },
        "Right Hand": {
            "wrist": 112,
            "thumb": list(range(113, 117)),
            "index": list(range(117, 121)),
            "middle": list(range(121, 125)),
            "ring": list(range(125, 129)),
            "pinky": list(range(129, 133)),
        }
    }

    print("=" * 80)
    print("DETAILED HAND TOPOLOGY ANALYSIS")
    print("=" * 80)

    for hand_name, parts in hands.items():
        print(f"\n{hand_name.upper()}")
        print("-" * 80)

        # Wrist analysis
        wrist_idx = parts["wrist"]
        wrist_diff = np.linalg.norm(apple_kps[:, wrist_idx, :] - truth_kps[:, wrist_idx, :], axis=-1)
        valid_mask = (apple_kps[:, wrist_idx, 0] > 0) & (truth_kps[:, wrist_idx, 0] > 0)
        if np.sum(valid_mask) > 0:
            print(f"  Wrist (idx {wrist_idx}): {np.mean(wrist_diff[valid_mask]):.2f}px")

        # Per-finger analysis
        for finger_name in ["thumb", "index", "middle", "ring", "pinky"]:
            finger_indices = parts[finger_name]

            # Calculate differences for this finger
            finger_diffs = []
            for idx in finger_indices:
                diff = np.linalg.norm(apple_kps[:, idx, :] - truth_kps[:, idx, :], axis=-1)
                valid = (apple_kps[:, idx, 0] > 0) & (truth_kps[:, idx, 0] > 0)
                if np.sum(valid) > 0:
                    finger_diffs.append(np.mean(diff[valid]))
                else:
                    finger_diffs.append(0.0)

            avg_error = np.mean([d for d in finger_diffs if d > 0])
            if avg_error > 0:
                print(f"  {finger_name.capitalize():8s}: {avg_error:6.2f}px  (joints: {finger_diffs})")

    # Calculate offset statistics
    print("\n" + "=" * 80)
    print("COORDINATE OFFSET ANALYSIS")
    print("=" * 80)

    for hand_name, parts in hands.items():
        print(f"\n{hand_name}")

        # Collect all hand points for this hand
        all_indices = [parts["wrist"]] + parts["thumb"] + parts["index"] + parts["middle"] + parts["ring"] + parts["pinky"]

        dx_list = []
        dy_list = []

        for idx in all_indices:
            valid = (apple_kps[:, idx, 0] > 0) & (truth_kps[:, idx, 0] > 0)
            if np.sum(valid) > 0:
                dx = truth_kps[valid, idx, 0] - apple_kps[valid, idx, 0]
                dy = truth_kps[valid, idx, 1] - apple_kps[valid, idx, 1]
                dx_list.extend(dx)
                dy_list.extend(dy)

        if dx_list:
            print(f"  Median Offset X: {np.median(dx_list):+.2f}px")
            print(f"  Median Offset Y: {np.median(dy_list):+.2f}px")
            print(f"  Mean Offset X:   {np.mean(dx_list):+.2f}px")
            print(f"  Mean Offset Y:   {np.mean(dy_list):+.2f}px")
            print(f"  Std Dev X:       {np.std(dx_list):.2f}px")
            print(f"  Std Dev Y:       {np.std(dy_list):.2f}px")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--apple', required=True, help='Path to Apple Vision pose JSON')
    parser.add_argument('--truth', required=True, help='Path to ground truth pose JSON')
    args = parser.parse_args()

    analyze_hand_topology(args.apple, args.truth)

