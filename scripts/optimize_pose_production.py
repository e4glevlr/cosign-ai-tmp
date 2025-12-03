"""
FINAL PRODUCTION OPTIMIZATION SCRIPT

Based on experimental findings:
- Apple hands have correct topology and work well
- Apple body has structural issues that calibration alone cannot fix
- Solution: For production without ground truth, use best-effort calibration
- For best results: Consider fine-tuning the model on Apple Vision data
"""
import json
import numpy as np
import argparse

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)

    if 'frames' in data:
        frames = sorted(data['frames'], key=lambda x: x['frame_index'])
        kps = []
        scores = []
        for f in frames:
            kps.append(f['keypoints'])
            scores.append(f['scores'])
        return np.array(kps)[:, np.newaxis, :, :], np.array(scores)[:, np.newaxis, :]
    elif 'keypoints' in data:
        kps = np.array(data['keypoints'])
        if 'scores' in data:
            scores = np.array(data['scores'])
        else:
            scores = np.ones(kps.shape[:-1])
        return kps, scores
    else:
        raise ValueError("Unknown format")

def optimize_apple_pose_production(apple_path, output_path):
    """
    Production optimization for Apple Vision pose without ground truth

    Applies empirically-determined offsets from BG1_S002 analysis
    """
    print(f"Loading Apple Pose: {apple_path}")
    apple_kps, apple_scores = load_json(apple_path)

    # Fix Frame 0 if empty
    if np.sum(apple_kps[0]) == 0 and len(apple_kps) > 1:
        apple_kps[0] = apple_kps[1]
        apple_scores[0] = apple_scores[1]
        print("Fixed empty Frame 0")

    # Apply empirically-determined offsets (from BG1_S002 analysis)
    # These are median offsets that work reasonably across frames
    upper_body_indices = [0, 3, 4, 5, 6, 7, 8, 9, 10]  # Model-used indices
    face_indices = list(range(23, 91))
    left_hand_indices = list(range(91, 112))
    right_hand_indices = list(range(112, 133))

    # Offsets from detailed analysis
    offset_upper_body_x, offset_upper_body_y = 4.56, 7.64  # Conservative estimate
    offset_face_x, offset_face_y = -0.79, 1.06
    offset_left_hand_x, offset_left_hand_y = 3.07, 8.91
    offset_right_hand_x, offset_right_hand_y = 5.52, 7.36

    mask = apple_kps > 0

    # Apply offsets
    for idx in upper_body_indices:
        apple_kps[:, :, idx, 0] = np.where(mask[:, :, idx, 0],
                                            apple_kps[:, :, idx, 0] + offset_upper_body_x,
                                            apple_kps[:, :, idx, 0])
        apple_kps[:, :, idx, 1] = np.where(mask[:, :, idx, 1],
                                            apple_kps[:, :, idx, 1] + offset_upper_body_y,
                                            apple_kps[:, :, idx, 1])

    apple_kps[:, :, face_indices, 0] = np.where(mask[:, :, face_indices, 0],
                                                  apple_kps[:, :, face_indices, 0] + offset_face_x,
                                                  apple_kps[:, :, face_indices, 0])
    apple_kps[:, :, face_indices, 1] = np.where(mask[:, :, face_indices, 1],
                                                  apple_kps[:, :, face_indices, 1] + offset_face_y,
                                                  apple_kps[:, :, face_indices, 1])

    apple_kps[:, :, left_hand_indices, 0] = np.where(mask[:, :, left_hand_indices, 0],
                                                       apple_kps[:, :, left_hand_indices, 0] + offset_left_hand_x,
                                                       apple_kps[:, :, left_hand_indices, 0])
    apple_kps[:, :, left_hand_indices, 1] = np.where(mask[:, :, left_hand_indices, 1],
                                                       apple_kps[:, :, left_hand_indices, 1] + offset_left_hand_y,
                                                       apple_kps[:, :, left_hand_indices, 1])

    apple_kps[:, :, right_hand_indices, 0] = np.where(mask[:, :, right_hand_indices, 0],
                                                        apple_kps[:, :, right_hand_indices, 0] + offset_right_hand_x,
                                                        apple_kps[:, :, right_hand_indices, 0])
    apple_kps[:, :, right_hand_indices, 1] = np.where(mask[:, :, right_hand_indices, 1],
                                                        apple_kps[:, :, right_hand_indices, 1] + offset_right_hand_y,
                                                        apple_kps[:, :, right_hand_indices, 1])

    print(f"Applied calibration offsets:")
    print(f"  Upper Body: ({offset_upper_body_x:+.2f}, {offset_upper_body_y:+.2f})")
    print(f"  Face: ({offset_face_x:+.2f}, {offset_face_y:+.2f})")
    print(f"  Left Hand: ({offset_left_hand_x:+.2f}, {offset_left_hand_y:+.2f})")
    print(f"  Right Hand: ({offset_right_hand_x:+.2f}, {offset_right_hand_y:+.2f})")

    # Save
    output_data = {
        "keypoints": apple_kps.tolist(),
        "scores": apple_scores.tolist()
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f)
    print(f"Saved optimized pose: {output_path}")

    print("\n" + "="*80)
    print("IMPORTANT NOTES:")
    print("="*80)
    print("1. Calibration improves alignment but may not achieve 100% accuracy")
    print("2. Apple body keypoints have structural differences from RTMPose")
    print("3. For production accuracy, consider fine-tuning the model on Apple data")
    print("4. See PROJECT_KNOWLEDGE_BASE.md for detailed findings")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize Apple Vision pose for UniSign')
    parser.add_argument('--apple', required=True, help='Input Apple Vision pose JSON')
    parser.add_argument('--output', required=True, help='Output optimized pose JSON')
    args = parser.parse_args()

    optimize_apple_pose_production(args.apple, args.output)

