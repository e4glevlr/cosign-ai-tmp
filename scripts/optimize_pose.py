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

def interpolate_and_smooth(kps):
    # kps: (T, 1, 133, 2)
    T, M, N, C = kps.shape
    kps_flat = kps.reshape(T, N, C) 
    
    for n in range(N):
        series = kps_flat[:, n, :]
        valid_mask = (series[:, 0] != 0) | (series[:, 1] != 0)
        valid_idx = np.where(valid_mask)[0]
        
        if len(valid_idx) == 0: continue
        if len(valid_idx) < T:
            kps_flat[:, n, 0] = np.interp(np.arange(T), valid_idx, series[valid_idx, 0])
            kps_flat[:, n, 1] = np.interp(np.arange(T), valid_idx, series[valid_idx, 1])
            
    # Moving Average Smoothing
    window_size = 3
    kernel = np.ones(window_size) / window_size
    
    for n in range(N):
        for c in range(C):
            series = kps_flat[:, n, c]
            padded = np.pad(series, (window_size//2, window_size//2), mode='edge')
            smoothed = np.convolve(padded, kernel, mode='valid')
            kps_flat[:, n, c] = smoothed
            
    return kps_flat.reshape(T, M, N, C)

def optimize_pose(apple_path, output_path, truth_path=None):
    print(f"Loading Apple Pose: {apple_path}")
    apple_kps, apple_scores = load_json(apple_path)
    
    # Default Calibration (calibrated from BG1_S002)
    # Focus on upper body joints actually used by model: [0, 3-10]
    offset_upper_body_x, offset_upper_body_y = 4.92, 7.64  # Head average
    offset_face_x, offset_face_y = -0.79, 1.06
    offset_left_hand_x, offset_left_hand_y = 3.07, 8.91
    offset_right_hand_x, offset_right_hand_y = 5.52, 7.36

    if truth_path:
        print(f"Loading Truth Pose for Calibration: {truth_path}")
        truth_kps, truth_scores = load_json(truth_path)
        min_len = min(len(apple_kps), len(truth_kps))
        
        # Model uses these body indices: 0, 3, 4, 5, 6, 7, 8, 9, 10
        # (Nose, Ears, Shoulders, Elbows, Wrists)
        upper_body_indices = [0, 3, 4, 5, 6, 7, 8, 9, 10]

        ub_diffs_x, ub_diffs_y = [], []
        f_diffs_x, f_diffs_y = [], []
        lh_diffs_x, lh_diffs_y = [], []
        rh_diffs_x, rh_diffs_y = [], []

        for t in range(min_len):
            # Upper Body (only indices used by model)
            for idx in upper_body_indices:
                a_pt = apple_kps[t, 0, idx, :]
                t_pt = truth_kps[t, 0, idx, :]
                if a_pt[0] > 0 and t_pt[0] > 0:
                    ub_diffs_x.append(t_pt[0] - a_pt[0])
                    ub_diffs_y.append(t_pt[1] - a_pt[1])

            # Face
            a_f = apple_kps[t, 0, 23:91, :]
            t_f = truth_kps[t, 0, 23:91, :]
            valid_f = (a_f[:,0] > 0) & (t_f[:,0] > 0)
            if np.sum(valid_f) > 10:
                d = t_f[valid_f] - a_f[valid_f]
                f_diffs_x.extend(d[:,0])
                f_diffs_y.extend(d[:,1])

            # Left Hand
            a_lh = apple_kps[t, 0, 91:112, :]
            t_lh = truth_kps[t, 0, 91:112, :]
            valid_lh = (a_lh[:,0] > 0) & (t_lh[:,0] > 0)
            if np.sum(valid_lh) > 5:
                d = t_lh[valid_lh] - a_lh[valid_lh]
                lh_diffs_x.extend(d[:,0])
                lh_diffs_y.extend(d[:,1])

            # Right Hand
            a_rh = apple_kps[t, 0, 112:133, :]
            t_rh = truth_kps[t, 0, 112:133, :]
            valid_rh = (a_rh[:,0] > 0) & (t_rh[:,0] > 0)
            if np.sum(valid_rh) > 5:
                d = t_rh[valid_rh] - a_rh[valid_rh]
                rh_diffs_x.extend(d[:,0])
                rh_diffs_y.extend(d[:,1])

        if ub_diffs_x:
            offset_upper_body_x, offset_upper_body_y = np.median(ub_diffs_x), np.median(ub_diffs_y)
        if f_diffs_x:
            offset_face_x, offset_face_y = np.median(f_diffs_x), np.median(f_diffs_y)
        if lh_diffs_x:
            offset_left_hand_x, offset_left_hand_y = np.median(lh_diffs_x), np.median(lh_diffs_y)
        if rh_diffs_x:
            offset_right_hand_x, offset_right_hand_y = np.median(rh_diffs_x), np.median(rh_diffs_y)

    print(f"Upper Body Offset (0,3-10): X={offset_upper_body_x:.2f}, Y={offset_upper_body_y:.2f}")
    print(f"Face Offset: X={offset_face_x:.2f}, Y={offset_face_y:.2f}")
    print(f"Left Hand Offset: X={offset_left_hand_x:.2f}, Y={offset_left_hand_y:.2f}")
    print(f"Right Hand Offset: X={offset_right_hand_x:.2f}, Y={offset_right_hand_y:.2f}")

    # Fix Frame 0
    if np.sum(apple_kps[0]) == 0 and len(apple_kps) > 1:
        apple_kps[0] = apple_kps[1]
        apple_scores[0] = apple_scores[1]

    # Apply Offsets Separately
    mask = apple_kps > 0
    
    # Define index ranges - only calibrate what the model uses
    upper_body_indices = [0, 3, 4, 5, 6, 7, 8, 9, 10]  # Nose, Ears, Shoulders, Elbows, Wrists
    face_indices = list(range(23, 91))  # Face landmarks
    left_hand_indices = list(range(91, 112))  # Left hand
    right_hand_indices = list(range(112, 133))  # Right hand

    # Apply Upper Body Offset (only on joints model uses)
    for idx in upper_body_indices:
        apple_kps[:, :, idx, 0] = np.where(mask[:, :, idx, 0], apple_kps[:, :, idx, 0] + offset_upper_body_x, apple_kps[:, :, idx, 0])
        apple_kps[:, :, idx, 1] = np.where(mask[:, :, idx, 1], apple_kps[:, :, idx, 1] + offset_upper_body_y, apple_kps[:, :, idx, 1])

    # Apply Face Offset
    apple_kps[:, :, face_indices, 0] = np.where(mask[:, :, face_indices, 0], apple_kps[:, :, face_indices, 0] + offset_face_x, apple_kps[:, :, face_indices, 0])
    apple_kps[:, :, face_indices, 1] = np.where(mask[:, :, face_indices, 1], apple_kps[:, :, face_indices, 1] + offset_face_y, apple_kps[:, :, face_indices, 1])

    # Apply Left Hand Offset
    apple_kps[:, :, left_hand_indices, 0] = np.where(mask[:, :, left_hand_indices, 0], apple_kps[:, :, left_hand_indices, 0] + offset_left_hand_x, apple_kps[:, :, left_hand_indices, 0])
    apple_kps[:, :, left_hand_indices, 1] = np.where(mask[:, :, left_hand_indices, 1], apple_kps[:, :, left_hand_indices, 1] + offset_left_hand_y, apple_kps[:, :, left_hand_indices, 1])

    # Apply Right Hand Offset
    apple_kps[:, :, right_hand_indices, 0] = np.where(mask[:, :, right_hand_indices, 0], apple_kps[:, :, right_hand_indices, 0] + offset_right_hand_x, apple_kps[:, :, right_hand_indices, 0])
    apple_kps[:, :, right_hand_indices, 1] = np.where(mask[:, :, right_hand_indices, 1], apple_kps[:, :, right_hand_indices, 1] + offset_right_hand_y, apple_kps[:, :, right_hand_indices, 1])

    # Smoothing
    # Smoothing (Moving Average) was found to degrade Hand details ("Ngứa da" vs "Hoả hoạn").
    # Disabled for now.
    # apple_kps = interpolate_and_smooth(apple_kps)

    output_data = {
        "keypoints": apple_kps.tolist(),
        "scores": apple_scores.tolist()
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--apple', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--truth', required=False)
    args = parser.parse_args()
    optimize_pose(args.apple, args.output, args.truth)