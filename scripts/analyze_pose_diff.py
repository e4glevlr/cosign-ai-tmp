import pickle
import json
import numpy as np
import pandas as pd
import sys

def analyze_poses(pkl_path, json_path):
    print(f"--- Analyzing Difference between {pkl_path} and {json_path} ---")

    # 1. Load Original Pose (PKL)
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    
    # PKL Structure usually: {'keypoints': (N, T, 133, 2), 'scores': ...} or just list of dicts
    # Let's inspect structure first
    if isinstance(pkl_data, dict):
        pkl_kps = pkl_data['keypoints']
        pkl_scores = pkl_data['scores']
    else:
        print("Unknown PKL format")
        return

    # Debug info
    print(f"Raw PKL Type: {type(pkl_kps)}")
    if isinstance(pkl_kps, list):
        print(f"Raw PKL Len: {len(pkl_kps)}")
        if len(pkl_kps) > 0:
            print(f"PKL Element 0 Type: {type(pkl_kps[0])}")
            if hasattr(pkl_kps[0], 'shape'):
                 print(f"PKL Element 0 Shape: {pkl_kps[0].shape}")
    
    # Convert to numpy - Be careful with ragged arrays
    # If list of arrays (M, 133, 2), we want to stack them
    if isinstance(pkl_kps, list):
        try:
            pkl_kps = np.stack(pkl_kps)
        except:
            print("Cannot stack PKL kps directly. Check consistency.")
            
    print(f"PKL Shape after stack: {pkl_kps.shape}")

    # Handle dimension (Person, Time, Points, Coords)
    # Expected (T, M, 133, 2) or (T, 133, 2)
    if len(pkl_kps.shape) == 4: 
        # Assume (T, M, 133, 2) -> (166, 1, 133, 2)
        # We want (T, 133, 2)
        pkl_kps = pkl_kps[:, 0, :, :]
    
    print(f"Final PKL Shape: {pkl_kps.shape}") 

    # 2. Load Apple Vision Pose (JSON)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        
    print(f"JSON Keys: {json_data.keys() if isinstance(json_data, dict) else 'List'}")
    
    # JSON Structure Handling
    if isinstance(json_data, dict) and 'frames' in json_data:
        # My generated format
        frames = sorted(json_data['frames'], key=lambda x: x['frame_index'])
        json_kps = []
        for frame in frames:
            json_kps.append(frame['keypoints']) 
        json_kps = np.array(json_kps)
    elif 'keypoints' in json_data:
         # Maybe direct format {keypoints: [...]}
         json_kps = np.array(json_data['keypoints'])
    else:
        print("Unknown JSON format")
        return
    
    # JSON might be (T, 1, 133, 2) or (T, 133, 2)
    if len(json_kps.shape) == 4:
        json_kps = json_kps[:, 0, :, :]
        
    print(f"JSON Shape (Raw): {json_kps.shape}")

    # 3. Alignment & Normalization Check
    # Assuming Video Resolution 1280x720 (Standard for this dataset)
    W, H = 1280, 720
    
    # Align Frame Counts (Truncate to min length)
    min_len = min(len(pkl_kps), len(json_kps))
    pkl_kps = pkl_kps[:min_len]
    json_kps = json_kps[:min_len]
    
    print(f"Analyzing first {min_len} frames.")

    # Convert JSON (Normalized) to Pixels
    # Check if already pixels
    json_max = np.max(json_kps)
    print(f"JSON Max Value: {json_max}")
    
    if json_max > 1.5:
        print("JSON data appears to be in Pixel Coordinates already. Skipping normalization.")
        vision_pixels = json_kps
    else:
        print("JSON data appears to be Normalized. Converting to Pixels.")
        vision_pixels = json_kps * np.array([W, H])
    
    # 4. Calculate Statistics
    
    # Define Keypoint Groups for clearer analysis
    groups = {
        "Nose (0)": [0],
        "Shoulders (5,6)": [5, 6],
        "Wrists (9,10)": [9, 10],
        "Left Hand Root (91)": [91],
        "Right Hand Root (112)": [112]
    }
    
    results = []
    
    for name, indices in groups.items():
        # Extract points for this group
        # pkl: (T, N_indices, 2)
        p_ref = pkl_kps[:, indices, :]
        p_hyp = vision_pixels[:, indices, :]
        
        # Calculate Diff
        diff = p_hyp - p_ref
        
        # Mean Error per axis
        mean_diff_x = np.mean(diff[:, :, 0])
        mean_diff_y = np.mean(diff[:, :, 1])
        
        # Euclidean Distance
        dist = np.linalg.norm(diff, axis=2)
        mean_dist = np.mean(dist)
        
        results.append({
            "Part": name,
            "Mean Dist (px)": f"{mean_dist:.2f}",
            "Offset X (px)": f"{mean_diff_x:.2f}",
            "Offset Y (px)": f"{mean_diff_y:.2f}",
            "Ref X (sample)": f"{p_ref[0,0,0]:.1f}",
            "Hyp X (sample)": f"{p_hyp[0,0,0]:.1f}",
            "Ref Y (sample)": f"{p_ref[0,0,1]:.1f}",
            "Hyp Y (sample)": f"{p_hyp[0,0,1]:.1f}"
        })

    df = pd.DataFrame(results)
    print("\n--- Comparison Results ---")
    print(df.to_string(index=False))
    
    # 5. Deep Dive on Y-Axis (Check flip correctness)
    # If Offset Y is massive (e.g., ~700px), then flip logic is wrong.
    # If Offset Y is moderate but negative/positive, it's a shift.
    
    print("\n--- Diagnosis ---")
    nose_diff_y = float(results[0]["Offset Y (px)"])
    if abs(nose_diff_y) > 300:
        print("CRITICAL: Y-axis seems inverted or scaled incorrectly.")
    elif abs(nose_diff_y) > 50:
        print("WARNING: Significant Y-axis shift detected. Calibration needed.")
    else:
        print("SUCCESS: Y-axis orientation looks correct. Minor fine-tuning needed.")

if __name__ == "__main__":
    # Ensure files exist
    import os
    if not os.path.exists("BG1_S014_pose.pkl"):
        print("Error: BG1_S014_pose.pkl not found.")
    elif not os.path.exists("BG1_S014_pose.json"):
        # Try fallback to output_pose.json if the user renamed it in prompt but not file
        if os.path.exists("output_pose.json"):
            print("Using output_pose.json as BG1_S014_pose.json not found.")
            analyze_poses("BG1_S014_pose.pkl", "output_pose.json")
        else:
             print("Error: JSON pose file not found.")
    else:
        # Default to comparing PKL with the generated output_pose.json
        analyze_poses("BG1_S014_pose.pkl", "output_pose.json")
