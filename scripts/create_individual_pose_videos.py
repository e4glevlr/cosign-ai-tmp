import cv2
import json
import pickle
import numpy as np
import sys

def draw_skeleton(img, kps, color, thickness=2):
    # kps: (133, 2)
    # Draw Points
    for i in range(kps.shape[0]):
        x, y = int(kps[i, 0]), int(kps[i, 1])
        # Only draw if point is not (0,0) and within image bounds
        if x == 0 and y == 0: continue
        if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]: continue
        cv2.circle(img, (x, y), 3, color, -1)

    # Define connections (Simplified COCO-WholeBody subset for visualization)
    # This might not perfectly match how Uni-Sign connects, but for visual comparison, it's enough.
    body_links = [
        # Torso
        (5,6), # Shoulders
        (5,11), (6,12), # Shoulders to Hips
        (11,12), # Hips
        (0,1), (1,2), (2,3), (3,4), # Face (approx)
        (0,5), (0,6), # Neck to shoulders
        # Arms
        (5,7), (7,9), # Left Arm
        (6,8), (8,10), # Right Arm
        # Legs (if available)
        (11,13), (13,15), # Left Leg
        (12,14), (14,16) # Right Leg
    ]
    
    for i, j in body_links:
        if i >= len(kps) or j >= len(kps): continue
        if kps[i,0] == 0 and kps[i,1] == 0: continue # Skip if start point is (0,0)
        if kps[j,0] == 0 and kps[j,1] == 0: continue # Skip if end point is (0,0)
        
        pt1 = (int(kps[i,0]), int(kps[i,1]))
        pt2 = (int(kps[j,0]), int(kps[j,1]))
        cv2.line(img, pt1, pt2, color, thickness)

    # Hands (Left: 91-111, Right: 112-132)
    # Standard COCO Hand mapping: 0(Wrist) -> 1-4(Thumb) -> 5-8(Index)...
    # Our kps indices are already COCO-WholeBody style
    
    def draw_hand(start_idx, wrist_body_idx):
        # Wrist to finger roots
        # wrist_body_idx is for the connection to the body (e.g., kps[9] for left wrist)
        # However, for hand drawing, 91 is the start of the hand KPs.
        # kps[start_idx] is the wrist of the hand
        
        finger_segments = [
            (0,1), (1,2), (2,3), (3,4), # Thumb (start_idx to start_idx+4)
            (5,6), (6,7), (7,8), # Index (start_idx+5 to start_idx+8)
            (9,10), (10,11), (11,12), # Middle
            (13,14), (14,15), (15,16), # Ring
            (17,18), (18,19), (19,20) # Little
        ]
        
        for i_offset, j_offset in finger_segments:
            i, j = start_idx + i_offset, start_idx + j_offset
            if i >= len(kps) or j >= len(kps): continue # Ensure index is valid
            if kps[i,0] == 0 and kps[i,1] == 0: continue
            if kps[j,0] == 0 and kps[j,1] == 0: continue
            
            pt1 = (int(kps[i,0]), int(kps[i,1]))
            pt2 = (int(kps[j,0]), int(kps[j,1]))
            cv2.line(img, pt1, pt2, color, 1)

        # Connection from body wrist to hand wrist
        # if kps[wrist_body_idx,0] != 0 and kps[start_idx,0] != 0:
        #     cv2.line(img, (int(kps[wrist_body_idx,0]), int(kps[wrist_body_idx,1])), (int(kps[start_idx,0]), int(kps[start_idx,1])), color, thickness)

    draw_hand(91, 9) # Left Hand, connects to body wrist 9
    draw_hand(112, 10) # Right Hand, connects to body wrist 10

def create_pose_video(video_path, pose_frames, output_path, pose_color, legend_text):
    print(f"Creating video: {output_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx < len(pose_frames):
            draw_skeleton(frame, pose_frames[frame_idx], pose_color, 2)
            cv2.putText(frame, legend_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pose_color, 2)
        
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"Saved video to {output_path}")

def create_individual_pose_videos(video_path, pkl_path, json_path):
    print(f"--- Preparing data for individual pose videos ---")

    # 1. Load Original Pose (PKL)
    with open(pkl_path, 'rb') as f:
        pkl_raw = pickle.load(f)
    pkl_kps = np.array(pkl_raw['keypoints']) 
    pkl_kps = pkl_kps[:, 0, :, :] # (T, 133, 2)
    
    # 2. Load Apple Vision Pose (JSON)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    frames_json_sorted = sorted(json_data['frames'], key=lambda x: x['frame_index'])
    W, H = json_data.get('width', 1280), json_data.get('height', 720)
    
    json_kps_processed = []
    for frame in frames_json_sorted:
        kp = np.array(frame['keypoints'])
        
        # Check if normalized (max <= 1.5) and de-normalize
        if np.max(kp) <= 1.5:
            kp = kp * np.array([W, H])
        
        # Apply Calibration (+6.5, +8.0)
        kp[..., 0] += 6.5
        kp[..., 1] += 8.0
            
        json_kps_processed.append(kp)

    # Align Frame Counts
    min_len = min(len(pkl_kps), len(json_kps_processed))
    pkl_kps = pkl_kps[:min_len]
    json_kps_processed = json_kps_processed[:min_len]

    # Create individual videos
    create_pose_video(video_path, json_kps_processed, "apple_vision_viz.mp4", (0, 255, 0), "Apple Vision (Calibrated)")
    create_pose_video(video_path, pkl_kps, "original_model_viz.mp4", (0, 0, 255), "Original Model (Ground Truth)")

if __name__ == "__main__":
    create_individual_pose_videos("BG1_S014_pose.avi", "BG1_S014_pose.pkl", "output_pose.json")
