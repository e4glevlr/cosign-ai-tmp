import cv2
import json
import pickle
import numpy as np
# import matplotlib.pyplot as plt

def draw_hand_structure(img, kps, offset_idx, color, label_prefix):
    # kps: (133, 2) array
    # Hand indices relative to start: 0(Wrist), 1-4(Thumb), 5-8(Index), 9-12(Middle), 13-16(Ring), 17-20(Little)
    
    wrist = kps[offset_idx]
    if wrist[0] == 0 and wrist[1] == 0: return # No hand detected

    # Define fingers
    fingers = [
        list(range(offset_idx + 1, offset_idx + 5)),   # Thumb
        list(range(offset_idx + 5, offset_idx + 9)),   # Index
        list(range(offset_idx + 9, offset_idx + 13)),  # Middle
        list(range(offset_idx + 13, offset_idx + 17)), # Ring
        list(range(offset_idx + 17, offset_idx + 21))  # Little
    ]
    
    # Draw Wrist
    cv2.circle(img, (int(wrist[0]), int(wrist[1])), 5, color, -1)
    cv2.putText(img, "W", (int(wrist[0])+5, int(wrist[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw Fingers
    for f_idx, finger in enumerate(fingers):
        # Connect Wrist to Finger Root
        root = kps[finger[0]]
        if root[0] != 0:
            cv2.line(img, (int(wrist[0]), int(wrist[1])), (int(root[0]), int(root[1])), color, 1)
        
        # Connect joints
        for i in range(len(finger) - 1):
            p1 = kps[finger[i]]
            p2 = kps[finger[i+1]]
            if p1[0] != 0 and p2[0] != 0:
                cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)
                cv2.circle(img, (int(p2[0]), int(p2[1])), 3, color, -1)
                
        # Label the Tip
        tip = kps[finger[-1]]
        if tip[0] != 0:
            cv2.putText(img, f"{f_idx+1}", (int(tip[0])+5, int(tip[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def analyze_hand_structure(pkl_path, json_path, output_img):
    print("Analyzing Hand Structure...")
    
    # 1. Load PKL (Original)
    with open(pkl_path, 'rb') as f:
        pkl_raw = pickle.load(f)
    pkl_kps = np.array(pkl_raw['keypoints'])
    pkl_kps = pkl_kps[:, 0, :, :] # (T, 133, 2)
    
    # 2. Load JSON (Apple Vision)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    frames_json = sorted(json_data['frames'], key=lambda x: x['frame_index'])
    
    # NOTE: JSON is now Global Pixels in Swift.
    # But we applied +6.5, +8.0 manually in Python scripts before.
    # Here we should apply it to match what the Inference sees.
    json_kps_processed = []
    for frame in frames_json:
        kp = np.array(frame['keypoints'])
        # Apply Calibration
        kp[..., 0] += 6.5
        kp[..., 1] += 8.0
        json_kps_processed.append(kp)
    
    json_kps_processed = np.array(json_kps_processed)

    # 3. Find a good frame (where both have hands)
    target_frame = -1
    for i in range(min(len(pkl_kps), len(json_kps_processed))):
        # Check Left Hand Wrist (91)
        p_ref = pkl_kps[i, 91]
        p_hyp = json_kps_processed[i, 91]
        if p_ref[0] > 0 and p_hyp[0] > 0:
            target_frame = i
            break
            
    if target_frame == -1:
        print("Could not find a frame with overlapping hands.")
        return

    print(f"Analyzing Frame: {target_frame}")
    
    # 4. Draw Comparison
    # Create a canvas (White background)
    H, W = 720, 1280
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    
    ref_kps = pkl_kps[target_frame]
    hyp_kps = json_kps_processed[target_frame]
    
    # Draw Left Hand (91)
    draw_hand_structure(canvas, ref_kps, 91, (0, 0, 255), "Ref") # Red
    draw_hand_structure(canvas, hyp_kps, 91, (0, 255, 0), "Hyp") # Green
    
    # Draw Right Hand (112)
    draw_hand_structure(canvas, ref_kps, 112, (0, 0, 255), "Ref") # Red
    draw_hand_structure(canvas, hyp_kps, 112, (0, 255, 0), "Hyp") # Green
    
    # Crop to hands area to see details better? No, full image is fine for context.
    
    # Add Legend
    cv2.putText(canvas, f"Frame {target_frame}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(canvas, "Red: Original (Reference)", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(canvas, "Green: Apple Vision (Hypothesis)", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(canvas, "Labels: W=Wrist, 1=Thumb...5=Little", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

    cv2.imwrite(output_img, canvas)
    print(f"Saved analysis image to {output_img}")

if __name__ == "__main__":
    analyze_hand_structure("BG1_S014_pose.pkl", "output_BG1_S014.json", "hand_comparison.png")
