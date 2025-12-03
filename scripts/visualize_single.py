import cv2
import json
import numpy as np
import argparse

def draw_skeleton(img, kps, color, thickness=2):
    # kps: (133, 2)
    # Draw Points
    for i in range(kps.shape[0]):
        x, y = int(kps[i, 0]), int(kps[i, 1])
        if x == 0 and y == 0: continue
        cv2.circle(img, (x, y), 3, color, -1)

    # Define connections (Simplified COCO-WholeBody subset)
    # Body
    body_links = [
        (0,1), (1,2), (2,3), (3,4), # Face contour approx
        (5,6), (5,7), (7,9), (6,8), (8,10), # Arms
        (11,12), (5,11), (6,12) # Torso
    ]
    
    for i, j in body_links:
        if i >= len(kps) or j >= len(kps): continue
        if kps[i,0] == 0 or kps[j,0] == 0: continue
        pt1 = (int(kps[i,0]), int(kps[i,1]))
        pt2 = (int(kps[j,0]), int(kps[j,1]))
        cv2.line(img, pt1, pt2, color, thickness)

    # Hands (Left: 91-111, Right: 112-132)
    def draw_hand(start_idx):
        wrist = start_idx
        finger_roots = [start_idx+1, start_idx+5, start_idx+9, start_idx+13, start_idx+17]
        for root in finger_roots:
             if kps[wrist,0] != 0 and kps[root,0] != 0:
                cv2.line(img, (int(kps[wrist,0]), int(kps[wrist,1])), (int(kps[root,0]), int(kps[root,1])), color, 1)
        
        for finger_base in finger_roots:
            for k in range(3):
                i, j = finger_base + k, finger_base + k + 1
                if kps[i,0] != 0 and kps[j,0] != 0:
                    cv2.line(img, (int(kps[i,0]), int(kps[i,1])), (int(kps[j,0]), int(kps[j,1])), color, 1)

    draw_hand(91)
    draw_hand(112)

def load_kps_from_any_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    if 'frames' in data:
        frames = sorted(data['frames'], key=lambda x: x['frame_index'])
        kps_list = []
        for f in frames:
            k = np.array(f['keypoints'])
            # If Apple Raw (normalized), de-normalize?
            # Assuming this script is used for Optimized/Truth files which are already pixel coords.
            kps_list.append(k)
        return kps_list
    elif 'keypoints' in data:
        # (T, 1, 133, 2) -> list of (133, 2)
        kps = np.array(data['keypoints'])
        return [kps[i, 0] for i in range(kps.shape[0])]
    else:
        raise ValueError("Unknown JSON format")

def visualize_single(video_path, json_path, output_path, label, color_str):
    print(f"Visualizing: {label} from {json_path}")
    
    frames_data = load_kps_from_any_json(json_path)
    
    # Color parsing
    if color_str == 'red': color = (0, 0, 255)
    elif color_str == 'green': color = (0, 255, 0)
    else: color = (255, 0, 0) # Blue

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
        
        if frame_idx < len(frames_data):
            draw_skeleton(frame, frames_data[frame_idx], color, 2)
        
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--json', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--color', default='green')
    args = parser.parse_args()
    
    visualize_single(args.video, args.json, args.output, args.label, args.color)
