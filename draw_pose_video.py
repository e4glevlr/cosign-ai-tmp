import sys
import os
import cv2
import numpy as np

# Add rtmlib to python path
sys.path.append(os.path.join(os.getcwd(), 'demo/rtmlib-main'))

from rtmlib import Wholebody, draw_skeleton

def main():
    video_path = 'BG1_S014.mp4'
    output_path = 'BG1_S014_pose.avi'
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return

    # Initialize Wholebody model
    # Using 'cpu' and 'onnxruntime' as seen in the demo, but can be adjusted if cuda is available
    # The user environment seems to have cuda-keyring, so maybe cuda is available, 
    # but let's stick to a safe default or try to detect. 
    # The demo uses device='cuda' by default in pose_extraction.py, but 'cpu' in wholebody_demo.py.
    # I'll use 'cpu' to be safe unless I see explicit instructions otherwise, or I can try 'cuda' if available.
    # Given the environment details show cuda-keyring, I'll try to use what's likely available, 
    # but 'cpu' is safest for a script that needs to run without error first.
    # However, pose_extraction.py defaults to cuda. Let's stick to cpu for reliability in this context unless specified.
    
    device = 'cpu'
    backend = 'onnxruntime'
    
    print(f"Initializing Wholebody model on {device}...")
    wholebody = Wholebody(
        to_openpose=False,
        mode='balanced', # balanced seems like a good middle ground
        backend=backend,
        device=device
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Setup VideoWriter
    # Using MJPG for maximum compatibility since H.264 (avc1) is not available in this environment
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{total_frames}...")

        # Extract poses
        keypoints, scores = wholebody(frame)

        # Draw poses
        # kpt_thr=0.3 is a common threshold, using 0.4 as seen in demo
        img_show = draw_skeleton(frame, keypoints, scores, kpt_thr=0.4)

        out.write(img_show)

    cap.release()
    out.release()
    print(f"Done! Output saved to {output_path}")

if __name__ == "__main__":
    main()