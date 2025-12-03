import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import json
import sys
import os
import argparse
import numpy as np
import subprocess
import uuid
import time
import cv2
import csv
from pathlib import Path
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from models import Uni_Sign
from datasets import S2T_Dataset_online
import utils as utils
from config import *

# --- Global Variables ---
model = None
device = None
args = None

def setup_model():
    global model, device, args
    print("Initializing Model...")
    
    parser = argparse.ArgumentParser('Uni-Sign Inference', parents=[utils.get_args_parser()])
    args, _ = parser.parse_known_args([])
    args.hidden_dim = 256
    args.rgb_support = False
    args.dataset = 'CSL_Daily'
    args.max_length = 300
    args.label_smoothing = 0.1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Uni_Sign(args=args)
    model.to(device)
    model.eval()
    
    checkpoint_path = "pretrained_weight/best_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)['model']
        model.load_state_dict(state_dict, strict=False)
    else:
        print("WARNING: Checkpoint not found! Results will be random.")

def get_video_duration(video_path):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()
        return duration
    except:
        return 0.0

def run_inference_on_file(video_path, temp_dir):
    global model, device, args
    
    input_path = Path(video_path)
    unique_id = str(uuid.uuid4())[:8]
    output_mov = temp_dir / f"temp_{unique_id}.mov"
    output_json = temp_dir / f"temp_{unique_id}.json"
    
    # 1. Pose Extraction (Swift)
    cmd = [
        "./PoseExtractor",
        str(input_path.absolute()),
        str(output_mov.absolute()),
        str(output_json.absolute())
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        return f"Error: Pose Extraction Failed"

    if not output_json.exists():
        return "Error: JSON not created"

    # 2. Load JSON & Preprocess
    try:
        with open(output_json, 'r') as f:
            data = json.load(f)
    except:
        return "Error: Read JSON Failed"
        
    keypoints = []
    scores = []
    frames = sorted(data['frames'], key=lambda x: x['frame_index'])
    
    if len(frames) == 0:
        return "Error: No Frames"

    for frame in frames:
        kp = np.array(frame['keypoints'], dtype=np.float32) # (133, 2)
        sc = np.array(frame['scores'], dtype=np.float32)    # (133,)
        
        # Logic Update: Swift now returns Global Pixels. No de-norm needed.
        # Calibration Only
        kp[..., 0] += 6.5
        kp[..., 1] += 8.0
        
        kp = kp[np.newaxis, ...] # (1, 133, 2)
        sc = sc[np.newaxis, ...] # (1, 133)
        
        keypoints.append(kp)
        scores.append(sc)
        
    pose_data = {"keypoints": keypoints, "scores": scores}
    
    # 3. Inference
    online_data = S2T_Dataset_online(args=args)
    online_data.rgb_data = str(input_path)
    online_data.pose_data = pose_data
    
    dataloader = DataLoader(
        online_data, batch_size=1, 
        collate_fn=online_data.collate_fn,
        sampler=torch.utils.data.SequentialSampler(online_data)
    )
    
    result_text = ""
    try:
        with torch.no_grad():
            tgt_pres = []
            for step, (src_input, tgt_input) in enumerate(dataloader):
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(device)

                stack_out = model(src_input, tgt_input)
                output = model.generate(stack_out, max_new_tokens=50, num_beams=4)
                for i in range(len(output)):
                    tgt_pres.append(output[i])
            
            if len(tgt_pres) > 0:
                tokenizer = model.mt5_tokenizer
                padding_value = tokenizer.eos_token_id
                pad_tensor = torch.ones(150 - len(tgt_pres[0])).to(device) * padding_value
                tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)
                tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
                decoded = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)
                result_text = decoded[0]
            else:
                result_text = ""
    except Exception as e:
        return f"Error: Inference - {str(e)}"
    
    # Cleanup
    if output_mov.exists(): output_mov.unlink()
    if output_json.exists(): output_json.unlink()
    
    return result_text

def main():
    setup_model()
    
    input_dir = "test"
    output_csv = "inference_results.csv"
    temp_dir = Path("temp_batch_inference")
    temp_dir.mkdir(exist_ok=True)
    
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, file))
    
    print(f"Found {len(video_files)} videos.")
    
    results = []
    
    for video_path in tqdm(video_files, desc="Processing Videos"):
        start_time = time.time()
        
        vid_duration = get_video_duration(video_path)
        prediction = run_inference_on_file(video_path, temp_dir)
        
        end_time = time.time()
        infer_time = end_time - start_time
        
        results.append({
            "File Name": os.path.basename(video_path),
            "Prediction": prediction,
            "Video Duration (s)": round(vid_duration, 2),
            "Inference Time (s)": round(infer_time, 2)
        })
        
    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["File Name", "Prediction", "Video Duration (s)", "Inference Time (s)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(f"Done! Results saved to {output_csv}")
    # Remove temp dir if empty
    try:
        temp_dir.rmdir()
    except:
        pass

if __name__ == "__main__":
    main()
