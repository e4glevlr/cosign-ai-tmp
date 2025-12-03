import gradio as gr
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
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from models import Uni_Sign
from datasets import S2T_Dataset_online
import utils as utils
from config import *

# --- Global Variables for Model ---
model = None
device = None
args = None

def setup_model():
    global model, device, args
    
    print("Initializing Model...")
    
    # Mimic the arguments required by Uni-Sign
    parser = argparse.ArgumentParser('Uni-Sign Inference', parents=[utils.get_args_parser()])
    # Add defaults manually if needed, but get_args_parser should cover most
    
    # Fake args object with defaults
    args, _ = parser.parse_known_args([])
    args.hidden_dim = 256
    args.rgb_support = False # Default to false for pure pose inference
    args.dataset = 'CSL_Daily'
    args.max_length = 300
    args.label_smoothing = 0.1
    
    # Force device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Model
    model = Uni_Sign(args=args)
    model.to(device)
    model.eval()
    
    # Load Checkpoint
    checkpoint_path = "pretrained_weight/best_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)['model']
        ret = model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded. Missing keys: {len(ret.missing_keys)}")
    else:
        print("WARNING: No checkpoint found at 'pretrained_weight/best_checkpoint.pth'. Model will use random weights.")

# Call setup once
setup_model()

def process_pipeline(video_path):
    global model, device, args
    
    if video_path is None:
        return "Please upload a video."
    
    # 1. Prepare Paths
    unique_id = str(uuid.uuid4())[:8]
    temp_dir = Path("temp_inference")
    temp_dir.mkdir(exist_ok=True)
    
    # Input video path (gradio provides a temp path usually)
    input_path = Path(video_path)
    output_mov = temp_dir / f"out_{unique_id}.mov"
    output_json = temp_dir / f"pose_{unique_id}.json"
    
    print(f"Processing video: {input_path}")
    
    # 2. Run PoseExtractor (Swift)
    # Command: ./PoseExtractor <input> <output_mov> <output_json>
    cmd = [
        "./bin/PoseExtractor",
        str(input_path.absolute()),
        str(output_mov.absolute()),
        str(output_json.absolute())
    ]
    
    print(f"Running Swift Extractor: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Swift Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Swift Error:", e.stderr)
        return f"Error during pose extraction: {e.stderr}"
    
    if not output_json.exists():
        return "Error: Pose JSON file was not created."
        
    # 3. Load Pose JSON
    try:
        with open(output_json, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return f"Error reading JSON: {e}"
        
    # width = data.get('width', 1920)
    # height = data.get('height', 1080)
    # print(f"De-normalizing UI data using video size: {width}x{height}")
        
    keypoints = []
    scores = []
    frames = sorted(data['frames'], key=lambda x: x['frame_index'])
    
    if len(frames) == 0:
        return "Error: No frames detected in video."

    for frame in frames:
        kp = np.array(frame['keypoints'], dtype=np.float32) # (133, 2)
        sc = np.array(frame['scores'], dtype=np.float32)    # (133,)
        
        # De-normalize: REMOVED - Swift output is already Global Pixels
        # kp = kp * np.array([width, height], dtype=np.float32)
        
        # Calibration
        kp[..., 0] += 6.5
        kp[..., 1] += 8.0
        
        # Add Person Dimension (T, 1, 133, C)
        kp = kp[np.newaxis, ...]
        sc = sc[np.newaxis, ...]
        
        keypoints.append(kp)
        scores.append(sc)
        
    pose_data = {"keypoints": keypoints, "scores": scores}
    
    # 4. Create Dataset & Dataloader
    # Hack: S2T_Dataset_online needs 'args' to have 'rgb_support'
    # We pass the global args we created
    
    online_data = S2T_Dataset_online(args=args)
    online_data.rgb_data = str(input_path) # Placeholder, not used if rgb_support=False
    online_data.pose_data = pose_data
    
    online_dataloader = DataLoader(
        online_data,
        batch_size=1,
        collate_fn=online_data.collate_fn,
        sampler=torch.utils.data.SequentialSampler(online_data)
    )
    
    # 5. Inference
    print("Running Inference...")
    target_dtype = torch.float32
    result_text = ""
    
    try:
        with torch.no_grad():
            tgt_pres = []
            for step, (src_input, tgt_input) in enumerate(online_dataloader):
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
                result_text = "No translation generated."
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Inference Error: {e}"
    
    # Cleanup Temp Files (Optional - maybe keep for debug)
    # output_mov.unlink(missing_ok=True)
    # output_json.unlink(missing_ok=True)
    
    return result_text

# --- Gradio Interface ---
with gr.Blocks(title="CoSign AI - Sign Language Translation") as demo:
    gr.Markdown("# 🤟 CoSign AI Demo")
    gr.Markdown("Upload a video containing sign language to translate it into text.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Input Video", sources=["upload"])
            submit_btn = gr.Button("Translate", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="Translation Result", lines=4)
            
    submit_btn.click(
        fn=process_pipeline,
        inputs=[video_input],
        outputs=[output_text]
    )

    gr.Markdown("---")
    gr.Markdown(f"Running on device: {device}")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
