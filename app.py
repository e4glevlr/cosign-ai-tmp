import gradio as gr
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import sys
import os
import subprocess
import uuid
from pathlib import Path
import json
import numpy as np

# Add src and scripts to path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))

# Import Model Components
from models import Uni_Sign
from datasets import S2T_Dataset_online
import utils as utils
from config import *
# Import Inference Logic directly to avoid subprocess overhead for model
from run_json_inference import load_pose_from_json

# --- Global Model Cache ---
MODEL = None
DEVICE = None
ARGS = None

def load_model():
    global MODEL, DEVICE, ARGS
    if MODEL is not None:
        return

    print("Loading Uni-Sign Model...")
    
    # Setup Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup Mock Args (matching training config)
    parser = argparse.ArgumentParser('Uni-Sign Inference', parents=[utils.get_args_parser()])
    ARGS, _ = parser.parse_known_args([])
    # Overrides
    ARGS.hidden_dim = 256 # Assuming default from config/training
    ARGS.rgb_support = False
    ARGS.dataset = 'CSL_Daily'
    
    # Initialize Model
    MODEL = Uni_Sign(args=ARGS)
    MODEL.to(DEVICE)
    MODEL.eval()
    
    # Load Weights
    checkpoint_path = "pretrained_weight/best_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        if 'model' in checkpoint:
            MODEL.load_state_dict(checkpoint['model'], strict=False)
        else:
            MODEL.load_state_dict(checkpoint, strict=False)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

def run_inference_logic(pose_data):
    global MODEL, DEVICE, ARGS
    
    # Create Dataset
    online_data = S2T_Dataset_online(args=ARGS)
    online_data.pose_data = pose_data
    # rgb_data is not needed for pose-only
    
    dataloader = DataLoader(
        online_data,
        batch_size=1,
        collate_fn=online_data.collate_fn,
        sampler=torch.utils.data.SequentialSampler(online_data)
    )
    
    # Inference Loop
    try:
        with torch.no_grad():
            tgt_pres = []
            for step, (src_input, tgt_input) in enumerate(dataloader):
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(DEVICE)

                stack_out = MODEL(src_input, tgt_input)
                output = MODEL.generate(stack_out, max_new_tokens=50, num_beams=4)

                for i in range(len(output)):
                    tgt_pres.append(output[i])
            
            if not tgt_pres:
                return "No output generated."

            tokenizer = MODEL.mt5_tokenizer
            padding_value = tokenizer.eos_token_id
            
            # Pad and Decode
            pad_tensor = torch.ones(150 - len(tgt_pres[0])).to(DEVICE) * padding_value
            tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)
            tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
            
            result = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)
            return result[0]
            
    except Exception as e:
        return f"Inference Error: {str(e)}"

def pipeline(video_file):
    if video_file is None:
        return None, "Please upload a video."
        
    # Ensure Model is Loaded
    load_model()
    
    # Paths
    temp_dir = Path("temp_inference")
    temp_dir.mkdir(exist_ok=True)
    unique_id = str(uuid.uuid4())[:8]
    
    input_video_path = Path(video_file)
    viz_video_path = temp_dir / f"viz_{unique_id}.mov"
    raw_json_path = temp_dir / f"raw_{unique_id}.json"
    opt_json_path = temp_dir / f"opt_{unique_id}.json"
    
    # 1. Run PoseExtractor
    # ./bin/PoseExtractor <input> <output_viz> <output_json>
    print(f"Extracting pose from {input_video_path}...")
    cmd_extract = [
        "./bin/PoseExtractor",
        str(input_video_path),
        str(viz_video_path),
        str(raw_json_path)
    ]
    
    try:
        subprocess.run(cmd_extract, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        return None, f"PoseExtractor Failed: {e.stderr.decode()}"
        
    if not raw_json_path.exists():
        return None, "PoseExtractor did not generate JSON."

    # 2. Run Optimization (Convert Apple -> RTMPose structure + Calibrate)
    # python scripts/optimize_pose.py --apple <raw> --output <opt>
    print("Optimizing pose data...")
    cmd_opt = [
        sys.executable, # Use current python
        "scripts/optimize_pose.py",
        "--apple", str(raw_json_path),
        "--output", str(opt_json_path)
    ]
    
    try:
        subprocess.run(cmd_opt, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        # Fallback to raw if optimization fails (though it shouldn't)
        print(f"Optimization failed: {e.stderr.decode()}. Using raw JSON.")
        opt_json_path = raw_json_path

    # 3. Run Inference
    print("Running Model Inference...")
    # Load the JSON data into memory
    pose_data = load_pose_from_json(str(opt_json_path))
    
    # Run inference logic
    translation = run_inference_logic(pose_data)
    
    return str(viz_video_path), translation

# --- Gradio App ---
import argparse 

# Create the Interface
with gr.Blocks(title="CoSign AI") as app:
    gr.Markdown("# CoSign AI: Sign Language Translation")
    gr.Markdown("Upload a video to translate Sign Language to Text.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Input Video", sources=["upload", "webcam"])
            btn_run = gr.Button("Translate", variant="primary")
        
        with gr.Column():
            video_output = gr.Video(label="Pose Visualization")
            text_output = gr.Textbox(label="Translation", lines=2, show_copy_button=True)
            
    btn_run.click(
        fn=pipeline,
        inputs=[video_input],
        outputs=[video_output, text_output]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
