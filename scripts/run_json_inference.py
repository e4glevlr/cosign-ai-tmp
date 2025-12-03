import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import json
import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Add current directory to path to find models, utils, config
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from models import Uni_Sign
import utils as utils
from datasets import S2T_Dataset_online
from config import *

def main(args):
    utils.set_seed(args.seed)

    # extract pose from JSON
    print(f"Loading pose from JSON: {args.pose_json}")
    pose_data = load_pose_from_json(args.pose_json)

    print(f"Creating dataset...")
    online_data = S2T_Dataset_online(args=args)
    online_data.rgb_data = args.online_video # Video path needed for RGB support if enabled
    online_data.pose_data = pose_data

    online_sampler = torch.utils.data.SequentialSampler(online_data)
    online_dataloader = DataLoader(online_data,
                                batch_size=1,
                                collate_fn=online_data.collate_fn,
                                sampler=online_sampler,)

    print(f"Creating model...")
    # Force CPU if no CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mock args for model creation if not present
    if not hasattr(args, 'hidden_dim'): args.hidden_dim = 768 # Default? Need to check config
    if not hasattr(args, 'dataset'): args.dataset = "CSL_Daily" # Default
    
    model = Uni_Sign(args=args)
    model.to(device)
    model.eval()

    if args.finetune and os.path.exists(args.finetune):
        print(f'Loading Checkpoint from {args.finetune}...')
        state_dict = torch.load(args.finetune, map_location=device, weights_only=True)['model']
        ret = model.load_state_dict(state_dict, strict=False)
        print('Missing keys:', len(ret.missing_keys))
    else:
        print("Warning: No checkpoint provided or found. Running with random weights (Output will be nonsense).")

    inference(online_dataloader, model, device)

def load_pose_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    keypoints = []
    scores = []

    # Case 1: Apple Vision Format (has 'frames' list)
    if 'frames' in data:
        print("Detected Apple Vision format.")
        width = data.get('width', 1920)
        height = data.get('height', 1080)
        # print(f"De-normalizing coordinates using video size: {width}x{height}")

        frames = sorted(data['frames'], key=lambda x: x['frame_index'])
        
        for frame in frames:
            kp = np.array(frame['keypoints'], dtype=np.float32)
            sc = np.array(frame['scores'], dtype=np.float32)
            
            # Calibration (Hardcoded fallback if not optimized)
            # If using optimized file, this branch won't be taken.
            # But for raw apple file, let's keep the rough offset or remove if we trust optimize script?
            # Let's keep it for backward compatibility but print warning.
            # kp[..., 0] += 6.5 
            # kp[..., 1] += 8.0 

            kp = kp[np.newaxis, ...]
            sc = sc[np.newaxis, ...]
            
            keypoints.append(kp)
            scores.append(sc)

    # Case 2: RTMPose / Optimized Format (has 'keypoints' list/tensor)
    elif 'keypoints' in data:
        print("Detected RTMPose/Optimized format.")
        # Shape is usually (T, 1, 133, 2)
        kps_raw = np.array(data['keypoints'], dtype=np.float32)
        scores_raw = np.array(data['scores'], dtype=np.float32)
        
        # Check dimensions
        if len(kps_raw.shape) == 4: # (T, M, N, C)
            # S2T_Dataset expects list of (1, 133, 2) arrays?
            # datasets.py: poses['keypoints'] is list of (1, 133, 2) arrays? 
            # Actually datasets.py takes the whole thing.
            # But this function returns a dict where keys are lists of arrays.
            
            for i in range(len(kps_raw)):
                keypoints.append(kps_raw[i]) # (1, 133, 2)
                scores.append(scores_raw[i]) # (1, 133)
        else:
            print(f"Unexpected shape: {kps_raw.shape}")
            
    else:
        raise ValueError("Unknown JSON format")
        
    return {"keypoints": keypoints, "scores": scores}

def inference(data_loader, model, device):
    model.eval()
    target_dtype = torch.float32 # Use float32 for CPU/Safety

    with torch.no_grad():
        tgt_pres = []
        
        print("Running inference...")
        for step, (src_input, tgt_input) in enumerate(data_loader):
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(device)

            stack_out = model(src_input, tgt_input)

            output = model.generate(stack_out,
                                    max_new_tokens=50,
                                    num_beams=4,
                                    )

            for i in range(len(output)):
                tgt_pres.append(output[i])

    tokenizer = model.mt5_tokenizer
    padding_value = tokenizer.eos_token_id

    if len(tgt_pres) > 0:
        pad_tensor = torch.ones(150 - len(tgt_pres[0])).to(device) * padding_value
        tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)
        tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
        result = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)
        print(f"Prediction result is: {result[0]}")
    else:
        print("No output generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Uni-Sign Inference', parents=[utils.get_args_parser()])
    
    # Add our specific args
    parser.add_argument('--pose_json', type=str, required=True, help='Path to JSON pose file from Swift')
    
    args = parser.parse_args()
    main(args)
