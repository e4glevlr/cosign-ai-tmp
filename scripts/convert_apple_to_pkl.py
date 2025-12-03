"""
Convert Apple Vision PoseExtractor JSON to pickle format for UniSign training.

Apple Vision JSON format:
{
    "height": 720,
    "width": 1280,
    "frames": [
        {
            "frame_index": 0,
            "keypoints": [[x, y], ...],  # 133 keypoints
            "scores": [score, ...]        # 133 scores
        },
        ...
    ]
}

UniSign pickle format:
{
    "keypoints": [frame0, frame1, ...],  # Each frame: (1, 133, 2)
    "scores": [frame0, frame1, ...]      # Each frame: (1, 133)
}
"""
import json
import pickle
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm


def convert_apple_json_to_pkl(json_path: str, pkl_path: str, fix_empty_frames: bool = True):
    """
    Convert a single Apple Vision JSON to UniSign pickle format.
    
    Args:
        json_path: Path to input Apple Vision JSON
        pkl_path: Path to output pickle file
        fix_empty_frames: If True, copy next valid frame to empty Frame 0
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle Apple Vision format with 'frames' key
    if 'frames' in data:
        frames = sorted(data['frames'], key=lambda x: x['frame_index'])
        
        keypoints = []
        scores = []
        
        for frame in frames:
            # Apple Vision: keypoints is [[x,y], ...] shape (133, 2)
            kp = np.array(frame['keypoints'])  # (133, 2)
            sc = np.array(frame['scores'])      # (133,)
            
            # UniSign expects (1, 133, 2) and (1, 133)
            keypoints.append(kp[np.newaxis, :, :])  # (1, 133, 2)
            scores.append(sc[np.newaxis, :])         # (1, 133)
        
        keypoints = np.array(keypoints)  # (T, 1, 133, 2)
        scores = np.array(scores)        # (T, 1, 133)
        
        # Fix empty Frame 0 (common Apple Vision initialization issue)
        if fix_empty_frames and len(keypoints) > 1:
            if np.sum(keypoints[0]) == 0:
                keypoints[0] = keypoints[1]
                scores[0] = scores[1]
        
        # Convert to list format expected by datasets.py
        pose_data = {
            'keypoints': [keypoints[i] for i in range(len(keypoints))],
            'scores': [scores[i] for i in range(len(scores))]
        }
    
    # Handle already-converted format
    elif 'keypoints' in data:
        kps = np.array(data['keypoints'])
        if 'scores' in data:
            scs = np.array(data['scores'])
        else:
            scs = np.ones(kps.shape[:-1])
        
        pose_data = {
            'keypoints': [kps[i] for i in range(len(kps))],
            'scores': [scs[i] for i in range(len(scs))]
        }
    else:
        raise ValueError(f"Unknown JSON format in {json_path}")
    
    # Save as pickle
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, 'wb') as f:
        pickle.dump(pose_data, f)
    
    return len(pose_data['keypoints'])


def batch_convert(input_dir: str, output_dir: str, pattern: str = "*.json"):
    """
    Convert all Apple Vision JSONs in a directory to pickle format.
    
    Args:
        input_dir: Directory containing Apple Vision JSON files
        output_dir: Directory to save pickle files
        pattern: Glob pattern for JSON files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_path.glob(pattern))
    
    if not json_files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return
    
    print(f"Converting {len(json_files)} files...")
    
    success = 0
    failed = 0
    
    for json_file in tqdm(json_files, desc="Converting"):
        try:
            pkl_name = json_file.stem + ".pkl"
            pkl_path = output_path / pkl_name
            
            num_frames = convert_apple_json_to_pkl(str(json_file), str(pkl_path))
            success += 1
            
        except Exception as e:
            print(f"\nFailed to convert {json_file.name}: {e}")
            failed += 1
    
    print(f"\nConversion complete: {success} success, {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Apple Vision JSON to UniSign pickle format')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Single file conversion
    single_parser = subparsers.add_parser('single', help='Convert a single file')
    single_parser.add_argument('--input', '-i', required=True, help='Input Apple Vision JSON file')
    single_parser.add_argument('--output', '-o', required=True, help='Output pickle file')
    
    # Batch conversion
    batch_parser = subparsers.add_parser('batch', help='Convert all files in a directory')
    batch_parser.add_argument('--input-dir', '-i', required=True, help='Input directory with JSON files')
    batch_parser.add_argument('--output-dir', '-o', required=True, help='Output directory for pickle files')
    batch_parser.add_argument('--pattern', '-p', default='*.json', help='Glob pattern (default: *.json)')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        num_frames = convert_apple_json_to_pkl(args.input, args.output)
        print(f"Converted {args.input} -> {args.output} ({num_frames} frames)")
    
    elif args.command == 'batch':
        batch_convert(args.input_dir, args.output_dir, args.pattern)
    
    else:
        parser.print_help()
