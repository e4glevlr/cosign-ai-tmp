#!/usr/bin/env python
"""
Create sample label files for Vietnamese Sign Language dataset.

Usage:
    python scripts/create_vsl_labels.py --input <annotations.json> --output data/VSL/
    
Or to create a test sample:
    python scripts/create_vsl_labels.py --test
"""
import pickle
import gzip
import json
import argparse
from pathlib import Path


def create_test_labels(output_dir: str):
    """Create sample label files for testing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample data
    data = {
        'BG1_S002': {
            'name': 'BG1_S002',
            'text': 'có hỏa hoạn, nguy hiểm nguy hiểm',
            'video_path': 'BG1_S002.mp4'
        }
    }
    
    for split in ['train', 'dev', 'test']:
        label_path = output_path / f'labels.{split}'
        with gzip.open(label_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Created: {label_path}")
    
    print(f"\nSample labels created in {output_dir}")


def convert_json_to_labels(input_json: str, output_dir: str, 
                           train_ratio: float = 0.8, 
                           dev_ratio: float = 0.1):
    """
    Convert JSON annotations to UniSign label format.
    
    Expected JSON format:
    [
        {"video": "video1.mp4", "text": "translation text"},
        ...
    ]
    
    Or:
    {
        "video1": {"text": "translation text", ...},
        ...
    }
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(input_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Normalize format
    if isinstance(raw_data, list):
        data = {}
        for item in raw_data:
            video_name = Path(item['video']).stem
            data[video_name] = {
                'name': video_name,
                'text': item['text'],
                'video_path': item['video']
            }
    else:
        data = {}
        for key, item in raw_data.items():
            video_name = Path(key).stem if '.' in key else key
            data[video_name] = {
                'name': video_name,
                'text': item.get('text', item.get('translation', '')),
                'video_path': item.get('video', item.get('video_path', f'{video_name}.mp4'))
            }
    
    # Split data
    keys = list(data.keys())
    n = len(keys)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    
    train_keys = keys[:n_train]
    dev_keys = keys[n_train:n_train + n_dev]
    test_keys = keys[n_train + n_dev:]
    
    splits = {
        'train': {k: data[k] for k in train_keys},
        'dev': {k: data[k] for k in dev_keys},
        'test': {k: data[k] for k in test_keys}
    }
    
    for split, split_data in splits.items():
        label_path = output_path / f'labels.{split}'
        with gzip.open(label_path, 'wb') as f:
            pickle.dump(split_data, f)
        print(f"Created: {label_path} ({len(split_data)} samples)")
    
    print(f"\nTotal: {n} samples")
    print(f"  Train: {len(train_keys)}")
    print(f"  Dev: {len(dev_keys)}")
    print(f"  Test: {len(test_keys)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create VSL label files')
    parser.add_argument('--input', '-i', type=str, help='Input JSON annotations file')
    parser.add_argument('--output', '-o', type=str, default='data/VSL/', 
                        help='Output directory')
    parser.add_argument('--test', action='store_true',
                        help='Create test sample labels')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Train split ratio')
    parser.add_argument('--dev-ratio', type=float, default=0.1,
                        help='Dev split ratio')
    
    args = parser.parse_args()
    
    if args.test:
        create_test_labels(args.output)
    elif args.input:
        convert_json_to_labels(args.input, args.output, 
                               args.train_ratio, args.dev_ratio)
    else:
        parser.print_help()
