#!/usr/bin/env python3
"""
Evaluate Apple Vision + UniSign model on test set.
Computes BLEU, ROUGE, accuracy and generates detailed report.
"""

import os
import sys
import json
import gzip
import pickle
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models import Uni_Sign
from datasets import load_part_kp


def convert_apple_json_to_pose(json_path: str) -> dict:
    """Convert Apple Vision JSON to pose format for model."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames = data.get('frames', [])
    if not frames:
        raise ValueError("No frames in JSON")
    
    num_frames = len(frames)
    num_keypoints = 133
    
    skeletons = np.zeros((num_frames, 1, num_keypoints, 2), dtype=np.float32)
    scores = np.zeros((num_frames, 1, num_keypoints), dtype=np.float32)
    
    for i, frame in enumerate(frames):
        keypoints = frame.get('keypoints', [])
        frame_scores = frame.get('scores', [])
        
        for idx, kp in enumerate(keypoints):
            if idx < num_keypoints and len(kp) >= 2:
                skeletons[i, 0, idx, 0] = kp[0]
                skeletons[i, 0, idx, 1] = kp[1]
                if idx < len(frame_scores):
                    scores[i, 0, idx] = frame_scores[idx]
    
    kps = load_part_kp(skeletons, scores, force_ok=True)
    return kps


def extract_pose_apple(video_path: str, output_json: str) -> bool:
    """Extract pose using Apple Vision PoseExtractor."""
    extractor = Path(__file__).parent / "bin" / "PoseExtractor"
    if not extractor.exists():
        raise FileNotFoundError(f"PoseExtractor not found at {extractor}")
    
    result = subprocess.run(
        [str(extractor), video_path, output_json],
        capture_output=True,
        text=True
    )
    return result.returncode == 0 and os.path.exists(output_json)


def compute_bleu(reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute BLEU scores (1-4)."""
    from collections import Counter
    
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    if len(hyp_tokens) == 0:
        return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}
    
    scores = {}
    for n in range(1, 5):
        if len(hyp_tokens) < n:
            scores[f'bleu{n}'] = 0.0
            continue
            
        # Get n-grams
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
        hyp_ngrams = Counter([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)])
        
        # Count matches
        matches = sum((ref_ngrams & hyp_ngrams).values())
        total = sum(hyp_ngrams.values())
        
        scores[f'bleu{n}'] = matches / total if total > 0 else 0.0
    
    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / len(hyp_tokens))) if len(hyp_tokens) > 0 else 0.0
    
    # Geometric mean for BLEU-4
    if all(scores[f'bleu{i}'] > 0 for i in range(1, 5)):
        scores['bleu'] = bp * np.exp(sum(np.log(scores[f'bleu{i}']) for i in range(1, 5)) / 4)
    else:
        scores['bleu'] = 0.0
    
    return scores


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L score."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0.0
    
    # LCS
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs = dp[m][n]
    precision = lcs / n
    recall = lcs / m
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    if len(ref_tokens) == 0:
        return 1.0 if len(hyp_tokens) > 0 else 0.0
    
    # Dynamic programming for edit distance
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    return dp[m][n] / m


def exact_match(reference: str, hypothesis: str) -> bool:
    """Check exact match after normalization."""
    ref_norm = ' '.join(reference.lower().split())
    hyp_norm = ' '.join(hypothesis.lower().split())
    return ref_norm == hyp_norm


def partial_match(reference: str, hypothesis: str, threshold: float = 0.5) -> bool:
    """Check if at least threshold% of reference words appear in hypothesis."""
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())
    
    if len(ref_words) == 0:
        return True
    
    overlap = len(ref_words & hyp_words)
    return overlap / len(ref_words) >= threshold


def load_model(checkpoint_path: str, device: str):
    """Load trained model."""
    # Create args object with all required attributes
    class Args:
        def __init__(self):
            self.hidden_dim = 256
            self.rgb_support = False
            self.dataset = 'VSL'
            self.label_smoothing = 0.2
    
    args = Args()
    model = Uni_Sign(args)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def run_inference(model, pose_data: dict, device: str) -> str:
    """Run model inference on pose data."""
    # Prepare batch - handle both tensor and numpy input
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.clone().detach().unsqueeze(0).float().to(device)
        else:
            return torch.from_numpy(x).unsqueeze(0).float().to(device)
    
    batch = {
        'body': to_tensor(pose_data['body']),
        'left': to_tensor(pose_data['left']),
        'right': to_tensor(pose_data['right']),
        'face_all': to_tensor(pose_data['face_all']),
    }
    
    # Add attention mask (all ones)
    seq_len = batch['body'].shape[1]  # T dimension
    batch['attention_mask'] = torch.ones(1, seq_len, dtype=torch.float32).to(device)
    
    # Create fake tgt_input for forward pass (needed for prefix generation)
    fake_tgt = {'gt_sentence': ['']}
    
    with torch.no_grad():
        # Get encoder output with fake target
        outputs = model(batch, fake_tgt)
        
        # Generate predictions
        generated = model.generate(
            outputs, 
            max_new_tokens=100,
            num_beams=4
        )
        
        # Decode
        text = model.mt5_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    
    return text


def main():
    parser = argparse.ArgumentParser(description='Evaluate on test set')
    parser.add_argument('--checkpoint', type=str, default='output/best_checkpoint_apple.pth')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--output', type=str, default='EVALUATION_REPORT.md')
    parser.add_argument('--use-cached-poses', action='store_true', 
                        help='Use cached PKL poses instead of re-extracting')
    args = parser.parse_args()
    
    print("=" * 60)
    print("UniSign + Apple Vision - Test Set Evaluation")
    print("=" * 60)
    
    # Load test set labels
    print("\n[1/4] Loading test set...")
    with gzip.open('vn_sentence_data/labels.test', 'rb') as f:
        test_labels = pickle.load(f)
    
    with gzip.open('vn_sentence_data/labels.train', 'rb') as f:
        train_labels = pickle.load(f)
    
    with gzip.open('vn_sentence_data/labels.dev', 'rb') as f:
        dev_labels = pickle.load(f)
    
    print(f"   Train: {len(train_labels)} samples")
    print(f"   Dev:   {len(dev_labels)} samples")
    print(f"   Test:  {len(test_labels)} samples")
    
    # Load model
    print("\n[2/4] Loading model...")
    model = load_model(args.checkpoint, args.device)
    print(f"   ✓ Model loaded on {args.device}")
    
    # Run evaluation
    print("\n[3/4] Running evaluation on test set...")
    results = []
    errors = []
    
    for name, info in tqdm(test_labels.items(), desc="Evaluating"):
        video_path = f"vn_sentence_data/video/{name}.mp4"
        ground_truth = info['text']
        
        try:
            # Get pose data
            if args.use_cached_poses:
                pkl_path = f"vn_sentence_data/pose_apple/{name}.pkl"
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        raw_data = pickle.load(f)
                    
                    # Check if already processed or raw format
                    if 'body' in raw_data:
                        pose_data = raw_data
                    else:
                        # Raw format with keypoints/scores - need to process
                        # Handle list format
                        if isinstance(raw_data['keypoints'], list):
                            skeletons = np.array(raw_data['keypoints'])  # (T, 1, 133, 2)
                            scores = np.array(raw_data['scores'])  # (T, 1, 133)
                        else:
                            skeletons = raw_data['keypoints']
                            scores = raw_data['scores']
                        pose_data = load_part_kp(skeletons, scores, force_ok=True)
                else:
                    # Extract and convert
                    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                        json_path = tmp.name
                    
                    if not extract_pose_apple(video_path, json_path):
                        raise Exception("Pose extraction failed")
                    
                    pose_data = convert_apple_json_to_pose(json_path)
                    os.unlink(json_path)
            else:
                # Always extract fresh
                with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                    json_path = tmp.name
                
                if not extract_pose_apple(video_path, json_path):
                    raise Exception("Pose extraction failed")
                
                pose_data = convert_apple_json_to_pose(json_path)
                os.unlink(json_path)
            
            # Run inference
            prediction = run_inference(model, pose_data, args.device)
            
            # Compute metrics
            bleu_scores = compute_bleu(ground_truth, prediction)
            rouge_l = compute_rouge_l(ground_truth, prediction)
            wer = word_error_rate(ground_truth, prediction)
            is_exact = exact_match(ground_truth, prediction)
            is_partial = partial_match(ground_truth, prediction, 0.5)
            
            results.append({
                'name': name,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'gloss': info.get('gloss', []),
                'bleu1': bleu_scores['bleu1'],
                'bleu2': bleu_scores['bleu2'],
                'bleu4': bleu_scores['bleu4'],
                'bleu': bleu_scores['bleu'],
                'rouge_l': rouge_l,
                'wer': wer,
                'exact_match': is_exact,
                'partial_match': is_partial,
            })
            
        except Exception as e:
            import traceback
            print(f"Error on {name}: {e}")
            traceback.print_exc()
            errors.append({'name': name, 'error': str(e)})
    
    # Compute aggregate metrics
    print("\n[4/4] Computing metrics...")
    
    if len(results) == 0:
        print("No results to compute!")
        return
    
    metrics = {
        'bleu1': np.mean([r['bleu1'] for r in results]),
        'bleu2': np.mean([r['bleu2'] for r in results]),
        'bleu4': np.mean([r['bleu4'] for r in results]),
        'bleu': np.mean([r['bleu'] for r in results]),
        'rouge_l': np.mean([r['rouge_l'] for r in results]),
        'wer': np.mean([r['wer'] for r in results]),
        'exact_match': np.mean([r['exact_match'] for r in results]) * 100,
        'partial_match_50': np.mean([r['partial_match'] for r in results]) * 100,
    }
    
    # Generate report
    report = f"""# UniSign + Apple Vision - Evaluation Report

## Overview

- **Date:** 2024-12-03
- **Model:** ST-GCN Encoder + mT5 Decoder (Fine-tuned on Apple Vision poses)
- **Checkpoint:** `{args.checkpoint}`
- **Device:** {args.device}

## Dataset Split

| Split | Samples |
|-------|---------|
| Train | {len(train_labels)} |
| Dev   | {len(dev_labels)} |
| Test  | {len(test_labels)} |
| **Total** | **{len(train_labels) + len(dev_labels) + len(test_labels)}** |

## Evaluation Results

### Aggregate Metrics

| Metric | Score |
|--------|-------|
| **BLEU-1** | {metrics['bleu1']:.4f} |
| **BLEU-2** | {metrics['bleu2']:.4f} |
| **BLEU-4** | {metrics['bleu4']:.4f} |
| **BLEU (combined)** | {metrics['bleu']:.4f} |
| **ROUGE-L** | {metrics['rouge_l']:.4f} |
| **WER** | {metrics['wer']:.4f} |
| **Exact Match** | {metrics['exact_match']:.2f}% |
| **Partial Match (≥50%)** | {metrics['partial_match_50']:.2f}% |

### Metric Explanations

- **BLEU (Bilingual Evaluation Understudy):** Measures n-gram overlap between prediction and reference. Higher is better (0-1).
- **ROUGE-L:** Measures longest common subsequence. Higher is better (0-1).
- **WER (Word Error Rate):** Edit distance normalized by reference length. Lower is better (0-∞).
- **Exact Match:** Percentage of predictions that exactly match the reference.
- **Partial Match:** Percentage of predictions where ≥50% of reference words appear.

## Sample Predictions

### Best Predictions (Highest ROUGE-L)

"""
    
    # Sort by ROUGE-L for best/worst
    sorted_results = sorted(results, key=lambda x: x['rouge_l'], reverse=True)
    
    report += "| # | Video | Ground Truth | Prediction | ROUGE-L |\n"
    report += "|---|-------|--------------|------------|--------|\n"
    for i, r in enumerate(sorted_results[:10]):
        gt_short = r['ground_truth'][:40] + "..." if len(r['ground_truth']) > 40 else r['ground_truth']
        pred_short = r['prediction'][:40] + "..." if len(r['prediction']) > 40 else r['prediction']
        report += f"| {i+1} | {r['name']} | {gt_short} | {pred_short} | {r['rouge_l']:.3f} |\n"
    
    report += """

### Worst Predictions (Lowest ROUGE-L)

"""
    report += "| # | Video | Ground Truth | Prediction | ROUGE-L |\n"
    report += "|---|-------|--------------|------------|--------|\n"
    for i, r in enumerate(sorted_results[-10:]):
        gt_short = r['ground_truth'][:40] + "..." if len(r['ground_truth']) > 40 else r['ground_truth']
        pred_short = r['prediction'][:40] + "..." if len(r['prediction']) > 40 else r['prediction']
        report += f"| {i+1} | {r['name']} | {gt_short} | {pred_short} | {r['rouge_l']:.3f} |\n"
    
    report += f"""

## Detailed Results

<details>
<summary>Click to expand all {len(results)} predictions</summary>

| Video | Ground Truth | Prediction | BLEU-1 | ROUGE-L | WER |
|-------|--------------|------------|--------|---------|-----|
"""
    
    for r in sorted(results, key=lambda x: x['name']):
        report += f"| {r['name']} | {r['ground_truth']} | {r['prediction']} | {r['bleu1']:.3f} | {r['rouge_l']:.3f} | {r['wer']:.3f} |\n"
    
    report += """
</details>

"""
    
    if errors:
        report += f"""## Errors

{len(errors)} samples failed during evaluation:

| Video | Error |
|-------|-------|
"""
        for e in errors:
            report += f"| {e['name']} | {e['error'][:50]}... |\n"
    
    report += f"""

## Analysis

### Key Observations

1. **BLEU-1 Score ({metrics['bleu1']:.4f}):** Measures unigram overlap. {"Good" if metrics['bleu1'] > 0.3 else "Needs improvement"} for low-resource setting.

2. **ROUGE-L Score ({metrics['rouge_l']:.4f}):** Measures sequence similarity. {"Reasonable" if metrics['rouge_l'] > 0.2 else "Low"} performance on unseen data.

3. **WER ({metrics['wer']:.4f}):** {"High" if metrics['wer'] > 0.7 else "Moderate"} word error rate indicates {"significant room for improvement" if metrics['wer'] > 0.7 else "reasonable accuracy"}.

4. **Exact Match ({metrics['exact_match']:.2f}%):** {"Very low" if metrics['exact_match'] < 5 else "Some"} exact matches, typical for translation tasks.

5. **Partial Match ({metrics['partial_match_50']:.2f}%):** {metrics['partial_match_50']:.1f}% of predictions capture at least half of the reference words.

### Recommendations for Improvement

1. **Data Augmentation:** Apply temporal augmentation, keypoint jittering, and video augmentation.
2. **More Training Data:** Current {len(train_labels)} samples is limited. Consider collecting more data.
3. **Hyperparameter Tuning:** Experiment with learning rates, batch sizes, and model architecture.
4. **Ensemble Methods:** Combine multiple models for better predictions.
5. **Pre-training:** Use larger sign language datasets for pre-training before fine-tuning.

## Conclusion

The Apple Vision + UniSign model achieved:
- **BLEU-4:** {metrics['bleu']:.4f}
- **ROUGE-L:** {metrics['rouge_l']:.4f}
- **WER:** {metrics['wer']:.4f}

on the {len(test_labels)}-sample test set. With only {len(train_labels)} training samples, the model shows {"promising" if metrics['bleu1'] > 0.2 else "initial"} capability to translate Vietnamese Sign Language to text.

---
*Generated by evaluate_test_set.py*
"""
    
    # Save report
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n📊 Results Summary:")
    print(f"   BLEU-1:        {metrics['bleu1']:.4f}")
    print(f"   BLEU-4:        {metrics['bleu']:.4f}")
    print(f"   ROUGE-L:       {metrics['rouge_l']:.4f}")
    print(f"   WER:           {metrics['wer']:.4f}")
    print(f"   Exact Match:   {metrics['exact_match']:.2f}%")
    print(f"   Partial Match: {metrics['partial_match_50']:.2f}%")
    print(f"\n📄 Full report saved to: {args.output}")
    
    # Save raw results as JSON
    json_output = args.output.replace('.md', '.json')
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'results': results,
            'errors': errors,
        }, f, ensure_ascii=False, indent=2)
    print(f"📊 Raw results saved to: {json_output}")


if __name__ == '__main__':
    main()
