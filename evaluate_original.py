#!/usr/bin/env python3
"""
Evaluate Original UniSign (RTMPose + RGB) on test set.
Uses the original checkpoint with rgb_support=True.

Usage:
    python evaluate_original.py --checkpoint pretrained_weight/best_checkpoint.pth --device cuda
"""

import os
import sys
import gzip
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models import Uni_Sign
from datasets import load_part_kp


def compute_bleu(reference: str, hypothesis: str) -> dict:
    """Compute BLEU scores."""
    from collections import Counter
    
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    if len(hyp_tokens) == 0:
        return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu4': 0.0, 'bleu': 0.0}
    
    scores = {}
    for n in range(1, 5):
        if len(hyp_tokens) < n:
            scores[f'bleu{n}'] = 0.0
            continue
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
        hyp_ngrams = Counter([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)])
        matches = sum((ref_ngrams & hyp_ngrams).values())
        total = sum(hyp_ngrams.values())
        scores[f'bleu{n}'] = matches / total if total > 0 else 0.0
    
    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / len(hyp_tokens))) if len(hyp_tokens) > 0 else 0.0
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
    """Compute Word Error Rate."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    if len(ref_tokens) == 0:
        return 1.0 if len(hyp_tokens) > 0 else 0.0
    
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
    ref_norm = ' '.join(reference.lower().split())
    hyp_norm = ' '.join(hypothesis.lower().split())
    return ref_norm == hyp_norm


def load_model(checkpoint_path: str, device: str, rgb_support: bool = True):
    """Load model with rgb_support for original checkpoint."""
    class Args:
        hidden_dim = 256
        dataset = 'VSL'
        label_smoothing = 0.2
    
    Args.rgb_support = rgb_support
    
    model = Uni_Sign(Args())
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
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


def load_pose_data(name: str, pose_dir: str) -> dict:
    """Load and process RTMPose data."""
    pkl_path = f"{pose_dir}/{name}.pkl"
    
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    if 'body' in raw_data:
        return raw_data
    
    keypoints_list = raw_data['keypoints']
    scores_list = raw_data['scores']
    
    first_kp = np.array(keypoints_list[0])
    
    if first_kp.ndim == 3 and first_kp.shape[0] > 1:
        # RTMPose format: (N_persons, 133, 2) - take first person
        skeletons = np.array([kp[0:1] for kp in keypoints_list])
        scores = np.array([s[0:1] for s in scores_list])
    else:
        skeletons = np.array(keypoints_list)
        scores = np.array(scores_list)
    
    return load_part_kp(skeletons, scores, force_ok=True)


def run_inference(model, pose_data: dict, device: str, rgb_support: bool = True) -> str:
    """Run model inference."""
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
    
    seq_len = batch['body'].shape[1]
    batch['attention_mask'] = torch.ones(1, seq_len, dtype=torch.float32).to(device)
    
    # Add RGB placeholders if rgb_support is enabled
    if rgb_support:
        # Create dummy RGB tensors and indices
        batch['left_hands'] = torch.zeros(1, 3, 112, 112).to(device)
        batch['right_hands'] = torch.zeros(1, 3, 112, 112).to(device)
        batch['left_sampled_indices'] = torch.zeros(1, dtype=torch.long).to(device)
        batch['right_sampled_indices'] = torch.zeros(1, dtype=torch.long).to(device)
        batch['left_rgb_len'] = [1]
        batch['right_rgb_len'] = [1]
        batch['left_skeletons_norm'] = torch.zeros(1, 21, 2).to(device)
        batch['right_skeletons_norm'] = torch.zeros(1, 21, 2).to(device)
    
    fake_tgt = {'gt_sentence': ['']}
    
    with torch.no_grad():
        outputs = model(batch, fake_tgt)
        generated = model.generate(outputs, max_new_tokens=100, num_beams=4)
        text = model.mt5_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    
    return text


def main():
    parser = argparse.ArgumentParser(description='Evaluate Original UniSign on test set')
    parser.add_argument('--checkpoint', type=str, default='pretrained_weight/best_checkpoint.pth',
                        help='Path to original checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda, mps, or cpu')
    parser.add_argument('--pose-dir', type=str, default='vn_sentence_data/pose',
                        help='Directory containing RTMPose PKL files')
    parser.add_argument('--output', type=str, default='EVALUATION_ORIGINAL.md',
                        help='Output report file')
    args = parser.parse_args()
    
    print("=" * 60)
    print("UniSign Original (RTMPose + RGB) - Test Set Evaluation")
    print("=" * 60)
    
    # Load test set
    print("\n[1/3] Loading test set...")
    with gzip.open('data/VSL/labels.test', 'rb') as f:
        test_labels = pickle.load(f)
    
    with gzip.open('data/VSL/labels.train', 'rb') as f:
        train_labels = pickle.load(f)
    
    print(f"   Train: {len(train_labels)} samples")
    print(f"   Test:  {len(test_labels)} samples")
    
    # Load model
    print("\n[2/3] Loading model...")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Device: {args.device}")
    print(f"   RGB Support: True (original model)")
    
    model = load_model(args.checkpoint, args.device, rgb_support=True)
    print("   ✓ Model loaded")
    
    # Evaluate
    print("\n[3/3] Evaluating on test set...")
    results = []
    errors = []
    
    for name, info in tqdm(test_labels.items(), desc="Evaluating"):
        ground_truth = info['text']
        
        try:
            pose_data = load_pose_data(name, args.pose_dir)
            prediction = run_inference(model, pose_data, args.device, rgb_support=True)
            
            bleu_scores = compute_bleu(ground_truth, prediction)
            rouge_l = compute_rouge_l(ground_truth, prediction)
            wer = word_error_rate(ground_truth, prediction)
            is_exact = exact_match(ground_truth, prediction)
            
            results.append({
                'name': name,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'bleu1': bleu_scores['bleu1'],
                'bleu4': bleu_scores['bleu4'],
                'bleu': bleu_scores['bleu'],
                'rouge_l': rouge_l,
                'wer': wer,
                'exact_match': is_exact,
            })
        except Exception as e:
            errors.append({'name': name, 'error': str(e)})
            print(f"\nError on {name}: {e}")
    
    # Compute metrics
    if len(results) == 0:
        print("\n❌ No results to compute!")
        return
    
    metrics = {
        'bleu1': np.mean([r['bleu1'] for r in results]),
        'bleu4': np.mean([r['bleu4'] for r in results]),
        'bleu': np.mean([r['bleu'] for r in results]),
        'rouge_l': np.mean([r['rouge_l'] for r in results]),
        'wer': np.mean([r['wer'] for r in results]),
        'exact_match': np.mean([r['exact_match'] for r in results]) * 100,
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n📊 Metrics on {len(results)} test samples:")
    print(f"   BLEU-1:        {metrics['bleu1']:.4f}")
    print(f"   BLEU-4:        {metrics['bleu4']:.4f}")
    print(f"   BLEU:          {metrics['bleu']:.4f}")
    print(f"   ROUGE-L:       {metrics['rouge_l']:.4f}")
    print(f"   WER:           {metrics['wer']:.4f}")
    print(f"   Exact Match:   {metrics['exact_match']:.2f}%")
    
    if errors:
        print(f"\n⚠️  Errors: {len(errors)} samples failed")
    
    # Generate report
    report = f"""# UniSign Original (RTMPose + RGB) - Evaluation Report

## Model Info

- **Checkpoint:** `{args.checkpoint}`
- **Device:** {args.device}
- **RGB Support:** True (original model uses EfficientNet-B0)
- **Pose Extraction:** RTMPose

## Dataset

| Split | Samples |
|-------|---------|
| Train | {len(train_labels)} |
| Test  | {len(test_labels)} |

## Results on Test Set

| Metric | Score |
|--------|-------|
| **BLEU-1** | {metrics['bleu1']:.4f} |
| **BLEU-4** | {metrics['bleu4']:.4f} |
| **BLEU** | {metrics['bleu']:.4f} |
| **ROUGE-L** | {metrics['rouge_l']:.4f} |
| **WER** | {metrics['wer']:.4f} |
| **Exact Match** | {metrics['exact_match']:.2f}% |

## Sample Predictions

### Best (Highest ROUGE-L)

| Video | Ground Truth | Prediction | ROUGE-L |
|-------|--------------|------------|---------|
"""
    
    sorted_results = sorted(results, key=lambda x: x['rouge_l'], reverse=True)
    for r in sorted_results[:10]:
        report += f"| {r['name']} | {r['ground_truth']} | {r['prediction']} | {r['rouge_l']:.3f} |\n"
    
    report += """

### Worst (Lowest ROUGE-L)

| Video | Ground Truth | Prediction | ROUGE-L |
|-------|--------------|------------|---------|
"""
    for r in sorted_results[-10:]:
        report += f"| {r['name']} | {r['ground_truth']} | {r['prediction']} | {r['rouge_l']:.3f} |\n"
    
    if errors:
        report += f"\n## Errors ({len(errors)} samples)\n\n"
        for e in errors[:10]:
            report += f"- {e['name']}: {e['error']}\n"
    
    report += "\n---\n*Generated by evaluate_original.py*\n"
    
    # Save report
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 Report saved to: {args.output}")
    
    # Save JSON
    import json
    json_output = args.output.replace('.md', '.json')
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'results': results, 'errors': errors}, f, ensure_ascii=False, indent=2)
    print(f"📊 Raw data saved to: {json_output}")


if __name__ == '__main__':
    main()
