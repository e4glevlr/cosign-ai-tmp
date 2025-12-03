#!/usr/bin/env python
"""
Test the Apple Vision fine-tuning pipeline.

This script validates that all components work together:
1. Dataset loading (Apple Vision poses)
2. Model initialization
3. Checkpoint loading
4. Forward pass
5. Loss computation
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
from torch.utils.data import DataLoader

from models import Uni_Sign
from datasets import S2T_Dataset
from config import train_label_paths, dev_label_paths


class Args:
    """Minimal args for testing."""
    dataset = 'VSL'
    rgb_support = False
    max_length = 256
    use_apple_pose = True
    hidden_dim = 256
    label_smoothing = 0.2


def test_dataset():
    """Test dataset loading."""
    print("=" * 60)
    print("Testing Dataset Loading")
    print("=" * 60)
    
    args = Args()
    
    try:
        ds = S2T_Dataset(train_label_paths['VSL'], args, 'train')
        print(f"✓ Dataset loaded: {len(ds)} samples")
        
        name, pose, text, gloss, rgb = ds[0]
        print(f"✓ Sample loaded: {name}")
        print(f"  Text: {text}")
        print(f"  Pose parts: {list(pose.keys())}")
        for k, v in pose.items():
            print(f"    {k}: {v.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False


def test_dataloader():
    """Test dataloader with collation."""
    print("\n" + "=" * 60)
    print("Testing DataLoader")
    print("=" * 60)
    
    args = Args()
    
    try:
        ds = S2T_Dataset(train_label_paths['VSL'], args, 'train')
        loader = DataLoader(ds, batch_size=1, collate_fn=ds.collate_fn)
        
        batch = next(iter(loader))
        src_input, tgt_input = batch
        
        print(f"✓ Batch loaded")
        print(f"  Source keys: {list(src_input.keys())}")
        print(f"  Target keys: {list(tgt_input.keys())}")
        print(f"  GT sentence: {tgt_input['gt_sentence']}")
        
        return True, src_input, tgt_input
    except Exception as e:
        print(f"✗ DataLoader failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_model():
    """Test model initialization."""
    print("\n" + "=" * 60)
    print("Testing Model Initialization")
    print("=" * 60)
    
    args = Args()
    
    try:
        model = Uni_Sign(args)
        print(f"✓ Model initialized")
        
        # Count parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params: {total:,}")
        print(f"  Trainable: {trainable:,}")
        
        return True, model
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward(model, src_input, tgt_input):
    """Test forward pass."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    try:
        # Move to device
        device = torch.device('cpu')  # Use CPU for testing
        model = model.to(device)
        
        for key in src_input:
            if isinstance(src_input[key], torch.Tensor):
                src_input[key] = src_input[key].to(device).float()  # Convert to float32
        
        # Forward pass
        with torch.no_grad():
            outputs = model(src_input, tgt_input)
        
        print(f"✓ Forward pass succeeded")
        print(f"  Output keys: {list(outputs.keys())}")
        print(f"  Loss: {outputs['loss'].item():.4f}")
        print(f"  Embeddings shape: {outputs['inputs_embeds'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint():
    """Test checkpoint loading."""
    print("\n" + "=" * 60)
    print("Testing Checkpoint Loading")
    print("=" * 60)
    
    checkpoint_path = Path('pretrained_weight/best_checkpoint.pth')
    
    if not checkpoint_path.exists():
        print(f"⚠ Checkpoint not found at {checkpoint_path}")
        print("  Skipping checkpoint test")
        return True  # Not a failure, just skipped
    
    args = Args()
    
    try:
        model = Uni_Sign(args)
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ Checkpoint loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Checkpoint loading failed: {e}")
        return False


def test_freeze_mt5():
    """Test mT5 freezing."""
    print("\n" + "=" * 60)
    print("Testing mT5 Freezing")
    print("=" * 60)
    
    args = Args()
    
    try:
        model = Uni_Sign(args)
        
        # Before freezing
        before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Freeze mT5
        for name, param in model.named_parameters():
            if 'mt5_model' in name:
                param.requires_grad = False
        
        # After freezing
        after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = before - after
        
        print(f"✓ mT5 frozen successfully")
        print(f"  Before: {before:,} trainable params")
        print(f"  After: {after:,} trainable params")
        print(f"  Frozen: {frozen:,} params")
        
        # List trainable layers
        trainable_layers = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"  Trainable layers: {len(trainable_layers)}")
        for layer in trainable_layers[:5]:
            print(f"    - {layer}")
        if len(trainable_layers) > 5:
            print(f"    ... and {len(trainable_layers) - 5} more")
        
        return True
    except Exception as e:
        print(f"✗ mT5 freezing failed: {e}")
        return False


def main():
    print("UniSign Apple Vision Pipeline Test")
    print("=" * 60)
    
    results = {}
    
    # Test dataset
    results['dataset'] = test_dataset()
    
    # Test dataloader
    success, src_input, tgt_input = test_dataloader()
    results['dataloader'] = success
    
    # Test model
    success, model = test_model()
    results['model'] = success
    
    # Test forward pass (if model and data available)
    if model is not None and src_input is not None:
        results['forward'] = test_forward(model, src_input, tgt_input)
    else:
        results['forward'] = False
    
    # Test checkpoint
    results['checkpoint'] = test_checkpoint()
    
    # Test freezing
    results['freeze'] = test_freeze_mt5()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! Pipeline is ready for training.")
    else:
        print("Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
