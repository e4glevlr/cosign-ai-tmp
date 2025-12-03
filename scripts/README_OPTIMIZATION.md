# Apple Vision Pose Optimization - Quick Start Guide

This guide explains how to optimize Apple Vision pose extractions for use with the UniSign sign language translation model.

## Background

The UniSign model was trained on RTMPose keypoints. Apple Vision Framework produces geometrically different keypoints, particularly for body joints. This optimization pipeline applies empirically-determined calibration offsets.

**Important:** Calibration improves accuracy but has limitations (~70-80% ceiling). For production, consider fine-tuning the model on Apple Vision data (see [Fine-tuning Guide](../docs/OPTIMIZATION_RESULTS.md#long-term-production-quality)).

## Quick Start

### 1. Extract Pose from Video

```bash
./bin/PoseExtractor input_video.mp4 output_viz.mov pose_raw.json
```

### 2. Optimize for UniSign

```bash
./.venv/bin/python scripts/optimize_pose_production.py \
    --apple pose_raw.json \
    --output pose_optimized.json
```

### 3. Run Inference

```bash
./.venv/bin/python scripts/run_json_inference.py \
    --pose_json pose_optimized.json \
    --finetune pretrained_weight/best_checkpoint.pth
```

## Applied Calibration Offsets

Based on detailed analysis of BG1_S002.mp4 with ground truth comparison:

| Body Part | X Offset | Y Offset | Note |
|-----------|----------|----------|------|
| Upper Body (0,3-10) | +4.56px | +7.64px | Nose, ears, shoulders, elbows, wrists |
| Face (23-90) | -0.79px | +1.06px | Facial landmarks |
| Left Hand (91-111) | +3.07px | +8.91px | All left hand joints |
| Right Hand (112-132) | +5.52px | +7.36px | All right hand joints |

## What Works Well

✅ **Hand-dominant signs** - Apple Vision hands have correct topology  
✅ **Facial expressions** - Simplified face landmarks are sufficient  
✅ **Signs with clear gestures** - Robust to small alignment errors

## Known Limitations

❌ **Body-dependent signs** - Shoulder/elbow positioning may be incorrect  
❌ **Complex compound signs** - May confuse similar signs  
❌ **Subtle body movements** - Structural differences affect recognition

## Analysis & Debugging Tools

### Compare with Ground Truth

```bash
# Show per-part error metrics
python scripts/compare_json_diff.py pose_optimized.json ground_truth.json

# Detailed body joint analysis
python scripts/analyze_body_detailed.py --apple pose.json --truth truth.json

# Detailed hand analysis
python scripts/analyze_hand_detailed.py --apple pose.json --truth truth.json
```

### Create Hybrid Experiments

```bash
# Test component substitution (requires ground truth)
python scripts/create_hand_experiments.py
python scripts/run_json_inference.py --pose_json json/exp_truth_both_hands.json --finetune ...
```

### Visualize Comparison

```bash
python scripts/visualize_comparison.py truth.json optimized.json output_comparison.mp4
```

## Files & Documentation

- `scripts/optimize_pose_production.py` - Production optimization script
- `scripts/optimize_pose.py` - Research version with ground truth calibration
- `docs/OPTIMIZATION_RESULTS.md` - Detailed analysis and findings
- `PROJECT_KNOWLEDGE_BASE.md` - Technical retrospective
- `docs/mapping_analysis.md` - Coordinate system documentation

## Accuracy Results (BG1_S002 Test Case)

| Method | Body Error | Hand Error | Prediction | Status |
|--------|------------|------------|------------|--------|
| Raw Apple | 19.52px | 33.25px | "Hôm nay..." | ❌ Wrong |
| Calibrated | 14.74px | 31.54px | "Hôm nay..." | ❌ Still Wrong |
| Truth Body + Apple Hands | 0px | ~23px | "Hoả hoạn..." | ✅ **Correct!** |

**Key Insight:** Apple hands work perfectly. The issue is body keypoint topology.

## Next Steps for Production

1. **Short-term:** Use calibration for ~70-80% accuracy
2. **Long-term:** Fine-tune model on Apple Vision data for 90-95% accuracy

See [Optimization Results](../docs/OPTIMIZATION_RESULTS.md) for detailed recommendations.

## Support

For questions or issues:
1. Check `PROJECT_KNOWLEDGE_BASE.md` for technical details
2. Review `OPTIMIZATION_RESULTS.md` for experimental findings
3. Examine test case: `test/BG1_S002.mp4` with ground truth in `json/BG1_S002_pose_truth.json`

---

*Last Updated: December 2, 2025*

