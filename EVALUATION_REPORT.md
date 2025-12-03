# UniSign + Apple Vision - Evaluation Report

## Overview

- **Date:** 2024-12-03
- **Model:** ST-GCN Encoder + mT5 Decoder (Fine-tuned on Apple Vision poses)
- **Checkpoint:** `output/best_checkpoint_apple.pth`
- **Device:** mps

## Dataset Split

| Split | Samples |
|-------|---------|
| Train | 493 |
| Dev   | 61 |
| Test  | 63 |
| **Total** | **617** |

## Evaluation Results

### Aggregate Metrics

| Metric | Score |
|--------|-------|
| **BLEU-1** | 0.3802 |
| **BLEU-2** | 0.3148 |
| **BLEU-4** | 0.3056 |
| **BLEU (combined)** | 0.3085 |
| **ROUGE-L** | 0.3749 |
| **WER** | 1.0812 |
| **Exact Match** | 30.16% |
| **Partial Match (≥50%)** | 33.33% |

### Metric Explanations

- **BLEU (Bilingual Evaluation Understudy):** Measures n-gram overlap between prediction and reference. Higher is better (0-1).
- **ROUGE-L:** Measures longest common subsequence. Higher is better (0-1).
- **WER (Word Error Rate):** Edit distance normalized by reference length. Lower is better (0-∞).
- **Exact Match:** Percentage of predictions that exactly match the reference.
- **Partial Match:** Percentage of predictions where ≥50% of reference words appear.

## Sample Predictions

### Best Predictions (Highest ROUGE-L)

| # | Video | Ground Truth | Prediction | ROUGE-L |
|---|-------|--------------|------------|--------|
| 1 | TN5_S019 | hãy mở nhanh cửa sổ ra | hãy mở nhanh cửa sổ ra | 1.000 |
| 2 | TN5_S033 | tôi ngứa da, hãy giúp đỡ tôi | tôi ngứa da, hãy giúp đỡ tôi | 1.000 |
| 3 | BG2_S025 | gọi xe cứu thương đi | gọi xe cứu thương đi | 1.000 |
| 4 | BG5_S008 | nơi trú ẩn ở đâu | nơi trú ẩn ở đâu | 1.000 |
| 5 | BG2_S034 | tôi đang bị sốt | tôi đang bị sốt | 1.000 |
| 6 | BG3_S005 | sóng thần đang tới, hãy di tản | sóng thần đang tới, hãy di tản | 1.000 |
| 7 | BG5_S033 | tôi ngứa da, hãy giúp đỡ tôi | tôi ngứa da, hãy giúp đỡ tôi | 1.000 |
| 8 | BG4_S019 | hãy mở nhanh cửa sổ ra | hãy mở nhanh cửa sổ ra | 1.000 |
| 9 | BG4_S023 | bệnh viện ở đâu? | bệnh viện ở đâu? | 1.000 |
| 10 | BG1_S007 | cảnh báo bây giờ đang có lũ | cảnh báo bây giờ đang có lũ | 1.000 |


### Worst Predictions (Lowest ROUGE-L)

| # | Video | Ground Truth | Prediction | ROUGE-L |
|---|-------|--------------|------------|--------|
| 1 | TN5_S038 | tôi yêu gia đình | có thể lấy nước uống ở đâu? | 0.000 |
| 2 | BG5_S038 | tôi yêu gia đình | có thể lấy nước uống ở đâu? | 0.000 |
| 3 | TN5_S041 | học chữ A, B | tôi đang bị sốt | 0.000 |
| 4 | TN3_S045 | @ | tôi ngứa da, hãy giúp đỡ tôi | 0.000 |
| 5 | HP1_S003 | có bão, nước đang dâng cao | hãy mở nhanh cửa sổ ra | 0.000 |
| 6 | BG5_S036 | cảm ơn vì bạn đã giúp | cảnh báo bây giờ đang có lũ | 0.000 |
| 7 | TN5_S036 | cảm ơn vì bạn đã giúp | cảnh báo bây giờ đang có lũ | 0.000 |
| 8 | TN4_S038 | tôi yêu gia đình | có thể lấy nước uống ở đâu? | 0.000 |
| 9 | BG5_S041 | học chữ A, B | tôi ngứa da, hãy giúp đỡ tôi | 0.000 |
| 10 | HP1_S016 | hãy chuẩn bị áo phao | có thể lấy nước uống ở đâu? | 0.000 |


## Detailed Results

<details>
<summary>Click to expand all 63 predictions</summary>

| Video | Ground Truth | Prediction | BLEU-1 | ROUGE-L | WER |
|-------|--------------|------------|--------|---------|-----|
| BG1_S007 | cảnh báo bây giờ đang có lũ | cảnh báo bây giờ đang có lũ | 1.000 | 1.000 | 0.000 |
| BG1_S027 | tôi bị chảy máy tay | tôi bị chảy máy tay | 1.000 | 1.000 | 0.000 |
| BG1_S029 | tôi đau đầu và chóng mặt | tôi đang đau bụng | 0.500 | 0.400 | 0.833 |
| BG1_S032 | tôi mệt mỏi do không ngủ | tôi mệt mỏi do không ngủ | 1.000 | 1.000 | 0.000 |
| BG1_S034 | tôi đang bị sốt | tôi đang bị sốt | 1.000 | 1.000 | 0.000 |
| BG1_S046 | ngày mai tôi đi chợ | hãy tìm người mất | 0.000 | 0.000 | 1.000 |
| BG1_S048 | 10 đôi giày đẹp | báo cáo có hỏa hoạn | 0.000 | 0.000 | 1.250 |
| BG2_S025 | gọi xe cứu thương đi | gọi xe cứu thương đi | 1.000 | 1.000 | 0.000 |
| BG2_S034 | tôi đang bị sốt | tôi đang bị sốt | 1.000 | 1.000 | 0.000 |
| BG2_S035 | xin chào, tôi tên là | tôi ngứa da, hãy giúp đỡ tôi | 0.143 | 0.167 | 1.400 |
| BG2_S043 | thức ăn có ngon không? | hôm nay có vui không vậy? | 0.167 | 0.182 | 1.000 |
| BG2_S049 | hôm nay cửa hàng mở | có hỏa hoạn, nguy hiểm nguy hiểm | 0.000 | 0.000 | 1.400 |
| BG3_S001 | lũ đang đến ngay bây giờ | lũ đang đến ngay bây giờ | 1.000 | 1.000 | 0.000 |
| BG3_S005 | sóng thần đang tới, hãy di tản | sóng thần đang tới, hãy di tản | 1.000 | 1.000 | 0.000 |
| BG3_S009 | giúp đỡ tôi, khẩn cấp khẩn cấp | tôi bị chảy máy tay | 0.000 | 0.000 | 1.000 |
| BG3_S010 | gọi cứu hộ ngay bây giờ đi | gọi cứu hộ ngay bây giờ đi | 1.000 | 1.000 | 0.000 |
| BG3_S021 | tôi đang đau bụng | tôi ngứa da, hãy giúp đỡ tôi | 0.143 | 0.182 | 1.500 |
| BG3_S034 | tôi đang bị sốt | tôi ngứa da, hãy giúp đỡ tôi | 0.143 | 0.182 | 1.500 |
| BG4_S001 | lũ đang đến ngay bây giờ | lũ đang đến ngay bây giờ | 1.000 | 1.000 | 0.000 |
| BG4_S019 | hãy mở nhanh cửa sổ ra | hãy mở nhanh cửa sổ ra | 1.000 | 1.000 | 0.000 |
| BG4_S023 | bệnh viện ở đâu? | bệnh viện ở đâu? | 1.000 | 1.000 | 0.000 |
| BG4_S026 | tôi ngất hãy giúp đỡ tôi | tôi ngứa da, hãy giúp đỡ tôi | 0.714 | 0.769 | 0.333 |
| BG5_S008 | nơi trú ẩn ở đâu | nơi trú ẩn ở đâu | 1.000 | 1.000 | 0.000 |
| BG5_S016 | hãy chuẩn bị áo phao | hãy chuẩn bị áo phao | 1.000 | 1.000 | 0.000 |
| BG5_S033 | tôi ngứa da, hãy giúp đỡ tôi | tôi ngứa da, hãy giúp đỡ tôi | 1.000 | 1.000 | 0.000 |
| BG5_S036 | cảm ơn vì bạn đã giúp | cảnh báo bây giờ đang có lũ | 0.000 | 0.000 | 1.167 |
| BG5_S038 | tôi yêu gia đình | có thể lấy nước uống ở đâu? | 0.000 | 0.000 | 1.750 |
| BG5_S041 | học chữ A, B | tôi ngứa da, hãy giúp đỡ tôi | 0.000 | 0.000 | 1.750 |
| BG5_S047 | tôi không biết | tôi ngứa da, hãy giúp đỡ tôi | 0.143 | 0.200 | 2.000 |
| HP1_S003 | có bão, nước đang dâng cao | hãy mở nhanh cửa sổ ra | 0.000 | 0.000 | 1.000 |
| HP1_S016 | hãy chuẩn bị áo phao | có thể lấy nước uống ở đâu? | 0.000 | 0.000 | 1.400 |
| HP1_S036 | cảm ơn vì bạn đã giúp | tôi ngứa da, hãy giúp đỡ tôi | 0.143 | 0.154 | 1.167 |
| HP1_S047 | tôi không biết | tôi ngứa da, hãy giúp đỡ tôi | 0.143 | 0.200 | 2.000 |
| HP2_S001 | lũ đang đến ngay bây giờ | cảnh báo bây giờ đang có lũ | 0.571 | 0.308 | 1.167 |
| HP2_S013 | con đường này nguy hiểm | tôi ngứa da, hãy giúp đỡ tôi | 0.000 | 0.000 | 1.400 |
| HP2_S018 | hãy tìm người mất | gọi cứu hộ ngay bây giờ đang có lũ | 0.000 | 0.000 | 2.250 |
| HP3_S013 | con đường này nguy hiểm | tôi ngứa da, hãy giúp đỡ tôi | 0.000 | 0.000 | 1.400 |
| HP3_S018 | hãy tìm người mất | tôi bị chảy máy tay | 0.000 | 0.000 | 1.250 |
| HP3_S022 | hãy gọi bác sĩ ngay bây giờ | tôi ngứa da, hãy giúp đỡ tôi | 0.143 | 0.143 | 1.000 |
| HP3_S037 | có thể lấy nước uống ở đâu? | cảnh báo bây giờ đang có lũ | 0.143 | 0.143 | 1.000 |
| HP3_S040 | tôi mua 1 cuốn sách | hôm nay cửa hàng mở | 0.000 | 0.000 | 1.000 |
| HP3_S043 | thức ăn có ngon không? | tôi ngứa da, hãy giúp đỡ tôi | 0.000 | 0.000 | 1.400 |
| HP3_S049 | hôm nay cửa hàng mở | hãy bảo vệ nhà tôi với | 0.000 | 0.000 | 1.200 |
| TN1_S009 | giúp đỡ tôi, khẩn cấp khẩn cấp | tôi ngứa da, hãy giúp đỡ tôi | 0.286 | 0.286 | 1.000 |
| TN1_S011 | có an toàn không? | tôi ngứa da, hãy giúp đỡ tôi | 0.000 | 0.000 | 1.750 |
| TN2_S004 | đang có động đất, hãy chạy nhanh đi | hãy chuẩn bị áo phao | 0.200 | 0.154 | 1.000 |
| TN2_S007 | cảnh báo bây giờ đang có lũ | cảnh báo bây giờ đang có lũ | 1.000 | 1.000 | 0.000 |
| TN2_S035 | xin chào, tôi tên là | tôi ngứa da, hãy giúp đỡ tôi | 0.143 | 0.167 | 1.400 |
| TN2_S040 | tôi mua 1 cuốn sách | tôi ngứa da, hãy giúp đỡ tôi | 0.143 | 0.167 | 1.200 |
| TN2_S046 | ngày mai tôi đi chợ | hãy tìm người mất | 0.000 | 0.000 | 1.000 |
| TN3_S045 | @ | tôi ngứa da, hãy giúp đỡ tôi | 0.000 | 0.000 | 7.000 |
| TN4_S029 | tôi đau đầu và chóng mặt | tôi đang đau bụng | 0.500 | 0.400 | 0.833 |
| TN4_S038 | tôi yêu gia đình | có thể lấy nước uống ở đâu? | 0.000 | 0.000 | 1.750 |
| TN4_S045 | @ | tôi ngứa da, hãy giúp đỡ tôi | 0.000 | 0.000 | 7.000 |
| TN5_S005 | sóng thần đang tới, hãy di tản | sóng thần đang tới, hãy di tản | 1.000 | 1.000 | 0.000 |
| TN5_S012 | nhà tôi bị ướt | có hỏa hoạn, nguy hiểm nguy hiểm | 0.000 | 0.000 | 1.750 |
| TN5_S019 | hãy mở nhanh cửa sổ ra | hãy mở nhanh cửa sổ ra | 1.000 | 1.000 | 0.000 |
| TN5_S024 | khẩn cấp sốt cao | tôi đang bị sốt | 0.250 | 0.250 | 1.000 |
| TN5_S033 | tôi ngứa da, hãy giúp đỡ tôi | tôi ngứa da, hãy giúp đỡ tôi | 1.000 | 1.000 | 0.000 |
| TN5_S036 | cảm ơn vì bạn đã giúp | cảnh báo bây giờ đang có lũ | 0.000 | 0.000 | 1.167 |
| TN5_S038 | tôi yêu gia đình | có thể lấy nước uống ở đâu? | 0.000 | 0.000 | 1.750 |
| TN5_S041 | học chữ A, B | tôi đang bị sốt | 0.000 | 0.000 | 1.000 |
| TN5_S042 | mẹ của tôi đang ở nhà | hãy bảo vệ nhà tôi với | 0.333 | 0.167 | 1.000 |

</details>



## Analysis

### Key Observations

1. **BLEU-1 Score (0.3802):** Measures unigram overlap. Good for low-resource setting.

2. **ROUGE-L Score (0.3749):** Measures sequence similarity. Reasonable performance on unseen data.

3. **WER (1.0812):** High word error rate indicates significant room for improvement.

4. **Exact Match (30.16%):** Some exact matches, typical for translation tasks.

5. **Partial Match (33.33%):** 33.3% of predictions capture at least half of the reference words.

### Recommendations for Improvement

1. **Data Augmentation:** Apply temporal augmentation, keypoint jittering, and video augmentation.
2. **More Training Data:** Current 493 samples is limited. Consider collecting more data.
3. **Hyperparameter Tuning:** Experiment with learning rates, batch sizes, and model architecture.
4. **Ensemble Methods:** Combine multiple models for better predictions.
5. **Pre-training:** Use larger sign language datasets for pre-training before fine-tuning.

## Conclusion

The Apple Vision + UniSign model achieved:
- **BLEU-4:** 0.3085
- **ROUGE-L:** 0.3749
- **WER:** 1.0812

on the 63-sample test set. With only 493 training samples, the model shows promising capability to translate Vietnamese Sign Language to text.

---
*Generated by evaluate_test_set.py*
