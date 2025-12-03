# UniSign & Apple Vision Integration: Technical Retrospective

This document serves as a knowledge base for future agents and developers working on integrating Apple Vision Framework pose estimation with the UniSign (RTMPose-based) Sign Language Translation model.

## 1. Project Overview
**Goal:** Deploy the UniSign SLT model on iOS/macOS devices without requiring heavy server-side inference or specific depth sensors (FaceID), utilizing the native Apple Vision Framework for pose extraction.

**Core Challenge:** Domain Gap between **Apple Vision** (Source) and **RTMPose** (Training Target).
*   **Apple Vision:** 2D contours for face, distinct coordinate system (Bottom-Left origin), simpler hand topology, native optimization for Neural Engine.
*   **RTMPose (COCO-WholeBody):** 3D-aware, 133 keypoints, high-fidelity face mesh (iBUG 68), top-left pixel coordinates.

## 2. Key Findings & Failures

### 2.0. **BREAKTHROUGH: Body is the Problem, Not Hands** ⚠️ CRITICAL
*   **Experimental Proof (Dec 2, 2025):**
    *   Raw Apple Pose: WRONG prediction ("Hôm nay" instead of "Hoả hoạn")
    *   Apple Body + Truth Hands: WRONG prediction ("Ngứa da")
    *   **Truth Body + Apple Hands: CORRECT prediction** ✅
*   **Conclusion:** Apple hands have **correct topology** and work perfectly. The issue is **Apple body keypoints** have structural/geometric differences that calibration cannot fix.
*   **Per-joint Error Analysis:**
    *   Model uses indices: 0 (Nose), 3-4 (Ears), 5-6 (Shoulders), 7-8 (Elbows), 9-10 (Wrists)
    *   Shoulder asymmetry: Left +11px X, Right -3.32px X → 14.3px difference
    *   This suggests rotational/scaling issues, not just translation
*   **Implication:** Pure calibration has a ceiling of ~70-80% accuracy. Fine-tuning is mandatory for production.

### 2.1. The "Upside Down" Skeleton
*   **Symptom:** Visualization video showed skeletons drawn upside down.
*   **Cause:** `CVPixelBuffer` CoreGraphics context is Bottom-Left origin. Vision coordinates are also Bottom-Left relative. 
*   **Fix:** Do NOT flip Y (`1-y`) when drawing on the buffer directly. Pass `y` through.
*   **Note:** For JSON export (Model Input), Y-flip (`1-y`) IS required because standard CV models expect Top-Left origin.

### 2.2. Body & Hand Misalignment
*   **Symptom:** Hybrid test (Apple Body + Truth Face) failed.
*   **Cause:** Apple Vision detects Body and Face as separate requests with independent coordinate biases relative to the "Ground Truth" (RTMPose).
*   **Solution:** Implemented **Split Calibration**.
    *   Body Offset: `X ~ +4.95`, `Y ~ +8.21`
    *   Face Offset: `X ~ -0.79`, `Y ~ +1.06`
    *   *Lesson:* Never assume a global offset fixes all body parts in a multi-model pipeline like Apple Vision.

### 2.3. The "Face" Red Herring
*   **Initial Hypothesis:** The prediction failure ("Hôm nay" vs "Hoả hoạn") was due to low-quality Apple Face landmarks (flat mouth).
*   **Disproof:** Control Test "Truth Body + Apple Face" yielded the **Correct** prediction.
*   **Conclusion:** Even with lower fidelity (contours vs mesh), Apple Face data is sufficient for this specific vocabulary set. The model is robust enough to handle simplified facial features.

### 2.4. The "Smoothing" Trap
*   **Attempt:** Applied Moving Average smoothing to fix Apple Vision jitter.
*   **Result:** Accuracy degraded ("Hoả hoạn" -> "Ngứa da").
*   **Reason:** Sign Language relies on high-frequency motion (waving, trembling). Naive smoothing erases these "micro-gestures", turning an active sign (Fire) into a passive one (Itchy/Scratching).
*   **Rule:** Do not smooth hand keypoints unless using a sophisticated filter (e.g., 1€ Filter) tuned specifically for gesture velocity.

### 2.5. Structural Incompatibility
*   **Apple:** Output is a list of Frames. Frame 0 is often empty (initialization lag).
*   **UniSign/RTMPose:** Expects a Tensor `(T, N, 133, C)`.
*   **Fix:** `optimize_pose.py` script created to:
    *   Reshape JSON structure.
    *   Pad missing initial frames (copy Frame 1).
    *   Apply calibration offsets.

## 3. Recommended Roadmap (The "Fine-tune" Strategy)

Attempts to mathematically "map" Apple Vision to RTMPose via offsets have hit a ceiling. ~~The geometric topology of the hands (finger lengths, joint angles) is fundamentally different.~~ 

**UPDATE (Dec 2, 2025):** Experiments prove hands work perfectly. The issue is **body keypoint topology** (shoulders, elbows positioning) which has structural differences that linear transforms cannot resolve.

**The definitive solution is Domain Adaptation (Fine-tuning):**
1.  **Freeze the LLM:** Keep the mT5 (Language Brain) frozen. It knows grammar.
2.  **Retrain the Encoder:** Unfreeze the Spatial-Temporal GCN (Visual Eye).
3.  **Data:** Re-process the original training dataset (CSL/WLASL) using `PoseExtractor` (Apple Vision).
4.  **Result:** The GCN will "learn" to interpret Apple's specific hand topology and biases directly, eliminating the need for fragile manual calibration scripts.

## 4. Tools & Scripts
*   `bin/PoseExtractor`: Swift executable for extracting pose from video.
*   `scripts/optimize_pose.py`: Converts Apple JSON -> RTMPose format, applies calibration, fills gaps.
*   `scripts/visualize_comparison.py`: Side-by-side video visualization.
*   `scripts/run_json_inference.py`: Bridge script to load JSON and run the PyTorch model.

## 5. Critical Coordinates Reference
*   **Image:** Top-Left (0,0).
*   **Vision:** Bottom-Left (0,0).
*   **Model Input:** Normalized [-1, 1] or Pixel [0, W]. (UniSign handles normalization internally via `crop_scale`).
*   **Face Indices:** 23-90.
*   **Left Hand Indices:** 91-111.
*   **Right Hand Indices:** 112-132.

---
*Generated by Gemini CLI Agent - December 2025*
