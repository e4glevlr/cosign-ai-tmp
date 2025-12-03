#!/bin/bash
"""
Batch extract poses from videos using Apple Vision PoseExtractor.

This script processes all videos in a directory and extracts poses
using the Apple Vision framework via the PoseExtractor binary.

Usage:
    ./scripts/batch_extract_apple.sh <video_dir> <output_dir> [--parallel N]
    
Example:
    ./scripts/batch_extract_apple.sh ./dataset/VSL/rgb_format ./dataset/VSL/pose_json_apple
"""

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <video_dir> <output_dir> [--no-viz]"
    echo ""
    echo "Arguments:"
    echo "  video_dir   Directory containing .mp4 video files"
    echo "  output_dir  Directory to save extracted pose JSON files"
    echo "  --no-viz    Skip visualization video generation (faster)"
    exit 1
fi

VIDEO_DIR="$1"
OUTPUT_DIR="$2"
NO_VIZ=false

if [ "$3" == "--no-viz" ]; then
    NO_VIZ=true
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
POSE_EXTRACTOR="$PROJECT_ROOT/bin/PoseExtractor"

# Check if PoseExtractor exists
if [ ! -f "$POSE_EXTRACTOR" ]; then
    echo "Error: PoseExtractor not found at $POSE_EXTRACTOR"
    echo "Please build PoseExtractor first."
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
if [ "$NO_VIZ" = false ]; then
    mkdir -p "$OUTPUT_DIR/viz"
fi

# Count videos
VIDEO_COUNT=$(find "$VIDEO_DIR" -name "*.mp4" -type f | wc -l | tr -d ' ')
echo "Found $VIDEO_COUNT videos in $VIDEO_DIR"

if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "No .mp4 files found in $VIDEO_DIR"
    exit 1
fi

# Process videos
PROCESSED=0
FAILED=0
START_TIME=$(date +%s)

echo ""
echo "Starting extraction..."
echo "========================================"

for VIDEO in "$VIDEO_DIR"/*.mp4; do
    if [ ! -f "$VIDEO" ]; then
        continue
    fi
    
    BASENAME=$(basename "$VIDEO" .mp4)
    JSON_PATH="$OUTPUT_DIR/${BASENAME}.json"
    
    # Skip if already exists
    if [ -f "$JSON_PATH" ]; then
        echo "[SKIP] $BASENAME (already exists)"
        ((PROCESSED++))
        continue
    fi
    
    if [ "$NO_VIZ" = true ]; then
        VIZ_PATH="/dev/null"
    else
        VIZ_PATH="$OUTPUT_DIR/viz/${BASENAME}.mov"
    fi
    
    echo -n "[$(($PROCESSED + 1))/$VIDEO_COUNT] $BASENAME ... "
    
    if "$POSE_EXTRACTOR" "$VIDEO" "$VIZ_PATH" "$JSON_PATH" 2>/dev/null; then
        echo "✓"
        ((PROCESSED++))
    else
        echo "✗ (failed)"
        ((FAILED++))
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "Extraction complete!"
echo "  Processed: $PROCESSED"
echo "  Failed:    $FAILED"
echo "  Time:      ${ELAPSED}s"
echo ""
echo "Output saved to: $OUTPUT_DIR"
echo ""
echo "Next step: Convert to pickle format for training"
echo "  python scripts/convert_apple_to_pkl.py batch -i $OUTPUT_DIR -o ${OUTPUT_DIR}_pkl"
