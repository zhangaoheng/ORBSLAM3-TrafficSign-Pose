#!/bin/bash
# RGB-D runner with auto-retry
# Keeps running until completion, saving per-map checkpoints every 1000 frames

RUN_DIR="/home/zah/ORB_SLAM3-master"
BIN="$RUN_DIR/landmarkslam/implement/slam_outside_real/build/run_real"
VOCAB="$RUN_DIR/Vocabulary/ORBvoc.txt"
CONFIG="$RUN_DIR/landmarkslam/implement/slam_outside_real/D456_RGBD.yaml"
DATA="$RUN_DIR/landmarkslam/implement/data/extracted_data_new"
ASSOC="$DATA/associations.txt"
RUN_NAME="rgbd_final"

MAX_RETRIES=5
RETRY=0

cd "$RUN_DIR"

while [ $RETRY -lt $MAX_RETRIES ]; do
    RETRY=$((RETRY + 1))
    echo ""
    echo "========================================"
    echo "  Attempt $RETRY / $MAX_RETRIES"
    echo "  $(date)"
    echo "========================================"
    
    $BIN rgbd "$VOCAB" "$CONFIG" "$DATA" "$ASSOC" "${RUN_NAME}_try${RETRY}"
    EXIT=$?
    
    echo ""
    echo "  Exit code: $EXIT at $(date)"
    
    if [ $EXIT -eq 0 ]; then
        echo "  🎉 SUCCESS! Run completed."
        echo "$(date) - Success on attempt $RETRY" >> "$DATA/rgbd_retry_log.txt"
        exit 0
    fi
    
    echo "  ⚠️  Crashed (exit $EXIT). Restarting in 5 seconds..."
    sleep 5
done

echo "  ❌ Failed after $MAX_RETRIES attempts"
echo "$(date) - Failed after $MAX_RETRIES attempts" >> "$DATA/rgbd_retry_log.txt"
exit 1
