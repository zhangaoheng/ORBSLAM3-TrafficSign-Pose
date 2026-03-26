#!/bin/bash
cd "$(dirname "$0")/.."

# Ensure output directory exists
mkdir -p output2

# Set paths
SETTINGS_FILE="landmarkslam.yaml"
IMAGE_FOLDER="data/20260321_111801/"
YOLO_RESULTS=$(ls -t /home/zah/ORB_SLAM3-master/landmarkslam/yolo11_pose_results/run_*/yolo_keypoints.txt 2>/dev/null | head -1)

if [ -z "$YOLO_RESULTS" ] || [ ! -f "$YOLO_RESULTS" ]; then
    echo "Error: Cannot find YOLO keypoints output (yolo_keypoints.txt)!"
    exit 1
fi
OUTPUT_PLY="output2/sign_cloud.ply"

echo "=========================================="
echo " Starting Sign Pose Computation & Meshing "
echo "=========================================="
echo "Settings : $SETTINGS_FILE"
echo "Images   : $IMAGE_FOLDER"
echo "YOLO     : $YOLO_RESULTS"
echo "Output   : $OUTPUT_PLY"
echo "=========================================="

# Check if the executable exists
if [ ! -f "build/compute_sign_pose" ]; then
    echo "Executable build/compute_sign_pose not found. Compiling..."
    cd build
    cmake ..
    make compute_sign_pose -j4
    cd ..
fi

# Run the executable
./build/compute_sign_pose "$SETTINGS_FILE" "$IMAGE_FOLDER" "$YOLO_RESULTS" "$OUTPUT_PLY"

echo "=========================================="
echo " Process Finished."
echo " You can view $OUTPUT_PLY with MeshLab."
echo "=========================================="
