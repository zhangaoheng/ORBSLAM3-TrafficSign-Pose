#!/bin/bash
cd /home/zah/ORB_SLAM3-master/landmarkslam
# Rebuild the C++ 
echo "Rebuilding C++ backend..."
cd build
cmake ..
make -j4
cd ..

YOLO_TXT=$(ls -t /home/zah/ORB_SLAM3-master/landmarkslam/yolo11_pose_results/run_*/yolo_keypoints.txt | head -1)

VOC_FILE="../Vocabulary/ORBvoc.txt"
SETTINGS_FILE="landmarkslam.yaml"
IMAGE_FOLDER="/home/zah/ORB_SLAM3-master/landmarkslam/data/20260321_111801"

echo "Running SLAM..."
./build/run_landmarkslam_yolo "$VOC_FILE" "$SETTINGS_FILE" "$IMAGE_FOLDER" "$YOLO_TXT"
