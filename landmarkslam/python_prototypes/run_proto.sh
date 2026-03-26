#!/bin/bash
cd "$(dirname "$0")/.."

# Find latest YOLO output
YOLO_RESULTS=$(ls -t yolo11_pose_results/run_*/yolo_keypoints.txt 2>/dev/null | head -1)

if [ -z "$YOLO_RESULTS" ]; then
    echo "Error: Cannot find YOLO keypoints output (yolo_keypoints.txt)!"
    exit 1
fi

echo "Using YOLO file: $YOLO_RESULTS"

# Activate Python environment
source /home/zah/ORB_SLAM3-master/landmarkslam/yolo_venv/bin/activate

# Run the python prototype
python python_prototypes/compute_sign_pose_proto.py \
    --settings landmarkslam.yaml \
    --images data/20260321_111801/ \
    --yolo "$YOLO_RESULTS" \
    --output output/slam_py_yolo/sign_cloud_py.txt

echo "================================================="
echo "Processing Finished. The script generated 3 output files:"
echo "  1) output/slam_py_yolo/sign_cloud_py.txt (Combined: Sign + Trajectory)"
echo "  2) output/slam_py_yolo/sign_cloud_py_sign.txt (Sign & Corners only)"
echo "  3) output/slam_py_yolo/sign_cloud_py_trajectory.txt (Camera Trajectory only)"
echo ""
echo "You can check the results using your visualization tools, for example:"
echo "python tools_scripts/visualize_txt.py"
echo "python tools_scripts/visualize_interactive.py"
