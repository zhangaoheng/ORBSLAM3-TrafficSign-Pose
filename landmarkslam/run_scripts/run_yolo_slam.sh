#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

echo "==========================================="
echo "   1. Building Landmark SLAM (YOLO Fusion) "
echo "==========================================="
mkdir -p build
cd build
cmake ..
make -j4
if [ $? -ne 0 ]; then
    echo "Error: 编译失败！"
    exit 1
fi
cd ..

echo "==========================================="
echo "   2. Running YOLO11 Pose Estimation       "
echo "==========================================="
echo "正在执行 YOLO 预测并生成角点数据..."
cd yolo11

# 激活虚拟环境以运行YOLO
source /home/zah/yolo_venv/bin/activate
python test_yolo11_pose.py
if [ $? -ne 0 ]; then
    echo "Error: YOLO 推理脚本运行失败！请确保你激活了 Python 环境 (yolo_venv)。"
    exit 1
fi
cd ..

# 动态查找最新生成的 yolo_keypoints.txt
YOLO_TXT=$(ls -t /home/zah/ORB_SLAM3-master/landmarkslam/yolo11_pose_results/run_*/yolo_keypoints.txt 2>/dev/null | head -1)

if [ -z "$YOLO_TXT" ] || [ ! -f "$YOLO_TXT" ]; then
    echo "Error: 找不到最新生成的 YOLO 检测结果 (yolo_keypoints.txt)！"
    exit 1
fi

echo ">> 成功找到最新 YOLO 角点数据: $YOLO_TXT"

# ==================================
# 3. 准备运行 C++ SLAM 融合程序
# ==================================
VOC_FILE="../Vocabulary/ORBvoc.txt"
SETTINGS_FILE="landmarkslam.yaml"
IMAGE_FOLDER="/home/zah/ORB_SLAM3-master/landmarkslam/data/20260321_111801" 

# 如果运行脚本时带了参数，可以直接用传入的参数覆盖原来的图片路径
if [ "$#" -eq 1 ]; then
    IMAGE_FOLDER=$1
fi

echo "==========================================="
echo "   3. Running SLAM with YOLO Integration   "
echo "==========================================="
echo "Vocabulary : $VOC_FILE"
echo "Settings   : $SETTINGS_FILE"
echo "Image Dir  : $IMAGE_FOLDER"
echo "YOLO Data  : $YOLO_TXT"
echo "-------------------------------------------"

mkdir -p log output

# 运行刚才修改的新版本融合程序
./build/run_landmarkslam_yolo "$VOC_FILE" "$SETTINGS_FILE" "$IMAGE_FOLDER" "$YOLO_TXT"

