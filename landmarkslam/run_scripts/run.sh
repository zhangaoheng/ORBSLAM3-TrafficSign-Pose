#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

echo "==========================================="
echo "   Building Landmark SLAM                  "
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

# 1. 词典文件路径 (这是 ORB_SLAM3 原生词典位置)
VOC_FILE="../Vocabulary/ORBvoc.txt"

# 2. 相机参数和配置文件路径
SETTINGS_FILE="landmarkslam_real.yaml"

# 3. 图像数据集文件夹路径 (请在这里修改为你的实际图像文件夹)
IMAGE_FOLDER="/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data" 

# 如果运行脚本时带了参数，可以直接用传入的参数覆盖原来的图片路径
if [ "$#" -eq 1 ]; then
    IMAGE_FOLDER=$1
fi

echo "==========================================="
echo "   Running Landmark SLAM (Pure ORB-SLAM3)  "
echo "==========================================="
echo "Vocabulary : $VOC_FILE"
echo "Settings   : $SETTINGS_FILE"
echo "Image Dir  : $IMAGE_FOLDER"
echo "-------------------------------------------"

n# 创建独立的日志文件夹以供 C++ 端使用
mkdir -p log

# 执行程序
./build/run_landmarkslam "$VOC_FILE" "$SETTINGS_FILE" "$IMAGE_FOLDER"
