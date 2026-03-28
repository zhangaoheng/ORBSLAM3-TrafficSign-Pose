#!/bin/bash

# 获取脚本所在目录并切换到项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

echo "==========================================="
echo "   1. Building Landmark SLAM (YOLO Fusion) "
echo "==========================================="
# 建议在运行前清理 build 缓存以确保代码改动生效
# rm -rf build/* mkdir -p build
cd build
cmake ..
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo "Error: 编译失败！"
    exit 1
fi
cd ..

# ============================================
# 2. 数据路径配置 (已切换为手动清洗后的固定路径)
# ============================================
# 这里指向你手动处理过的文件夹和对应的关键点文件
DATA_DIR="/home/zah/ORB_SLAM3-master/landmarkslam/data/20260321_111801"
YOLO_TXT="/home/zah/ORB_SLAM3-master/landmarkslam/yolo11_pose_results/run_20260322_202936/yolo_keypoints.txt"

# 检查文件是否存在
if [ ! -f "$YOLO_TXT" ]; then
    echo "Error: 找不到手动清洗的 YOLO 数据文件: $YOLO_TXT"
    exit 1
fi

echo ">> 使用手动清洗后的数据源: $DATA_DIR"
echo ">> 关键点文件: $YOLO_TXT"

# ==================================
# 3. 准备运行 C++ SLAM 融合程序
# ==================================
VOC_FILE="../Vocabulary/ORBvoc.txt"
SETTINGS_FILE="landmarkslam.yaml"
# 使用与数据源配套的图片文件夹
IMAGE_FOLDER="$DATA_DIR" 

# 如果运行脚本时手动传入了其他图片路径，则覆盖默认路径
if [ "$#" -eq 1 ]; then
    IMAGE_FOLDER=$1
fi

echo "==========================================="
echo "   3. Running SLAM with Fixed Manual Data  "
echo "==========================================="
echo "Vocabulary  : $VOC_FILE"
echo "Settings    : $SETTINGS_FILE"
echo "Image Dir   : $IMAGE_FOLDER"
echo "YOLO Data   : $YOLO_TXT"
echo "-------------------------------------------"

# 确保输出目录存在
mkdir -p log output output2

# 运行融合程序
./build/run_landmarkslam_yolo "$VOC_FILE" "$SETTINGS_FILE" "$IMAGE_FOLDER" "$YOLO_TXT"

if [ $? -eq 0 ]; then
    echo "==========================================="
    echo "   🎉 运行成功！结果已保存至 output2 目录"
    echo "==========================================="
else
    echo "Error: 程序运行过程中出现异常。"
fi