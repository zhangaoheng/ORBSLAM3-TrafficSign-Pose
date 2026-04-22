#!/bin/bash

# ==========================================
# ORB-SLAM3 单目+IMU 自动编译与运行脚本
# ==========================================

# 1. 定义工作目录
IMPLEMENT_DIR="/home/zah/ORB_SLAM3-master/landmarkslam/implement"
BUILD_DIR="$IMPLEMENT_DIR/build"

# 1.1 定义你要处理的数据集前缀 (便于你随便切换 lines1 还是 lines2)
DATA_SEQ="lines1"

# 【⭐ 新增】定义你想保存结果的专属文件夹路径
# 你可以随意修改这个路径，比如放到 data 文件夹下
OUTPUT_DIR="/home/zah/ORB_SLAM3-master/landmarkslam/implement/output"

# 2. 定义运行参数路径 (全部使用绝对路径，拒绝任何路径歧义)
VOCAB_PATH="/home/zah/ORB_SLAM3-master/Vocabulary/ORBvoc.txt"
CONFIG_PATH="/home/zah/ORB_SLAM3-master/landmarkslam/D456.yaml"
CAM0_PATH="/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/$DATA_SEQ/cam0/data"
TIMES_PATH="/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/$DATA_SEQ/times.txt"
IMU_PATH="/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/$DATA_SEQ/imu0/data.csv"
OUTPUT_PREFIX="lines1_slam_traj"

echo "=========================================="
echo ">>> [1/3] 进入编译环境并检查更新..."
echo "=========================================="

# 如果 build 文件夹不存在，自动创建
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit

# 自动编译 (如果有新的 C++ 代码改动，会自动编译；如果没有改动，会瞬间跳过)
cmake ..
make -j4

# 拦截编译错误：如果 make 失败，立刻停止脚本，不往下运行
if [ $? -ne 0 ]; then
    echo -e "\n❌ 编译失败！请检查上方 C++ 报错信息。"
    exit 1
fi
echo -e "✅ 编译成功或已经是最新版本！\n"

echo "=========================================="
echo ">>> [2/3] 准备输出目录并运行 ORB-SLAM3..."
echo "=========================================="

# 【⭐ 新增核心逻辑】: 检查并进入你自己指定的保存目录
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR" || exit

echo ">>> ⏳ 提示: 加载 140MB 的字典需要一些时间，请耐心等待..."

# 3. 执行 SLAM 核心程序 
# (注意这里加上了 $BUILD_DIR/ 前缀，因为我们现在不在 build 目录里了)
"$BUILD_DIR/run_mono_imu" \
    "$VOCAB_PATH" \
    "$CONFIG_PATH" \
    "$CAM0_PATH" \
    "$TIMES_PATH" \
    "$IMU_PATH" \
    "$OUTPUT_PREFIX"

echo -e "\n=========================================="
echo ">>> [3/3] 运行结束！"
echo "=========================================="
echo ">>> 您的完整轨迹文件已干净、安全地保存在您指定的路径："
echo "    📂 $OUTPUT_DIR/all_frames_${OUTPUT_PREFIX}.txt"
echo "    📂 $OUTPUT_DIR/keyframes_${OUTPUT_PREFIX}.txt"
echo "=========================================="