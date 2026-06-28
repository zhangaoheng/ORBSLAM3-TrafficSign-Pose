#!/bin/bash
# ==========================================
# ORB-SLAM3 单目+IMU — 20260613_140500 数据集
# 自动编译与运行脚本
# ==========================================
# 用法: bash run_140500.sh
# ==========================================

set -e  # 遇到错误即退出

IMPLEMENT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$IMPLEMENT_DIR/build"
DATA_DIR="$IMPLEMENT_DIR/data"
EUROC_DIR="$DATA_DIR/euroc_20260613_140500"
OUTPUT_DIR="$IMPLEMENT_DIR/output"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ==========================================
# 步骤 1: 提取数据 (db3 → EuRoC 格式)
# ==========================================
echo "=========================================="
echo ">>> [1/4] 提取 ROS2 bag 数据..."
echo "=========================================="

if [ ! -f "$DATA_DIR/euroc_20260613_140500/times.txt" ]; then
    python3 "$DATA_DIR/extract_20260613_140500.py"
    if [ $? -ne 0 ]; then
        echo "❌ 数据提取失败！"
        exit 1
    fi
    echo "✅ 数据提取完成！"
else
    echo "⏭️  已提取过，跳过提取步骤。"
    echo "   如需重新提取，请删除 $EUROC_DIR 再运行。"
fi
echo ""

# ==========================================
# 步骤 2: 编译
# ==========================================
echo "=========================================="
echo ">>> [2/4] 编译 ORB-SLAM3 程序..."
echo "=========================================="

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit

cmake ..
make -j4

if [ $? -ne 0 ]; then
    echo "❌ 编译失败！"
    exit 1
fi
echo "✅ 编译成功！"
echo ""

# ==========================================
# 步骤 3: 运行 ORB-SLAM3 单目+IMU
# ==========================================
echo "=========================================="
echo ">>> [3/4] 运行 ORB-SLAM3 单目+IMU..."
echo "=========================================="

VOCAB_PATH="$IMPLEMENT_DIR/../../Vocabulary/ORBvoc.txt"
CONFIG_PATH="$DATA_DIR/D456i_140500.yaml"
TIMES_PATH="$EUROC_DIR/times.txt"
TRAJECTORY_NAME="dataset_140500"

echo ">>> ⏳ 加载字典和初始化 SLAM 需要一些时间，请耐心等待..."
echo "    数据集: $EUROC_DIR"
echo "    配置:   $CONFIG_PATH"
echo ""

cd "$OUTPUT_DIR" || exit

"$BUILD_DIR/run_mono_imu_140500" \
    "$VOCAB_PATH" \
    "$CONFIG_PATH" \
    "$EUROC_DIR" \
    "$TIMES_PATH" \
    "$TRAJECTORY_NAME"

echo ""
echo "=========================================="
echo ">>> [4/4] 运行结束！"
echo "=========================================="
echo ">>> 轨迹文件已保存至:"
echo "    📂 $OUTPUT_DIR/AllFrames_${TRAJECTORY_NAME}.txt"
echo "    📂 $OUTPUT_DIR/KeyFrames_${TRAJECTORY_NAME}.txt"
echo "=========================================="
