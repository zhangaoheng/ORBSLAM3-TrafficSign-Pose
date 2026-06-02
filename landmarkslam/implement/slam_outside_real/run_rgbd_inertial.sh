#!/bin/bash
# ============================================================
# run_rgbd_inertial.sh — 一键编译 + 运行 RGB-D + IMU
# ============================================================
set -e
cd "$(dirname "$0")/../../.."
PROJECT_DIR="$(pwd)"
echo "=========================================="
echo "  ORB-SLAM3 RGB-D + IMU 一键运行"
echo "  项目目录: $PROJECT_DIR"
echo "=========================================="

# 1. 编译
echo ""
echo "[1/2] 编译..."
BUILD_DIR="$PROJECT_DIR/landmarkslam/implement/slam_outside_real/build"
mkdir -p "$BUILD_DIR"
cmake -S "$PROJECT_DIR/landmarkslam/implement/slam_outside_real" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
make -C "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -1
echo "  ✅ 编译完成"

# 2. 运行
echo ""
echo "[2/2] 运行 RGB-D + IMU..."
echo ""

# ===== 配置区域 =====
# 修改这里的路径指向你的数据
DATA_DIR="$PROJECT_DIR/landmarkslam/implement/data/extracted_data"
ASSOC_FILE="$DATA_DIR/associations.txt"
IMU_FILE="$DATA_DIR/imu.txt"
CONFIG_FILE="$PROJECT_DIR/landmarkslam/implement/slam_outside_real/D456_RGBD.yaml"
VOCAB_FILE="$PROJECT_DIR/Vocabulary/ORBvoc.txt"
BIN="$BUILD_DIR/run_real"
OUTPUT_NAME="rgbd_inertial_$(date +%Y%m%d_%H%M%S)"
# ===================

# 检查必要文件
for f in "$BIN" "$VOCAB_FILE" "$CONFIG_FILE" "$DATA_DIR" "$ASSOC_FILE" "$IMU_FILE"; do
    if [ ! -e "$f" ]; then
        echo "  ❌ 缺少: $f"
        exit 1
    fi
done

echo "  数据: $DATA_DIR"
echo "  关联: $ASSOC_FILE"
echo "  IMU:  $IMU_FILE"
echo "  输出: $OUTPUT_NAME"
echo ""

"$BIN" rgbd_inertial "$VOCAB_FILE" "$CONFIG_FILE" "$DATA_DIR" "$ASSOC_FILE" "$IMU_FILE" "$OUTPUT_NAME"

echo ""
echo "=========================================="
echo "  🏁 运行结束"
echo "=========================================="
