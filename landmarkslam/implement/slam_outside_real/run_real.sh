#!/bin/bash
# ==============================================================
# run_real.sh — RealSense D456 实采数据一键 SLAM 处理
#
# 用法:
#   ./run_real.sh                    # 单目模式 (默认)
#   ./run_real.sh mono               # 单目模式
#   ./run_real.sh rgbd               # RGB-D 模式
#   ./run_real.sh [mode] run_xxx     # 指定数据目录
#   ./run_real.sh [mode] /全/路径     # 任意路径
# ==============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
ORB_SLAM3_DIR="$SCRIPT_DIR/../../.."
VOCAB="$ORB_SLAM3_DIR/Vocabulary/ORBvoc.txt"

# --- 参数解析 ---
MODE="mono"  # 默认单目
DATA_ARG=""

if [ -z "$1" ]; then
    :  # 双默认
elif [ "$1" = "mono" ] || [ "$1" = "rgbd" ]; then
    MODE="$1"
    DATA_ARG="$2"
else
    DATA_ARG="$1"
fi

if [ "$MODE" = "rgbd" ]; then
    CONFIG="$SCRIPT_DIR/D456_RGBD.yaml"
else
    CONFIG="$SCRIPT_DIR/D456_mono.yaml"
fi

# --- 解析数据路径 ---
if [ -z "$DATA_ARG" ]; then
    LATEST=$(ls -1d "$SCRIPT_DIR/../data/real/run_"* 2>/dev/null | sort -r | head -1)
    if [ -z "$LATEST" ]; then
        echo "❌ 未找到 data/real/run_* 目录"
        exit 1
    fi
    DATA_DIR="$LATEST"
elif [[ "$DATA_ARG" == /* ]]; then
    DATA_DIR="$DATA_ARG"
else
    DATA_DIR="$SCRIPT_DIR/../data/real/$DATA_ARG"
fi

DATA_DIR="$(cd "$DATA_DIR" 2>/dev/null && pwd)"
[ -d "$DATA_DIR" ] || { echo "❌ 目录不存在: $DATA_DIR"; exit 1; }

echo "=========================================="
echo "  模式:   $MODE"
echo "  数据:   $(basename "$DATA_DIR")"
echo "=========================================="

# --- 准备输入文件 ---
if [ "$MODE" = "mono" ]; then
    IN_FILE="$DATA_DIR/times.txt"
    if [ ! -f "$IN_FILE" ]; then
        echo "生成 times.txt (从 associations.txt 提取 RGB)..."
        awk '{print $1, $2}' "$DATA_DIR/associations.txt" > "$IN_FILE"
        echo "  → $(wc -l < "$IN_FILE") 帧"
    fi
    echo "  RGB:    $DATA_DIR/rgb/"
    echo "  时间戳: $IN_FILE"
else
    IN_FILE="$DATA_DIR/associations.txt"
    [ -f "$IN_FILE" ] || { echo "❌ 缺少: $IN_FILE"; exit 1; }
    echo "  RGB:    $DATA_DIR/rgb/"
    echo "  Depth:  $DATA_DIR/depth/"
    echo "  关联:   $IN_FILE"
fi

# --- 编译 ---
echo ""
echo "[1/3] 编译..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null
make -j"$(nproc)" 2>&1 | tail -1

# --- 运行 ---
echo ""
echo "[2/3] 运行 ORB-SLAM3 ($MODE)..."
echo ""

OUTPUT_DIR="$SCRIPT_DIR/../output"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

"$BUILD_DIR/run_real" \
    "$MODE" \
    "$VOCAB" \
    "$CONFIG" \
    "$DATA_DIR" \
    "$IN_FILE" \
    "$(basename "$DATA_DIR")"

echo ""
echo "[3/3] ✅ 完成!"
ls -1 "$OUTPUT_DIR"/AllFrames_* "$OUTPUT_DIR"/KeyFrames_* 2>/dev/null
