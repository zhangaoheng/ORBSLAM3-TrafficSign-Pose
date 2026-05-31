#!/bin/bash
# ==============================================================
# run_real.sh — RealSense D456 实采数据一键 SLAM 处理
#
# 用法:
#   ./run_real.sh                                 # 单目 (默认)
#   ./run_real.sh mono                            # 单目
#   ./run_real.sh imu_monocular                   # 单目+IMU
#   ./run_real.sh rgbd                            # RGB-D
#   ./run_real.sh rgbd_inertial                   # RGB-D+IMU
#   ./run_real.sh [mode] run_xxx                  # 指定数据目录名
#   ./run_real.sh [mode] /全/路径                 # 绝对路径
# ==============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
ORB_SLAM3_DIR="$SCRIPT_DIR/../../.."
VOCAB="$ORB_SLAM3_DIR/Vocabulary/ORBvoc.txt"

# --- 参数解析 ---
MODE="mono"
DATA_ARG=""

if [ -z "$1" ]; then
    :  # 默认 mono
elif [ "$1" = "mono" ] || [ "$1" = "imu_monocular" ] || [ "$1" = "rgbd" ] || [ "$1" = "rgbd_inertial" ]; then
    MODE="$1"
    DATA_ARG="$2"
else
    DATA_ARG="$1"
fi

# --- 选择配置文件 ---
case "$MODE" in
    mono)           CONFIG="$SCRIPT_DIR/D456_mono.yaml" ;;
    imu_monocular)  CONFIG="$SCRIPT_DIR/D456_mono_imu.yaml" ;;
    rgbd)           CONFIG="$SCRIPT_DIR/D456_RGBD.yaml" ;;
    rgbd_inertial)  CONFIG="$SCRIPT_DIR/D456_RGBD.yaml" ;;
esac

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
DATA_NAME="$(basename "$DATA_DIR")"
[ -d "$DATA_DIR" ] || { echo "❌ 目录不存在: $DATA_DIR"; exit 1; }

echo "=========================================="
echo "  模式:   $MODE"
echo "  数据:   $DATA_NAME"
echo "  目录:   $DATA_DIR"
echo "=========================================="

# --- 准备输入文件 ---
IN_FILE=""
IMU_FILE=""

if [ "$MODE" = "mono" ] || [ "$MODE" = "imu_monocular" ]; then
    IN_FILE="$DATA_DIR/times.txt"
    if [ ! -f "$IN_FILE" ] || [ ! -s "$IN_FILE" ]; then
        if [ -f "$DATA_DIR/associations.txt" ]; then
            echo "生成 times.txt (从 associations.txt 提取 RGB)..."
            awk '{print $1, $2}' "$DATA_DIR/associations.txt" > "$IN_FILE"
            echo "  → $(wc -l < "$IN_FILE") 帧"
        else
            echo "❌ 缺少 times.txt 或 associations.txt"
            exit 1
        fi
    fi
    echo "  RGB:    $DATA_DIR/rgb/"
    echo "  时间戳: $IN_FILE"
fi

if [ "$MODE" = "imu_monocular" ] || [ "$MODE" = "rgbd_inertial" ]; then
    for f in "$DATA_DIR/imu.txt" "$DATA_DIR/imu_data.csv"; do
        [ -f "$f" ] && { IMU_FILE="$f"; break; }
    done
    if [ -z "$IMU_FILE" ]; then
        echo "❌ 缺少 IMU 文件 (imu.txt 或 imu_data.csv)"
        exit 1
    fi
    echo "  IMU:    $IMU_FILE"
fi

if [ "$MODE" = "rgbd" ] || [ "$MODE" = "rgbd_inertial" ]; then
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
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
make -C "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -1

# --- 运行 ---
echo ""
echo "[2/3] 运行 ORB-SLAM3 ($MODE)..."
echo ""

# 切到 output 目录运行，轨迹文件和日志保存在那里
OUTPUT_DIR="$SCRIPT_DIR/../output"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# 构造命令
CMD="$BUILD_DIR/run_real $MODE $VOCAB $CONFIG $DATA_DIR $IN_FILE"
[ -n "$IMU_FILE" ] && CMD="$CMD $IMU_FILE"
CMD="$CMD $DATA_NAME"

echo ">> $CMD"
echo ""
eval "$CMD"

echo ""
echo "[3/3] ✅ 完成!"
# 找最新的 runs 目录
LATEST_RUN=$(ls -1d "$DATA_DIR/runs/"*/ 2>/dev/null | sort -r | head -1)
if [ -n "$LATEST_RUN" ]; then
    echo "📁 输出目录: $LATEST_RUN"
    ls -1 "$LATEST_RUN"
fi