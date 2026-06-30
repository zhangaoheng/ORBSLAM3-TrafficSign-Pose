#!/bin/bash
# ============================================================
# 步骤3: 运行 ORB-SLAM3 离线单目+IMU
# 用法: ./scripts/3_run_slam.sh /path/to/euroc_dataset
# ============================================================
set -e

if [ $# -lt 1 ]; then
    echo "用法: ./scripts/3_run_slam.sh /path/to/euroc_dataset"
    echo "示例: ./scripts/3_run_slam.sh dataset_rosbag2_monoimu_20260615_204133"
    exit 1
fi

DATASET="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VOCAB="$PROJECT_DIR/Vocabulary/ORBvoc.txt"
CONFIG="$PROJECT_DIR/Examples/Monocular-Inertial/RealSense_D456.yaml"
EXEC="$PROJECT_DIR/Examples/Monocular-Inertial/mono_inertial_euroc"
LIB_PATH="/opt/ros/lyrical/lib/x86_64-linux-gnu"

echo "═══════════════════════════════════════════"
echo "  ORB-SLAM3 离线单目+IMU"
echo "═══════════════════════════════════════════"
echo "  数据集: $DATASET"
echo "  词库: $VOCAB"
echo "  配置: $CONFIG"
echo ""
echo "  🖥️  Viewer 窗口显示建图结果"
echo "  🛑  Ctrl+C 安全退出"
echo "═══════════════════════════════════════════"

export LD_LIBRARY_PATH="$LIB_PATH:$LD_LIBRARY_PATH"
"$EXEC" "$VOCAB" "$CONFIG" "$DATASET" "$DATASET/trajectory.txt"
