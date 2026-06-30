#!/bin/bash
# ============================================================
# 实时 SLAM: 直连 D456 相机实时建图
# 用法: ./scripts/run_realtime.sh [轨迹输出.txt]
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VOCAB="$PROJECT_DIR/Vocabulary/ORBvoc.txt"
CONFIG="$PROJECT_DIR/Examples/Monocular-Inertial/RealSense_D456.yaml"
EXEC="$PROJECT_DIR/Examples/Monocular-Inertial/mono_inertial_realsense_D456"
LIB_PATH="/opt/ros/lyrical/lib/x86_64-linux-gnu"

echo "═══════════════════════════════════════════"
echo "  ORB-SLAM3 实时单目+IMU — D456"
echo "═══════════════════════════════════════════"
echo "  ⏳ 启动后静置相机 1-2 秒初始化 IMU"
echo "  🖥️  Viewer 窗口实时显示 3D 地图"
echo "  🛑  Ctrl+C 安全退出"
echo "═══════════════════════════════════════════"

export LD_LIBRARY_PATH="$LIB_PATH:$LD_LIBRARY_PATH"

if [ "$EUID" -ne 0 ]; then
    exec sudo env LD_LIBRARY_PATH="$LIB_PATH" "$EXEC" "$VOCAB" "$CONFIG" "$@"
else
    exec "$EXEC" "$VOCAB" "$CONFIG" "$@"
fi
