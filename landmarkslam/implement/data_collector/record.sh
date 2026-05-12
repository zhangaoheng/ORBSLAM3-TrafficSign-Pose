#!/usr/bin/env bash
# ==============================================================================
# 🔴 一键数据采集
# 用法:
#   bash record.sh sunny_run1            # 默认 GPS 串口 /dev/ttyACM0
#   bash record.sh night_run1 /dev/ttyUSB0  # 指定 GPS 串口
# ==============================================================================
set -e

NAME="${1:?请指定实验名称，如: bash record.sh sunny_run1}"
GPS_PORT="${2:-/dev/ttyACM0}"
GPS_BAUD="${3:-115200}"

VENV_PYTHON="/home/zah/ORB_SLAM3-master/landmarkslam/yolo_venv/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  🌀 数据采集"
echo "  实验名称: $NAME"
echo "  GPS 串口: $GPS_PORT"
echo "========================================"

"$VENV_PYTHON" "$SCRIPT_DIR/collect_data.py" \
    --name "$NAME" \
    --gps-port "$GPS_PORT" \
    --gps-baud "$GPS_BAUD"
