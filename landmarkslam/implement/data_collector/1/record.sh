#!/usr/bin/env bash
# ==============================================================================
# 🔴 一键同步采集 + 实时 ORB-SLAM3
# 用法:
#   bash record.sh                          # 按启动时间自动命名
#   bash record.sh sunny_run1               # 手动指定名称
#   bash record.sh night_run1 /dev/ttyUSB0  # 指定 GPS 串口
# 默认保存到:
#   /run/media/ehuy/文件夹/recordings/live_YYYYmmdd_HHMMSS
# ==============================================================================
set -Eeuo pipefail

NAME="${1:-}"
GPS_PORT="${2:-auto}"
GPS_BAUD="${3:-115200}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
STORAGE_MOUNT="/run/media/ehuy/文件夹"
OUTPUT_ROOT="${RECORDING_ROOT:-$STORAGE_MOUNT/recordings}"
# One hour of 848x480@30 lossless monocular PNG plus IMU/GPS is about 60 GiB
# on the existing dataset. Keep additional headroom for scene complexity.
MIN_FREE_GB="${MIN_FREE_GB:-80}"

# Never silently fall back to the system disk if the recording disk is absent.
MOUNT_TARGET="$(findmnt -n -o TARGET -T "$STORAGE_MOUNT" 2>/dev/null || true)"
if [[ ! -d "$STORAGE_MOUNT" ]] || [[ "$MOUNT_TARGET" != "$STORAGE_MOUNT" ]]; then
    echo "错误：录制盘未挂载：$STORAGE_MOUNT" >&2
    echo "请先在文件管理器中挂载‘文件夹’磁盘，再重新运行。" >&2
    exit 2
fi

MOUNT_OPTIONS="$(findmnt -n -o OPTIONS -T "$STORAGE_MOUNT")"
if [[ ",$MOUNT_OPTIONS," == *,ro,* ]]; then
    echo "错误：录制盘当前为只读挂载，不能保存数据：$STORAGE_MOUNT" >&2
    echo "请修复磁盘或重新以读写方式挂载后再运行。" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
if [[ ! -w "$OUTPUT_ROOT" ]]; then
    echo "错误：录制目录不可写：$OUTPUT_ROOT" >&2
    exit 2
fi

AVAILABLE_BYTES="$(df -PB1 "$OUTPUT_ROOT" | awk 'NR == 2 {print $4}')"
MIN_FREE_BYTES=$((MIN_FREE_GB * 1024 * 1024 * 1024))
if (( AVAILABLE_BYTES < MIN_FREE_BYTES )); then
    AVAILABLE_GB=$((AVAILABLE_BYTES / 1024 / 1024 / 1024))
    echo "错误：目标盘仅剩 ${AVAILABLE_GB} GiB；一小时录制至少预留 ${MIN_FREE_GB} GiB。" >&2
    exit 2
fi

if [[ -x "$PROJECT_ROOT/gps_venv/bin/python3" ]]; then
    PYTHON="$PROJECT_ROOT/gps_venv/bin/python3"
else
    PYTHON="${PYTHON:-python3}"
fi

echo "========================================"
echo "  🌀 数据采集"
if [[ -n "$NAME" ]]; then
    echo "  实验名称: $NAME"
else
    echo "  实验名称: 自动按启动时间生成"
fi
echo "  GPS 串口: $GPS_PORT"
echo "  保存位置: $OUTPUT_ROOT"
echo "  可用空间: $((AVAILABLE_BYTES / 1024 / 1024 / 1024)) GiB"
echo "========================================"

COMMAND=(
    "$PYTHON" "$SCRIPT_DIR/record_d456_orbslam_live.py"
    --mode auto
    --out "$OUTPUT_ROOT"
    --gps-port "$GPS_PORT"
    --gps-baud "$GPS_BAUD"
)
if [[ -n "$NAME" ]]; then
    COMMAND+=(--name "$NAME")
fi
exec "${COMMAND[@]}"
