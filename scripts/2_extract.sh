#!/bin/bash
# ============================================================
# 步骤2: 提取 bag → EuRoC 格式
# 用法: ./scripts/2_extract.sh /path/to/bag_dir [输出目录]
# ============================================================
set -e

if [ $# -lt 1 ]; then
    echo "用法: ./scripts/2_extract.sh /path/to/bag_dir [输出目录]"
    echo "示例: ./scripts/2_extract.sh ~/rosbag2_monoimu_20260615_204133"
    echo "      ./scripts/2_extract.sh ~/rosbag2_monoimu_20260615_204133 数据集名称"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "═══════════════════════════════════════════"
echo "  提取 bag → EuRoC 格式"
echo "═══════════════════════════════════════════"
echo "  Bag: $1"
echo "  输出: ${2:-dataset_$(basename "$1")}"
echo ""

/home/ehuy/orbslam3_venv/bin/python3 "$SCRIPT_DIR/extract_bag.py" "$@"
