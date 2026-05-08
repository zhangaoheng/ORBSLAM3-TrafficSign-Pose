#!/usr/bin/env bash
# ==============================================================================
# 🔄 ORB-SLAM3 Looming 运行脚本
# 用法:
#   ./run_all.sh              — 先 batch 评估，再交互式度量恢复
#   ./run_all.sh batch        — 仅 batch 评估（安静模式，无 GUI）
#   ./run_all.sh interactive  — 仅交互式度量恢复（有 GUI）
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="/home/zah/ORB_SLAM3-master/landmarkslam/yolo_venv/bin/python"
TEST_PY="$SCRIPT_DIR/test.py"

MODE="${1:-all}"

run_batch() {
    echo "========================================"
    echo "  📊 批量评估模式"
    echo "========================================"
    "$VENV_PYTHON" "$TEST_PY" -q
    echo ""
}

run_interactive() {
    echo "========================================"
    echo "  🖥️  交互式度量恢复模式"
    echo "========================================"
    "$VENV_PYTHON" "$TEST_PY"
}

case "$MODE" in
    batch)
        run_batch
        ;;
    interactive)
        run_interactive
        ;;
    all)
        run_batch
        run_interactive
        ;;
    *)
        echo "❌ 未知模式: $MODE"
        echo "用法: $0 {batch|interactive|all}"
        exit 1
        ;;
esac
