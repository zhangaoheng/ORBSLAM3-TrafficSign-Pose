#!/usr/bin/env bash
# ==============================================================================
# 🌀 ORB-SLAM3 Looming 数据安装脚本
#
# 用途: 在新设备上配置数据（从原始设备 rsync，或仅下载 .bag 后由 pipeline 重新生成）
#
# 数据需求:
#   必需: 20260426_104450.bag  (1.7 GB)  — 原始 RealSense 录制文件
#   可选: lines_all/  lines1/  lines2/     — 处理后的数据（可由 pipeline 自动生成）
#
# 最小依赖方案: 只传输 .bag 文件 → 运行 pipeline 重新生成其余数据
#   1. rsync <source>:data/deepseek/20260426_104450.bag  data/deepseek/
#   2. cd pipeline && python run_pipeline.py
#
# 完整传输方案: 同步整个数据目录
#   1. rsync -avz --progress <source>:data/deepseek/  data/deepseek/
#   2. 直接使用现有数据，无需 pipeline 重新生成
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # ORB_SLAM3-master/
DATA_DIR="$PROJECT_ROOT/landmarkslam/implement/data/deepseek"
PIPELINE_DIR="$PROJECT_ROOT/landmarkslam/implement/pipeline"

echo "================================================"
echo "  🌀 Looming 数据安装脚本"
echo "  数据目录: $DATA_DIR"
echo "  流水线:   $PIPELINE_DIR"
echo "================================================"
echo ""

# ---- 检查当前状态 ----
BAG_FILE="$DATA_DIR/20260426_104450.bag"
HAS_BAG=false
HAS_LINES_ALL=false
HAS_LINES1=false
HAS_LINES2=false

[ -f "$BAG_FILE" ] && HAS_BAG=true
[ -d "$DATA_DIR/lines_all/rgb" ] && [ "$(ls -A "$DATA_DIR/lines_all/rgb" 2>/dev/null)" ] && HAS_LINES_ALL=true
[ -d "$DATA_DIR/lines1/rgb" ] && [ "$(ls -A "$DATA_DIR/lines1/rgb" 2>/dev/null)" ] && HAS_LINES1=true
[ -d "$DATA_DIR/lines2/rgb" ] && [ "$(ls -A "$DATA_DIR/lines2/rgb" 2>/dev/null)" ] && HAS_LINES2=true

echo "📊 当前数据状态:"
echo "   .bag 文件:         $HAS_BAG"
echo "   lines_all/ (完整): $HAS_LINES_ALL"
echo "   lines1/ (序列1):    $HAS_LINES1"
echo "   lines2/ (序列2):    $HAS_LINES2"
echo ""

if $HAS_BAG && $HAS_LINES_ALL && $HAS_LINES1 && $HAS_LINES2; then
    echo "✅ 数据已完整！无需额外操作。"
    echo "   如果 pipeline 尚未运行, 可执行:"
    echo "     cd $PIPELINE_DIR && python run_pipeline.py"
    exit 0
fi

# ---- 安装指导 ----
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  方式 A（推荐）: rsync 从现有设备同步"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  在已有数据的设备上执行:"
echo ""
ORIG_DIR="$DATA_DIR"
echo "    rsync -avz --progress \\"
echo "        user@source-machine:${ORIG_DIR}/ \\"
echo "        ${ORIG_DIR}/"
echo ""
echo "  或者只传输 .bag 文件（其他由 pipeline 生成）:"
echo "    rsync -avz --progress \\"
echo "        user@source-machine:${ORIG_DIR}/20260426_104450.bag \\"
echo "        ${ORIG_DIR}/"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  方式 B: 使用 pipeline 从 .bag 重新生成"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  前提: 已有 .bag 文件"
echo ""
echo "    cd $PIPELINE_DIR"
echo "    python run_pipeline.py"
echo ""
echo "  这将自动执行:"
echo "    1️⃣  从 .bag 提取 → lines_all/"
echo "    2️⃣  运行 ORB-SLAM3 → 轨迹文件"
echo "    3️⃣  分割序列 → lines1/ + lines2/"
echo "    4️⃣  ROI 标注 → saved_rois.txt"
echo "    5️⃣  批量评估 → 图表/报告"
echo ""
echo "  其中 1️⃣  需要 pyrealsense2 库:"
echo "    pip install pyrealsense2"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📦 数据各组件大小参考"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   20260426_104450.bag     1.7 GB  (原始录制文件，必需)"
echo "   lines_all/rgb/          468 MB  (由 1️⃣  生成)"
echo "   lines_all/depth/        204 MB  (由 1️⃣  生成)"
echo "   lines1/rgb/             128 MB  (由 3️⃣  生成)"
echo "   lines1/depth/           62 MB   (由 3️⃣  生成)"
echo "   lines2/rgb/             123 MB  (由 3️⃣  生成)"
echo "   lines2/depth/           59 MB   (由 3️⃣  生成)"
echo "   ─────────────────────────────────"
echo "   总计 ≈ 2.8 GB"
echo ""

# ---- 快速验证 ----
if $HAS_BAG; then
    echo "🔧 .bag 文件已存在，可立即运行 pipeline:"
    echo "    cd $PIPELINE_DIR && python run_pipeline.py"
elif ! $HAS_BAG && ! $HAS_LINES1; then
    echo "⚠️  未检测到任何数据，请通过 rsync 获取。"
    echo "   最小方案: 只传 .bag 文件，其余由 pipeline 生成。"
fi
