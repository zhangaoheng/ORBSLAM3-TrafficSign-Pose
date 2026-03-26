#!/bin/bash
# MapViewer 窗口录制脚本
# 使用方法: ./record_mapviewer.sh [输出文件路径]

OUTPUT_FILE="${1:-/tmp/mapviewer_recording.mp4}"
echo "开始录制 MapViewer 窗口..."
echo "按 Ctrl+C 停止录制"
echo "输出文件: $OUTPUT_FILE"
echo ""

# 等待用户确认窗口已打开
read -p "请确保 MapViewer 窗口已打开，按 Enter 继续..."

# 使用 ffmpeg 录制整个屏幕
ffmpeg -f x11grab -video_size 1920x1080 -framerate 30 -i $DISPLAY \
       -vcodec libx264 -preset ultrafast -crf 23 \
       -y "$OUTPUT_FILE"

echo ""
echo "✓ 录制完成: $OUTPUT_FILE"
