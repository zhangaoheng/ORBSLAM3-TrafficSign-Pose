#!/bin/bash
# ============================================================
# 步骤1: 录制 D456 单目+IMU bag
# 用法: ./scripts/1_record.sh [录制秒数]
# ============================================================
set -e

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BAG_DIR="$HOME/rosbag2_monoimu_${TIMESTAMP}"
DURATION="${1:-30}"

echo "═══════════════════════════════════════════"
echo "  启动 D456 相机 + 录制 ${DURATION} 秒"
echo "═══════════════════════════════════════════"
echo "  Bag 输出: $BAG_DIR"
echo ""

# 启动相机
echo "[1/3] 启动相机..."
sudo bash -c 'source /opt/ros/lyrical/setup.bash && source /home/ehuy/realsense_ws/install/setup.bash && exec ros2 launch realsense2_camera rs_launch.py \
    enable_depth:=false enable_infra1:=false enable_infra2:=false \
    enable_color:=true enable_gyro:=true enable_accel:=true \
    unite_imu_method:=2 enable_sync:=true \
    rgb_camera.color_profile:=848x480x30' &
CAM_PID=$!

# 等待 IMU 话题就绪
echo "[2/3] 等待相机就绪..."
for i in $(seq 1 25); do
    sleep 1
    if bash -c 'source /opt/ros/lyrical/setup.bash && ros2 topic list' 2>/dev/null | grep -q "/camera/camera/imu"; then
        echo "  ✅ 相机就绪（含 IMU）"
        sleep 1
        break
    fi
done

# 录制
echo "[3/3] 开始录制 ${DURATION} 秒，请移动相机采集数据..."
sudo bash -c "source /opt/ros/lyrical/setup.bash && source /home/ehuy/realsense_ws/install/setup.bash && exec ros2 bag record \
    --output ${BAG_DIR} --max-bag-size 500000000 \
    --topics /camera/camera/color/image_raw /camera/camera/imu /camera/camera/color/camera_info" &
REC_PID=$!

sleep "$DURATION"

# 停止
echo ""
echo "⏹  停止录制..."
kill $REC_PID 2>/dev/null; wait $REC_PID 2>/dev/null || true
kill $CAM_PID 2>/dev/null; wait $CAM_PID 2>/dev/null || true

echo ""
echo "✅ 录制完成！"
echo "   Bag: $BAG_DIR"
du -sh "$BAG_DIR" 2>/dev/null
