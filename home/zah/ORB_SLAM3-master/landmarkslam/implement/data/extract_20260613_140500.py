#!/usr/bin/env python3
"""
从 ROS2 .db3 bag 提取单目+IMU数据，转为 ORB-SLAM3 EuRoC 格式。

数据集: 20260613_140500
设备: Intel RealSense D456 (252122301043)
Color: 848x480 rgb8, ~30fps, 3530帧
IMU: Gyro ~200Hz, Accel ~100Hz

用法:
  python3 extract_20260613_140500.py

输出结构:
  euroc_20260613_140500/
    mav0/
      cam0/data/    (时间戳命名的 PNG 图像)
      imu0/data.csv (EuRoC 格式 IMU: timestamp,wx,wy,wz,ax,ay,az)
    times.txt       (每行一个纳秒时间戳)
"""

import sqlite3
import struct
import sys
import os
import cv2
import numpy as np
from pathlib import Path
from io import BytesIO

# ============================================================
# 配置 — 针对 20260613_140500 数据集
# ============================================================
DB_PATH = "//wsl.localhost/Ubuntu-22.04/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/20260613_140500.db3"
OUT_DIR = "//wsl.localhost/Ubuntu-22.04/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/euroc_20260613_140500"

# 该数据集的 Topic ID (从 sqlite topics 表查询得到)
TOPIC_ID_COLOR = 106   # /device_0/sensor_1/Color_0/image/data (3530 msgs)
TOPIC_ID_GYRO  = 112   # /device_0/sensor_2/Gyro_0/imu/data   (23440 msgs)
TOPIC_ID_ACCEL = 109   # /device_0/sensor_2/Accel_0/imu/data  (11806 msgs)


# ============================================================
# CDR 反序列化 (ROS2 序列化格式)
# ============================================================

class CDRReader:
    """Little-endian CDR reader for ROS2 messages."""
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 4  # Skip 4-byte encapsulation header

    def align(self, n=4):
        self.pos = (self.pos + n - 1) & ~(n - 1)

    def uint8(self):
        val = struct.unpack_from('<B', self.data, self.pos)[0]
        self.pos += 1
        return val

    def uint32(self):
        self.align(4)
        val = struct.unpack_from('<I', self.data, self.pos)[0]
        self.pos += 4
        return val

    def int32(self):
        self.align(4)
        val = struct.unpack_from('<i', self.data, self.pos)[0]
        self.pos += 4
        return val

    def float32(self):
        self.align(4)
        val = struct.unpack_from('<f', self.data, self.pos)[0]
        self.pos += 4
        return val

    def float64(self):
        self.align(8)
        val = struct.unpack_from('<d', self.data, self.pos)[0]
        self.pos += 8
        return val

    def string(self):
        """Read a CDR string (uint32 length + chars)."""
        self.align(4)
        length = struct.unpack_from('<I', self.data, self.pos)[0]
        self.pos += 4
        s = self.data[self.pos:self.pos+length].decode('utf-8', errors='replace').rstrip('\x00')
        self.pos += length
        return s

    def skip(self, n):
        self.pos += n


def parse_header(cdr: CDRReader):
    """Read std_msgs/Header: stamp(sec,nanosec) + frame_id(string)."""
    sec = cdr.int32()
    nanosec = cdr.uint32()
    frame_id = cdr.string()
    return sec, nanosec, frame_id


def parse_imu_msg_empirical(data: bytes, is_gyro_topic: bool):
    """Parse sensor_msgs/msg/Imu using empirically determined byte offsets.
    
    对 D456 的该 bag，IMU 消息结构:
    - Header: 4+4+string (sec=0, nanosec=传感器时间)
    - orientation: 4*4=16 bytes (dummy, unreal)
    - orientation_covariance: 36 bytes (dummy)
    - angular_velocity: 3*8=24 bytes (float64 LE) at offset ~132
    - angular_velocity_covariance: 36 bytes
    - linear_acceleration: 3*8=24 bytes (float64 LE) at offset ~228
    
    返回 (t_ns, gx, gy, gz, ax, ay, az)
    """
    # Read header timestamp
    sec = struct.unpack_from('<i', data, 4)[0]
    nanosec = struct.unpack_from('<I', data, 8)[0]
    t_ns = sec * 1_000_000_000 + nanosec
    
    # Read angular_velocity (gyro) at empirical offsets
    gx = struct.unpack_from('<d', data, 132)[0]
    gy = struct.unpack_from('<d', data, 140)[0]
    gz = struct.unpack_from('<d', data, 148)[0]
    
    # Read linear_acceleration at empirical offsets
    ax = struct.unpack_from('<d', data, 228)[0]
    ay = struct.unpack_from('<d', data, 236)[0]
    az = struct.unpack_from('<d', data, 244)[0]
    
    return t_ns, gx, gy, gz, ax, ay, az


def parse_image_msg(data: bytes):
    """Parse sensor_msgs/msg/Image from CDR.
    返回 (t_ns, width, height, encoding, step, image_data)
    """
    cdr = CDRReader(data)
    sec, nanosec, _ = parse_header(cdr)

    height = cdr.uint32()
    width = cdr.uint32()
    encoding = cdr.string()
    is_bigendian = cdr.uint8()
    step = cdr.uint32()

    # image data: uint32 length + raw bytes
    data_len = cdr.uint32()
    img_data = cdr.data[cdr.pos:cdr.pos+data_len]

    t_ns = sec * 1_000_000_000 + nanosec
    return t_ns, width, height, encoding, step, img_data


# ============================================================
# 主提取流程
# ============================================================

def main():
    out_path = Path(OUT_DIR)
    mav0 = out_path / "mav0"
    cam0_data = mav0 / "cam0" / "data"
    imu0_dir = mav0 / "imu0"

    cam0_data.mkdir(parents=True, exist_ok=True)
    imu0_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 数据库: {DB_PATH}")
    print(f"📁 输出目录: {out_path}")

    if not Path(DB_PATH).exists():
        print(f"❌ 数据库文件不存在: {DB_PATH}")
        return 1

    print(f"   文件大小: {Path(DB_PATH).stat().st_size / 1e9:.2f} GB")
    print()

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cursor = conn.cursor()

    # ========== 1. 提取 IMU 数据 ==========
    print("=" * 60)
    print("🔧 [1/2] 提取 IMU 数据 (Gyro + Accel)...")
    print("=" * 60)

    gyro_readings = {}   # t_ns -> (gx, gy, gz)
    accel_readings = {}  # t_ns -> (ax, ay, az)

    # 读取 Gyro
    cursor.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (TOPIC_ID_GYRO,)
    )
    for db_ts, data in cursor:
        t_ns, gx, gy, gz, ax, ay, az = parse_imu_msg_empirical(data, is_gyro_topic=True)
        gyro_readings[t_ns] = (gx, gy, gz)

    # 读取 Accel
    cursor.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (TOPIC_ID_ACCEL,)
    )
    for db_ts, data in cursor:
        t_ns, gx, gy, gz, ax, ay, az = parse_imu_msg_empirical(data, is_gyro_topic=False)
        accel_readings[t_ns] = (ax, ay, az)

    print(f"  Gyro 测量: {len(gyro_readings)} 条")
    print(f"  Accel 测量: {len(accel_readings)} 条")

    if len(gyro_readings) < 2 or len(accel_readings) < 2:
        print("  ❌ IMU 数据不足")
        conn.close()
        return 1

    # 融合: 以 Gyro 时间戳为基准，对 Accel 做线性插值
    gyro_times = sorted(gyro_readings.keys())
    accel_times = sorted(accel_readings.keys())

    accel_t_arr = np.array(accel_times, dtype=np.float64)
    accel_x_arr = np.array([accel_readings[t][0] for t in accel_times], dtype=np.float64)
    accel_y_arr = np.array([accel_readings[t][1] for t in accel_times], dtype=np.float64)
    accel_z_arr = np.array([accel_readings[t][2] for t in accel_times], dtype=np.float64)

    gyro_t_arr = np.array(gyro_times, dtype=np.float64)

    interp_ax = np.interp(gyro_t_arr, accel_t_arr, accel_x_arr)
    interp_ay = np.interp(gyro_t_arr, accel_t_arr, accel_y_arr)
    interp_az = np.interp(gyro_t_arr, accel_t_arr, accel_z_arr)

    # 写入 data.csv (EuRoC 格式)
    imu_csv = imu0_dir / "data.csv"
    with open(imu_csv, 'w') as f:
        f.write("#timestamp_ns,wx,wy,wz,ax,ay,az\n")
        for i, t_ns in enumerate(gyro_times):
            gx, gy, gz = gyro_readings[t_ns]
            ax = interp_ax[i]
            ay = interp_ay[i]
            az = interp_az[i]
            f.write(f"{t_ns},{gx:.9f},{gy:.9f},{gz:.9f},{ax:.9f},{ay:.9f},{az:.9f}\n")

    print(f"  ✅ IMU 融合完成: {len(gyro_times)} 条同步测量")
    print(f"  保存至: {imu_csv}")

    # ========== 2. 提取彩色图像 ==========
    print()
    print("=" * 60)
    print("🎨 [2/2] 提取彩色图像 ({:.0f}x{:.0f})...".format(848, 480))
    print("=" * 60)

    times_file = out_path / "times.txt"
    times_fp = open(times_file, 'w')

    cursor.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (TOPIC_ID_COLOR,)
    )

    img_count = 0
    first_t_ns = None
    last_t_ns = None
    for db_ts, data in cursor:
        t_ns, width, height, encoding, step, img_raw = parse_image_msg(data)

        if first_t_ns is None:
            first_t_ns = t_ns
            print(f"  第一帧: timestamp={t_ns}, {width}x{height}, encoding={encoding}, step={step}")

        # 解码图像数据
        if encoding == 'rgb8':
            img_np = np.frombuffer(img_raw, dtype=np.uint8).reshape((height, width, 3))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif encoding == 'bgr8':
            img_np = np.frombuffer(img_raw, dtype=np.uint8).reshape((height, width, 3))
            img_bgr = img_np
        elif encoding == 'rgba8':
            img_np = np.frombuffer(img_raw, dtype=np.uint8).reshape((height, width, 4))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        elif encoding == 'mono8':
            img_bgr = np.frombuffer(img_raw, dtype=np.uint8).reshape((height, width))
        else:
            print(f"  ⚠ 未知编码 '{encoding}'，尝试原始读取...")
            img_np = np.frombuffer(img_raw, dtype=np.uint8)
            if len(img_np) == height * width * 3:
                img_bgr = img_np.reshape((height, width, 3))
            else:
                print(f"  ❌ 无法解码 ({len(img_np)} bytes)")
                continue

        # 保存为 PNG (EuRoC 命名: 纳秒时间戳.png)
        filename = f"{t_ns}.png"
        filepath = cam0_data / filename
        cv2.imwrite(str(filepath), img_bgr)

        # 写入 times.txt (EuRoC 格式: 纳秒时间戳)
        times_fp.write(f"{t_ns}\n")

        img_count += 1
        last_t_ns = t_ns
        if img_count % 500 == 0:
            print(f"  ⏳ 已提取 {img_count} 帧...")

    times_fp.close()
    conn.close()

    print(f"  ✅ 图像提取完成: {img_count} 帧")
    print(f"  保存至: {cam0_data}")
    print(f"  times.txt → {times_file}")
    if first_t_ns and last_t_ns:
        duration_s = (last_t_ns - first_t_ns) / 1e9
        print(f"  时长: {duration_s:.2f} 秒")
        if img_count > 1:
            print(f"  平均帧率: {img_count / duration_s:.1f} FPS")

    # ========== 最终统计 ==========
    print()
    print("=" * 60)
    print("📊 提取完成！")
    print("=" * 60)
    print(f"  🎯 图像: {img_count} 帧")
    print(f"  🎯 IMU:  {len(gyro_times)} 条")
    print(f"  📁 输出: {out_path}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
