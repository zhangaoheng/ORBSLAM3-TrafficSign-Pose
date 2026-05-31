#!/usr/bin/env python3
"""
改良版：从 ROS1 bag 提取 RealSense D456 数据
=============================================
原则：能导出多少是多少，不爆内存，逐帧处理。

用法:
  python3 extract_bag.py <bag_path> <output_dir>
"""

import sys
import os
import struct
from collections import deque

import cv2
import numpy as np
from rosbags.rosbag1 import Reader


# ============================================================
# 配置
# ============================================================
TOPIC_LEFT  = '/device_0/sensor_0/Infrared_1/image/data'
TOPIC_RIGHT = '/device_0/sensor_0/Infrared_2/image/data'
TOPIC_GYRO  = '/device_0/sensor_2/Gyro_0/imu/data'
TOPIC_ACCEL = '/device_0/sensor_2/Accel_0/imu/data'

MAX_GYRO_CACHE = 2000      # Gyro 缓存上限（防止内存无限增长）
PRINT_INTERVAL = 500        # 每 N 条消息打印一次进度


def parse_image(rawdata):
    """手动解析 ROS1 sensor_msgs/Image 二进制"""
    pos = 0
    _, sec, nsec, flen = struct.unpack_from('<IIII', rawdata, pos)
    pos += 16
    frame_id = rawdata[pos:pos+flen]
    pos += flen

    ts_ns = int(sec * 1e9 + nsec)

    height, width, elen = struct.unpack_from('<III', rawdata, pos)
    pos += 12
    encoding = rawdata[pos:pos+elen].decode('utf-8').strip('\x00')
    pos += elen
    _, step, dlen = struct.unpack_from('<BII', rawdata, pos)
    pos += 9
    img_bytes = rawdata[pos:pos+dlen]

    if encoding in ('mono8', '8UC1'):
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = arr.reshape((height, width))
    elif encoding in ('mono16', '16UC1'):
        arr = np.frombuffer(img_bytes, dtype=np.uint16)
        img = cv2.convertScaleAbs(arr.reshape((height, width)), alpha=(255.0/65535.0))
    elif encoding == 'rgb8':
        arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape((height, width, 3))
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        return ts_ns, None

    return ts_ns, img


def parse_imu(rawdata):
    """解析 ROS1 IMU 消息，返回 (timestamp_ns, wx,wy,wz, ax,ay,az)"""
    pos = 0
    _, sec, nsec, flen = struct.unpack_from('<IIII', rawdata, pos)
    pos += 16 + flen  # skip frame_id

    # orientation (4 doubles)
    pos += 32
    # orientation_cov (9 doubles)
    pos += 72
    # angular_velocity (3 doubles)
    wx, wy, wz = struct.unpack_from('<ddd', rawdata, pos)
    pos += 24
    # angular_velocity_cov (9 doubles)
    pos += 72
    # linear_acceleration (3 doubles)
    ax, ay, az = struct.unpack_from('<ddd', rawdata, pos)

    ts_ns = int(sec * 1e9 + nsec)
    return ts_ns, (wx, wy, wz), (ax, ay, az)


def extract_bag(bag_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    cam0_dir = os.path.join(output_dir, "cam0", "data")
    cam1_dir = os.path.join(output_dir, "cam1", "data")
    imu_dir  = os.path.join(output_dir, "imu0")
    os.makedirs(cam0_dir, exist_ok=True)
    os.makedirs(cam1_dir, exist_ok=True)
    os.makedirs(imu_dir, exist_ok=True)

    # — 输出文件 —
    times_path  = os.path.join(output_dir, "times.txt")
    assoc_path  = os.path.join(output_dir, "associations.txt")
    imu_csv_path = os.path.join(imu_dir, "data.csv")
    imu_euroc_path = os.path.join(output_dir, "imu.txt")  # Euroc 格式，供 run_real.cc 使用

    f_times = open(times_path, 'w')
    f_assoc = open(assoc_path, 'w')
    f_imu   = open(imu_csv_path, 'w')
    f_imu_euroc = open(imu_euroc_path, 'w')
    f_imu.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
                "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],"
                "a_RS_S_z [m s^-2]\n")
    f_imu_euroc.write("# timestamp_ns w_RS_S_x w_RS_S_y w_RS_S_z a_RS_S_x a_RS_S_y a_RS_S_z\n")
    f_imu.flush()

    # — 计数器 —
    cnt = {'left': 0, 'right': 0, 'imu': 0, 'total': 0, 'errors': 0}

    # — Gyro 缓存（定长队列，防止内存爆炸）—
    gyro_cache = deque()   # 元素: (timestamp_ns, (wx,wy,wz))

    print(f"📂 Bag: {bag_path}")
    print(f"📁 Output: {output_dir}")
    print("-" * 40)

    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            cnt['total'] += 1

            if cnt['total'] % PRINT_INTERVAL == 0:
                print(f"  ⏳ 已处理 {cnt['total']} 条消息 | "
                      f"左目 {cnt['left']} | 右目 {cnt['right']} | "
                      f"IMU {cnt['imu']} | 错误 {cnt['errors']}")

            try:
                topic = connection.topic

                # ——— 图像 ———
                if topic in (TOPIC_LEFT, TOPIC_RIGHT):
                    ts_ns, img = parse_image(rawdata)
                    if img is None:
                        cnt['errors'] += 1
                        continue

                    fname = f"{ts_ns}.png"
                    if topic == TOPIC_LEFT:
                        cv2.imwrite(os.path.join(cam0_dir, fname), img)
                        ts_s = ts_ns / 1e9
                        f_times.write(f"{ts_s:.6f} cam0/data/{fname}\n")
                        f_assoc.write(f"{ts_s:.6f} cam0/data/{fname}\n")
                        cnt['left'] += 1
                    else:
                        cv2.imwrite(os.path.join(cam1_dir, fname), img)
                        cnt['right'] += 1

                # ——— Gyro ———
                elif topic == TOPIC_GYRO:
                    ts_ns, gyro, _ = parse_imu(rawdata)
                    gyro_cache.append((ts_ns, gyro))
                    # 超过上限就丢弃最旧的
                    if len(gyro_cache) > MAX_GYRO_CACHE:
                        gyro_cache.popleft()

                # ——— Accel ———
                elif topic == TOPIC_ACCEL:
                    ts_ns, _, accel = parse_imu(rawdata)

                    # 找时间最近的 Gyro
                    best = None
                    best_dt = None
                    for gt, gw in gyro_cache:
                        dt = abs(gt - ts_ns)
                        if best_dt is None or dt < best_dt:
                            best_dt = dt
                            best = (gt, gw)

                    # 时间差小于 5ms 才认为有效
                    if best is not None and best_dt < 5_000_000:
                        gt, (wx, wy, wz) = best
                        ax, ay, az = accel
                        f_imu.write(f"{ts_ns},{wx},{wy},{wz},{ax},{ay},{az}\n")
                        f_imu_euroc.write(f"{ts_ns} {wx:.9f} {wy:.9f} {wz:.9f} {ax:.9f} {ay:.9f} {az:.9f}\n")
                        cnt['imu'] += 1

                        # 清理已配对的 Gyro（删除该时间戳之前的所有 gyro）
                        while gyro_cache and gyro_cache[0][0] <= gt:
                            gyro_cache.popleft()

            except Exception as e:
                cnt['errors'] += 1
                if cnt['errors'] <= 5:
                    print(f"  ⚠ 消息 {cnt['total']} 解析失败: {e}")

    # — 收尾 —
    f_times.close()
    f_assoc.close()
    f_imu.close()
    f_imu_euroc.close()

    print("-" * 40)
    print(f"✅ 提取完成！总消息: {cnt['total']}")
    print(f"📸 左目 (cam0): {cnt['left']} 张")
    print(f"📸 右目 (cam1): {cnt['right']} 张")
    print(f"✈️ IMU (imu0):  {cnt['imu']} 条")
    if cnt['errors']:
        print(f"⚠  解析失败: {cnt['errors']} 条（已跳过）")

    return cnt


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        bag = sys.argv[1]
        out = sys.argv[2]
    else:
        bag = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/lines2.bag"
        out = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/lines2"
    extract_bag(bag, out)
