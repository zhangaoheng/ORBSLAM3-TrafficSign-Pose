#!/usr/bin/env python3
"""
从 ROS 1/2 rosbag 中提取 RGB/Depth/IMU 数据，
生成 ORB-SLAM3 可直接使用的数据目录结构。

用法:
  python3 extract_bag.py /path/to/bag /path/to/output_dir
"""

import sys
import csv
import json
import pathlib
import struct
import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.parent /
                        'landmarkslam' / 'yolo_venv' / 'lib' / 'python3.10' / 'site-packages'))
from rosbags.highlevel import AnyReader

# ============================================================
TOPIC_COLOR       = '/device_0/sensor_1/Color_0/image/data'
TOPIC_DEPTH       = '/device_0/sensor_0/Depth_0/image/data'
TOPIC_ACCEL       = '/device_0/sensor_2/Accel_0/imu/data'
TOPIC_GYRO        = '/device_0/sensor_2/Gyro_0/imu/data'
TOPIC_CAMERA_INFO = '/device_0/sensor_1/Color_0/info/camera_info'
# ============================================================


def parse_ros1_image(data: bytes):
    """手动解析 ROS 1 sensor_msgs/Image"""
    pos = 0
    # Header
    seq      = struct.unpack_from('<I', data, pos)[0]; pos += 4
    secs     = struct.unpack_from('<I', data, pos)[0]; pos += 4
    nsecs    = struct.unpack_from('<I', data, pos)[0]; pos += 4
    flen     = struct.unpack_from('<I', data, pos)[0]; pos += 4
    frame_id = data[pos:pos+flen].decode('utf-8');    pos += flen
    # Image fields
    height   = struct.unpack_from('<I', data, pos)[0]; pos += 4
    width    = struct.unpack_from('<I', data, pos)[0]; pos += 4
    elen     = struct.unpack_from('<I', data, pos)[0]; pos += 4
    encoding = data[pos:pos+elen].decode('utf-8');    pos += elen
    bigendian = struct.unpack_from('<B', data, pos)[0]; pos += 1
    step     = struct.unpack_from('<I', data, pos)[0]; pos += 4
    img_data = data[pos:pos + step * height]

    class Image:
        pass

    img = Image()
    img.width = width
    img.height = height
    img.encoding = encoding
    img.step = step
    img.data = img_data
    img.header = type('h', (), {})()
    img.header.stamp = type('s', (), {})()
    img.header.stamp.sec = secs
    img.header.stamp.nanosec = nsecs
    return img


def ros_time_to_sec(header_stamp) -> float:
    return header_stamp.sec + header_stamp.nanosec / 1e9


def save_image(msg, save_path: str):
    """保存图像消息为 PNG"""
    if msg.encoding == 'rgb8':
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif msg.encoding == 'bgr8':
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    elif msg.encoding in ('16UC1', 'mono16'):
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    elif msg.encoding == '32FC1':
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        arr = (arr * 1000).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    cv2.imwrite(save_path, arr)


BAG_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/20260528_153824.bag"
OUTPUT_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data"


def main():
    if len(sys.argv) >= 2:
        bag_path = pathlib.Path(sys.argv[1])
    else:
        bag_path = pathlib.Path(BAG_PATH)

    if len(sys.argv) >= 3:
        out_dir = pathlib.Path(sys.argv[2])
    else:
        out_dir = pathlib.Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'rgb').mkdir(exist_ok=True)
    (out_dir / 'depth').mkdir(exist_ok=True)

    print(f"📂 Bag: {bag_path}")
    print(f"📁 Output: {out_dir}")

    with AnyReader([bag_path]) as reader:
        conns = {c.topic: c for c in reader.connections}

        # ============================================================
        # 1) 相机内参
        # ============================================================
        if TOPIC_CAMERA_INFO in conns:
            for conn, _, data in reader.messages(connections=[conns[TOPIC_CAMERA_INFO]]):
                try:
                    msg = reader.deserialize(data, conn.msgtype)
                    intrinsics = {
                        "width": msg.width, "height": msg.height,
                        "fx": msg.K[0], "fy": msg.K[4],
                        "cx": msg.K[2], "cy": msg.K[5],
                        "distortion_model": msg.distortion_model,
                        "distortion_coeffs": list(msg.D),
                    }
                    with open(out_dir / 'camera_intrinsics.json', 'w') as f:
                        json.dump(intrinsics, f, indent=2)
                    print(f"  ✅ 相机内参: fx={intrinsics['fx']:.3f} fy={intrinsics['fy']:.3f}")
                except Exception as e:
                    print(f"  ⚠ 读取相机内参失败: {e}")
                break

        # ============================================================
        # 2) 提取 RGB 和 Depth（手动解析 Image）
        # ============================================================
        color_list = []
        depth_list = []

        for conn, _, data in reader.messages(connections=[
            conns.get(TOPIC_COLOR), conns.get(TOPIC_DEPTH)
        ]):
            if conn is None:
                continue
            try:
                img = parse_ros1_image(data) if not reader.is2 else reader.deserialize(data, conn.msgtype)
            except Exception:
                # fallback
                try:
                    img = reader.deserialize(data, conn.msgtype)
                except:
                    continue
            t = ros_time_to_sec(img.header.stamp)
            if conn.topic == TOPIC_COLOR:
                color_list.append((t, img))
            elif conn.topic == TOPIC_DEPTH:
                depth_list.append((t, img))

        # 按时间排序
        color_list.sort(key=lambda x: x[0])
        depth_list.sort(key=lambda x: x[0])
        depth_ts_list = [t for t, _ in depth_list]

        print(f"  📷 Color: {len(color_list)} 帧")
        print(f"  📷 Depth: {len(depth_list)} 帧")

        if len(color_list) == 0:
            print("❌ 无彩色图像，退出")
            return

        cam_csv = out_dir / 'camera_timestamps.csv'
        assoc_file = out_dir / 'associations.txt'

        # 对每张 Color 找最近的 Depth 做时间对齐
        with open(cam_csv, 'w', newline='') as fcsv, \
             open(assoc_file, 'w') as fassoc:
            writer = csv.writer(fcsv)
            writer.writerow(['frame_id', 'timestamp_ms'])

            for i, (ct, cmsg) in enumerate(color_list):
                # 找最近 Depth
                if depth_ts_list:
                    nearest_dt = min(depth_ts_list, key=lambda dt: abs(dt - ct))
                    dmsg = next(d for t, d in depth_list if t == nearest_dt)
                else:
                    dmsg = None

                frame_id = f"{i:06d}.png"
                ts_ms = ct * 1000

                writer.writerow([frame_id, f"{ts_ms:.4f}"])
                fassoc.write(f"{ct:.6f} rgb/{frame_id} {ct:.6f} depth/{frame_id}\n")

                save_image(cmsg, str(out_dir / 'rgb' / frame_id))
                if dmsg:
                    save_image(dmsg, str(out_dir / 'depth' / frame_id))

                if (i + 1) % 100 == 0:
                    print(f"  ⏳ 已处理 {i + 1}/{len(color_list)} 帧...")

        print(f"  ✅ 图像保存完成: {len(color_list)} 帧")

        # ============================================================
        # 3) 提取 IMU (修复时间戳对齐与插值)
        # ============================================================
        accel_list = []
        gyro_list = []

        # 收集 Accel
        if TOPIC_ACCEL in conns:
            for _, _, data in reader.messages(connections=[conns[TOPIC_ACCEL]]):
                msg = reader.deserialize(data, conns[TOPIC_ACCEL].msgtype)
                t = ros_time_to_sec(msg.header.stamp)
                accel_list.append([t, msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

        # 收集 Gyro
        if TOPIC_GYRO in conns:
            for _, _, data in reader.messages(connections=[conns[TOPIC_GYRO]]):
                msg = reader.deserialize(data, conns[TOPIC_GYRO].msgtype)
                t = ros_time_to_sec(msg.header.stamp)
                gyro_list.append([t, msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

        if not accel_list or not gyro_list:
            print("❌ 缺少 IMU 数据，跳过 IMU 提取")
            return

        print(f"  🔄 正在对齐 IMU 数据 (Accel: {len(accel_list)}, Gyro: {len(gyro_list)})...")
        
        accel_arr = np.array(accel_list)
        gyro_arr = np.array(gyro_list)

        # 提取时间戳
        t_accel = accel_arr[:, 0]
        t_gyro = gyro_arr[:, 0]

        # 将 Accel (低频) 线性插值到 Gyro (高频) 的时间戳上
        interp_ax = np.interp(t_gyro, t_accel, accel_arr[:, 1])
        interp_ay = np.interp(t_gyro, t_accel, accel_arr[:, 2])
        interp_az = np.interp(t_gyro, t_accel, accel_arr[:, 3])

        # 保存为 ORB-SLAM3 推荐的标准 Euroc imu.txt 格式
        imu_txt = out_dir / 'imu.txt'
        with open(imu_txt, 'w') as f:
            f.write("# timestamp_ns w_x w_y w_z a_x a_y a_z\n")
            for i in range(len(t_gyro)):
                t_ns = int(t_gyro[i] * 1e9)  # 转为纳秒
                wx, wy, wz = gyro_arr[i, 1], gyro_arr[i, 2], gyro_arr[i, 3]
                ax, ay, az = interp_ax[i], interp_ay[i], interp_az[i]
                
                # 写入格式: 纳秒时间戳, gyro(xyz), accel(xyz)
                f.write(f"{t_ns} {wx:.6f} {wy:.6f} {wz:.6f} {ax:.6f} {ay:.6f} {az:.6f}\n")

        print(f"  ✅ IMU 对齐并保存完成: 成功合并为 {len(t_gyro)} 条同步数据")

    print(f"\n📊 总结: Color={len(color_list)} Depth={len(depth_list)} IMU_Sync={len(gyro_list)}")
    print(f"🎉 完成!")


if __name__ == '__main__':
    main()