#!/usr/bin/env python3
"""
从 ROS 1/2 rosbag 中提取 RGB/Depth/IMU 数据，
生成 ORB-SLAM3 可直接使用的数据目录结构。

用法:
  python3 extract_bag.py /path/to/bag /path/to/output_dir
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import csv
import json
import pathlib
import struct
import cv2
import numpy as np

try:
    from rosbags.highlevel import AnyReader
except ImportError:
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


BAG_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/20260529_114122.bag"
OUTPUT_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data"


def extract_timestamps_only(bag_path, out_dir):
    """只读时间戳 + IMU，不碰图片（图片已存在时用这个）"""
    assoc_file = out_dir / 'associations.txt'
    times_file = out_dir / 'times.txt'
    imu_file   = out_dir / 'imu.txt'

    print(f"⏩ 快速模式: 只提取时间戳 + IMU, 不重复保存图片")
    print(f"📂 Bag: {bag_path}")
    print(f"📁 Output: {out_dir}")

    with AnyReader([bag_path]) as reader:
        conns = {c.topic: c for c in reader.connections}

        # --- 收集 Color + Depth 真实时间戳 ---
        color_ts = []
        depth_ts = []
        for conn, _, data in reader.messages(connections=[
            conns.get(TOPIC_COLOR), conns.get(TOPIC_DEPTH)
        ]):
            if conn is None:
                continue
            try:
                img = parse_ros1_image(data) if not reader.is2 else reader.deserialize(data, conn.msgtype)
            except:
                try:
                    img = reader.deserialize(data, conn.msgtype)
                except:
                    continue
            t = ros_time_to_sec(img.header.stamp)
            if conn.topic == TOPIC_COLOR:
                color_ts.append(t)
            elif conn.topic == TOPIC_DEPTH:
                depth_ts.append(t)

    color_ts.sort()
    depth_ts.sort()
    n = min(len(color_ts), len(depth_ts))
    print(f"  Color: {len(color_ts)}  Depth: {len(depth_ts)}  对齐: {n}")

    # --- 写 associations.txt（真实时间戳）---
    with open(assoc_file, 'w') as f:
        for i in range(n):
            f.write(f"{color_ts[i]:.6f} rgb/{i:06d}.png {depth_ts[i]:.6f} depth/{i:06d}.png\n")
    print(f"  ✅ associations.txt ({n} 行)")

    # --- 写 times.txt（mono 用）---
    with open(times_file, 'w') as f:
        for i in range(n):
            f.write(f"{color_ts[i]:.6f} rgb/{i:06d}.png\n")
    print(f"  ✅ times.txt ({n} 行)")

    # --- 提取 IMU ---
    extract_imu(bag_path, out_dir, reader=None)

    print(f"🎉 完成！时间戳已对齐")


def extract_imu(bag_path, out_dir, reader=None):
    """提取 IMU 并保存为 Euroc 格式"""
    imu_file = out_dir / 'imu.txt'
    accel_list, gyro_list = [], []

    def _do_extract(r):
        conns = {c.topic: c for c in r.connections}
        if TOPIC_ACCEL in conns:
            for _, _, data in r.messages(connections=[conns[TOPIC_ACCEL]]):
                msg = r.deserialize(data, conns[TOPIC_ACCEL].msgtype)
                t = ros_time_to_sec(msg.header.stamp)
                accel_list.append([t, msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        if TOPIC_GYRO in conns:
            for _, _, data in r.messages(connections=[conns[TOPIC_GYRO]]):
                msg = r.deserialize(data, conns[TOPIC_GYRO].msgtype)
                t = ros_time_to_sec(msg.header.stamp)
                gyro_list.append([t, msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

    if reader is not None:
        _do_extract(reader)
    else:
        with AnyReader([bag_path]) as r:
            _do_extract(r)

    if not accel_list or not gyro_list:
        print("❌ 缺少 IMU 数据")
        return

    print(f"  🔄 对齐 IMU (Accel: {len(accel_list)}, Gyro: {len(gyro_list)})...")
    accel_arr = np.array(accel_list)
    gyro_arr = np.array(gyro_list)
    t_accel, t_gyro = accel_arr[:, 0], gyro_arr[:, 0]

    interp_ax = np.interp(t_gyro, t_accel, accel_arr[:, 1])
    interp_ay = np.interp(t_gyro, t_accel, accel_arr[:, 2])
    interp_az = np.interp(t_gyro, t_accel, accel_arr[:, 3])

    with open(imu_file, 'w') as f:
        f.write("# timestamp_ns w_x w_y w_z a_x a_y a_z\n")
        for i in range(len(t_gyro)):
            t_ns = int(t_gyro[i] * 1e9)
            wx, wy, wz = gyro_arr[i, 1], gyro_arr[i, 2], gyro_arr[i, 3]
            ax, ay, az = interp_ax[i], interp_ay[i], interp_az[i]
            f.write(f"{t_ns} {wx:.6f} {wy:.6f} {wz:.6f} {ax:.6f} {ay:.6f} {az:.6f}\n")
    print(f"  ✅ imu.txt ({len(t_gyro)} 条)")


def main():
    quick_mode = '--quick' in sys.argv
    if quick_mode:
        sys.argv.remove('--quick')

    if len(sys.argv) >= 2:
        bag_path = pathlib.Path(sys.argv[1])
    else:
        bag_path = pathlib.Path(BAG_PATH)

    if len(sys.argv) >= 3:
        out_dir = pathlib.Path(sys.argv[2])
    else:
        out_dir = pathlib.Path(OUTPUT_DIR)

    if quick_mode:
        extract_timestamps_only(bag_path, out_dir)
        return

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
        # 2) 流式提取 RGB 和 Depth — 边读边存，内存只留时间戳
        # ============================================================
        color_frames = []    # 只存 (timestamp, frame_id)
        depth_frames = []    # 只存 (timestamp, frame_id)
        color_idx = 0
        depth_idx = 0

        cam_csv = out_dir / 'camera_timestamps.csv'
        assoc_file = out_dir / 'associations.txt'
        fcsv = open(cam_csv, 'w', newline='')
        fassoc = open(assoc_file, 'w')
        csv_writer = csv.writer(fcsv)
        csv_writer.writerow(['frame_id', 'timestamp_ms'])

        for conn, _, data in reader.messages(connections=[
            conns.get(TOPIC_COLOR), conns.get(TOPIC_DEPTH)
        ]):
            if conn is None:
                continue
            try:
                img = parse_ros1_image(data) if not reader.is2 else reader.deserialize(data, conn.msgtype)
            except Exception:
                try:
                    img = reader.deserialize(data, conn.msgtype)
                except:
                    continue

            t = ros_time_to_sec(img.header.stamp)

            if conn.topic == TOPIC_COLOR:
                frame_id = f"{color_idx:06d}.png"
                # 立即存盘 — 不占内存
                save_image(img, str(out_dir / 'rgb' / frame_id))
                color_frames.append((t, frame_id))
                color_idx += 1
                if color_idx % 100 == 0:
                    print(f"  ⏳ 已提取 Color: {color_idx} 帧...")

            elif conn.topic == TOPIC_DEPTH:
                frame_id = f"{depth_idx:06d}.png"
                save_image(img, str(out_dir / 'depth' / frame_id))
                depth_frames.append((t, frame_id))
                depth_idx += 1
                if depth_idx % 100 == 0:
                    print(f"  ⏳ 已提取 Depth: {depth_idx} 帧...")

        print(f"  📷 Color: {color_idx} 帧")
        print(f"  📷 Depth: {depth_idx} 帧")

        if color_idx == 0:
            print("❌ 无彩色图像，退出")
            fcsv.close(); fassoc.close()
            return

        # 排序（rosbag 消息基本有序，但排序确保不出问题）
        color_frames.sort(key=lambda x: x[0])
        depth_frames.sort(key=lambda x: x[0])

        # 时序匹配：对每张 Color 找最近 Depth，写 associations.txt
        depth_ts_list = [t for t, _ in depth_frames]
        depth_map = dict(depth_frames)  # timestamp -> filename

        for i, (ct, cfile) in enumerate(color_frames):
            if depth_ts_list:
                nearest_dt = min(depth_ts_list, key=lambda dt: abs(dt - ct))
                dframe = depth_map[nearest_dt]
            else:
                dframe = None

            ts_ms = ct * 1000
            csv_writer.writerow([cfile, f"{ts_ms:.4f}"])
            fassoc.write(f"{ct:.6f} rgb/{cfile} {ct:.6f} depth/{dframe}\n")

        fcsv.close()
        fassoc.close()
        print(f"  ✅ 图像提取 + 时序关联完成: {color_idx} 帧")

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