#!/usr/bin/env python3
"""提取 ROS2 bag (单目+IMU) → EuRoC 格式（内存友好，逐帧写盘）"""
import sys, cv2, numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader

def main():
    if len(sys.argv) < 2:
        print("用法: extract_bag.py <bag_dir> [输出目录]")
        sys.exit(1)

    bag_path = Path(sys.argv[1])
    out_dir  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"dataset_{bag_path.name}")

    cam0_dir = out_dir / 'mav0' / 'cam0' / 'data'
    imu0_dir = out_dir / 'mav0' / 'imu0'
    cam0_dir.mkdir(parents=True, exist_ok=True)
    imu0_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pass 1: 只收集时间戳 + IMU 数据，不存图像 ----
    image_timestamps = []
    imu_data = []
    first_imu_ts = None

    with AnyReader([bag_path]) as reader:
        conns = {c.topic: c for c in reader.connections}

        if '/camera/camera/imu' in conns:
            for _, _, raw in reader.messages(connections=[conns['/camera/camera/imu']]):
                msg = reader.deserialize(raw, conns['/camera/camera/imu'].msgtype)
                ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                if first_imu_ts is None:
                    first_imu_ts = ts
                imu_data.append((ts, msg.angular_velocity.x, msg.angular_velocity.y,
                    msg.angular_velocity.z, msg.linear_acceleration.x,
                    msg.linear_acceleration.y, msg.linear_acceleration.z))

        if '/camera/camera/color/image_raw' in conns:
            for _, _, raw in reader.messages(connections=[conns['/camera/camera/color/image_raw']]):
                msg = reader.deserialize(raw, conns['/camera/camera/color/image_raw'].msgtype)
                ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                image_timestamps.append(ts)

    # 截掉 IMU 启动前的帧
    cutoff = first_imu_ts or 0
    valid_ts = [t for t in image_timestamps if t >= cutoff]
    print(f"  IMU: {len(imu_data)} 条, 图像: {len(valid_ts)} 帧")

    # ---- Pass 2: 逐帧读图像，边读边写，不堆积内存 ----
    img_idx = 0
    valid_set = set(valid_ts)
    with AnyReader([bag_path]) as reader:
        conns = {c.topic: c for c in reader.connections}
        if '/camera/camera/color/image_raw' in conns:
            for _, _, raw in reader.messages(connections=[conns['/camera/camera/color/image_raw']]):
                msg = reader.deserialize(raw, conns['/camera/camera/color/image_raw'].msgtype)
                ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                if ts not in valid_set:
                    continue
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                ts_ns = int(ts * 1e9)
                cv2.imwrite(str(cam0_dir / f'{ts_ns}.png'), img)
                img_idx += 1
                if img_idx % 500 == 0:
                    print(f"  写入图像: {img_idx}/{len(valid_ts)}")

    # ---- 写元数据 ----
    with open(out_dir / 'mav0' / 'cam0' / 'data.csv', 'w') as f:
        f.write('#timestamp [ns],filename\n')
        for t in valid_ts:
            ts_ns = int(t * 1e9)
            f.write(f'{ts_ns},{ts_ns}.png\n')

    with open(out_dir / 'mav0' / 'imu0' / 'data.csv', 'w') as f:
        f.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n')
        for ts, wx, wy, wz, ax, ay, az in imu_data:
            f.write(f'{int(ts*1e9)},{wx:.9f},{wy:.9f},{wz:.9f},{ax:.9f},{ay:.9f},{az:.9f}\n')

    with open(out_dir / 'times.txt', 'w') as f:
        for t in valid_ts:
            f.write(f'{int(t * 1e9)}\n')

    print(f"✅ 提取完成 → {out_dir}")

if __name__ == '__main__':
    main()
