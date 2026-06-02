#!/usr/bin/env python3
"""
【混合架构版】ORB-SLAM3 数据集一键提取工具 (RGB-D-Inertial)
- 图像部分：调用 pyrealsense2 执行严苛的“时间戳同步” + “空间对齐(Alignment)”
- IMU 部分：调用 rosbags 执行高频“无损提取” + “线性插值”

用法:
  python3 extract_bag_slam.py /path/to/bag /path/to/output_dir
"""

import sys
import io
import os
import pathlib
import struct
import cv2
import numpy as np
import pyrealsense2 as rs

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 尝试导入 rosbags，保留你原有的环境变量配置
try:
    from rosbags.highlevel import AnyReader
except ImportError:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.parent /
                            'landmarkslam' / 'yolo_venv' / 'lib' / 'python3.10' / 'site-packages'))
    from rosbags.highlevel import AnyReader

# ============================================================
# 默认配置路径 (如果在命令行未提供，将使用这里的默认值)
# ============================================================
BAG_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/20260529_114122.bag"
OUTPUT_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data"

# IMU 的 ROS 话题名 (RealSense Viewer 默认)
TOPIC_ACCEL = '/device_0/sensor_2/Accel_0/imu/data'
TOPIC_GYRO  = '/device_0/sensor_2/Gyro_0/imu/data'
# ============================================================

def ros_time_to_sec(header_stamp) -> float:
    """ROS 时间戳转为秒"""
    return header_stamp.sec + header_stamp.nanosec / 1e9

def extract_imu_rosbags(bag_path, out_dir):
    """
    使用 rosbags 提取并对齐 IMU 数据 (Accel 插值到 Gyro)
    输出标准 EuRoC 格式: imu.txt
    """
    print("\n" + "="*50)
    print("🚀 [阶段一] 开始使用 rosbags 提取 IMU 数据...")
    print("="*50)
    
    imu_file = out_dir / 'imu.txt'
    accel_list, gyro_list = [], []

    try:
        with AnyReader([bag_path]) as reader:
            conns = {c.topic: c for c in reader.connections}
            
            # 1. 收集加速度计 (Accel)
            if TOPIC_ACCEL in conns:
                for _, _, data in reader.messages(connections=[conns[TOPIC_ACCEL]]):
                    msg = reader.deserialize(data, conns[TOPIC_ACCEL].msgtype)
                    t = ros_time_to_sec(msg.header.stamp)
                    accel_list.append([t, msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            
            # 2. 收集陀螺仪 (Gyro)
            if TOPIC_GYRO in conns:
                for _, _, data in reader.messages(connections=[conns[TOPIC_GYRO]]):
                    msg = reader.deserialize(data, conns[TOPIC_GYRO].msgtype)
                    t = ros_time_to_sec(msg.header.stamp)
                    gyro_list.append([t, msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
                    
    except Exception as e:
        print(f"❌ 读取 IMU 失败，请检查 ROSBag 格式或话题名: {e}")
        return 0

    if not accel_list or not gyro_list:
        print("❌ 未在 bag 中找到 IMU 话题数据，跳过 IMU 提取。")
        return 0

    print(f" 🔄 正在执行线性插值对齐 (Accel: {len(accel_list)} 条, Gyro: {len(gyro_list)} 条)...")
    
    accel_arr = np.array(accel_list)
    gyro_arr = np.array(gyro_list)
    t_accel, t_gyro = accel_arr[:, 0], gyro_arr[:, 0]

    # 将 Accel (通常低频) 插值到 Gyro (通常高频) 的时间轴上
    interp_ax = np.interp(t_gyro, t_accel, accel_arr[:, 1])
    interp_ay = np.interp(t_gyro, t_accel, accel_arr[:, 2])
    interp_az = np.interp(t_gyro, t_accel, accel_arr[:, 3])

    # 写入 imu.txt
    with open(imu_file, 'w') as f:
        f.write("# timestamp_ns w_x w_y w_z a_x a_y a_z\n")
        for i in range(len(t_gyro)):
            t_ns = int(t_gyro[i] * 1e9)  # EuRoC 要求时间戳为纳秒
            wx, wy, wz = gyro_arr[i, 1], gyro_arr[i, 2], gyro_arr[i, 3]
            ax, ay, az = interp_ax[i], interp_ay[i], interp_az[i]
            f.write(f"{t_ns} {wx:.6f} {wy:.6f} {wz:.6f} {ax:.6f} {ay:.6f} {az:.6f}\n")
            
    print(f" ✅ IMU 提取完成！生成同步 IMU 数据 {len(t_gyro)} 条。")
    return len(t_gyro)


def extract_align_images_rs(bag_path, out_dir):
    """
    使用 pyrealsense2 提取 RGB-D 图像并执行绝对的【空间对齐】与【时间同步过滤】
    """
    print("\n" + "="*50)
    print("🚀 [阶段二] 开始使用 pyrealsense2 执行 RGB-D 空间对齐与提取...")
    print("="*50)
    
    rgb_dir = out_dir / 'rgb'
    depth_dir = out_dir / 'depth'
    rgb_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    
    assoc_file = open(out_dir / 'associations.txt', 'w')
    
    # 1. 初始化 Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)
    
    # 我们只需要提取图像，IMU已经交给 rosbags 提取了
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)

    # 2. 核心：创建对齐对象 (将深度投影至彩色坐标系)
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"❌ PyRealSense2 启动 pipeline 失败: {e}")
        return 0

    frame_count = 0
    valid_pairs = 0
    MAX_TIME_DIFF = 0.02 # 20ms 时间同步容忍阈值
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            
            # 3. 核心：执行底层 C++ 空间重投影对齐
            aligned_frames = align.process(frames)
            
            aligned_depth = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth or not color_frame:
                continue
                
            frame_count += 1
            
            # 时间戳获取 (毫秒转为秒)
            t_color = color_frame.get_timestamp() / 1000.0
            t_depth = aligned_depth.get_timestamp() / 1000.0
            
            # 4. 时间同步容错检查
            if abs(t_color - t_depth) > MAX_TIME_DIFF:
                # 丢弃差异过大的帧，宁缺毋滥，防止 SLAM 崩溃
                continue
                
            # 转为 numpy 以便 OpenCV 保存
            depth_image = np.asanyarray(aligned_depth.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            
            # 命名使用时间戳或序号，这里使用 6 位序号
            file_name = f"{valid_pairs:06d}.png"
            
            cv2.imwrite(str(rgb_dir / file_name), color_image_bgr)
            cv2.imwrite(str(depth_dir / file_name), depth_image)
            
            # 写入 associations.txt (保留各自真实的时间戳)
            assoc_file.write(f"{t_color:.6f} rgb/{file_name} {t_depth:.6f} depth/{file_name}\n")
            
            valid_pairs += 1
            if valid_pairs % 100 == 0:
                print(f" ⏳ 已成功提取并对齐 {valid_pairs} 对 RGB-D 图像...")
                
    except RuntimeError:
        # 当 bag 文件播放完毕时，SDK 正常抛出 RuntimeError
        pass
    except Exception as e:
        print(f"⚠ 提取图像时发生异常: {e}")
    finally:
        pipeline.stop()
        assoc_file.close()
        
    print(f" ✅ 图像提取完成！共读取 {frame_count} 帧，成功同步对齐 {valid_pairs} 对。")
    return valid_pairs


def main():
    # 接收命令行参数
    bag_path_str = sys.argv[1] if len(sys.argv) >= 2 else BAG_PATH
    out_dir_str = sys.argv[2] if len(sys.argv) >= 3 else OUTPUT_DIR

    bag_path = pathlib.Path(bag_path_str)
    out_dir = pathlib.Path(out_dir_str)

    if not bag_path.exists():
        print(f"❌ 错误: Bag 文件不存在 -> {bag_path}")
        sys.exit(1)

    # 创建根输出目录
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 目标数据包: {bag_path}")
    print(f"📁 提取输出至: {out_dir}")

    # 调用重构后的两个核心函数
    imu_count = extract_imu_rosbags(bag_path, out_dir)
    rgbd_count = extract_align_images_rs(bag_path, out_dir)

    # 打印最终统计（修复了之前的崩溃 Bug）
    print("\n" + "="*50)
    print("📊 数据集提取统计汇总")
    print("="*50)
    print(f" 🎯 同步并空间对齐的 RGB-D 帧: {rgbd_count} 对")
    print(f" 🎯 连续高频 IMU 同步数据: {imu_count} 条")
    
    if rgbd_count > 0 and imu_count > 0:
        print("\n🎉 完美！您的数据集已具备 ORB-SLAM3 RGB-D-Inertial 模式的高优运行条件。")
    else:
        print("\n⚠ 注意：某些数据流提取为空，请检查上述日志或话题名配置。")


if __name__ == '__main__':
    main()