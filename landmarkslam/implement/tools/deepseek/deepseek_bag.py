#!/usr/bin/env python3
"""
适用：包含 Color / Depth / IMU 的 RealSense D456 .bag 文件。
一键自动化：对齐、提取 RGB/Depth/IMU，并自动生成 ORB-SLAM3 所需的 associations.txt。
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import time
import glob

# ==================== 路径配置 ====================
BAG_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/deepseek/20260426_104450.bag"
OUTPUT_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/deepseek/lines_all"

# ==================== 备用内参（如果自动获取失败）====================
DEFAULT_FX = 643.4   
DEFAULT_FY = 642.3
DEFAULT_CX = 657.4
DEFAULT_CY = 367.5
# ====================================================================

def generate_associations(data_dir):
    """自动生成 ORB-SLAM3 RGB-D 模式所需的 4 列 associations.txt"""
    print("\n🔗 开始生成 TUM 格式关联文件...")
    files = glob.glob(os.path.join(data_dir, "rgb", "*.png"))
    files.sort() # 确保按时间顺序排列
    
    assoc_file = os.path.join(data_dir, "associations.txt")
    with open(assoc_file, 'w') as f:
        for filepath in files:
            filename = os.path.basename(filepath)
            # 提取数字部分（纳秒），转换为秒
            ns_str = filename.split('.')[0]
            sec = float(ns_str) / 1e9 
            
            # 严格按照 TUM 格式：时间戳 rgb路径 时间戳 depth路径
            f.write(f"{sec:.6f} rgb/{filename} {sec:.6f} depth/{filename}\n")
            
    print(f"✅ associations.txt 生成完毕，共包含 {len(files)} 帧对应关系！")

def main():
    rgb_dir = os.path.join(OUTPUT_DIR, 'rgb')
    depth_dir = os.path.join(OUTPUT_DIR, 'depth')
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    imu_path = os.path.join(OUTPUT_DIR, 'imu.txt')
    intrinsics_path = os.path.join(OUTPUT_DIR, 'camera_intrinsics.json')

    print(f"📁 加载 bag: {BAG_FILE}")
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 禁止包文件循环播放
    config.enable_device_from_file(BAG_FILE, repeat_playback=False)
    
    profile = pipeline.start(config)

    # 全速回放模式
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    # 检查流信息
    print("📋 Bag 中包含的流：")
    has_color, has_depth, has_imu = False, False, False
    for s in profile.get_streams():
        name = s.stream_name()
        if s.is_video_stream_profile():
            vp = s.as_video_stream_profile()
            print(f"   {name:15s} {vp.width()}x{vp.height()} @ {vp.fps()}fps")
            if name == 'Color': has_color = True
            if name == 'Depth': has_depth = True
        elif s.is_motion_stream_profile():
            print(f"   {name:15s} (IMU motion data)")
            if name in ['Gyro', 'Accel']: has_imu = True

    if not has_color:
        print("❌ 错误：bag 中没有 Color 流，无法提取彩色图像。")
        pipeline.stop()
        return

    # 深度到彩色对齐
    align = rs.align(rs.stream.color) if (has_color and has_depth) else None
    depth_scale = 0.001

    # 获取内参
    camera_params = {}
    try:
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        camera_params = {
            'fx': intr.fx, 'fy': intr.fy,
            'cx': intr.ppx, 'cy': intr.ppy,
            'width': intr.width, 'height': intr.height
        }
        print(f"✅ 自动获取内参: fx={intr.fx:.3f}, fy={intr.fy:.3f}, cx={intr.ppx:.3f}, cy={intr.ppy:.3f}")
    except:
        print("⚠️ 自动获取内参失败，使用备用值")
        camera_params = {'fx': DEFAULT_FX, 'fy': DEFAULT_FY, 'cx': DEFAULT_CX, 'cy': DEFAULT_CY, 'width': 1280, 'height': 720}

    with open(intrinsics_path, 'w') as f:
        json.dump(camera_params, f, indent=4)

    # 初始化 IMU 文件
    imu_f = open(imu_path, 'w')
    imu_f.write("# timestamp_ns w_x w_y w_z a_x a_y a_z\n")

    last_gyro = [0.0, 0.0, 0.0]
    last_accel = [0.0, 0.0, 0.0]

    frame_idx = 0
    imu_count = 0
    start_time = time.time()
    print("🚀 开始全速提取 RGB、Depth 和 IMU 数据...")

    try:
        while True:
            success, frames = pipeline.try_wait_for_frames(2000)
            if not success or not frames:
                print("⏳ 等待帧超时，尝试最后一次...")
                success, frames = pipeline.try_wait_for_frames(3000)
                if not success or not frames:
                    print("🏁 没有更多帧，bag 文件处理完毕。")
                    break

            # 1. ==== 处理 IMU 数据 ====
            gyro_frame = frames.first_or_default(rs.stream.gyro)
            accel_frame = frames.first_or_default(rs.stream.accel)
            
            if gyro_frame or accel_frame:
                if gyro_frame:
                    gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                    last_gyro = [gyro_data.x, gyro_data.y, gyro_data.z]
                    imu_ts = int(gyro_frame.get_timestamp() * 1e6)
                
                if accel_frame:
                    accel_data = accel_frame.as_motion_frame().get_motion_data()
                    last_accel = [accel_data.x, accel_data.y, accel_data.z]
                    if not gyro_frame: 
                        imu_ts = int(accel_frame.get_timestamp() * 1e6)
                
                imu_f.write(f"{imu_ts} {last_gyro[0]:.6f} {last_gyro[1]:.6f} {last_gyro[2]:.6f} "
                            f"{last_accel[0]:.6f} {last_accel[1]:.6f} {last_accel[2]:.6f}\n")
                imu_count += 1

            # 2. ==== 处理图像数据 ====
            if align:
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
            else:
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

            if not color_frame:
                continue

            timestamp_ns = int(color_frame.get_timestamp() * 1e6)
            
            # 保存彩色图
            color_img = np.asanyarray(color_frame.get_data())
            cv2.imwrite(os.path.join(rgb_dir, f"{timestamp_ns}.png"), color_img)

            # 保存深度图
            if depth_frame:
                depth_img = np.asanyarray(depth_frame.get_data())
                depth_mm = (depth_img * depth_scale * 1000).astype(np.uint16)
                cv2.imwrite(os.path.join(depth_dir, f"{timestamp_ns}.png"), depth_mm)

            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                print(f"📦 已提取图像: {frame_idx} 帧 | IMU: {imu_count} 条，用时 {elapsed:.1f}s")

    except Exception as e:
        print(f"❌ 提取出错: {e}")
    finally:
        pipeline.stop()
        imu_f.close()
        total_time = time.time() - start_time
        print(f"\n✅ 提取彻底完成！耗时 {total_time:.1f}s")
        print(f"   RGB 图像: {rgb_dir} (数量: {len(os.listdir(rgb_dir))})")
        print(f"   深度图:   {depth_dir} (数量: {len(os.listdir(depth_dir))})")
        print(f"   IMU 数据: {imu_path} (数量: {imu_count} 行)")
        
        # 🔥 在数据提取完成后，自动生成 ORB-SLAM3 所需的 associations.txt
        generate_associations(OUTPUT_DIR)

if __name__ == "__main__":
    main()