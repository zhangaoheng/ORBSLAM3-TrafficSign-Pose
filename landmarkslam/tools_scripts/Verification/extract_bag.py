import pyrealsense2 as rs
import numpy as np
import cv2
import os


# 提取bag包
# ================= 你的配置 =================
BAG_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/data/20260410_200107.bag" 
SAVE_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/output/extracted_frames" 

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ================= 初始化 =================
pipeline = rs.pipeline()
config = rs.config()

# repeat_playback=False 表示播完一遍就自动停止
rs.config.enable_device_from_file(config, BAG_FILE, repeat_playback=False) 

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
print(f"当前深度比例尺: 1 像素值 = {depth_scale} 米")

frame_count = 0
saved_count = 0

print("\n[无 GUI 自动抽帧模式启动]：")
print("正在后台飞速解析录像包，每隔 30 帧（约 1 秒）自动保存一张...")

try:
    while True:
        # 当 bag 包读到结尾时，wait_for_frames 会抛出 RuntimeError
        frames = pipeline.wait_for_frames()

        # 物理对齐！
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        frame_count += 1

        # ⭐️ 每隔 30 帧自动存一次
        if frame_count % 30 == 0:
            saved_count += 1
            
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            color_filename = os.path.join(SAVE_DIR, f"color_{saved_count:02d}.png")
            depth_filename = os.path.join(SAVE_DIR, f"depth_{saved_count:02d}.npy")

            cv2.imwrite(color_filename, color_image)
            np.save(depth_filename, depth_image)
            
            print(f"✅ 成功提取第 {saved_count:02d} 组数据 (位于录像第 {frame_count} 帧)")

except RuntimeError:
    # 捕捉到文件结尾的异常，优雅退出
    print(f"\n🎉 录像包读取完毕！共提取了 {saved_count} 组数据。")
    print(f"👉 现在请打开 Windows 里的 {SAVE_DIR} 文件夹，挑选出你需要的 4 个角度的照片吧！")

finally:
    pipeline.stop()