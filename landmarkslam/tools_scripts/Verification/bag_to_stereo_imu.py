import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from queue import Queue
import sys

# 提取双目imu数据

# ================= 配置路径 =================
BAG_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/data/stereo_imu.bag" 
OUT_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/data/stereo_imu_dataset" 

os.system(f"rm -rf {OUT_DIR}/cam0 {OUT_DIR}/cam1")
os.makedirs(os.path.join(OUT_DIR, "cam0"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "cam1"), exist_ok=True)

f_imu = open(os.path.join(OUT_DIR, "imu.txt"), 'w')
f_times = open(os.path.join(OUT_DIR, "times.txt"), 'w')
# TUM-VI 格式说明：纳秒时间戳, 陀螺仪xyz, 加速度xyz
f_imu.write("#timestamp,wx,wy,wz,ax,ay,az\n")

data_queue = Queue(maxsize=20000) 

def frame_callback(frame):
    try:
        if frame.is_frameset():
            fs = frame.as_frameset()
            ir1 = fs.get_infrared_frame(1)
            ir2 = fs.get_infrared_frame(2)
            if ir1 and ir2:
                img1 = np.asanyarray(ir1.get_data()).copy()
                img2 = np.asanyarray(ir2.get_data()).copy()
                ts = ir1.get_timestamp() / 1000.0
                data_queue.put(('img_pair', ts, img1, img2))
                
        if frame.is_motion_frame():
            m = frame.as_motion_frame()
            name = m.get_profile().stream_name()
            ts = m.get_timestamp() / 1000.0
            d = m.get_motion_data()
            data_queue.put(('imu', name, ts, d.x, d.y, d.z))
            
        elif frame.is_video_frame() and not frame.is_frameset():
            v = frame.as_video_frame()
            prof = v.get_profile()
            if prof.stream_type() == rs.stream.infrared:
                img = np.asanyarray(v.get_data()).copy()
                data_queue.put(('img_single', prof.stream_index(), v.get_timestamp()/1000.0, img))
    except Exception as e:
        pass 

# ================= 启动管道 =================
pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, BAG_FILE, repeat_playback=False)

print("\n🚀 正在启动【图像与IMU 纳秒级彻底对齐】提取...")
profile = pipeline.start(config, frame_callback)

playback = profile.get_device().as_playback()
playback.set_real_time(True) 

# ================= 消费者主线程 =================
last_accel = [0.0, 0.0, 0.0]
img_count = 0
imu_count = 0

time.sleep(1)

try:
    while True:
        while not data_queue.empty():
            item = data_queue.get()
            
            if item[0] == 'imu':
                name, ts, x, y, z = item[1], item[2], item[3], item[4], item[5]
                if "Accel" in name:
                    last_accel = [x, y, z]
                elif "Gyro" in name:
                    # IMU 时间戳变纳秒
                    ts_nano = int(ts * 1e9)
                    f_imu.write(f"{ts_nano},{x:.6f},{y:.6f},{z:.6f},{last_accel[0]:.6f},{last_accel[1]:.6f},{last_accel[2]:.6f}\n")
                    imu_count += 1
                    
            elif item[0] == 'img_pair':
                ts, img1, img2 = item[1], item[2], item[3]
                # ⚠️ 致命修复：图像时间戳也统一变成纯纳秒整数！
                ts_nano_str = str(int(ts * 1e9)) 
                cv2.imwrite(os.path.join(OUT_DIR, f"cam0/{ts_nano_str}.png"), img1)
                cv2.imwrite(os.path.join(OUT_DIR, f"cam1/{ts_nano_str}.png"), img2)
                f_times.write(ts_nano_str + "\n")
                img_count += 1
                
            elif item[0] == 'img_single':
                idx, ts, img = item[1], item[2], item[3]
                # ⚠️ 致命修复：图像时间戳也统一变成纯纳秒整数！
                ts_nano_str = str(int(ts * 1e9))
                if idx == 1:
                    cv2.imwrite(os.path.join(OUT_DIR, f"cam0/{ts_nano_str}.png"), img)
                    f_times.write(ts_nano_str + "\n")
                    img_count += 1
                elif idx == 2:
                    cv2.imwrite(os.path.join(OUT_DIR, f"cam1/{ts_nano_str}.png"), img)
        
        print(f"   📥 落盘进度: 图像 {img_count} 帧 | IMU {imu_count} 条", end='\r')

        if playback.current_status() == rs.playback_status.stopped and data_queue.empty():
            break
        time.sleep(0.01)

except KeyboardInterrupt:
    pass
finally:
    f_imu.close()
    f_times.close()
    print(f"\n\n📊 最终完美对齐战果:")
    print(f"   图像总数: {img_count} 帧")
    print(f"   IMU 总数: {imu_count} 条")
    os._exit(0)