import pyrealsense2 as rs
import numpy as np
import cv2
import serial
import os
import time
import threading
import logging
from datetime import datetime

# =========================================================
# 1. 核心配置与参数设定
# =========================================================
# --- 数据输出路径 (严格遵循 EuRoC/TUM 数据集格式) ---
DATASET_ROOT = "./captured_dataset"
SESSION_DIR = os.path.join(DATASET_ROOT, datetime.now().strftime("realsense_session_%Y%m%d_%H%M%S"))
CAM0_DIR = os.path.join(SESSION_DIR, "cam0", "data") # 左红外 (IR1) 保存路径
CAM1_DIR = os.path.join(SESSION_DIR, "cam1", "data") # 右红外 (IR2) 保存路径
GPS_FILE = os.path.join(SESSION_DIR, "gps_data.csv") # GPS 数据记录文件

os.makedirs(CAM0_DIR, exist_ok=True)
os.makedirs(CAM1_DIR, exist_ok=True)

# --- 硬件端口配置 ---
GPS_SERIAL_PORT = "/dev/ttyUSB0"  # GPS 串口节点 (Windows下为 'COM3' 等)
GPS_BAUDRATE = 9600               # GPS 波特率

# --- RealSense D456 双目配置 ---
# D456 支持 848x480 或 1280x800 的红外流。848x480@30fps 是跑 SLAM 最稳定、带宽压力最小的黄金配置。
RS_WIDTH = 848
RS_HEIGHT = 480
RS_FPS = 30

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s')
logger = logging.getLogger(__name__)

stop_event = threading.Event()

# =========================================================
# 2. GPS 异步采集线程 (保持不变，使用系统时钟对齐)
# =========================================================
def gps_reader_thread():
    logger.info(f"[GPS] 正在尝试连接串口: {GPS_SERIAL_PORT} (波特率: {GPS_BAUDRATE})...")
    try:
        ser = serial.Serial(GPS_SERIAL_PORT, GPS_BAUDRATE, timeout=1)
        logger.info(f"[GPS] 串口连接成功！开始采集。")
        
        with open(GPS_FILE, 'w') as f:
            f.write("sys_timestamp_s,nmea_sentence,latitude,longitude,altitude\n")
            
    except serial.SerialException as e:
        logger.error(f"❌ [GPS] 串口连接失败: {e}. (如果不用GPS可忽略此报错)")
        return

    while not stop_event.is_set():
        try:
            line = ser.readline().decode('ascii', errors='replace').strip()
            if line:
                ts = time.time() # 使用 Linux 主机系统时钟
                
                lat, lon, alt = "", "", ""
                if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                    parts = line.split(',')
                    if len(parts) > 9 and parts[2] and parts[4] and parts[9]:
                        lat = parts[2] + parts[3] 
                        lon = parts[4] + parts[5] 
                        alt = parts[9]            

                with open(GPS_FILE, 'a') as f:
                    f.write(f"{ts:.6f},{line},{lat},{lon},{alt}\n")
                    
                if lat and lon: 
                    logger.info(f"🛰️ [GPS] 定位更新 -> Lat: {lat}, Lon: {lon}, Alt: {alt}m")
                    
        except Exception as e:
            logger.warning(f"[GPS] 读取异常: {e}")
            
    ser.close()
    logger.info("[GPS] 采集线程已安全退出。")

# =========================================================
# 3. RealSense D456 硬件同步采集线程
# =========================================================
def realsense_thread():
    logger.info("[Camera] 正在初始化 Intel RealSense D456 (请求左/右双目红外流)...")
    
    # 初始化 RealSense 管线
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 开启 Infrared 1 (左目) 和 Infrared 2 (右目)
    # 使用 Y8 (8位灰度图) 格式，这是最纯粹的双目特征提取格式
    config.enable_stream(rs.stream.infrared, 1, RS_WIDTH, RS_HEIGHT, rs.format.y8, RS_FPS)
    config.enable_stream(rs.stream.infrared, 2, RS_WIDTH, RS_HEIGHT, rs.format.y8, RS_FPS)
    
    try:
        # 启动相机流
        profile = pipeline.start(config)
        
        # 强制关闭红外发射器 (Laser Emitter)。
        # 【极其重要】：如果在室外采集，或者对反光标志牌采集，必须关闭激光散斑发射器，否则画面上全是麻点，严重干扰 LSD 提边！
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
            logger.info("[Camera] 已主动关闭红外散斑发射器 (Laser Emitter Disabled) 以获取纯净图像。")
            
        logger.info(f"[Camera] RealSense 启动成功！硬件分辨率: {RS_WIDTH}x{RS_HEIGHT} @ {RS_FPS} FPS")
    except Exception as e:
        logger.error(f"❌ [Camera] RealSense 启动失败: {e}。请检查 USB3.0 线缆是否插紧！")
        stop_event.set()
        return

    # 创建时间戳索引文件
    cam0_csv_path = os.path.join(SESSION_DIR, "cam0", "data.csv")
    cam1_csv_path = os.path.join(SESSION_DIR, "cam1", "data.csv")
    with open(cam0_csv_path, 'w') as f0, open(cam1_csv_path, 'w') as f1:
        f0.write("#timestamp[ns],filename\n")
        f1.write("#timestamp[ns],filename\n")

    frame_count = 0
    start_time = time.time()
    
    while not stop_event.is_set():
        try:
            # 阻塞等待，直到拿到完全对齐的硬件级同步帧
            frames = pipeline.wait_for_frames()
            
            # 提取左红外和右红外帧
            ir1_frame = frames.get_infrared_frame(1)
            ir2_frame = frames.get_infrared_frame(2)
            
            if not ir1_frame or not ir2_frame:
                continue
                
            # 为了跟 GPS 的系统时间严丝合缝对齐，我们这里采用系统时钟 (也可以用 frames.get_timestamp() 硬件时钟)
            sys_ts = time.time() 
            ts_ns = int(sys_ts * 1e9) # 转化为纳秒整数 (EuRoC 标准)
            img_name = f"{ts_ns}.png"
            
            # 将 RealSense 的内存数据转化为 Numpy 数组 (灰度图)
            img_left = np.asanyarray(ir1_frame.get_data())
            img_right = np.asanyarray(ir2_frame.get_data())
            
            # 写入 SSD 硬盘
            cv2.imwrite(os.path.join(CAM0_DIR, img_name), img_left)
            cv2.imwrite(os.path.join(CAM1_DIR, img_name), img_right)
            
            # 记录时间戳索引
            with open(cam0_csv_path, 'a') as f0, open(cam1_csv_path, 'a') as f1:
                f0.write(f"{ts_ns},{img_name}\n")
                f1.write(f"{ts_ns},{img_name}\n")
                
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                real_fps = frame_count / elapsed
                logger.info(f"📷 [Camera] 已采集 {frame_count} 帧硬件同步双目图像 | 实时保存帧率: {real_fps:.2f} FPS")
                
        except Exception as e:
            logger.warning(f"[Camera] 抓取异常: {e}")

    pipeline.stop()
    logger.info("[Camera] 硬件管线已关闭，采集线程安全退出。")

# =========================================================
# 4. 主系统守护进程
# =========================================================
def main():
    logger.info("="*60)
    logger.info(" 🚀 [数据采集系统启动] Intel RealSense D456 + GPS 硬件同步采集引擎")
    logger.info(f" 📂 当前实验数据保存目录: {SESSION_DIR}")
    logger.info("="*60)
    
    t_gps = threading.Thread(target=gps_reader_thread, name="GPS-Thread")
    t_gps.daemon = True 
    t_gps.start()
    
    t_cam = threading.Thread(target=realsense_thread, name="Camera-Thread")
    t_cam.daemon = True
    t_cam.start()
    
    try:
        while True:
            time.sleep(0.5)
            if not t_cam.is_alive(): 
                break
    except KeyboardInterrupt:
        logger.info("\n" + "="*60)
        logger.info("🛑 收到手动中断信号 (Ctrl+C)，正在将缓存刷入硬盘...")
        logger.info("="*60)
        
    finally:
        stop_event.set()
        t_gps.join(timeout=2)
        t_cam.join(timeout=2)
        logger.info("🎉 采集圆满结束！所有的图像和定位数据已安全落盘。")

if __name__ == "__main__":
    main()