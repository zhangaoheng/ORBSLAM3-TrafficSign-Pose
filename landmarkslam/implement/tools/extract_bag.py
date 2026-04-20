import os
import cv2
import numpy as np
import struct
from rosbags.rosbag1 import Reader

def extract_d456_bag_bulletproof(bag_path, output_dir):
    """
    终极防弹版：手动解构 ROS1 二进制数据，绕过 rosbags 的内存对齐报错！
    """
    topics_of_interest = [
        '/device_0/sensor_0/Infrared_1/image/data',  # 左目
        '/device_0/sensor_0/Infrared_2/image/data'   # 右目
    ]

    cam0_dir = os.path.join(output_dir, "cam0", "data")
    cam1_dir = os.path.join(output_dir, "cam1", "data")
    os.makedirs(cam0_dir, exist_ok=True)
    os.makedirs(cam1_dir, exist_ok=True)

    count_l, count_r = 0, 0
    print(f"🚀 开始硬核解析数据包: {bag_path}")

    with Reader(bag_path) as reader:
        for connection, _, rawdata in reader.messages():
            if connection.topic in topics_of_interest:
                try:
                    # ========================================================
                    # 🚀 终极硬核解析：手动拆解二进制数据，绕过 rosbags Bug
                    # ========================================================
                    pos = 0
                    
                    # 1. 解析 Header (seq, stamp_sec, stamp_nsec, frame_id_len)
                    _, stamp_sec, stamp_nsec, frame_id_len = struct.unpack_from('<IIII', rawdata, pos)
                    pos += 16
                    pos += frame_id_len  # 跳过 frame_id 字符串
                    
                    # 2. 解析图像元数据 (height, width, encoding_len)
                    height, width, encoding_len = struct.unpack_from('<III', rawdata, pos)
                    pos += 12
                    
                    # 3. 解析 encoding 字符串
                    encoding = rawdata[pos:pos+encoding_len].decode('utf-8').strip('\x00')
                    pos += encoding_len
                    
                    # 4. 解析 data_len (is_bigendian: uint8, step: uint32, data_len: uint32)
                    is_bigendian, step, data_len = struct.unpack_from('<BII', rawdata, pos)
                    pos += 9
                    
                    # 5. 精准提取像素内存，不管末尾多余的 padding 字节！
                    img_binary = rawdata[pos:pos+data_len]
                    
                    # 生成完美硬件时间戳
                    timestamp = f"{stamp_sec}.{stamp_nsec:09d}"

                    # ------------------------------------------
                    # 将字节流转为 OpenCV 图像矩阵
                    # ------------------------------------------
                    if encoding in ['mono8', '8UC1']:
                        img_array = np.frombuffer(img_binary, dtype=np.uint8)
                        channels = 1
                    elif encoding in ['mono16', '16UC1']:
                        img_array = np.frombuffer(img_binary, dtype=np.uint16)
                        channels = 1
                    else:
                        img_array = np.frombuffer(img_binary, dtype=np.uint8)
                        channels = 3 # 假设其他为彩色

                    # 矩阵重塑
                    if channels == 1:
                        image = img_array.reshape((height, width))
                    else:
                        image = img_array.reshape((height, width, channels))
                        if encoding == 'rgb8': 
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # 16位红外映射为 8位 可见格式
                    if image.dtype == np.uint16:
                        image = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))

                    # ------------------------------------------
                    # 保存图片
                    # ------------------------------------------
                    save_filename = f"{timestamp}.png"
                    if connection.topic == topics_of_interest[0]:
                        cv2.imwrite(os.path.join(cam0_dir, save_filename), image)
                        count_l += 1
                    else:
                        cv2.imwrite(os.path.join(cam1_dir, save_filename), image)
                        count_r += 1

                except Exception as e:
                    print(f"⚠️ 解析跳过一帧，原因: {e}")

    print("-" * 30)
    print(f"✅ 提取完成！")
    print(f"📸 左目 (cam0): {count_l} 张")
    print(f"📸 右目 (cam1): {count_r} 张")

if __name__ == '__main__':
    MY_BAG_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/data/Homography.bag"
    MY_SAVE_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/data/Homography"
    
    if not os.path.exists(MY_BAG_FILE):
        print(f"❌ 找不到文件: {MY_BAG_FILE}")
    else:
        extract_d456_bag_bulletproof(MY_BAG_FILE, MY_SAVE_DIR)