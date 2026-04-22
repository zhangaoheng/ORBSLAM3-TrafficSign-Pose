from rosbags.rosbag1 import Reader
import cv2
import os
import numpy as np
import struct

def extract_d456_bag_bulletproof(bag_path, output_dir):
    """
    终极防弹版：手动解构 ROS1 二进制数据，绕过 rosbags 的内存对齐报错！
    """
    img_topics = [
        '/device_0/sensor_0/Infrared_1/image/data',  # 左目
        '/device_0/sensor_0/Infrared_2/image/data'   # 右目
    ]
    imu_topics = [
        '/device_0/sensor_2/Gyro_0/imu/data',
        '/device_0/sensor_2/Accel_0/imu/data'
    ]

    cam0_dir = os.path.join(output_dir, "cam0", "data")
    cam1_dir = os.path.join(output_dir, "cam1", "data")
    imu_dir = os.path.join(output_dir, "imu0")
    
    os.makedirs(cam0_dir, exist_ok=True)
    os.makedirs(cam1_dir, exist_ok=True)
    os.makedirs(imu_dir, exist_ok=True)

    times_txt_path = os.path.join(output_dir, "times.txt")
    f_times = open(times_txt_path, 'w')

    imu_csv_path = os.path.join(imu_dir, "data.csv")
    with open(imu_csv_path, 'w') as f_imu:
        f_imu.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")

    count_l, count_r, count_imu = 0, 0, 0
    gyro_data = {}

    print(f"🚀 开始硬核解析数据包: {bag_path}")
    print("-" * 30)

    # 用 rosbags 的 Reader 作为外壳遍历消息，用 struct 解析内容
    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic in img_topics or connection.topic in imu_topics:
                try:
                    pos = 0
                    _, stamp_sec, stamp_nsec, frame_id_len = struct.unpack_from('<IIII', rawdata, pos)
                    pos += 16
                    pos += frame_id_len  # 跳过 frame_id 字符串
                    
                    timestamp_ns = int(stamp_sec * 1e9 + stamp_nsec)
                    
                    # ⚠️ 必须用整数的纳秒字符串！
                    timestamp_str = str(timestamp_ns)

                    if connection.topic in img_topics:
                        height, width, encoding_len = struct.unpack_from('<III', rawdata, pos)
                        pos += 12
                        encoding = rawdata[pos:pos+encoding_len].decode('utf-8').strip('\x00')
                        pos += encoding_len
                        is_bigendian, step, data_len = struct.unpack_from('<BII', rawdata, pos)
                        pos += 9
                        img_binary = rawdata[pos:pos+data_len]
                        
                        if encoding in ['mono8', '8UC1']:
                            img_array = np.frombuffer(img_binary, dtype=np.uint8)
                            channels = 1
                        elif encoding in ['mono16', '16UC1']:
                            img_array = np.frombuffer(img_binary, dtype=np.uint16)
                            channels = 1
                        else:
                            img_array = np.frombuffer(img_binary, dtype=np.uint8)
                            channels = 3 

                        if channels == 1:
                            image = img_array.reshape((height, width))
                        else:
                            image = img_array.reshape((height, width, channels))
                            if encoding == 'rgb8': 
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        if image.dtype == np.uint16:
                            image = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))
                            
                        save_filename = f"{timestamp_str}.png"
                        if connection.topic == img_topics[0]:
                            cv2.imwrite(os.path.join(cam0_dir, save_filename), image)
                            count_l += 1
                            f_times.write(f"{timestamp_str}\n")
                        else:
                            cv2.imwrite(os.path.join(cam1_dir, save_filename), image)
                            count_r += 1

                    elif connection.topic in imu_topics:
                        pos += 104  # 32 + 72
                        wx, wy, wz = struct.unpack_from('<ddd', rawdata, pos)
                        pos += 24
                        pos += 72
                        ax, ay, az = struct.unpack_from('<ddd', rawdata, pos)
                        
                        if connection.topic == imu_topics[0]:
                            gyro_data[timestamp_ns] = (wx, wy, wz)
                            
                        elif connection.topic == imu_topics[1]:
                            if gyro_data:
                                closest_time = min(gyro_data.keys(), key=lambda k: abs(k - timestamp_ns))
                                if abs(closest_time - timestamp_ns) < 5_000_000:
                                    g_wx, g_wy, g_wz = gyro_data[closest_time]
                                    with open(imu_csv_path, 'a') as f_imu:
                                        f_imu.write(f"{timestamp_ns},{g_wx},{g_wy},{g_wz},{ax},{ay},{az}\n")
                                    count_imu += 1
                                    
                                    keys_to_del = [k for k in gyro_data.keys() if k <= closest_time]
                                    for k in keys_to_del:
                                        del gyro_data[k]

                except Exception as e:
                    pass
    f_times.close()
    print("-" * 30)
    print(f"✅ 提取完成！")
    print(f"📸 左目 (cam0): {count_l} 张")
    print(f"�� 右目 (cam1): {count_r} 张")
    print(f"✈️ IMU (imu0): {count_imu} 条数据 -> 存入 imu0/data.csv")

if __name__ == '__main__':
    MY_BAG_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/lines2.bag"
    # 🔥 注意这里的 MY_SAVE_DIR 是没有 implement 的路径，这样能和你的 shell 脚本绝对对齐！
    MY_SAVE_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/lines2"
    extract_d456_bag_bulletproof(MY_BAG_FILE, MY_SAVE_DIR)
