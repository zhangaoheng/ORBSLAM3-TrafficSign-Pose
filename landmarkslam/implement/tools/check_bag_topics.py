from rosbags.rosbag1 import Reader

def list_all_topics(bag_path):
    print(f"🔍 正在扫描数据包中的所有话题: {bag_path}")
    topics_found = {}
    with Reader(bag_path) as reader:
        for connection in reader.connections:
            topic = connection.topic
            msgtype = connection.msgtype
            if topic not in topics_found:
                topics_found[topic] = msgtype
                print(f" - ��️ 话题: {topic}  |  类型: {msgtype}")
    print("-" * 40)
    print("扫描完成！可以检查是否有类似 IMU、Gyro 或 Accel 的话题。")

if __name__ == '__main__':
    MY_BAG_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/lines1.bag"
    list_all_topics(MY_BAG_FILE)
