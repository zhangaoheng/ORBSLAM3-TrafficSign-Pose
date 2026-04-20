from rosbags.rosbag1 import Reader

# 替换成你的真实路径
bag_path = "/home/zah/ORB_SLAM3-master/landmarkslam/data/Homography.bag"

print(f"🔍 正在扫描 Bag 包: {bag_path}")
print("-" * 50)

try:
    with Reader(bag_path) as reader:
        print(f"{'Topic 名称':<40} | {'数据类型':<30} | {'消息数量'}")
        print("-" * 90)
        for connection in reader.connections:
            print(f"{connection.topic:<40} | {connection.msgtype:<30} | {connection.msgcount}")
except Exception as e:
    print(f"❌ 读取失败: {e}")