#!/usr/bin/env python3
"""订阅 /camera/camera/color/image_raw，cv2.imshow 实时查看（不影响录制）"""
import os
import threading
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data  # [新增] 导入传感器 QoS

class ImageViewer(Node):
    def __init__(self):
        super().__init__('image_viewer')
        self.bridge = CvBridge()
        self._latest = None
        self._lock = threading.Lock()
        self._running = True
        
        # [修改] 使用 qos_profile_sensor_data 替换 10
        self.sub = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw', 
            self.cb, 
            qos_profile_sensor_data
        )
        self.get_logger().info('Viewer ready, waiting for frames...')

    def cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self._lock:
                self._latest = frame
        except Exception as e:
            self.get_logger().error(f'Frame error: {e}')

def main():
    rclpy.init()
    node = ImageViewer()

    # ROS2 spin 在后台线程
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # OpenCV GUI 在主线程（避免 Qt 线程冲突）
    cv2.namedWindow('D456 Camera', cv2.WINDOW_NORMAL)
    count = 0
    try:
        while rclpy.ok() and node._running:
            with node._lock:
                frame = node._latest
            if frame is not None:
                count += 1
                if count == 1:
                    node.get_logger().info(f'Receiving frames: {frame.shape[1]}x{frame.shape[0]}')
                cv2.imshow('D456 Camera', frame)
            
            # waitKey 必须存在，否则窗口会无响应
            if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 退出
                break
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()