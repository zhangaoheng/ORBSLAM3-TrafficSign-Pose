#!/usr/bin/env python3
"""订阅 /gps/fix，实时记录 CSV + 弹窗显示二维轨迹（经纬度）"""
import csv
import time
import sys
import threading
from pathlib import Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class GpsRecorder(Node):
    def __init__(self, csv_path):
        super().__init__('gps_recorder')
        self.sub = self.create_subscription(NavSatFix, '/gps/fix', self.cb, 10)
        self.f = open(csv_path, 'w', newline='')
        self.w = csv.writer(self.f)
        self.w.writerow(['timestamp', 'latitude', 'longitude', 'altitude', 'status'])
        self.lats, self.lons = [], []
        self._lock = threading.Lock()
        self._running = True
        self.get_logger().info(f'Recording to {csv_path}')

    def cb(self, msg):
        ts = time.time()
        self.w.writerow([ts, msg.latitude, msg.longitude, msg.altitude, msg.status.status])
        self.f.flush()
        with self._lock:
            self.lats.append(msg.latitude)
            self.lons.append(msg.longitude)

    def shutdown(self):
        self._running = False
        self.f.close()
        self.get_logger().info('Stopped')

def main():
    rclpy.init()
    csv_path = sys.argv[1] if len(sys.argv) > 1 else f'gps_{time.strftime("%Y%m%d_%H%M%S")}.csv'
    node = GpsRecorder(Path(csv_path).resolve())

    # ROS2 spin 后台线程
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # matplotlib 轨迹窗口（主线程）
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('GPS Trajectory')
    line, = ax.plot([], [], 'b-', linewidth=1.5)
    point, = ax.plot([], [], 'ro', markersize=4)

    try:
        while rclpy.ok() and node._running:
            with node._lock:
                if node.lats:
                    line.set_data(node.lons, node.lats)
                    point.set_data([node.lons[-1]], [node.lats[-1]])
                    ax.relim()
                    ax.autoscale_view()
            plt.pause(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        plt.close('all')
        rclpy.shutdown()

if __name__ == '__main__':
    main()
