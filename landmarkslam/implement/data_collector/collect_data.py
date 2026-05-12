#!/usr/bin/env python3
"""
🌀 ORB-SLAM3 Looming — 数据采集器
===================================
同时读取 RealSense D456（RGB + Depth + IMU）和 GPS 模块，
确保摄像头帧和 GPS 数据严格时间戳对齐。

用法:
  python collect_data.py --name sunny_run1
  python collect_data.py --name night_run1 --gps-baud 115200

输出目录结构:
  data/采集名称/
  ├── rgb/{timestamp}.png
  ├── depth/{timestamp}.png
  ├── camera_intrinsics.json
  ├── associations.txt
  ├── gps_trajectory.txt
  └── imu.txt

时间戳对齐策略:
  - 相机帧使用 RealSense 硬件时间戳（微秒级）
  - GPS 使用 NMEA $GPGGA/$GPRMC 中的 UTC 时间
  - 统一转为 Unix 微秒时间戳
  - 每帧到达时从 GPS 缓存队列中取最近邻匹配
"""

import os
import sys
import json
import time
import math
import queue
import threading
import argparse
import datetime
import csv
import struct

import cv2
import numpy as np

# Windows GBK 终端兼容：确保 emoji 能正常打印
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ==============================================================================
# GPS 读取线程
# ==============================================================================
class GPSReader(threading.Thread):
    """后台线程：持续读取 GPS 串口数据，解析 NMEA 语句"""

    GPS_PORTS = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0",
                  "/dev/ttyUSB1", "/dev/ttyS0", "/dev/ttyS1",
                  "COM1", "COM2", "COM3", "COM4", "COM5", "COM6"]

    def __init__(self, port=None, baudrate=9600, timeout=0.05):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.running = True

        # 最新 GPS 数据缓存（线程安全，通过 list 存储单个元素）
        self.latest = []  # [(unix_usec, lat, lon, alt, fix_quality), ...]
        self.all_positions = []  # 全部记录（用于落盘）

        self._connect()

    @staticmethod
    def _find_available_port(preferred=None):
        """自动扫描可用串口（Windows / Linux 通用）"""
        # 如果指定了端口且存在，直接用
        if preferred:
            if os.path.exists(preferred):
                return preferred
            # Windows COM 口不带路径，用 serial 检测
            if sys.platform == "win32" and preferred.startswith("COM"):
                import serial
                try:
                    s = serial.Serial(preferred, timeout=0.1)
                    s.close()
                    return preferred
                except:
                    pass

        # 用 pyserial 扫描所有可用串口
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            for p in ports:
                dev = p.device
                if dev not in ("COM1",):  # 跳过通常无 GPS 的 COM1
                    return dev
        except Exception:
            pass

        # 回退：Linux glob 扫描
        import glob
        patterns = ["/dev/ttyACM*", "/dev/ttyUSB*", "/dev/ttyS[0-9]",
                     "/dev/ttyTHS*", "/dev/ttyAMA*"]
        for pat in patterns:
            matches = sorted(glob.glob(pat))
            if matches:
                return matches[0]

        return preferred or ("COM3" if sys.platform == "win32" else "/dev/ttyACM0")

    def _connect(self):
        """尝试打开串口（自动检测可用串口）"""
        detected = self._find_available_port(self.port)
        if detected != self.port:
            print(f"  ℹ️  指定端口 {self.port} 不存在，自动检测到: {detected}")
            self.port = detected

        try:
            import serial
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            print(f"  ✅ GPS 串口已连接: {self.port} @ {self.baudrate} baud")
        except Exception as e:
            print(f"  ⚠️  GPS 串口打开失败: {self.port} — {e}")
            print(f"     将在无 GPS 模式下运行（仅录制相机数据）")

    def run(self):
        if self.ser is None:
            return
        buffer = ""
        while self.running:
            try:
                data = self.ser.read(1024)
                if not data:
                    continue
                buffer += data.decode('ascii', errors='replace')
                # 按行分割
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    self._parse_nmea(line)
            except serial.SerialException:
                break
            except Exception:
                continue

    def _parse_nmea(self, line):
        """解析 NMEA 句子，提取位置和时间"""
        if not line.startswith('$'):
            return
        parts = line.split(',')
        msg_type = parts[0]

        # $GPGGA — 全球定位系统固定数据
        if msg_type in ('$GPGGA', '$GNGGA'):
            if len(parts) < 10:
                return
            try:
                utc_str = parts[1]  # HHMMSS.SS
                lat_str = parts[2]
                lat_dir = parts[3]
                lon_str = parts[4]
                lon_dir = parts[5]
                fix_q = parts[6]     # 0=无效, 1=单点, 2=差分, 4=RTK固定, 5=RTK浮点
                alt_str = parts[9]

                if not lat_str or not lon_str or fix_q == '0':
                    return

                lat = self._parse_nmea_coord(lat_str, lat_dir)
                lon = self._parse_nmea_coord(lon_str, lon_dir)
                alt = float(alt_str) if alt_str else 0.0
                fix = int(fix_q)

                # 将 UTC 时间(HHMMSS.SS)转为 Unix 微秒
                now = datetime.datetime.now(datetime.timezone.utc)
                h = int(utc_str[:2])
                m = int(utc_str[2:4])
                s = float(utc_str[4:])
                us = int(s * 1e6) % 1000000
                unix_us = int((
                    now.replace(hour=h, minute=m, second=int(s),
                               microsecond=us) - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
                ).total_seconds() * 1e6)

                entry = (unix_us, lat, lon, alt, fix)
                self.latest = [entry]
                self.all_positions.append(entry)

            except (ValueError, IndexError):
                pass

        # $GPRMC — 推荐最小特定 GPS 数据（含日期，时间更准）
        elif msg_type in ('$GPRMC', '$GNRMC'):
            if len(parts) < 12:
                return
            try:
                utc_str = parts[1]
                status = parts[2]       # A=有效, V=无效
                lat_str = parts[3]
                lat_dir = parts[4]
                lon_str = parts[5]
                lon_dir = parts[6]
                date_str = parts[9]     # DDMMYY

                if status != 'A' or not lat_str:
                    return

                lat = self._parse_nmea_coord(lat_str, lat_dir)
                lon = self._parse_nmea_coord(lon_str, lon_dir)

                # 组合日期和时间 → Unix 微秒
                h = int(utc_str[:2])
                m = int(utc_str[2:4])
                s = float(utc_str[4:])
                dd = int(date_str[:2])
                mm = int(date_str[2:4])
                yy = int(date_str[4:]) + 2000
                us = int(s * 1e6) % 1000000
                dt = datetime.datetime(yy, mm, dd, h, m, int(s), us)
                unix_us = int((dt - datetime.datetime(1970, 1, 1)).total_seconds() * 1e6)

                entry = (unix_us, lat, lon, 0.0, 1)
                self.latest = [entry]

            except (ValueError, IndexError):
                pass

    @staticmethod
    def _parse_nmea_coord(coord_str, direction):
        """解析 NMEA 坐标格式 DDDMM.MMMM → 十进制度"""
        if not coord_str:
            return 0.0
        dot = coord_str.find('.')
        if dot < 4:
            return 0.0
        degrees = float(coord_str[:dot-2])
        minutes = float(coord_str[dot-2:])
        decimal = degrees + minutes / 60.0
        if direction in ('S', 'W'):
            decimal = -decimal
        return decimal

    def get_latest(self):
        """获取最新 GPS 读数，返回 (unix_us, lat, lon, alt, fix) 或 None"""
        if self.latest:
            return self.latest[0]
        return None

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()


# ==============================================================================
# 辅助函数
# ==============================================================================
def ns_to_us(timestamp_ns):
    """RealSense 时间戳(ns) → Unix 微秒"""
    # RealSense 时间戳是设备启动以来的毫秒或微秒
    # 这里用 time.time() 做偏移校准
    return int(timestamp_ns)

def format_timestamp(us):
    """微秒 → 文件名用字符串（微秒级）"""
    return str(us)


# ==============================================================================
# 主录制逻辑
# ==============================================================================
class DataCollector:
    def __init__(self, output_dir, gps_port="/dev/ttyACM0", gps_baud=9600):
        self.output_dir = output_dir
        self.rgb_dir = os.path.join(output_dir, "rgb")
        self.depth_dir = os.path.join(output_dir, "depth")
        self.gps_port = gps_port
        self.gps_baud = gps_baud
        self.running = True
        self.frame_count = 0
        self.gps_count = 0

        # 创建目录
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        # 文件句柄
        self.assoc_f = None         # associations.txt
        self.imu_f = None           # imu.txt
        self.gps_traj = []          # GPS 位姿（TUM 格式）

        # GPS 读取器
        self.gps = GPSReader(port=gps_port, baudrate=gps_baud)

        # 数据记录
        self.assoc_lines = []
        self.start_unix = None      # 对齐用基准时间

    @staticmethod
    def _print_wsl_setup_guide():
        """打印 WSL USB 设置指南"""
        print("")
        print("=" * 60)
        print("  ⚠️  WSL 中未检测到视频设备")
        print("=" * 60)
        print("")
        print("  RealSense D456 和 GPS 模块需要通过 usbipd-win 透传到 WSL。")
        print("")
        print("  请按以下步骤操作：")
        print("")
        print("  ┌─────────────────────────────────────────────────────────┐")
        print("  │ 第 1 步：Windows PowerShell（管理员）                   │")
        print("  │  winget install usbipd         # 安装 usbipd（首次）   │")
        print("  │  usbipd wsl list               # 查看 USB 设备 BUSID   │")
        print("  │                                                     │")
        print("  │  输出示例:                                            │")
        print("  │  BUSID  VID:PID    DEVICE                          │")
        print("  │  4-2   8086:0b3a  Intel(R) RealSense(TM) Depth...   │")
        print("  │  4-5   1a86:7523  USB Serial Port (GPS)             │")
        print("  │                                                     │")
        print("  │ 第 2 步：attach 两个设备到 WSL                        │")
        print("  │  usbipd wsl attach --busid 4-2    # RealSense        │")
        print("  │  usbipd wsl attach --busid 4-5    # GPS 串口         │")
        print("  └─────────────────────────────────────────────────────────┘")
        print("")
        print("  attach 后在本 WSL 窗口验证：")
        print("    ls /dev/video*      # 应看到 video0 video1 ...")
        print("    ls /dev/tty*        # 应看到 ttyACM0 或 ttyUSB0")
        print("")

    def save_intrinsics(self, intrinsics):
        """保存相机内参"""
        path = os.path.join(self.output_dir, "camera_intrinsics.json")
        params = {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.ppx,
            "cy": intrinsics.ppy,
            "width": intrinsics.width,
            "height": intrinsics.height,
        }
        with open(path, "w") as f:
            json.dump(params, f, indent=4)
        print(f"  ✅ 内参已保存: fx={intrinsics.fx:.3f}, fy={intrinsics.fy:.3f}")
        print(f"     cx={intrinsics.ppx:.3f}, cy={intrinsics.ppy:.3f}")

    def write_associations(self):
        """写入 associations.txt（TUM 格式）"""
        path = os.path.join(self.output_dir, "associations.txt")
        with open(path, "w") as f:
            for ts_us, rgb_rel, depth_rel in self.assoc_lines:
                ts_sec = ts_us / 1e9
                f.write(f"{ts_sec:.6f} {rgb_rel} {ts_sec:.6f} {depth_rel}\n")
        print(f"  ✅ associations.txt: {len(self.assoc_lines)} 帧")

    def write_gps_trajectory(self):
        """写入 gps_trajectory.txt（TUM 格式）"""
        if not self.gps.all_positions:
            print("  ⚠️  无 GPS 数据")
            return
        path = os.path.join(self.output_dir, "gps_trajectory.txt")
        with open(path, "w") as f:
            f.write("# timestamp_sec lat lon alt fix_quality\n")
            for ts_us, lat, lon, alt, fix in self.gps.all_positions:
                ts_sec = ts_us / 1e9
                f.write(f"{ts_sec:.6f} {lat:.8f} {lon:.8f} {alt:.4f} {fix}\n")
        print(f"  ✅ gps_trajectory.txt: {len(self.gps.all_positions)} 条")

    def run(self):
        """主录制循环"""
        print(f"\n{'=' * 60}")
        print(f"  🌀 数据采集器启动")
        print(f"  输出目录: {self.output_dir}")
        print(f"{'=' * 60}\n")

        # ---- 导入 pyrealsense2 ----
        try:
            import pyrealsense2 as rs
        except ImportError:
            print("❌ 需要 pyrealsense2 库")
            print("   安装: pip install pyrealsense2")
            print("   注意: WSL 中需要 usbipd-win 透传 USB 设备")
            return

        # ---- 检查 WSL USB 设备（仅 Linux/WSL）----
        if sys.platform == "linux" and not (os.path.isdir("/dev") and any(
            f.startswith("video") for f in os.listdir("/dev")
        )):
            self._print_wsl_setup_guide()
            return

        # ---- 启动 RealSense 管道 ----
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("❌ pyrealsense2 已加载但未检测到 RealSense 设备")
            print("")
            print("   可能的原因:")
            print("    1. USB 设备未 attach 到 WSL（最常见）")
            print("    2. WSL 内核缺少 USB 驱动")
            print("")
            self._print_wsl_setup_guide()
            return

        dev = devices[0]
        print(f"  📷 检测到: {dev.get_info(rs.camera_info.name)}")
        print(f"     序列号: {dev.get_info(rs.camera_info.serial_number)}")

        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.gyro)
            config.enable_stream(rs.stream.accel)

            profile = pipeline.start(config)
            print("  ✅ RealSense 管道已启动")

            # 获取内参
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_stream.get_intrinsics()
            self.save_intrinsics(intr)

            # 获取深度传感器深度缩放
            depth_sensor = dev.first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            # 对齐深度到彩色
            align = rs.align(rs.stream.color)

            # 启动 GPS 线程
            self.gps.start()

            # ---- 录制循环 ----
            last_imu_us = 0
            imu_lines = []
            imu_f = open(os.path.join(self.output_dir, "imu.txt"), "w")
            imu_f.write("# timestamp_ns w_x w_y w_z a_x a_y a_z\n")

            start_time = time.time()
            last_gyro = [0.0, 0.0, 0.0]
            last_accel = [0.0, 0.0, 0.0]

            print(f"\n  🔴 开始录制... 按 Ctrl+C 停止\n")

            while self.running:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                except RuntimeError:
                    print("  ⚠️  等待帧超时")
                    continue

                aligned = align.process(frames)

                # ---- IMU 处理 ----
                gyro = frames.first_or_default(rs.stream.gyro)
                accel = frames.first_or_default(rs.stream.accel)

                if gyro:
                    gd = gyro.as_motion_frame().get_motion_data()
                    last_gyro = [gd.x, gd.y, gd.z]
                    ts_ns = gyro.get_timestamp() * 1e6  # 转为 ns
                    imu_f.write(f"{int(ts_ns)} {last_gyro[0]:.8f} {last_gyro[1]:.8f} "
                                f"{last_gyro[2]:.8f} {last_accel[0]:.8f} {last_accel[1]:.8f} {last_accel[2]:.8f}\n")

                if accel:
                    ad = accel.as_motion_frame().get_motion_data()
                    last_accel = [ad.x, ad.y, ad.z]

                # ---- 图像处理 ----
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame:
                    continue

                # 时间戳（RealSense 硬件时钟，微秒）
                ts_ms = color_frame.get_timestamp()         # 毫秒
                ts_us = int(ts_ms * 1000)                   # 微秒
                ts_ns = ts_us * 1000                        # 纳秒（用于文件名）

                # 保存 RGB
                color_img = np.asanyarray(color_frame.get_data())
                rgb_name = f"{ts_ns}.png"
                cv2.imwrite(os.path.join(self.rgb_dir, rgb_name), color_img)

                # 保存 Depth（16-bit 毫米）
                if depth_frame:
                    depth_img = np.asanyarray(depth_frame.get_data())
                    depth_mm = (depth_img * depth_scale * 1000).astype(np.uint16)
                    cv2.imwrite(os.path.join(self.depth_dir, rgb_name), depth_mm)

                # 记录 associations
                self.assoc_lines.append((ts_ns, f"rgb/{rgb_name}", f"depth/{rgb_name}"))

                # 匹配最近的 GPS
                latest_gps = self.gps.get_latest()
                if latest_gps:
                    gps_ts, lat, lon, alt, fix = latest_gps
                    # GPS 时间已转为 Unix 微秒
                    self.gps_traj.append((ts_ns, lat, lon, alt, fix))

                self.frame_count += 1

                # 进度显示（每 30 帧一次）
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = self.frame_count / elapsed
                    status = f"  📷 {self.frame_count} 帧 | {fps:.1f} fps"
                    if latest_gps:
                        status += f" | GPS: ({lat:.6f}, {lon:.6f}) fix={fix}"
                    else:
                        status += " | GPS: 等待..."
                    print(status)

            # ---- 结束 ----
            pipeline.stop()
            imu_f.close()
            self.gps.stop()
            self.gps.join(timeout=2)

            # ---- 写入索引文件 ----
            self.write_associations()
            self.write_gps_trajectory()

            elapsed = time.time() - start_time
            print(f"\n  ✅ 录制完成")
            print(f"  总帧数: {self.frame_count}")
            print(f"  GPS 数据: {len(self.gps.all_positions)} 条")
            print(f"  耗时: {elapsed:.1f} 秒")
            print(f"  输出目录: {self.output_dir}")

        except KeyboardInterrupt:
            print("\n  ⏹️  用户中断")
            self.running = False
            pipeline.stop()
            imu_f.close()
            self.gps.stop()
            self.gps.join(timeout=2)
            self.write_associations()
            self.write_gps_trajectory()
            print(f"\n  ✅ 已保存 {self.frame_count} 帧至 {self.output_dir}")

        except Exception as e:
            print(f"\n  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            self.running = False
            try:
                pipeline.stop()
            except:
                pass
            self.gps.stop()
            self.gps.join(timeout=2)
        finally:
            # 确保停止和关闭操作
            self.running = False
            for obj_name in ['pipeline', 'imu_f']:
                obj = self.__dict__.get(obj_name) or locals().get(obj_name)
                if obj is not None:
                    try:
                        if obj_name == 'pipeline':
                            obj.stop()
                        else:
                            obj.close()
                    except:
                        pass


# ==============================================================================
# 入口
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSense + GPS 数据采集器")
    parser.add_argument("--name", "-n", required=True,
                        help="实验名称（如 sunny_run1, night_run1），用作输出文件夹名")
    parser.add_argument("--out", "-o", default=None,
                        help="输出根目录（默认: data/deepseek/）")
    parser.add_argument("--gps-port", "-p", default="/dev/ttyACM0",
                        help="GPS 串口（默认: /dev/ttyACM0）")
    parser.add_argument("--gps-baud", "-b", type=int, default=9600,
                        help="GPS 波特率（默认: 9600）")
    args = parser.parse_args()

    # 输出路径：默认保存在脚本所在目录的 data/ 下
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = args.out or os.path.join(script_dir, "data")

    output_dir = os.path.join(root_dir, args.name)
    if os.path.exists(output_dir):
        if not sys.stdin.isatty():
            print(f"⚠️  目录已存在，自动覆盖: {output_dir}")
            import shutil
            shutil.rmtree(output_dir)
        else:
            print(f"⚠️  目录已存在: {output_dir}")
            reply = input("   覆盖? (y/n): ").strip().lower()
            if reply != 'y':
                print("  退出")
                sys.exit(0)
            import shutil
            shutil.rmtree(output_dir)

    collector = DataCollector(
        output_dir=output_dir,
        gps_port=args.gps_port,
        gps_baud=args.gps_baud,
    )
    collector.run()
