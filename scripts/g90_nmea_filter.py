#!/usr/bin/env python3
"""G90 完全体: 多线程双向通信 (读 NMEA 发布 Fix + 连网拉取 RTCM(带GGA回传) 写入串口)"""
import serial
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
import threading
import socket
import base64
import time

# ==========================================
# [NTRIP (RTK 账号) 配置信息]
# ==========================================
NTRIP_IP = "103.45.160.41"      
NTRIP_PORT = 8003               # 注意：8003 是 CGCS2000 坐标系，8002 是 WGS84
NTRIP_MOUNTPOINT = "AUTO"      
NTRIP_USER = "qxzh068444"       
NTRIP_PASS = "57355"            
# ==========================================

def nmea_checksum_ok(sentence):
    if not sentence.startswith('$') or '*' not in sentence:
        return False
    body, _, cs = sentence[1:].partition('*')
    calc = 0
    for c in body:
        calc ^= ord(c)
    try:
        return calc == int(cs, 16)
    except ValueError:
        return False

def parse_gga(sentence):
    parts = sentence.split(',')
    if len(parts) < 15:
        return None
    try:
        lat_raw = float(parts[2]) if parts[2] else 0
        lat_dir = parts[3]
        lon_raw = float(parts[4]) if parts[4] else 0
        lon_dir = parts[5]
        quality = int(parts[6]) if parts[6] else 0
        alt = float(parts[9]) if parts[9] else 0
    except (ValueError, IndexError):
        return None
    
    lat = lat_raw / 100.0
    lat = (int(lat) + (lat - int(lat)) * 100.0 / 60.0)
    if lat_dir == 'S': lat = -lat
    lon = lon_raw / 100.0
    lon = (int(lon) + (lon - int(lon)) * 100.0 / 60.0)
    if lon_dir == 'W': lon = -lon
    return (lat, lon, alt, quality)

STATUS_CN = {0: '无效/搜星中', 1: '单点定位', 2: '伪距差分', 4: '固定解(RTK Fix)', 5: '浮点解(RTK Float)'}

class G90Driver(Node):
    def __init__(self):
        super().__init__('g90_driver')
        self.pub = self.create_publisher(NavSatFix, '/gps/fix', 10)
        
        self.latest_gga = None  # [新增] 用于保存最新的 GGA 报文以供回传
        
        try:
            self.ser = serial.Serial('/dev/wheeltec_gnss', 115200, timeout=1)
            self.ser.reset_input_buffer()
            self.get_logger().info('G90 串口已连接.')
        except Exception as e:
            self.get_logger().error(f'串口打开失败: {e}')
            return

        self.ntrip_running = True
        self.ntrip_thread = threading.Thread(target=self.ntrip_worker, daemon=True)
        self.ntrip_thread.start()

        self.timer = self.create_timer(0.02, self.read_loop)
        self._tick = 0

    def ntrip_worker(self):
        auth_str = f"{NTRIP_USER}:{NTRIP_PASS}"
        auth_b64 = base64.b64encode(auth_str.encode('ascii')).decode('ascii')
        
        request = (
            f"GET /{NTRIP_MOUNTPOINT} HTTP/1.0\r\n"
            f"User-Agent: NTRIP Python Client\r\n"
            f"Authorization: Basic {auth_b64}\r\n\r\n"
        )

        while self.ntrip_running:
            self.get_logger().info("正在连接 NTRIP 服务器...")
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10.0)
                sock.connect((NTRIP_IP, NTRIP_PORT))
                sock.sendall(request.encode('ascii'))
                
                response = sock.recv(1024).decode('ascii', errors='ignore')
                if "200 OK" not in response and "ICY 200" not in response:
                    self.get_logger().error(f"NTRIP 认证失败: {response.splitlines()[0]}")
                    time.sleep(5)
                    continue
                
                self.get_logger().info("✅ NTRIP 连接成功！等待模块初始定位以回传 GGA...")
                
                # 设置短超时，为了能非阻塞地循环发送 GGA 和 接收 RTCM
                sock.settimeout(1.0) 
                last_gga_send_time = 0
                
                while self.ntrip_running:
                    # 1. [关键修复] 每 10 秒向服务器上报一次最新的 GGA 位置
                    if self.latest_gga and (time.time() - last_gga_send_time > 10):
                        try:
                            # 必须加回车换行符
                            sock.sendall((self.latest_gga + '\r\n').encode('ascii'))
                            last_gga_send_time = time.time()
                            # 调试用：可以取消注释下面这行看是否在发送
                            # self.get_logger().info("↑ 已向基站上报当前位置，请求专属差分数据") 
                        except Exception as e:
                            break # 发送失败，跳出循环重新连接
                    
                    # 2. 接收服务器下发的 RTCM 差分数据，并写入串口
                    try:
                        rtcm_data = sock.recv(1024)
                        if rtcm_data:
                            self.ser.write(rtcm_data) 
                    except socket.timeout:
                        continue # 超时是正常的，继续循环
                    except Exception:
                        break # 网络真断了，重新连接
                    
            except Exception as e:
                self.get_logger().warning(f"NTRIP 网络断开，准备重连... ({e})")
            finally:
                sock.close()
                time.sleep(3)

    def read_loop(self):
        try:
            while self.ser.in_waiting:
                line = self.ser.readline().decode('ascii', errors='ignore').strip()
                if not line.startswith('$G') or not nmea_checksum_ok(line) or 'GGA' not in line:
                    continue
                
                # [新增] 只要是格式正确的 GGA，就保存下来给网络线程回传用
                self.latest_gga = line 
                
                result = parse_gga(line)
                if result is None: continue
                
                lat, lon, alt, quality = result
                self._tick += 1

                if quality == 0:
                    if self._tick % 20 == 0:
                        self.get_logger().warning('仍在搜星中 (Quality=0)...')
                    continue

                fix = NavSatFix()
                fix.header.stamp = self.get_clock().now().to_msg()
                fix.header.frame_id = 'gps'
                fix.latitude = lat
                fix.longitude = lon
                fix.altitude = alt
                fix.status.status = quality
                self.pub.publish(fix)

                if self._tick % 30 == 0:
                    cn = STATUS_CN.get(quality, f'未知({quality})')
                    if quality == 4:
                        self.get_logger().info(f'🟢 {cn} | 纬度={lat:.6f} 经度={lon:.6f} 高程={alt:.2f}m')
                    else:
                        self.get_logger().info(f'🟡 {cn} | 纬度={lat:.6f} 经度={lon:.6f} 高程={alt:.2f}m')

        except Exception as ex:
            self.get_logger().error(f'串口读取异常: {ex}')

    def destroy_node(self):
        self.ntrip_running = False
        super().destroy_node()

def main():
    rclpy.init()
    node = G90Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()