# 数据采集器

## WSL 连接 USB 设备

WSL2 需要 usbipd-win 才能访问 USB 设备。

### Windows 端（管理员 PowerShell）

```powershell
# 1. 安装 usbipd-win（首次）
winget install usbipd

# 2. 列出 USB 设备
usbipd wsl list

# 3. 找到 RealSense 和 GPS 串口的 BUSID，attach 到 WSL
usbipd wsl attach --busid <RealSense-BUSID>
usbipd wsl attach --busid <GPS-BUSID>
```

### WSL 端验证

```bash
ls /dev/video*    # 确认 RealSense 摄像头
ls /dev/ttyACM*   # 确认 GPS 串口（也可能是 ttyUSB0）
```

## 采集命令

```bash
# 安装依赖
pip install pyrealsense2 pyserial numpy opencv-python

# 基本用法
python collect_data.py --name sunny_run1

# 指定 GPS 串口和波特率
python collect_data.py --name sunny_run1 --gps-port /dev/ttyUSB0 --gps-baud 9600

# 指定输出目录
python collect_data.py --name night_run1 --out /path/to/output
```

## 采集过程

1. 启动后自动开始录制 RGB + Depth + IMU
2. GPS 数据在后台自动同步
3. 按 `Ctrl+C` 停止录制
4. 采集结果自动保存到 `data/deepseek/实验名称/`

## 输出结构

```
data/deepseek/sunny_run1/
├── rgb/{timestamp}.png         # RGB 图像
├── depth/{timestamp}.png       # 深度图
├── camera_intrinsics.json      # 相机内参
├── associations.txt            # RGB-D 关联文件（TUM 格式）
├── gps_trajectory.txt          # GPS 轨迹
└── imu.txt                     # IMU 数据
```
