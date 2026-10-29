# 数据采集器

## 推荐：智能模式同步录制 + 实时 ORB-SLAM3

`record_d456_orbslam_live.py` 使用一个 RealSense pipeline 同时完成在线
ORB-SLAM3 单目惯性跟踪和 EuRoC 格式数据保存。在线算法和离线数据使用完全相同
的图像与 IMU 时间戳，不会重复打开 D456。

默认 `--mode auto` 会自动发现 GPS 串口、检测图形会话、按需编译实时
程序，并优先选择 `live` 模式。只有实时程序无法构建时才回退为
`record`，且会明确打印原因。GPS 默认是实验必需项；如果是无 GPS 调试，
需要显式加 `--gps-port none --allow-no-gps`。

```bash
cd /home/ehuy/ORBSLAM3-TrafficSign-Pose

# 最简单：自动选择模式、GPS 和 Viewer，并按启动时间命名
landmarkslam/implement/data_collector/record.sh

# 等价的完整命令
gps_venv/bin/python3 \
  landmarkslam/implement/data_collector/record_d456_orbslam_live.py \
  --mode auto --out recordings \
  --gps-port auto --gps-baud 115200

# 强制实时建图、固定 120 秒；无显示器时关闭 Viewer
gps_venv/bin/python3 \
  landmarkslam/implement/data_collector/record_d456_orbslam_live.py \
  --mode live --name live_road_02 --duration 120 --viewer off

# 只录制数据，不运行 ORB-SLAM3
gps_venv/bin/python3 \
  landmarkslam/implement/data_collector/record_d456_orbslam_live.py \
  --mode record --name raw_road_03 --gps-port auto
```

每次实验只保存 `mav0/cam0` 单目彩色图像、`mav0/imu0/data.csv`、GPS
同步结果、逐帧即时 `realtime_odometry.csv`、优化后的 `trajectory_euroc.txt` 和
`keyframes_euroc.txt`，并自动生成当前 D456 的内外参文件与
`run_offline.sh`。指定 `--gps-port` 后还会保存 NMEA 定位以及逐图像 GPS 对齐表。
深度流不会启动，也不会创建 `depth/` 和 RGB-D `associations.txt`，从而降低
一小时录制的容量、CPU 编码压力和安全退出时的写盘积压。
SLAM 地图和 Current Frame 只使用 ORB-SLAM3 自带的 Pangolin Viewer，不再
绘制替代地图。没有 `/dev/dri/renderD128` 时会自动让 Pangolin 使用 llvmpipe
软件渲染。GPS 使用独立的 `GPS Live` 窗口显示接收状态、卫星数、定位质量、
经纬高和实时轨迹；可用 `--gps-viewer auto|on|off` 控制。Pangolin 开启时
`--preview on` 不会再创建重复的自制 SLAM/相机窗口。
如果本次运动不足以完成视觉惯性初始化，原始图像和 IMU 仍会完整保留，但不会
伪造优化轨迹。统一的 `metadata.json` 中 `experiment_ready=true` 才表示
ORB-SLAM3 初始化和 GPS 同步质量都通过。

GPS 串口收到 NMEA 但 GGA 的 `fix_quality=0` 时，程序会明确提示到室外等待
卫星锁定；这种情况表示 GPS 在线但尚无位置，不会再误报为损坏语句。

## 推荐：同步录制 D456 单目、IMU 和 GPS

`record_d456_gps_synced.py` 是新的采集入口。它使用 D456 设备时钟保存相机和
IMU 数据，使用 NMEA UTC 保存 GPS，并在录制结束后生成逐图像 GPS 对齐表和
ORB-SLAM3 EuRoC 格式数据。

依赖为 `pyrealsense2 pyserial opencv-python numpy`。当前环境若提示缺少
`pyrealsense2`，需要先在用于采集的 Python 环境中安装 RealSense Python
绑定。

```bash
cd /home/ehuy/ORBSLAM3-TrafficSign-Pose

# 先运行不需要硬件的解析与同步自测
gps_venv/bin/python3 \
  landmarkslam/implement/data_collector/record_d456_gps_synced.py --self-test

# 录制到 recordings/road_01，Ctrl+C 停止
gps_venv/bin/python3 \
  landmarkslam/implement/data_collector/record_d456_gps_synced.py \
  --name road_01 \
  --out recordings \
  --gps-port auto \
  --gps-baud 115200

# 固定录制 120 秒
gps_venv/bin/python3 \
  landmarkslam/implement/data_collector/record_d456_gps_synced.py \
  --name road_01 --duration 120 --gps-port /dev/ttyACM0
```

录制开始后先静置约 2 秒，随后必须让设备产生平移和转动，给惯性初始化提供
充分激励。结束后检查 `metadata.json` 中的
`quality.orbslam_ready` 和 `quality.gps_sync_ready`。两项通过后可运行
输出目录中的：

```bash
./run_orbslam.sh
```

没有将 GPS PPS 接入相机时，只能实现经过误差测量的软件时钟同步，不能实现
物理零误差硬同步。同步拟合残差和每帧 GPS 时间差都会保存在输出文件中。

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
