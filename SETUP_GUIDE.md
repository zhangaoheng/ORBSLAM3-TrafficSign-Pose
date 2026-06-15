# ORB-SLAM3 + D456 RealSense 完整安装与使用流程

> 适用环境：Ubuntu 26.04 LTS / ROS2 Lyrical / D456 相机
> 整理时间：2026-06-15

---

## 一、安装 ORB-SLAM3

### 1.1 系统依赖

```bash
sudo apt-get update
sudo apt-get install -y cmake git build-essential
sudo apt-get install -y libeigen3-dev
sudo apt-get install -y libboost-dev libboost-serialization-dev libboost-filesystem-dev libboost-thread-dev
sudo apt-get install -y libssl-dev pkg-config
sudo apt-get install -y libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev
```

### 1.2 OpenCV

```bash
sudo apt-get install -y libopencv-dev
# 验证
pkg-config --modversion opencv4   # 应为 4.x
```

### 1.3 Pangolin（从源码编译，需要 C++14 兼容）

```bash
# 依赖
sudo apt-get install -y libglew-dev libglu1-mesa-dev libegl1-mesa-dev libwayland-dev libxrandr-dev libxkbcommon-dev libepoxy-dev

# 下载最新版
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j4
sudo cmake --install .
sudo ldconfig
```

### 1.4 编译 ORB-SLAM3

> ⚠️ 关键：项目可能缺少 CMakeLists.txt 和 Vocabulary 文件，需要从官方仓库补全

```bash
# 下载官方源码包提取缺失文件
curl -sL https://github.com/UZ-SLAMLab/ORB_SLAM3/archive/refs/heads/master.tar.gz -o orbslam3.tar.gz

# 提取 CMakeLists.txt 和 Thirdparty 构建文件
tar -xzf orbslam3.tar.gz --strip-components=1 \
  "ORB_SLAM3-master/CMakeLists.txt" \
  "ORB_SLAM3-master/Thirdparty/DBoW2/CMakeLists.txt" \
  "ORB_SLAM3-master/Thirdparty/g2o/CMakeLists.txt" \
  "ORB_SLAM3-master/Thirdparty/Sophus/CMakeLists.txt" \
  "ORB_SLAM3-master/Thirdparty/Sophus/SophusConfig.cmake.in"

# 提取 Vocabulary
tar -xzf orbslam3.tar.gz --strip-components=2 "ORB_SLAM3-master/Vocabulary/ORBvoc.txt.tar.gz"
tar -xf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary/
```

**C++ 标准修改（关键！）**：Pangolin master 需要 C++14。编辑 `CMakeLists.txt`，把 C++11 检测改成 C++14：

```cmake
# 替换原有 # Check C++11 or C++0x support 部分
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX14)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
elseif(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()
```

**编译：**

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build . -j4
```

---

## 二、安装 ROS2 与 D456 RealSense 驱动

### 2.1 ROS2 环境

```bash
# 确认 ROS2 版本
ls /opt/ros/
# 本文基于 ROS2 Lyrical (Ubuntu 26.04)

# 添加到 bashrc
echo 'source /opt/ros/lyrical/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

### 2.2 安装 librealsense2 SDK

```bash
# ROS2 Lyrical 有预编译包
sudo apt-get install -y ros-lyrical-librealsense2

# 验证相机检测
rs-enumerate-devices
# 应能看到 D456 信息
```

### 2.3 编译 realsense-ros 驱动

```bash
# 创建工作区
mkdir -p ~/realsense_ws/src
cd ~/realsense_ws/src

# 克隆源码
git clone https://github.com/IntelRealSense/realsense-ros.git

# 安装编译工具和依赖
sudo apt-get install -y python3-colcon-common-extensions
sudo apt-get install -y ros-lyrical-diagnostic-updater

# 修复 lark 解析库版本（ROS2 Lyrical 兼容性问题）
pip3 install --break-system-packages 'lark<1.2.0'

# 修复 CMakeLists.txt 添加 Lyrical 支持
# 编辑 realsense-ros/realsense2_camera/CMakeLists.txt
# 在 kilted 分支后添加：
# elseif("$ENV{ROS_DISTRO}" STREQUAL "lyrical")
#   message(STATUS "Build for ROS2 Lyrical")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKILTED")
#   set(SOURCES "${SOURCES}" src/ros_param_backend.cpp)

# 编译（注意 Python 环境问题）
cd ~/realsense_ws
colcon build --packages-select realsense2_camera_msgs --executor sequential --cmake-args -DPython3_ROOT_DIR=/usr
source install/setup.bash
colcon build --packages-select realsense2_camera --executor sequential --cmake-args -DPython3_ROOT_DIR=/usr

# 添加到 bashrc
echo 'source ~/realsense_ws/install/setup.bash' >> ~/.bashrc
source ~/.bashrc

# 验证
ros2 pkg list | grep realsense
# 应显示: realsense2_camera  realsense2_camera_msgs
```

> ⚠️ **Python 环境坑**：如果系统有 Anaconda，`python3` 可能指向 Anaconda 的旧版本 Python。ORB-SLAM3 编译时需用 `/usr/bin/python3.14`，colcon 编译时需设 `-DPython3_ROOT_DIR=/usr`。

### 2.4 IMU 权限问题

D456 的 BMI085 IMU 通过 IIO sysfs 访问，普通用户权限不足。两种解法：

**解法 A（推荐）：用 root 运行**
```bash
su
source /opt/ros/lyrical/setup.bash && source ~/realsense_ws/install/setup.bash
ros2 launch realsense2_camera rs_launch.py ...
```

**解法 B：创建 udev 规则**
```bash
cat > /etc/udev/rules.d/99-realsense-hid.rules << 'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="8086", ATTR{idProduct}=="0b5c", MODE="0666"
ACTION=="add", KERNEL=="iio*", RUN+="/bin/chmod 0666 /sys/%p/scan_elements/in_*"
EOF
udevadm control --reload-rules
# 需要拔插 USB 生效
```

---

## 三、启动相机（SLAM 优化参数）

```bash
ros2 launch realsense2_camera rs_launch.py \
    enable_depth:=false \
    enable_infra1:=false \
    enable_infra2:=false \
    enable_color:=true \
    enable_gyro:=true \
    enable_accel:=true \
    unite_imu_method:=2 \
    enable_sync:=true \
    initial_reset:=true \
    rgb_camera.color_profile:=848x480x30
```

参数说明：
- `enable_depth:=false` — 禁用深度，节省 USB 带宽
- `unite_imu_method:=2` — 融合 accel+gyro 为 200Hz IMU 话题
- `enable_sync:=true` — 硬件同步
- `initial_reset:=true` — 启动时复位，减少掉线

相机启动后的话题：
```
/camera/camera/color/image_raw   # 彩色图 848×480 @30fps
/camera/camera/imu               # 融合 IMU @200Hz
/camera/camera/color/camera_info # 相机内参
```

---

## 四、录制数据集

`~/record_mono_imu.sh`：

```bash
#!/bin/bash
OUT_DIR="$HOME/rosbag2_$(date +%Y%m%d_%H%M%S)"
source /opt/ros/lyrical/setup.bash
source ~/realsense_ws/install/setup.bash

echo "录制单目+IMU: $OUT_DIR"
echo "按 Ctrl+C 停止"

ros2 bag record \
    --output "$OUT_DIR" \
    --topics \
    /camera/camera/color/image_raw \
    /camera/camera/imu \
    /camera/camera/color/camera_info
```

使用（分两个终端）：
```
# 终端1：启动相机
su -c 'source /opt/ros/lyrical/setup.bash && source ~/realsense_ws/install/setup.bash && ros2 launch realsense2_camera rs_launch.py enable_depth:=false enable_infra1:=false enable_infra2:=false enable_color:=true enable_gyro:=true enable_accel:=true unite_imu_method:=2 enable_sync:=true initial_reset:=true rgb_camera.color_profile:=848x480x30'

# 终端2：录制（也需 root）
su
source /opt/ros/lyrical/setup.bash && source ~/realsense_ws/install/setup.bash
bash ~/record_mono_imu.sh
```

录完按 `Ctrl+C`，数据保存在 `~/rosbag2_YYYYMMDD_HHMMSS/` 目录。

---

## 五、提取数据跑 ORB-SLAM3

### 5.1 提取脚本 `extract_bag.py`

提取 bag 中的图像和 IMU 到 EuRoC 格式。核心逻辑：

```python
# 1. 遍历 bag，读取图像和 IMU 话题
# 2. 将图像保存为 PNG（RGB→BGR 转换）
# 3. IMU 数据（accel+gyro 通过 unite_imu_method:=2 已融合在同一话题）
# 4. ⚠️ 关键：截掉 IMU 融合话题启动前的图像帧
#    （融合 IMU 比图像晚约 2.2 秒启动）
# 5. 输出: mav0/cam0/data/*.png, mav0/cam0/data.csv, mav0/imu0/data.csv, times.txt
```

### 5.2 配置文件 `config.yaml`

```yaml
Camera.type: "PinHole"
Camera1.fx: 426.43      # 数据集自带内参
Camera1.fy: 425.729
Camera1.cx: 435.525
Camera1.cy: 244.974
Camera1.k1: -0.056287
Camera1.k2: 0.065668
Camera1.p1: 0.000058
Camera1.p2: 0.000832
Camera.width: 848
Camera.height: 480
Camera.fps: 30
Camera.RGB: 1

# IMU-相机外参（使用 kalibr 联合标定结果）
IMU.T_b_c1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: f
   data: [0.99998408, -0.00343815, -0.00447398, -0.02734393,
          0.00342386, 0.99998902, -0.00319964, -0.00010614,
          0.00448493, 0.00318427, 0.99998487, -0.03650441,
          0.0, 0.0, 0.0, 1.0]

# IMU 噪声（从 kalibr 标定结果提取）
IMU.NoiseGyro: 0.0002
IMU.NoiseAcc: 0.01
IMU.GyroWalk: 0.000001
IMU.AccWalk: 0.0002
IMU.Frequency: 200.0
```

### 5.3 运行 ORB-SLAM3（离线模式）

```bash
./Examples/Monocular-Inertial/mono_inertial_euroc \
  Vocabulary/ORBvoc.txt \
  path/to/config.yaml \
  path/to/sequence_folder \
  path/to/times.txt
```

---

## 六、实时 ORB-SLAM3

基于 `Examples/Monocular-Inertial/mono_inertial_realsense_D435i.cc` 改造，适配 D456：

**修改点：**
1. 红外流→彩色流：`RS2_STREAM_INFRARED` → `RS2_STREAM_COLOR`，848×480 RGB8
2. 图像格式：`CV_8U` → `CV_8UC3` + RGB2BGR 转换
3. 传感器配置：简化，移除 D435i 特有的索引配置
4. 关键修复：启动后等待至少 1 秒 IMU 数据才创建 SLAM 系统

**编译（需要设置 realsense2_DIR）：**
```bash
cmake .. -Drealsense2_DIR=/opt/ros/lyrical/lib/x86_64-linux-gnu/cmake/realsense2
cmake --build . -j4 --target mono_inertial_realsense_D456
```

**运行：**
```bash
su -c 'source /opt/ros/lyrical/setup.bash && ./Examples/Monocular-Inertial/mono_inertial_realsense_D456 Vocabulary/ORBvoc.txt config.yaml'
```

> 注意：运行前先静置相机 1-2 秒，让 IMU 初始化积累足够的静止数据。

---

## 七、常见问题

### 7.1 IMU 启动延迟
融合 IMU 话题（`/camera/camera/imu`）比图像晚约 2 秒启动。
- 离线提取：脚本自动截掉无 IMU 的图像帧
- 实时模式：代码等待 IMU 数据就绪后再建图

### 7.2 IMU 权限问题
```
Failed to open scan_element ... Permission denied
Hid device is busy!
```
→ 用 root 运行，或配置 udev 规则

### 7.3 USB 掉线
```
xioctl(VIDIOC_S_FMT) failed, errno=16 Device or resource busy
The device has been disconnected!
```
→ 加 `initial_reset:=true`，换 USB 3.0 端口，不要用 Hub

### 7.4 Python 路径
Anaconda 会劫持 `python3`，CMake 构建时需指定 `-DPython3_ROOT_DIR=/usr`

### 7.5 CMake 版本兼容
Ubuntu 26.04 的 CMake 4.x 会警告旧版本策略，编译时加：
```
-DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

---

## 八、文件清单

```
~/ORBSLAM3-TrafficSign-Pose/
├── Vocabulary/ORBvoc.txt          # ORB 词库
├── build/                         # 编译目录
├── Examples/Monocular-Inertial/
│   ├── mono_inertial_euroc        # 离线模式（已编译）
│   └── mono_inertial_realsense_D456  # 实时模式（已编译）
├── landmarkslam/implement/data/
│   ├── extract_bag.py             # Bag→EuRoC 提取脚本
│   └── rosbag2_*/                 # 提取好的数据集
│       ├── config.yaml            # 相机+IMU 配置
│       ├── times.txt              # 图像时间戳
│       └── mav0/
│           ├── cam0/data/*.png    # 图像
│           └── imu0/data.csv      # IMU 数据

~/realsense_ws/
├── src/realsense-ros/             # realsense-ros 源码
├── install/                       # 编译产物
└── build/

~/record_mono_imu.sh               # 一键录制脚本
~/rosbag2_*/                       # 录制的 raw bag
```
