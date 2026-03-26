# MapViewer 窗口录制 - 完整指南

## 📹 自动录制功能（已集成）

程序运行时会自动启动 ffmpeg 录制 MapViewer 窗口。

### 运行流程

```bash
cd /home/zah/ORB_SLAM3-master/test
./test_slam
```

运行时会看到：
```
等待 MapViewer 窗口启动...
✓ 启动 MapViewer 窗口录制
  命令: ffmpeg -f x11grab -video_size 1920x1080 ...
  输出: /tmp/mapviewer_recording.mp4
  日志: /tmp/ffmpeg_mapviewer.log
```

### 检查录制状态

如果提示"未找到MapViewer 视频"，检查日志：

```bash
cat /tmp/ffmpeg_mapviewer.log
```

常见问题：
- **权限问题**: 确保 X11 允许录制
- **显示器未找到**: WSL 需要配置 X Server
- **分辨率不匹配**: 调整 test.cc 中的 `-video_size`

### 验证 ffmpeg 工作正常

测试 ffmpeg 是否能录制：

```bash
# 测试录制 5 秒
ffmpeg -f x11grab -video_size 1920x1080 -framerate 30 -i $DISPLAY \
       -t 5 -vcodec libx264 -preset ultrafast /tmp/test_record.mp4

# 检查文件
ls -lh /tmp/test_record.mp4
```

## 🎬 手动录制方案（备选）

如果自动录制失败，使用提供的脚本：

### 方法 1: 使用录制脚本

```bash
# 在一个终端启动录制
./record_mapviewer.sh /tmp/my_recording.mp4

# 在另一个终端运行 SLAM
cd test
./test_slam

# SLAM 完成后，按 Ctrl+C 停止录制
```

### 方法 2: 使用 OBS Studio

1. 安装 OBS Studio
2. 设置捕获窗口为 "ORB-SLAM3: Map Viewer"
3. 手动开始/停止录制

### 方法 3: 使用 SimpleScreenRecorder

```bash
sudo apt-get install simplescreenrecorder
simplescreenrecorder
```

选择录制区域为 MapViewer 窗口。

## 🔧 自定义录制参数

修改 [test/test.cc](test/test.cc#L208) 中的参数：

```cpp
string ffmpeg_cmd = "ffmpeg -f x11grab "
                   "-video_size 1920x1080 "    // 分辨率
                   "-framerate 30 "             // 帧率
                   "-i $DISPLAY "               // 显示器
                   "-vcodec libx264 "           // 编码器
                   "-preset ultrafast "         // 速度预设
                   "-crf 23 "                   // 质量 (18-28)
                   "-y " + mapviewerVideoFile + " > " + log_file + " 2>&1 &";
```

### 录制特定窗口（而非全屏）

使用 `xwininfo` 获取窗口ID：

```bash
xwininfo -name "Map Viewer"
```

然后修改命令：

```cpp
"-i $DISPLAY+[WINDOW_ID]"  // 替换 [WINDOW_ID]
```

## 📊 输出结果

成功运行后，输出目录包含：

```
./outputs/run_20260123_HHMMSS/
├── pointcloud.csv            ← 点云数据
├── camera_trajectory.csv     ← 相机轨迹
├── runway_cut.mp4           ← 原始视频
└── mapviewer_recording.mp4  ← MapViewer 录制视频 ✨
```

## 🐛 故障排除

### 问题 1: "未找到MapViewer 视频"

**原因**: ffmpeg 没有生成视频文件

**解决步骤**:

1. 检查 ffmpeg 日志:
   ```bash
   cat /tmp/ffmpeg_mapviewer.log
   ```

2. 验证 ffmpeg 已安装:
   ```bash
   ffmpeg -version
   ```

3. 测试 X11grab:
   ```bash
   ffmpeg -f x11grab -video_size 800x600 -i $DISPLAY -t 2 /tmp/test.mp4
   ```

4. 检查进程是否运行:
   ```bash
   ps aux | grep ffmpeg
   ```

### 问题 2: WSL 环境下 X11 不工作

**解决方案**: 使用 VcXsrv 或 X410

1. 在 Windows 安装 VcXsrv
2. 启动 XLaunch (禁用访问控制)
3. 在 WSL 中设置:
   ```bash
   export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
   export LIBGL_ALWAYS_INDIRECT=1
   ```

### 问题 3: 录制的视频是黑屏

**原因**: OpenGL 窗口可能无法被 X11grab 捕获

**解决**: 
- 方法 1: 使用 OBS Studio (支持 OpenGL 窗口捕获)
- 方法 2: 修改 Pangolin 使用软件渲染 (较慢)

### 问题 4: 录制影响性能

**调整录制参数**:

```cpp
"-preset superfast"     // 改为 superfast
"-crf 28"              // 降低质量 (增大数值)
"-framerate 15"        // 降低帧率
```

## 💡 最佳实践

1. **首次运行**: 先测试 5 秒短视频确认 ffmpeg 工作
2. **长时间运行**: 使用 `-crf 25` 平衡质量和文件大小
3. **高质量录制**: 使用 `-crf 18 -preset slow`（完成后转码）
4. **调试**: 将 ffmpeg 输出重定向到屏幕：
   ```cpp
   // 临时去掉 "> log 2>&1 &"，直接运行看输出
   system("ffmpeg -f x11grab ... -y output.mp4");
   ```

## 📝 技术说明

- **X11grab**: 直接从 X11 显示服务器捕获像素
- **后台运行**: `&` 让 ffmpeg 在后台录制，不阻塞 SLAM
- **停止录制**: `pkill -INT ffmpeg` 发送中断信号优雅停止
- **等待时间**: 2 秒确保视频文件完整写入

