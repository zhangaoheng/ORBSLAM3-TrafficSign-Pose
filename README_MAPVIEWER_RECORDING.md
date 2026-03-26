# MapViewer 窗口录制功能

## 功能说明

程序已自动集成 MapViewer 窗口录制功能，运行时会：

1. **启动 SLAM 系统**后等待 3 秒，让 MapViewer 窗口完全显示
2. **自动启动 ffmpeg** 录制整个屏幕（捕获 MapViewer 窗口）
3. **处理完成后自动停止录制**并将视频保存到输出目录

## 输出内容

每次运行会创建带时间戳的输出目录 `./outputs/run_YYYYMMDD_HHMMSS/`，包含：

- ✅ `pointcloud.csv` - 点云数据（X,Y,Z,R,G,B）
- ✅ `camera_trajectory.csv` - 相机轨迹（timestamp,tx,ty,tz,qx,qy,qz,qw）
- ✅ `runway_cut.mp4` - 原始输入视频的副本
- ✅ `mapviewer_recording.mp4` - **MapViewer 窗口录制的视频**

## 前置要求

需要安装 ffmpeg：
```bash
sudo apt-get install ffmpeg
```

## 录制参数

- 分辨率：1241x829（默认 MapViewer 窗口大小）
- 帧率：30 FPS
- 编码器：H.264 (libx264)
- 质量：CRF 23（高质量）
- 预设：ultrafast（实时录制优化）

##注意事项

1. ffmpeg 使用 X11grab 录制整个屏幕的 `:0.0` 显示器
2. 如果 MapViewer 窗口大小或位置变化，可能需要调整 ffmpeg 参数
3. 录制会在后台进行，不影响 SLAM 运行性能
4. 程序结束时会自动停止 ffmpeg 进程并等待视频写入完成

## 自定义录制区域（可选）

如需指定录制区域，修改 `test/test.cc` 中的 ffmpeg_cmd：

```cpp
string ffmpeg_cmd = "ffmpeg -y -f x11grab -video_size WIDTHxHEIGHT -i :0.0+X_OFFSET,Y_OFFSET "
                   "-r 30 -vcodec libx264 -preset ultrafast -crf 23 " + mapviewerVideoFile + " > /dev/null 2>&1 &";
```

例如录制从坐标 (100, 50) 开始的 1024x768 区域：
```cpp
-video_size 1024x768 -i :0.0+100,50
```
