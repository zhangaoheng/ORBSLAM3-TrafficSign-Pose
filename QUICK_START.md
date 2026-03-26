# 🚀 快速开始 - MapViewer 录制

## ✅ 现在可以使用了！

ffmpeg 已安装并测试通过。现在运行程序会自动录制 MapViewer 窗口。

## 立即运行

```bash
cd /home/zah/ORB_SLAM3-master/test
./test_slam
```

## 查看结果

程序运行完成后：

```bash
# 查看输出目录
ls -lh ./outputs/run_*/

# 应该看到 4 个文件：
# ✅ pointcloud.csv
# ✅ camera_trajectory.csv  
# ✅ runway_cut.mp4
# ✅ mapviewer_recording.mp4  ⭐ 新增！
```

## 如果仍然没有 mapviewer_recording.mp4

检查日志找出原因：

```bash
cat /tmp/ffmpeg_mapviewer.log
```

常见原因：
1. **X11 显示问题** - 在 WSL 需要 X Server (VcXsrv/X410)
2. **录制区域太大** - 修改为更小的分辨率
3. **权限问题** - 确保可以访问 /tmp 目录

## 手动录制（备选方案）

如果自动录制不工作，手动启动：

```bash
# 终端 1: 启动录制
./record_mapviewer.sh

# 终端 2: 运行 SLAM
cd test && ./test_slam
```

## 详细文档

- 📖 [完整录制指南](MAPVIEWER_RECORDING_GUIDE.md)
- 🔧 [实现总结](README_MAPVIEWER_RECORDING.md)

---

**提示**: 第一次运行建议先运行几秒钟测试，确认录制正常工作后再完整运行。
