# 🚀 快速开始指南

## 📦 已准备好的文件

✅ `test.cc` - 包含 ROI 四边形掩膜功能的主程序
✅ `build.sh` - 自动编译脚本
✅ `run.sh` - 自动运行脚本
✅ `test_slam` - 编译后的可执行文件
✅ NPY 文件集合 - 包含 732 帧的角点坐标

## ⚡ 最快开始方式

```bash
cd /home/zah/ORB_SLAM3-master/test
./run.sh
```

这将自动执行：
1. ✓ 检查编译状态
2. ✓ 如果需要重新编译会自动编译
3. ✓ 运行程序处理视频

## 🔧 分步骤执行

### 步骤 1: 编译
```bash
cd /home/zah/ORB_SLAM3-master/test
./build.sh
```

**预期输出**:
```
=========================================
开始编译 ORB-SLAM3 测试程序
=========================================
项目根目录: /home/zah/ORB_SLAM3-master
执行 CMake 配置...
编译项目...
...
编译成功!
测试可执行文件已生成: ./test_slam
```

### 步骤 2: 运行
```bash
cd /home/zah/ORB_SLAM3-master/test
../build/test_slam
```

**预期输出**:
```
Starting ORB-SLAM3 test with ROI masking...
...
Successfully loaded corner points from npy file
Corner points: (1696.71,461.41) (1731.74,461.12) (1751.88,692.52) (1675.11,690.42)
```

## 🎮 运行时操作

- **'q' 或 'ESC'** - 停止程序
- **任何其他键** - 继续处理

## 📊 ROI 可视化

程序运行时会显示一个 "ROI Region" 窗口：
- 🟢 **绿线** - 四边形边界
- 🔴 **红点** - 四个角点

## ✅ 验证功能

### 测试 NPY 读取
```bash
cd /home/zah/ORB_SLAM3-master/test
python3 test_npy_read.py
```

**预期输出**:
```
Frame 1:
  Corner Points (refined_kpts):
    Corner 0: (1696.71, 461.41)
    Corner 1: (1731.74, 461.12)
    Corner 2: (1751.88, 692.52)
    Corner 3: (1675.11, 690.42)
  Confidence: 0.498212069272995
  Has Detection: True
...
```

## 📝 相关文档

- **README_ROI.md** - 功能详细说明
- **IMPLEMENTATION_SUMMARY.md** - 完整技术文档
- **test.cc** - 源代码（包含详细注释）

## 🐛 问题排查

### 问题: NPY 文件读取失败
**解决**: 确保 numpy 已安装
```bash
pip3 install numpy
```

### 问题: 编译出错
**解决**: 清除旧的编译文件
```bash
cd /home/zah/ORB_SLAM3-master
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### 问题: 没有图形窗口显示
**解决**: 确保系统支持 X11 或远程显示
```bash
export DISPLAY=:0  # 如果需要的话
```

## 📈 下一步

1. 查看 `README_ROI.md` 了解完整功能
2. 检查 `test.cc` 中的代码了解实现细节
3. 修改 NPY 目录路径以处理其他数据集
4. 优化 ORB 特征提取器参数以改进特征点检测

## 💡 快速参考

| 命令 | 说明 |
|------|------|
| `./build.sh` | 编译程序 |
| `./run.sh` | 编译并运行 |
| `../build/test_slam` | 直接运行 |
| `python3 test_npy_read.py` | 测试 NPY 读取 |
| `cat README_ROI.md` | 查看详细文档 |

## 🎯 成功标志

✅ 编译完成，无错误
✅ 程序启动，输出日志
✅ 成功读取 NPY 文件，显示 4 个角点
✅ ROI 窗口显示四边形区域
✅ SLAM 开始处理视频帧

---

**需要帮助?** 查看 `IMPLEMENTATION_SUMMARY.md` 获取完整技术文档！

