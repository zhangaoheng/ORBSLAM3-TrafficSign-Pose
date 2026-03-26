# ORB-SLAM3 ROI 掩膜功能说明

## 功能简介

新版 `test.cc` 添加了从 NPY 文件读取四个角点坐标的功能，实现在窗口化界面中只显示四边形区域内的特征点和特征点云。

## 工作原理

1. **NPY 文件读取**: 程序会自动读取对应帧的 NPY 文件，其中包含四个角点的像素坐标
   - NPY 文件格式: `{frame_idx, raw_kpts, refined_kpts, conf, has_detection}`
   - 使用 `refined_kpts` 作为四个角点坐标（如果不存在则使用 `raw_kpts`）

2. **四边形掩膜创建**: 根据四个角点创建一个四边形掩膜
   - 使用 `cv::fillPoly` 在掩膜上填充四边形区域
   - 掩膜外的像素被设置为 0（黑色）

3. **特征点过滤**: 在进行 SLAM 跟踪前，将帧的 ROI 外部像素设为 0
   - 这确保特征提取器只在 ROI 区域内找到特征点
   - 地图点的显示也会受到影响，因为它们基于这些特征点

4. **可视化**: 在处理第一帧时会显示一个 ROI 区域窗口
   - 绿线显示四边形边界
   - 红点显示四个角点

## 文件结构

```
/home/zah/ORB_SLAM3-master/
├── test/
│   ├── test.cc                    # 主程序（包含新的ROI功能）
│   ├── test.yaml                  # 相机标定参数
│   ├── build.sh                   # 编译脚本
│   ├── run.sh                     # 运行脚本
│   └── videos/
│       ├── Dataset_Final_165123.mp4  # 视频文件
│       └── 0123232/
│           └── Dataset_Final_165123_npy/  # NPY文件目录
│               ├── frame_000001.npy
│               ├── frame_000002.npy
│               └── ...
```

## 使用方法

### 方式 1: 使用编译脚本
```bash
cd /home/zah/ORB_SLAM3-master/test
./build.sh    # 编译
```

### 方式 2: 使用运行脚本（自动编译+运行）
```bash
cd /home/zah/ORB_SLAM3-master/test
./run.sh
```

### 方式 3: 手动编译和运行
```bash
cd /home/zah/ORB_SLAM3-master/build
make -j4

cd /home/zah/ORB_SLAM3-master/test
../build/test_slam
```

## 关键函数说明

### 1. `loadCornerPointsFromNpyPython()`
- 使用 Python 脚本读取 NPY 文件
- 返回 4 个 `cv::Point2f` 的向量
- 这是首选方法，因为可以正确处理 pickle 对象

### 2. `createQuadrilateralMask()`
- 创建与帧同尺寸的掩膜矩阵
- 四边形内部为 255（白色），外部为 0（黑色）
- 用于后续的像素过滤

### 3. `pointInQuadrilateral()`
- 判断单个点是否在四边形内部
- 使用叉积方法检测点与四边形的位置关系
- 当前未直接使用，但可用于高级功能扩展

## 输出示例

```
Starting ORB-SLAM3 test with ROI masking...
Vocabulary file: ../Vocabulary/ORBvoc.txt
Parameter file: ./test.yaml
Video file: /home/zah/ORB_SLAM3-master/test/videos/Dataset_Final_165123.mp4
NPY directory: /home/zah/ORB_SLAM3-master/test/videos/0123232/Dataset_Final_165123_npy
SLAM system initialized
Video opened successfully
Frame width: 1920
Frame height: 1440
FPS: 30
Total frames: 732
Processing frame 1
Successfully loaded corner points from npy file
Corner points: (1696.714355,461.414948) (1731.744141,461.115600) (1751.884033,692.520203) (1675.114502,690.415955)
Processing frame 30
...
```

## 注意事项

1. **NPY 文件必须存在**: 确保 NPY 文件路径正确且包含四个角点坐标
2. **Python 环境**: 需要安装 numpy
   ```bash
   pip3 install numpy
   ```
3. **坐标范围**: 确保 NPY 文件中的坐标不超过视频帧的尺寸
4. **性能影响**: 应用掩膜可能会略微降低处理速度

## 问题排查

### NPY 文件读取失败
- 检查文件路径是否正确
- 确保 Python3 和 numpy 已安装
- 查看错误日志信息

### 特征点太少
- 检查 ROI 区域是否过小
- 增加 ORB 提取器的特征点阈值
- 验证视频质量

### 可视化窗口没有显示
- 确保系统支持图形输出
- 检查 OpenCV 的 GUI 支持

## 扩展功能建议

1. **动态ROI**: 从所有帧的 NPY 文件中读取四个角点，允许随时间变化
2. **多ROI**: 支持多个不同的四边形区域
3. **绘图优化**: 在 FrameDrawer 中只绘制 ROI 内的特征点
4. **配置文件**: 使用配置文件指定 NPY 目录路径
5. **自适应掩膜**: 基于帧内容动态调整 ROI

