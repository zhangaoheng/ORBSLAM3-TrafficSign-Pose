# ORB-SLAM3 Looming Metric Recovery

基于 ORB-SLAM3 的 Looming 深度估计与度量位姿恢复流水线。通过平面目标的 Looming 膨胀效应，从单目 SLAM 轨迹中恢复绝对深度和度量位姿。

---

## 1. 环境准备与代码拉取

```bash
# 克隆代码（包含子模块）
git clone --recursive https://github.com/your-repo/ORB_SLAM3.git
cd ORB_SLAM3

# 编译 ORB-SLAM3 本体
chmod +x build.sh
./build.sh

# Python 虚拟环境
python3 -m venv landmarkslam/yolo_venv
source landmarkslam/yolo_venv/bin/activate
pip install opencv-python numpy matplotlib pyyaml torch kornia scipy pyrealsense2
```

### Python 依赖

以下是上述命令安装的包：
opencv-python
numpy
matplotlib
pyyaml
torch
kornia
scipy
pyrealsense2       # Stage 1 bag 提取需要
```

---

## 2. 数据获取

原始数据（RealSense D456 `.bag` 录制文件，约 1.7 GB）不在 git 仓库中，需要从已有数据的设备传输。

### 最小方案：只传 .bag，其余由 pipeline 自动生成

```bash
# 在已有数据的设备上执行 rsync
rsync -avz --progress \
    user@source-machine:/path/to/ORB_SLAM3-master/landmarkslam/implement/data/deepseek/20260426_104450.bag \
    landmarkslam/implement/data/deepseek/

# 传输后运行流水线自动生成其余数据
cd landmarkslam/implement/pipeline
python run_pipeline.py
```

### 完整传输：同步整个数据目录

```bash
rsync -avz --progress \
    user@source-machine:/path/to/ORB_SLAM3-master/landmarkslam/implement/data/deepseek/ \
    landmarkslam/implement/data/deepseek/
```

### 验证数据状态

```bash
bash landmarkslam/implement/pipeline/setup_data.sh
```

该脚本会检测 `20260426_104450.bag`、`lines_all/`、`lines1/`、`lines2/` 是否齐全，并给出后续操作指引。

### 数据组成

| 组件 | 大小 | 来源 |
|------|------|------|
| `20260426_104450.bag` | 1.7 GB | 原始 RealSense D456 录制（RealSense Viewer 或 ROS bag） |
| `lines_all/rgb/` + `depth/` | ~672 MB | Stage 1 从 .bag 提取 |
| `lines_all/AllFrames_trajectory.txt` | ~300 KB | Stage 2 ORB-SLAM3 生成 |
| `lines1/` | ~190 MB | Stage 3 从 lines_all 分割 |
| `lines2/` | ~181 MB | Stage 3 从 lines_all 分割 |

---

## 3. 一键流水线

所有路径和参数集中管理在 `pipeline/pipeline_config.yaml` 中。流水线自动跳过已完成阶段。

```bash
cd landmarkslam/implement/pipeline

# 批量评估模式（默认）：遍历所有帧对，输出统计和图表
python run_pipeline.py

# 交互模式：GUI 选帧 → LoFTR → 3D 可视化
python run_pipeline.py --mode interactive

# 全模式：先 batch 后 interactive
python run_pipeline.py --mode full

# 跳过指定阶段
python run_pipeline.py --skip 1 2 3
```

### 5 个阶段

| 阶段 | 功能 | 输入 | 输出 |
|------|------|------|------|
| 1️⃣ Bag 提取 | 从 .bag 提取 RGB/Depth/IMU | `20260426_104450.bag` | `lines_all/`（rgb, depth, imu, 内参） |
| 2️⃣ ORB-SLAM3 建图 | 运行 `rgbd_tum` 生成真值轨迹 | `lines_all/` | `AllFrames_trajectory.txt` |
| 3️⃣ 序列分割 | 自动切割两个子序列 | `lines_all/` | `lines1/` + `lines2/`（含轨迹） |
| 4️⃣ ROI 标注 | 框选目标区域并缓存 | `lines1/rgb/` | `saved_rois.txt` |
| 5️⃣ 主流水线 | 批量验证或交互度量恢复 | 前述所有输出 | 图表、报告、JSON 结果 |

---

## 4. 项目结构

```
ORB_SLAM3-master/
├── Examples/RGB-D/
│   ├── rgbd_tum              # 预编译的 RGB-D 轨迹生成器
│   └── TUM1.yaml             # 相机参数模板（pipeline 动态覆盖内参）
│
├── Vocabulary/ORBvoc.txt     # ORB 词典（编译时下载）
│
└── landmarkslam/implement/
    ├── pipeline/                         # 🆕 一键流水线
    │   ├── pipeline_config.yaml          #   集中配置（所有路径 + 参数）
    │   ├── run_pipeline.py               #   5 阶段引擎
    │   └── setup_data.sh                 #   数据安装引导
    │
    ├── main/                             # 主实验脚本
    │   ├── test.py                       #   核心：batch 评估 + 交互度量恢复
    │   ├── config.yaml                   #   pipeline 自动同步的配置文件
    │   └── legacy/                       #   旧版独立脚本（备查）
    │
    ├── mid_FOE_Z_d/                      # Looming 深度模块
    │   ├── pure_looming_depth.py         #   Looming 公式、轨迹加载、FOE
    │   └── imgs_mid.py                   #   ROI 序列标注与缓存
    │
    ├── tools/                            # 工具库
    │   ├── mid.py                        #   几何工具（线段检测、矩形中心）
    │   ├── deepseek/                     #   数据提取工具
    │   │   ├── deepseek_bag.py           #     .bag → lines_all
    │   │   └── deepseek_cut_all.py       #     交互式序列分割
    │   └── ...
    │
    └── data/deepseek/                    # 数据目录（gitignored）
        ├── 20260426_104450.bag           #   原始录制文件（需 rsync）
        ├── lines_all/                    #   Stage 1 产物
        ├── lines1/                       #   Stage 3 产物
        └── lines2/                       #   Stage 3 产物
```

---

## 5. 流水线细节

### Stage 1 — Bag 提取

使用 RealSense SDK 从 `.bag` 文件中提取对齐的 RGB 和 Depth 图像，同时提取 IMU 数据。自动写入`associations.txt`（TUM 格式，供 ORB-SLAM3 使用）。

```bash
# 也可手动运行
python tools/deepseek/deepseek_bag.py
```

### Stage 2 — ORB-SLAM3 轨迹生成

运行 `Examples/RGB-D/rgbd_tum` 二进制文件，输入 `lines_all/` 目录和 `associations.txt`，输出 `AllFrames_trajectory.txt`（TUM 格式的完整轨迹）。

ORB-SLAM3 的相机 YAML 配置文件由 pipeline 根据 `pipeline_config.yaml` 的 `camera` 段动态生成。

### Stage 3 — 序列分割

从 `lines_all/` 按配置的帧索引范围切割出 `lines1/` 和 `lines2/`，自动复制 RGB/Depth 图像、生成 `associations.txt` 和 `trajectory.txt`。

在 `pipeline_config.yaml` 中配置：
```yaml
split:
  seq1_start: 0
  seq1_end: 243
  seq2_start: 260
  seq2_end: 496
```

### Stage 4 — ROI 标注

对 `lines1/rgb/` 逐帧运行视觉标注，框选目标平面区域并缓存为 `saved_rois.txt`。

```bash
# 也可手动运行
python -c "from mid_FOE_Z_d.imgs_mid import process_sequence_with_cached_rois; process_sequence_with_cached_rois('data/deepseek/lines1/rgb')"
```

### Stage 5 — 主流水线

**Batch 模式**：遍历 `lines1` 所有连续帧对（间隔 `frame_step`），计算 Looming Z 并与深度真值对比，输出：

```
batch_plots/
├── 01_depth_comparison.png      # Z_gt vs Z_looming 散点 + Bland-Altman
├── 02_error_histogram.png       # 绝对误差分布
├── 03_error_timeseries.png      # 逐帧误差趋势
├── 04_error_vs_depth.png        # 相对误差 vs 真实深度
├── 05_d_comparison.png          # 平面距离 d 对比
├── 06_dr_vs_error.png           # 膨胀量-误差关联
└── batch_report.md              # Markdown 报告

batch_results.json               # 详细统计结果
experiment_results.txt           # 文本日志
```

**Interactive 模式**：GUI 选帧 → Looming 测距 → LoFTR 匹配 → 单应性分解 → 3D 正交验证 → 绝对平移恢复 → 精度评估。

---

## 6. 关键算法参数

在 `pipeline_config.yaml` 中调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `camera.fx/fy/cx/cy` | 426.37 / 425.67 / 435.53 / 244.97 | RealSense D456 内参 |
| `algorithm.frame_step` | 15 | Looming 帧间步长 |
| `algorithm.dr_threshold` | 5.0 | 最小膨胀量（像素） |
| `algorithm.max_z_threshold` | 30.0 | 最大合理深度（米） |
| `algorithm.min_delta_d` | 0.02 | 最小帧间运动（米） |
| `algorithm.loftr_model` | outdoor | LoFTR 预训练模型 |

---

## 7. 在新设备上的完整部署流程

```bash
# ==== 1. 拉取代码 ====
git clone <your-repo-url>
cd ORB_SLAM3-master
git checkout master

# ==== 2. 初始化环境 ====
python3 -m venv landmarkslam/yolo_venv
source landmarkslam/yolo_venv/bin/activate
pip install torch kornia opencv-python numpy matplotlib pyyaml scipy pyrealsense2

# ==== 3. 获取数据 ====
mkdir -p landmarkslam/implement/data/deepseek
rsync -avz --progress user@source-machine:.../data/deepseek/20260426_104450.bag \
    landmarkslam/implement/data/deepseek/

# ==== 4. 运行流水线 ====
cd landmarkslam/implement/pipeline
python run_pipeline.py
```

流水线会自动检测已有数据，跳过已完成阶段，从 .bag 开始逐步生成所有文件。
