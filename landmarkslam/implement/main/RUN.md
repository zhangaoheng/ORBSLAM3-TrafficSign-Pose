# 运行指南

## 环境准备

```bash
cd /home/zah/ORB_SLAM3-master/landmarkslam/implement/main
```

## 一键运行脚本 `run_all.sh`

### 运行所有模式（默认）

先执行 batch 安静批量评估，完成后自动启动交互式度量恢复：

```bash
./run_all.sh
```

等价于：

```bash
./run_all.sh all
```

### 仅批量评估

安静模式，遍历所有帧对计算 Looming 深度，输出统计报告和图表，不弹出 GUI：

```bash
./run_all.sh batch
```

后台运行（不占用终端）：

```bash
nohup ./run_all.sh batch > batch_output.log 2>&1 &
```

### 仅交互式度量恢复

启动 GUI 选帧、LoFTR 匹配、3D 可视化窗口：

```bash
./run_all.sh interactive
```

---

## 手动运行（不通过脚本）

```bash
# Python 虚拟环境
source /home/zah/ORB_SLAM3-master/landmarkslam/yolo_venv/bin/activate

# 批量评估
python test.py -q

# 交互式度量恢复
python test.py
```
