# GPS v1 分支 — 常用命令速查

---

## 一、轨迹可视化

```bash
~/ORBSLAM3-TrafficSign-Pose/gps_venv/bin/python3 \
  landmarkslam/implement/tools/visualize_trajectory.py \
  <轨迹文件.txt>
```

示例：

```bash
~/ORBSLAM3-TrafficSign-Pose/gps_venv/bin/python3 \
  landmarkslam/implement/tools/visualize_trajectory.py \
  traj_20260627_202350_crash_1782567496970465277.txt
```

> 弹出四合一窗口：3D 轨迹、XY 鸟瞰（带朝向箭头）、XZ 侧视图、各分量时间曲线。
> 控制台同时打印尺度报告（总距离、跨度、平均速度）。

---

## 二、可视化所有分割段（不同颜色叠加）

```bash
~/ORBSLAM3-TrafficSign-Pose/gps_venv/bin/python3 \
  landmarkslam/implement/tools/viz_split_segments.py \
  <split_output目录>
```

示例：

```bash
~/ORBSLAM3-TrafficSign-Pose/gps_venv/bin/python3 \
  landmarkslam/implement/tools/viz_split_segments.py \
  split_output
```

## 三、交互式手动剪裁（图片+轨迹联动）

```bash
~/ORBSLAM3-TrafficSign-Pose/gps_venv/bin/python3 \
  landmarkslam/implement/tools/interactive_split.py \
  <轨迹文件.txt> <图片目录> [输出目录]
```

示例：

```bash
~/ORBSLAM3-TrafficSign-Pose/gps_venv/bin/python3 \
  landmarkslam/implement/tools/interactive_split.py \
  traj_20260627_202350_crash_1782567496970465277.txt \
  test1 \
  manual_split
```

> 操作：←→ 移动帧 | **S** 标记起点 | **E** 标记终点 | **回车** 保存当前段并继续 | **ESC** 退出
> 左侧显示当前图片，右侧显示轨迹（当前帧青点、起点绿方块、终点红方块、选区橙色线）

## 四、四角点标注

```bash
LD_LIBRARY_PATH="" ~/ORBSLAM3-TrafficSign-Pose/gps_venv/bin/python3 \
  landmarkslam/implement/tools/annotate_corners.py \
  <图片目录/rgb> [标注文件路径]
```

示例：

```bash
LD_LIBRARY_PATH="" ~/ORBSLAM3-TrafficSign-Pose/gps_venv/bin/python3 \
  landmarkslam/implement/tools/annotate_corners.py \
  manual_split/seg_manual_0/rgb
```

> 鼠标依次点击路牌的 4 个角点（顺时针），满 4 个自动保存。
> 按键: A/D 切换图片 | 空格 开始标注 | Q 保存退出

## 五、按轨迹断点分割数据集

```bash
python3 landmarkslam/implement/tools/split_by_traj_jump.py \
  <轨迹文件> <输出目录> --image-dir <数据集目录>
```

示例：

```bash
python3 landmarkslam/implement/tools/split_by_traj_jump.py \
  traj_20260627_202350_crash_1782567496970465277.txt \
  split_output --image-dir test1
```
