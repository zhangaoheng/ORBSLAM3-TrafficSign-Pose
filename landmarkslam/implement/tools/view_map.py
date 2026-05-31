#!/usr/bin/env python3
"""
点云 + 轨迹联合可视化工具

用法:
  # 可视化轨迹文件
  python view_map.py <trajectory.txt>
  
  # 可视化轨迹 + 点云
  python view_map.py <trajectory.txt> --cloud <pointcloud.ply>

  # 查看 extract_data 结果
  python view_map.py ../data/extracted_data/runs/2026-05-30_19-43-36_mono/trajectory.txt

支持格式:
  - 轨迹: TUM 格式 (timestamp tx ty tz qx qy qz qw), EuRoC 格式
  - 点云: PLY 格式 (ascii)
"""

import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.widgets import Button


# ──────────────────────────────────────────────
# 1. 加载 PLY 点云
# ──────────────────────────────────────────────
def load_ply(path):
    """加载 ASCII PLY 点云文件。支持 vertex 含 x y z 和可选 r g b。"""
    if not os.path.exists(path):
        print(f"  ⚠️  点云文件不存在: {path}")
        return None, None

    with open(path, "r") as f:
        lines = f.readlines()

    # 解析 header
    header_end = 0
    vertex_count = 0
    has_color = False
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        if "property uchar red" in line or "property uchar blue" in line:
            has_color = True
        if line.strip() == "end_header":
            header_end = i + 1
            break

    if vertex_count == 0:
        return None, None

    # 读取点数据
    points = []
    colors = []
    for line in lines[header_end:header_end + vertex_count]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        points.append([x, y, z])
        if has_color and len(parts) >= 6:
            r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
            colors.append([r / 255.0, g / 255.0, b / 255.0])

    pts = np.array(points)
    cols = np.array(colors) if colors else None
    print(f"  ✅ 点云: {len(pts)} 个点, 颜色={'有' if cols is not None else '无'}")
    return pts, cols


# ──────────────────────────────────────────────
# 2. 加载 TUM / EuRoC 轨迹
# ──────────────────────────────────────────────
def load_trajectory(path):
    times, poses, quats = [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            t = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            times.append(t)
            poses.append([tx, ty, tz])
            quats.append([qx, qy, qz, qw])

    poses = np.array(poses)
    quats = np.array(quats)
    times = np.array(times)

    print(f"  ✅ 轨迹: {len(poses)} 帧")
    if len(poses) > 0:
        xr = poses[:, 0].max() - poses[:, 0].min()
        yr = poses[:, 1].max() - poses[:, 1].min()
        zr = poses[:, 2].max() - poses[:, 2].min()
        dur = times[-1] - times[0]
        print(f"     范围: X={xr:.2f}m, Y={yr:.2f}m, Z={zr:.2f}m, 时长={dur:.1f}s")

    return times, poses, quats


# ──────────────────────────────────────────────
# 3. 3D 可视化
# ──────────────────────────────────────────────
class MapViewer:
    def __init__(self, traj_path, cloud_path=None):
        self.traj_path = traj_path
        self.cloud_path = cloud_path
        self.times = None
        self.poses = None
        self.quats = None
        self.cloud_pts = None
        self.cloud_cols = None

    def load(self):
        print("📂 加载数据...")
        self.times, self.poses, self.quats = load_trajectory(self.traj_path)

        if self.cloud_path:
            self.cloud_pts, self.cloud_cols = load_ply(self.cloud_path)

        if self.poses is None or len(self.poses) == 0:
            print("❌ 轨迹加载失败")
            sys.exit(1)

    def show(self):
        self.load()

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"3D 点云与轨迹可视化\n{os.path.basename(self.traj_path)}",
                     fontsize=14, fontweight="bold")

        ax = fig.add_subplot(111, projection="3d")

        # ── 1. 画点云 ──
        if self.cloud_pts is not None and len(self.cloud_pts) > 0:
            if self.cloud_cols is not None:
                ax.scatter(self.cloud_pts[:, 0], self.cloud_pts[:, 1], self.cloud_pts[:, 2],
                           c=self.cloud_cols, s=1.0, alpha=0.8, rasterized=True)
            else:
                ax.scatter(self.cloud_pts[:, 0], self.cloud_pts[:, 1], self.cloud_pts[:, 2],
                           c="gray", s=1.0, alpha=0.6, rasterized=True)
            print(f"  🎨 点云已绘制 ({len(self.cloud_pts)} 点)")

        # ── 2. 画轨迹 ──
        if self.poses is not None and len(self.poses) > 0:
            t_norm = (self.times - self.times[0]) / (self.times[-1] - self.times[0] + 1e-9)
            ax.scatter(self.poses[:, 0], self.poses[:, 1], self.poses[:, 2],
                       c=t_norm, cmap="plasma", s=4, alpha=0.9, label="Trajectory")
            ax.plot(self.poses[:, 0], self.poses[:, 1], self.poses[:, 2],
                    color="gray", alpha=0.3, linewidth=0.8)

            # 起点（绿）和终点（红）
            ax.scatter(self.poses[0, 0], self.poses[0, 1], self.poses[0, 2],
                       c="green", s=80, marker="o", label="Start", edgecolors="white", linewidth=1)
            ax.scatter(self.poses[-1, 0], self.poses[-1, 1], self.poses[-1, 2],
                       c="red", s=80, marker="x", label="End", linewidth=2)

        # ── 3. 坐标轴 ──
        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Y (m)", fontsize=11)
        ax.set_zlabel("Z (m)", fontsize=11)
        ax.legend(fontsize=10, loc="upper left")

        # 自动设置视角
        if self.poses is not None and len(self.poses) > 0:
            mid = self.poses.mean(axis=0)
            ax.view_init(elev=25, azim=-60)

        plt.tight_layout()
        print("\n🎨 窗口已打开（可拖拽旋转/缩放，关闭窗口退出）")
        print("   🖱️  左键拖拽旋转 | 滚轮缩放 | 右键平移")
        plt.show()


# ──────────────────────────────────────────────
# 4. 命令行入口
# ──────────────────────────────────────────────
def find_data_dirs(base):
    """在 trajectory.txt 所在目录找可能的点云文件"""
    candidates = []
    parent = os.path.dirname(os.path.abspath(base))
    # 同一目录下的 .ply
    for f in os.listdir(parent):
        if f.endswith(".ply"):
            candidates.append(os.path.join(parent, f))
    # 如果 runs/ 目录下有 analysis_results/
    analysis = os.path.join(os.path.dirname(os.path.dirname(parent)), "analysis_results")
    if os.path.isdir(analysis):
        for f in os.listdir(analysis):
            if f.endswith(".ply"):
                candidates.append(os.path.join(analysis, f))
    # 项目根下的 analysis_results/
    root_analysis = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(parent))), "analysis_results")
    if os.path.isdir(root_analysis):
        for f in os.listdir(root_analysis):
            if f.endswith(".ply"):
                candidates.append(os.path.join(root_analysis, f))
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="ORB-SLAM3 点云与轨迹3D可视化工具"
    )
    parser.add_argument("trajectory", nargs="?", default=None,
                        help="轨迹文件路径 (TUM/EuRoC格式)")
    parser.add_argument("--cloud", "-c", default=None,
                        help="PLY 点云文件路径")
    parser.add_argument("--auto", "-a", action="store_true",
                        help="自动搜索轨迹目录附近的点云文件")
    args = parser.parse_args()

    # 默认路径：最新的 extracted_data 运行结果
    if args.trajectory is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default = os.path.join(script_dir, "..", "data", "extracted_data",
                               "runs", "2026-05-30_19-43-36_mono", "trajectory.txt")
        if os.path.exists(default):
            args.trajectory = default
        else:
            print("❌ 请指定轨迹文件路径")
            print("   用法: python view_map.py <trajectory.txt> [--cloud <cloud.ply>]")
            sys.exit(1)

    if not os.path.exists(args.trajectory):
        print(f"❌ 找不到轨迹文件: {args.trajectory}")
        sys.exit(1)

    # 自动搜索点云
    cloud_path = args.cloud
    if cloud_path is None and args.auto:
        candidates = find_data_dirs(args.trajectory)
        if candidates:
            cloud_path = candidates[0]
            if len(candidates) > 1:
                print(f"📂 找到多个点云文件，使用第一个: {os.path.basename(cloud_path)}")
        else:
            print("  (未找到 PLY 点云文件，仅显示轨迹)")

    if cloud_path and not os.path.exists(cloud_path):
        print(f"  ⚠️  点云文件不存在: {cloud_path}")
        cloud_path = None

    viewer = MapViewer(args.trajectory, cloud_path)
    viewer.show()


if __name__ == "__main__":
    main()
