#!/usr/bin/env python3
"""
AllFrames_trajectory.txt 可视化工具
读取 TUM 格式的轨迹文件，用 Matplotlib 绘制：
  - 3D 轨迹图（物理尺度，单位：米）
  - XY 鸟瞰图（俯视）
  - XZ 侧视图（前进方向）
  - 各分量随时间变化图
  - 尺度验证摘要
"""

import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_tum_trajectory(path):
    """
    读取 TUM 格式轨迹文件。
    每行: timestamp tx ty tz qx qy qz qw
    返回:
        times:  (N,) 秒
        poses:  (N, 3) 位置 xyz，单位米
        quats:  (N, 4) 四元数 xyzw
    """
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
    return np.array(times), np.array(poses), np.array(quats)


def compute_stats(poses, times):
    """返回轨迹统计信息。"""
    # 总位移（起点到终点直线距离）
    total_displacement = np.linalg.norm(poses[-1] - poses[0])

    # 累计路径长度
    diffs = np.diff(poses, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_distance = np.sum(segment_lengths)

    # 范围
    x_range = poses[:, 0].max() - poses[:, 0].min()
    y_range = poses[:, 1].max() - poses[:, 1].min()
    z_range = poses[:, 2].max() - poses[:, 2].min()

    # 时间跨度
    duration = times[-1] - times[0]

    # 平均速度
    avg_speed = total_distance / duration if duration > 0 else 0

    return {
        "total_displacement": total_displacement,
        "total_distance": total_distance,
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
        "duration": duration,
        "avg_speed": avg_speed,
        "num_frames": len(poses),
    }


def print_scale_report(stats, path):
    """打印尺度报告。"""
    print("=" * 70)
    print(" 📊 AllFrames_trajectory 轨迹尺度报告")
    print("=" * 70)
    print(f" 文件: {path}")
    print(f" 帧数: {stats['num_frames']}")
    print(f" 时间跨度: {stats['duration']:.2f} s")
    print("-" * 70)
    print(f" 🌍 空间范围（物理单位：米）")
    print(f"   X 方向跨度: {stats['x_range']:.3f} m")
    print(f"   Y 方向跨度: {stats['y_range']:.3f} m")
    print(f"   Z 方向跨度: {stats['z_range']:.3f} m")
    print(f"   起点 → 终点直线距离: {stats['total_displacement']:.3f} m")
    print(f"   总路径长度: {stats['total_distance']:.3f} m")
    print(f"   平均速度: {stats['avg_speed']:.3f} m/s")
    print("-" * 70)

    # 物理尺度判断
    max_range = max(stats["x_range"], stats["y_range"], stats["z_range"])
    if max_range > 0.1:
        print(" ✅ 结论: 轨迹具有真实物理尺度（RGBD SLAM，单位为米）")
        print(f"    场景最大跨度 ≈ {max_range:.2f} m，符合真实世界尺度。")
    elif max_range > 0.001:
        print(" ⚠️  结论: 轨迹尺度偏小（可能是无尺度的单目 SLAM 或缩放后数据）。")
    else:
        print(" ❌ 结论: 轨迹尺度极小，基本无物理尺度信息。")
    print("=" * 70)


def make_heading_arrows(poses, quats, step=20, size=0.03):
    """
    从四元数中提取朝向（绕 Y 轴的 yaw），生成方向箭头。
    返回 (px, py, pz, dx, dz) 用于 XY 平面上的箭头。
    """
    px, py, pz = [], [], []
    dx, dz = [], []

    for i in range(0, len(poses), step):
        px.append(poses[i, 0])
        py.append(poses[i, 1])
        pz.append(poses[i, 2])

        R = R_scipy.from_quat(quats[i]).as_matrix()
        forward = R @ np.array([0, 0, 1])  # 相机坐标系 Z 轴朝前
        dx.append(forward[0] * size)
        dz.append(forward[2] * size)

    return np.array(px), np.array(py), np.array(pz), np.array(dx), np.array(dz)


def visualize_trajectory(times, poses, quats, stats, save_to=None):
    """绘制四合一可视化窗口。"""
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("AllFrames Trajectory — RGBD SLAM (real metric scale, meters)", fontsize=14, fontweight="bold")

    # 颜色按时间映射
    t_norm = (times - times[0]) / (times[-1] - times[0] + 1e-9)

    # --- 子图 1: 3D 轨迹 ---
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.set_title("3D Trajectory (m)")
    sc1 = ax1.scatter(poses[:, 0], poses[:, 1], poses[:, 2],
                      c=t_norm, cmap="plasma", s=2, alpha=0.8)
    ax1.plot(poses[:, 0], poses[:, 1], poses[:, 2], color="gray", alpha=0.3, linewidth=0.5)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.invert_zaxis()
    fig.colorbar(sc1, ax=ax1, label="Normalized Time", shrink=0.6)

    # --- 子图 2: XY 鸟瞰图 + 朝向箭头 ---
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Top-Down View (XY plane)")
    ax2.scatter(poses[:, 0], poses[:, 1], c=t_norm, cmap="plasma", s=3, alpha=0.8)
    ax2.plot(poses[:, 0], poses[:, 1], color="gray", alpha=0.3, linewidth=0.5)

    # 每隔 step=15 帧画一个朝向箭头
    step = max(1, len(poses) // 30)
    px, py, pz, quiver_dx, quiver_dz = make_heading_arrows(poses, quats, step=step, size=0.04)
    ax2.quiver(px, py, quiver_dx, quiver_dz, angles="xy", scale_units="xy",
               scale=1, color="red", width=0.003, alpha=0.7)

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, alpha=0.3)

    # --- 子图 3: XZ 侧视图 ---
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("Side View (XZ plane — forward direction)")
    ax3.scatter(poses[:, 0], poses[:, 2], c=t_norm, cmap="plasma", s=3, alpha=0.8)
    ax3.plot(poses[:, 0], poses[:, 2], color="gray", alpha=0.3, linewidth=0.5)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Z (forward, m)")
    ax3.invert_yaxis()  # Z 朝前，通常朝下画
    ax3.grid(True, alpha=0.3)

    # --- 子图 4: 各分量随时间变化 ---
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("Position Components vs Time")
    t_rel = times - times[0]
    ax4.plot(t_rel, poses[:, 0], label="X (m)", linewidth=1.2)
    ax4.plot(t_rel, poses[:, 1], label="Y (m)", linewidth=1.2)
    ax4.plot(t_rel, poses[:, 2], label="Z (m)", linewidth=1.2)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Position (m)")
    ax4.legend(loc="best")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
        print(f"\n💾 图表已保存至: {save_to}")

    print("\n🎨 正在显示交互式图表（可拖拽旋转 3D 视图，关闭窗口退出）...")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="可视化 AllFrames_trajectory.txt（TUM 格式，有真实物理尺度）"
    )
    parser.add_argument(
        "trajectory_file",
        nargs="?",
        default=None,
        help="TUM 轨迹文件路径（默认: data/deepseek/lines_all/AllFrames_trajectory.txt）",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="保存图表到指定路径（如 trajectory.png）",
    )
    args = parser.parse_args()

    # 默认路径
    if args.trajectory_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(
            script_dir, "..", "data", "deepseek", "lines_all", "AllFrames_trajectory.txt"
        )
        args.trajectory_file = os.path.normpath(default_path)

    if not os.path.exists(args.trajectory_file):
        print(f"❌ 找不到文件: {args.trajectory_file}")
        print("   用法: python visualize_trajectory.py <trajectory_file>")
        sys.exit(1)

    print(f"📂 正在加载: {args.trajectory_file}")
    times, poses, quats = load_tum_trajectory(args.trajectory_file)

    if len(poses) == 0:
        print("❌ 轨迹文件为空或格式不正确。")
        sys.exit(1)

    stats = compute_stats(poses, times)
    print_scale_report(stats, args.trajectory_file)
    visualize_trajectory(times, poses, quats, stats, save_to=args.save)


if __name__ == "__main__":
    main()