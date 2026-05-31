#!/usr/bin/env python3
"""
轨迹拼接工具 — 将 ORB-SLAM3 多个地图的关键帧轨迹拼接为连续轨迹。

原理：地图 N 的最后几帧 和 地图 N+1 的第一帧 在物理上是连续的，
通过最小化拼接点处的位姿跳变来对齐各地图坐标系。
"""

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_trajectory(path):
    """加载 TUM 格式轨迹文件"""
    times, poses, quats = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) < 8: continue
            t = float(parts[0])
            if t > 1e14: t /= 1e9
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            times.append(t)
            poses.append([tx, ty, tz])
            quats.append([qx, qy, qz, qw])
    return np.array(times), np.array(poses), np.array(quats)

def find_map_files(run_dir):
    """查找目录中所有 map_X_trajectory.txt 文件"""
    files = []
    for f in sorted(os.listdir(run_dir)):
        if f.startswith("map_") and f.endswith("_trajectory.txt"):
            map_id = int(f.split("_")[1])
            files.append((map_id, os.path.join(run_dir, f)))
    return sorted(files, key=lambda x: x[0])

def align_trajectories(poses_ref, poses_align, align_window=5):
    """通过最小化拼接点窗口内的平均距离，计算对齐变换"""
    if len(poses_ref) < align_window or len(poses_align) < align_window:
        return np.eye(4)
    
    # 取 ref 的最后 align_window 帧和 align 的前 align_window 帧
    ref_tail = poses_ref[-align_window:]
    align_head = poses_align[:align_window]
    
    # 计算中心点
    ref_center = np.mean(ref_tail, axis=0)
    align_center = np.mean(align_head, axis=0)
    
    # 平移 = ref 中心 - align 中心
    translation = ref_center - align_center
    
    # 假设旋转近似为 Identity（短时间内的旋转变化不大）
    T = np.eye(4)
    T[:3, 3] = translation
    
    return T

def stitch_trajectories(run_dir, output_path):
    """拼接主函数"""
    map_files = find_map_files(run_dir)
    if not map_files:
        print(f"❌ 在 {run_dir} 中未找到地图轨迹文件")
        return
    
    print(f"📂 找到 {len(map_files)} 个地图轨迹文件")
    
    all_times = []
    all_poses = []
    all_quats = []
    
    transform_world = np.eye(4)  # 累积的世界变换
    
    for i, (map_id, filepath) in enumerate(map_files):
        times, poses, quats = load_trajectory(filepath)
        print(f"  地图{map_id}: {len(times)} 关键帧")
        
        if len(times) < 2:
            continue
        
        if i == 0:
            # 第一个地图，保持原始坐标系
            transform_world = np.eye(4)
        else:
            # 对齐到上一个地图的末尾
            ref_tail = np.array(all_poses[-5:]) if len(all_poses) >= 5 else np.array(all_poses)
            T_align = align_trajectories(ref_tail, poses)
            transform_world = transform_world @ T_align
        
        # 应用变换
        for j in range(len(poses)):
            p = np.append(poses[j], 1.0)
            p_aligned = transform_world @ p
            all_times.append(times[j])
            all_poses.append(p_aligned[:3])
            all_quats.append(quats[j])
    
    all_poses = np.array(all_poses)
    all_times = np.array(all_times)
    all_quats = np.array(all_quats)
    
    # 保存拼接轨迹
    with open(output_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i in range(len(all_times)):
            f.write(f"{all_times[i]:.6f} {all_poses[i,0]:.7f} {all_poses[i,1]:.7f} {all_poses[i,2]:.7f} "
                    f"{all_quats[i,0]:.7f} {all_quats[i,1]:.7f} {all_quats[i,2]:.7f} {all_quats[i,3]:.7f}\n")
    
    print(f"\n✅ 拼接完成: {len(all_times)} 帧 → {output_path}")
    
    # 统计
    total_dist = np.sum(np.linalg.norm(np.diff(all_poses, axis=0), axis=1))
    dur = all_times[-1] - all_times[0]
    print(f"   总路程: {total_dist:.2f}m")
    print(f"   时间跨度: {dur:.1f}s")
    print(f"   平均速度: {total_dist/dur*3.6:.1f} km/h")
    
    # 生成可视化
    fig = plt.figure(figsize=(16, 6))
    
    ax1 = fig.add_subplot(121, projection="3d")
    t_norm = (all_times - all_times[0]) / (all_times[-1] - all_times[0] + 1e-9)
    ax1.scatter(all_poses[:,0], all_poses[:,1], all_poses[:,2], 
                c=t_norm, cmap="plasma", s=2, alpha=0.8)
    ax1.plot(all_poses[:,0], all_poses[:,1], all_poses[:,2], "gray", alpha=0.3, linewidth=0.5)
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.set_zlabel("Z (m)")
    ax1.set_title("拼接后的完整轨迹")
    
    ax2 = fig.add_subplot(122)
    ax2.scatter(all_poses[:,0], all_poses[:,1], c=t_norm, cmap="plasma", s=3, alpha=0.8)
    ax2.plot(all_poses[:,0], all_poses[:,1], "gray", alpha=0.3, linewidth=0.5)
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    ax2.set_title("XY 俯视图")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, alpha=0.3)
    
    viz_path = output_path.replace(".txt", "_viz.png")
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    print(f"   可视化: {viz_path}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python stitch_maps.py <运行目录> <输出路径>")
        print("示例: python stitch_maps.py runs/2026-05-31_16-57-06_rgbd stitched_trajectory.txt")
        sys.exit(1)
    
    stitch_trajectories(sys.argv[1], sys.argv[2])
