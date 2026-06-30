#!/usr/bin/env python3
"""
自动修复轨迹四元数脚本
用途: 仅读取已分割文件夹中的轨迹时间戳，从原始轨迹中匹配真实的四元数并覆盖修复。
用法: python3 fix_trajectory.py <原始轨迹.txt> <manual_split目录>
"""
import sys, os, numpy as np

if len(sys.argv) < 3:
    print("用法: python3 fix_trajectory.py <原始KeyFrameTrajectory.txt> <manual_split目录>")
    sys.exit(1)

orig_traj_path = sys.argv[1]
split_dir = sys.argv[2]

# 1. 加载原始轨迹 (含真实的四元数)
print("⏳ 正在加载原始轨迹...")
traj = np.loadtxt(orig_traj_path)
traj_ts = traj[:, 0]
if traj_ts[0] < 1e12:
    traj_ts = traj_ts * 1e9

if traj.shape[1] >= 8:
    traj_poses = traj[:, 1:8] # [x, y, z, qx, qy, qz, qw]
else:
    print("❌ 原始轨迹没有四元数列！请检查文件。")
    sys.exit(1)

# 2. 遍历你已经生成的 manual_split 文件夹
fixed_count = 0
for seg_name in sorted(os.listdir(split_dir)):
    seg_dir = os.path.join(split_dir, seg_name)
    bad_traj_path = os.path.join(seg_dir, 'trajectory.txt')

    if not os.path.isdir(seg_dir) or not os.path.exists(bad_traj_path):
        continue

    # 读取由于硬编码导致四元数为 0 0 0 1 的坏轨迹
    bad_traj = np.loadtxt(bad_traj_path)
    if len(bad_traj) == 0: 
        continue
    
    # 获取单列的情况防崩
    if bad_traj.ndim == 1:
        bad_traj = bad_traj.reshape(1, -1)

    bad_ts = bad_traj[:, 0]
    good_lines = []

    # 3. 对齐时间戳，提取真实四元数
    for ts in bad_ts:
        # 寻找距离当前时间戳最近的原始轨迹点
        idx = np.argmin(np.abs(traj_ts - ts))
        px, py, pz, qx, qy, qz, qw = traj_poses[idx]
        
        # 写入修复后的行 (保留原时间戳，加上真实的 XYZ 和四元数)
        good_lines.append(f"{ts:.0f} {px:.6f} {py:.6f} {pz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

    # 4. 覆盖重写轨迹文件
    with open(bad_traj_path, 'w') as f:
        f.writelines(good_lines)

    print(f"✅ 已修复: {seg_name}/trajectory.txt (共 {len(good_lines)} 帧)")
    fixed_count += 1

print(f"\n🎉 修复完成！共修复了 {fixed_count} 个数据段的轨迹，未触碰任何图片数据。")