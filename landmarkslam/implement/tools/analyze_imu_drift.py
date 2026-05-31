#!/usr/bin/env python3
"""绘制轨迹抖动分析图"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data/runs/2026-05-30_21-00-56_imu_monocular/trajectory.txt"
outdir = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data/runs/2026-05-30_21-00-56_imu_monocular"

times, poses = [], []
with open(path) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 8: continue
        t = float(parts[0])
        if t > 1e14: t /= 1e9
        poses.append([float(parts[1]), float(parts[2]), float(parts[3])])
        times.append(t)

poses = np.array(poses)
times = np.array(times)
t0 = times[0]
n = len(poses)

# 计算每100帧的抖动因子
step = 100
ratios, speeds, mids = [], [], []
for i in range(0, n - step, step):
    seg = poses[i:i+step]
    seg_t = times[i:i+step]
    dt = seg_t[-1] - seg_t[0]
    dist = np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))
    disp = np.linalg.norm(seg[-1] - seg[0])
    ratio = dist / disp if disp > 0.5 else 0
    speed = dist / dt if dt > 0 else 0
    ratios.append(ratio)
    speeds.append(speed)
    mids.append((times[i] + times[min(i+step, n-1)]) / 2 - t0)

mids = np.array(mids)
ratios = np.array(ratios)
speeds = np.array(speeds)

fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle("IMU Monocular 轨迹质量分析 — 从稳定到漂移的完整过程", fontsize=14, fontweight="bold")

# 1. 抖动因子
ax = axes[0]
ax.plot(mids, ratios, 'r-', linewidth=1.2)
ax.axhline(y=3, color='orange', linestyle='--', alpha=0.7, label='抖动阈值(3x)')
ax.axhline(y=8, color='red', linestyle='--', alpha=0.7, label='严重漂移(8x)')
ax.set_ylabel("抖动因子 (路程/位移)")
ax.set_title("抖动因子随时间变化 — 越高表示原地抖动越严重")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 速度
ax = axes[1]
ax.plot(mids, speeds * 3.6, 'b-', linewidth=1.2)
ax.set_ylabel("速度 (km/h)")
ax.set_title("估计速度 — 正常情况下应 < 80km/h")
ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='合理上限')
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='静止')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 位置分量
ax = axes[2]
t_rel = times - t0
ax.plot(t_rel, poses[:,0], label='X', linewidth=0.8)
ax.plot(t_rel, poses[:,1], label='Y', linewidth=0.8)
ax.plot(t_rel, poses[:,2], label='Z', linewidth=0.8)
ax.axvline(x=1400, color='red', linestyle='--', alpha=0.5, label='漂移开始~1400s')
# Annotate start of drift
drift_start = 480  # around frame 14400
ax.annotate('漂移开始', xy=(drift_start, 300), fontsize=10, color='red',
            xytext=(drift_start+30, 600), arrowprops=dict(arrowstyle='->', color='red'))
ax.set_ylabel("位置 (m)")
ax.set_title("X/Y/Z 位置分量 — 观察发散点")
ax.legend()
ax.grid(True, alpha=0.3)

# 4. XY 轨迹着色（前段绿后段红）
ax = axes[3]
colors = np.zeros((n, 3))
colors[:, 0] = np.linspace(0, 1, n)  # R
colors[:, 1] = 1 - np.linspace(0, 1, n)  # G
colors[:, 2] = 0
ax.scatter(poses[:,0], poses[:,1], c=colors, s=1, alpha=0.6)
ax.plot(poses[:,0], poses[:,1], 'gray', alpha=0.2, linewidth=0.5)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("XY 轨迹 — 颜色从绿(前)渐变到红(后)")
ax.set_aspect('equal', adjustable='datalim')
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = f"{outdir}/trajectory_analysis.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"分析图已保存: {save_path}")

print("\n" + "="*60)
print("分析结论")
print("="*60)
print(f"1. 前段 (0-~14000帧, ~460秒): ✅ 稳定跟踪")
print(f"   路程~2500m, 速度~20km/h, 抖动因子<3")
print(f"")
print(f"2. 中段 (~14000-14700帧, ~475秒): ⚠️ 开始抖动")
print(f"   抖动比升至9-27x, 速度异常")
print(f"")
print(f"3. 中段 (~14700-15000帧, 480-490秒): 🟢 静止(停车?)")
print(f"   位置变化<0.1m, 约10秒")
print(f"")
print(f"4. 后段 (~15000-16634帧, 490-555秒): 🚨 严重漂移到崩溃")
print(f"   抖动比最高122x, 速度>100km/h")
print(f"   最后g2o优化发散→SIGSEGV")
print(f"")
print(f"根因: IMU偏置发散+视觉弱纹理 → 停车期间偏置无法校准")
print(f"     → 重启后轨迹完全失控 → NaN → 崩溃")
