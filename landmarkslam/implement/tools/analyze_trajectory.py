#!/usr/bin/env python3
"""分析 IMU Monocular 轨迹分段质量"""
import numpy as np

path = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data/runs/2026-05-30_21-00-56_imu_monocular/trajectory.txt"

times, poses = [], []
with open(path) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 8: continue
        t = float(parts[0])
        if t > 1e14: t /= 1e9
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        times.append(t)
        poses.append([x, y, z])

poses = np.array(poses)
times = np.array(times)
t0 = times[0]
n = len(poses)

print(f"总帧数: {n}, 总时间: {times[-1]-t0:.1f}s")
print(f"{'='*80}")
print(f"{'段':>5} {'帧范围':>12} {'时间段(s)':>16} {'路程(m)':>8} {'X范围':>6} {'Y范围':>6} {'Z范围':>6} {'速度m/s':>7} {'漂移?':>6}")
print(f"{'='*80}")

segments = 20
for i in range(segments):
    start = i * n // segments
    end = min((i+1) * n // segments, n)
    if end - start < 2: continue

    seg = poses[start:end]
    st = times[start:end]
    dt = st[-1] - st[0]
    dx = seg[:,0].max() - seg[:,0].min()
    dy = seg[:,1].max() - seg[:,1].min()
    dz = seg[:,2].max() - seg[:,2].min()
    dist = np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))
    speed = dist / dt if dt > 0 else 0

    # 判断是否异常: 速度突变 > 50m/s 或 路程/位移比 > 3 (来回抖动)
    disp = np.linalg.norm(seg[-1] - seg[0])
    ratio = dist / disp if disp > 0.1 else 0
    drift = ""
    if speed > 30: drift = "⚠速度"
    if ratio > 5: drift = "⚠抖动"

    print(f"{i+1:4d}  {start:6d}-{end:<6d} {st[0]-t0:7.1f}-{st[-1]-t0:<7.1f} {dist:8.1f} {dx:6.1f} {dy:6.1f} {dz:6.1f} {speed:7.1f} {drift:>6}")

# 最后500帧详细分析
print(f"\n{'='*80}")
print(f"最后500帧详细分析:")
last500 = poses[-500:]
last_t = times[-500:]
dx_l = last500[:,0].max() - last500[:,0].min()
dy_l = last500[:,1].max() - last500[:,1].min()
dz_l = last500[:,2].max() - last500[:,2].min()
dist_l = np.sum(np.linalg.norm(np.diff(last500, axis=0), axis=1))
dt_l = last_t[-1] - last_t[0]
print(f"  路程: {dist_l:.1f}m, 位移: {np.linalg.norm(last500[-1]-last500[0]):.1f}m")
print(f"  X: {dx_l:.1f}m, Y: {dy_l:.1f}m, Z: {dz_l:.1f}m")
print(f"  速度: {dist_l/dt_l:.1f} m/s ({dist_l/dt_l*3.6:.0f} km/h)")
if dist_l / max(np.linalg.norm(last500[-1]-last500[0]), 0.1) > 5:
    print("  ❌ 最后500帧异常抖动！路程远大于位移")
else:
    print("  ✅ 最后500帧正常")
