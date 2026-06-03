#!/usr/bin/env python3
"""六条轨迹展示 — SLAM ×2 + GPS ×2 + 融合对齐轨迹"""
import os, numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def load_traj(path):
    ts, poses = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) < 8: continue
            ts.append(float(parts[0]))
            poses.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(ts), np.array(poses)

def load_gps_enu(path):
    ts, lats, lons, alts = [], [], [], []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            if len(p) < 4: continue
            try: ts.append(float(p[0])); lats.append(float(p[1])); lons.append(float(p[2])); alts.append(float(p[3]))
            except: pass
    ts=np.array(ts); lats=np.array(lats); lons=np.array(lons); alts=np.array(alts)
    v = alts > 0
    ts, lats, lons, alts = ts[v], lats[v], lons[v], alts[v]
    lat0, lon0 = lats[0], lons[0]
    a=6378137.0; e2=0.00669437999014
    sin_lat0=np.sin(np.radians(lat0))
    N0=a/np.sqrt(1-e2*sin_lat0**2)
    x0=(N0+alts[0])*np.cos(np.radians(lat0))*np.cos(np.radians(lon0))
    y0=(N0+alts[0])*np.cos(np.radians(lat0))*np.sin(np.radians(lon0))
    z0=(N0*(1-e2)+alts[0])*np.sin(np.radians(lat0))
    sin_lat=np.sin(np.radians(lats))
    N=a/np.sqrt(1-e2*sin_lat**2)
    x=(N+alts)*np.cos(np.radians(lats))*np.cos(np.radians(lons))
    y=(N+alts)*np.cos(np.radians(lats))*np.sin(np.radians(lons))
    z=(N*(1-e2)+alts)*np.sin(np.radians(lats))
    dx,dy,dz=x-x0,y-y0,z-z0
    sl,cl=np.sin(np.radians(lat0)),np.cos(np.radians(lat0))
    so,co=np.sin(np.radians(lon0)),np.cos(np.radians(lon0))
    e=-so*dx+co*dy; n=-sl*co*dx-sl*so*dy+cl*dz
    return ts, np.column_stack([e, n])

base = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/test"
main_dir = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/main"

# 自动找最新 run 目录
runs_dir = os.path.join(main_dir, "runs")
run_dir = None
if os.path.isdir(runs_dir):
    run_dirs = sorted(os.listdir(runs_dir))
    if run_dirs:
        run_dir = os.path.join(runs_dir, run_dirs[-1])
        print(f"📁 最新 run: {run_dir}")

datasets = [
    ("Map 03 SLAM",      "blue",   load_traj(f"{base}/map_3_trajectory.txt")),
    ("Map 19 SLAM",      "red",    load_traj(f"{base}/map_19_trajectory.txt")),
    ("Map 03 GPS",       "cyan",   load_gps_enu(f"{base}/map_03/gps_segment.csv")),
    ("Map 19 GPS",       "orange", load_gps_enu(f"{base}/map_19/gps_segment.csv")),
]

# 叠加图（如果对齐数据存在）
aligned_files = [
    os.path.join(run_dir, "trajectory_03.txt") if run_dir else "",
    os.path.join(run_dir, "trajectory_19_aligned.txt") if run_dir else "",
]
has_aligned = all(os.path.exists(f) for f in aligned_files)
if has_aligned:
    _, p03 = load_traj(aligned_files[0])
    _, p19 = load_traj(aligned_files[1])
    print(f"  对齐数据: Map03={len(p03)} KFs, Map19_aligned={len(p19)} KFs")

# 显示各段
for name, color, (ts, pts) in datasets:
    dist = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    dur = ts[-1] - ts[0] if len(ts) > 1 else 0
    print(f"  {name}: {len(pts)} pts, {dist:.1f}m, {dur:.1f}s")

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(f"{name}  ({dist:.1f}m, {dist/dur*3.6 if dur>0 else 0:.1f} km/h)", fontsize=14, fontweight="bold")
    ax = fig.add_subplot(1, 1, 1)
    tn = (ts - ts[0]) / (ts[-1] - ts[0] + 1e-9)
    ax.scatter(pts[:, 0], pts[:, 1], c=tn, cmap="plasma", s=10, alpha=0.8)
    ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.4, linewidth=1)
    ax.scatter(*pts[0, :2],  c="green", s=100, marker="o", label="Start")
    ax.scatter(*pts[-1, :2], c="red",   s=100, marker="x", label="End")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    out = f"{base}/{name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"    → {out}")

# ===== 融合轨迹叠加图 =====
if has_aligned:
    print(f"\n  Merged: {len(p03)}+{len(p19)} KFs")

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Trajectory Alignment: Map 03 + Map 19 (Merged)", fontsize=14, fontweight="bold")
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(p03[:, 0], p03[:, 1], "b-", linewidth=2, alpha=0.8, label="Map 03 (ref)")
    ax.plot(p19[:, 0], p19[:, 1], "r-", linewidth=2, alpha=0.8, label="Map 19 (aligned)")
    ax.scatter(p03[0, 0], p03[0, 1], c="green", s=150, marker="o", label="Start 03")
    ax.scatter(p03[-1,0], p03[-1,1], c="green", s=100, marker="x")
    ax.scatter(p19[0, 0], p19[0, 1], c="darkred", s=150, marker="o", label="Start 19")
    ax.scatter(p19[-1,0], p19[-1,1], c="darkred", s=100, marker="x")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    out = f"{base}/merged_trajectories.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"    → {out}")
else:
    print("\n  ⚠️ 对齐数据不存在，先运行 test.py 生成文件")

# ===== GPS 叠加图 =====
print(f"\n  GPS Overlay")
_, g1 = load_gps_enu(f"{base}/map_03/gps_segment.csv")
_, g2 = load_gps_enu(f"{base}/map_19/gps_segment.csv")

fig = plt.figure(figsize=(10, 8))
fig.suptitle("GPS Overlay: Map 03 + Map 19", fontsize=14, fontweight="bold")
ax = fig.add_subplot(1, 1, 1)
ax.plot(g1[:, 0], g1[:, 1], "c-", linewidth=2, alpha=0.8, label="GPS 03")
ax.plot(g2[:, 0], g2[:, 1], "orange", linewidth=2, alpha=0.8, label="GPS 19")
ax.scatter(g1[0, 0], g1[0, 1], c="green", s=150, marker="o", label="Start 03")
ax.scatter(g1[-1,0], g1[-1,1], c="green", s=100, marker="x")
ax.scatter(g2[0, 0], g2[0, 1], c="darkred", s=150, marker="o", label="Start 19")
ax.scatter(g2[-1,0], g2[-1,1], c="darkred", s=100, marker="x")
ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
ax.legend(); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
out = f"{base}/gps_overlay.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"    → {out}")

print("\n✅ Close each window to continue...")
plt.show()
