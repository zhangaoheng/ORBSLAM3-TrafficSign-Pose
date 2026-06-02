#!/home/zah/ORB_SLAM3-master/landmarkslam/yolo_venv/bin/python3
"""可视化 RGB-D 28 个地图轨迹"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_traj(path):
    times, poses = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) < 8: continue
            t = float(parts[0])
            if t > 1e14: t /= 1e9
            poses.append([float(parts[1]), float(parts[2]), float(parts[3])])
            times.append(t)
    return np.array(times), np.array(poses)

base = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data_new/runs/2026-05-31_21-48-05_rgbd"
map_files = sorted([f for f in os.listdir(base) if f.startswith("map_") and f.endswith("_trajectory.txt")],
                   key=lambda x: int(x.split("_")[1]))

maps = []
for mf in map_files:
    t, p = load_traj(os.path.join(base, mf))
    mid = int(mf.split("_")[1])
    dist = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
    maps.append((mid, t, p, dist))

n = len(maps)
print(f"Loaded {n} maps")
colors = plt.cm.tab20(np.linspace(0, 1, max(n, 20)))

# =============================================
# FIGURE 1: XY overlay of ALL 28 maps
# =============================================
fig1, ax1 = plt.subplots(figsize=(16, 12))
fig1.suptitle(f"RGB-D: All {n} Map Trajectories Overlay", fontsize=14, fontweight="bold")

for i, (mid, t, p, dist) in enumerate(maps):
    c = colors[i % 20]
    label = f"Map {mid}" if dist > 1 else ""
    ax1.plot(p[:,0], p[:,1], color=c, linewidth=1.2, alpha=0.7, label=label)
    ax1.scatter(p[0,0], p[0,1], color=c, s=15, marker="o", zorder=3)
    ax1.text(p[0,0], p[0,1], str(mid), fontsize=7, color=c, alpha=0.8)

# Main trajectory
main_traj = os.path.join(base, "trajectory.txt")
if os.path.exists(main_traj):
    _, mp = load_traj(main_traj)
    ax1.plot(mp[:,0], mp[:,1], "k-", linewidth=2, alpha=0.5, label="Main trajectory")

ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.legend(fontsize=6, ncol=3, loc="upper right")
ax1.set_aspect("equal", adjustable="datalim")
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base, "all_maps_overlay.png"), dpi=150, bbox_inches="tight")
print(f"Saved: all_maps_overlay.png")

# =============================================
# FIGURE 2: Largest maps in detail
# =============================================
# Sort by distance, take top 6
top_maps = sorted(maps, key=lambda x: x[3], reverse=True)[:6]
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10), subplot_kw={"projection": "3d"})
fig2.suptitle("Top 6 Largest Map Trajectories", fontsize=14, fontweight="bold")

for i, (mid, t, p, dist) in enumerate(top_maps):
    ax = axes2[i // 3][i % 3]
    tn = (t - t[0]) / (t[-1] - t[0] + 1e-9)
    ax.scatter(p[:,0], p[:,1], p[:,2], c=tn, cmap="plasma", s=5, alpha=0.8)
    ax.plot(p[:,0], p[:,1], p[:,2], "gray", alpha=0.3)
    ax.set_title(f"Map {mid}: {dist:.0f}m, {len(t)} KFs", fontsize=9)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

plt.tight_layout()
plt.savefig(os.path.join(base, "top6_maps_3d.png"), dpi=150, bbox_inches="tight")
print(f"Saved: top6_maps_3d.png")

# =============================================
# FIGURE 3: Stats summary
# =============================================
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.axis("off")

lines = ["RGB-D Run Summary (71,500 frames, 28 maps)", "="*50]
lines.append(f"{'Map':>5} {'KFs':>6} {'Dist(m)':>8} {'Time(s)':>8}")
lines.append("-"*35)
total_kf = 0
total_dist = 0
for mid, t, p, dist in maps:
    dur = t[-1] - t[0] if len(t) > 1 else 0
    lines.append(f"{mid:5d} {len(t):6d} {dist:8.1f} {dur:8.1f}")
    total_kf += len(t)
    total_dist += dist
lines.append("-"*35)
lines.append(f"{'Total':>5} {total_kf:6d} {total_dist:8.1f}")

ax3.text(0.05, 0.95, "\n".join(lines), fontsize=8, fontfamily="monospace",
         verticalalignment="top", transform=ax3.transAxes)
plt.savefig(os.path.join(base, "maps_summary.png"), dpi=150, bbox_inches="tight")
print(f"Saved: maps_summary.png")

print(f"\nImages saved to: {base}")
print("Close windows to exit...")
plt.show()
