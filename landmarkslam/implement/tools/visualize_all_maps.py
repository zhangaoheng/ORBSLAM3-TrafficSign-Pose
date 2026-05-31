#!/home/zah/ORB_SLAM3-master/landmarkslam/yolo_venv/bin/python3
"""可视化所有 7 段地图轨迹"""
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

base = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data_new/runs/2026-05-31_20-52-36_mono"

# Load all 7 maps
map_files = sorted([f for f in os.listdir(base) if f.startswith("map_") and f.endswith("_trajectory.txt")])
maps = []
for mf in map_files:
    t, p = load_traj(os.path.join(base, mf))
    mid = int(mf.split("_")[1])
    dist = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
    maps.append((mid, t, p))
    print(f"  Map {mid}: {len(t)} KFs, path {dist:.1f}m")

n_maps = len(maps)
colors = plt.cm.tab10(np.linspace(0, 1, n_maps))

# ========================================================
# FIGURE 1: 7 individual 3D views (2x4 grid, last slot empty)
# ========================================================
fig1 = plt.figure(figsize=(22, 10))
fig1.suptitle("Individual Map Trajectories (Mono RGB-D)", fontsize=14, fontweight="bold")

for i, (mid, t, p) in enumerate(maps):
    ax = fig1.add_subplot(2, 4, i + 1, projection="3d")
    t_norm = (t - t[0]) / (t[-1] - t[0] + 1e-9)
    ax.scatter(p[:,0], p[:,1], p[:,2], c=t_norm, cmap="plasma", s=5, alpha=0.8)
    ax.plot(p[:,0], p[:,1], p[:,2], "gray", alpha=0.3, linewidth=0.5)
    ax.set_title(f"Map {mid} ({len(t)} KFs)", fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

# Last subplot: info
ax = fig1.add_subplot(2, 4, 8)
ax.axis("off")
info = "\n".join([f"Map {mid}: {len(t):4d} KFs, {np.sum(np.linalg.norm(np.diff(p,axis=0),axis=1)):.1f}m" for mid, t, p in maps])
total_kf = sum(len(t) for _,t,_ in maps)
total_dist = sum(np.sum(np.linalg.norm(np.diff(p,axis=0),axis=1)) for _,_,p in maps)
ax.text(0.1, 0.7, info, fontsize=9, verticalalignment="top", family="monospace")
ax.text(0.1, 0.05, f"Total: {total_kf} KFs, {total_dist:.1f}m", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(base, "all_maps_3d.png"), dpi=150, bbox_inches="tight")
print(f"\n  Saved: all_maps_3d.png")

# ========================================================
# FIGURE 2: XY overlay of all 7 maps
# ========================================================
fig2, ax2 = plt.subplots(figsize=(14, 10))
fig2.suptitle("All Map Trajectories Overlay (XY View)", fontsize=14, fontweight="bold")

for i, (mid, t, p) in enumerate(maps):
    label = f"Map {mid} ({len(t)} KFs, {np.sum(np.linalg.norm(np.diff(p,axis=0),axis=1)):.1f}m)"
    ax2.scatter(p[:,0], p[:,1], c=[colors[i]], s=8, alpha=0.6, label=label)
    ax2.plot(p[:,0], p[:,1], color=colors[i], alpha=0.4, linewidth=0.8)

ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.legend(fontsize=9, loc="upper left")
ax2.set_aspect("equal", adjustable="datalim")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base, "all_maps_xy.png"), dpi=150, bbox_inches="tight")
print(f"  Saved: all_maps_xy.png")

# ========================================================
# FIGURE 3: Each map in its own window for detailed inspection
# ========================================================
print(f"\n  Opening separate windows for each map...")
for mid, t, p in maps:
    fig3 = plt.figure(figsize=(12, 8))
    fig3.suptitle(f"Map {mid} Trajectory", fontsize=14)

    ax1 = fig3.add_subplot(2, 2, 1, projection="3d")
    t_norm = (t - t[0]) / (t[-1] - t[0] + 1e-9)
    ax1.scatter(p[:,0], p[:,1], p[:,2], c=t_norm, cmap="plasma", s=8, alpha=0.8)
    ax1.plot(p[:,0], p[:,1], p[:,2], "gray", alpha=0.3)
    ax1.set_title("3D View")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    ax2 = fig3.add_subplot(2, 2, 2)
    ax2.scatter(p[:,0], p[:,1], c=t_norm, cmap="plasma", s=10, alpha=0.8)
    ax2.plot(p[:,0], p[:,1], "gray", alpha=0.3)
    ax2.set_title("XY View")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, alpha=0.3)

    ax3 = fig3.add_subplot(2, 2, 3)
    ax3.scatter(p[:,0], p[:,2], c=t_norm, cmap="plasma", s=10, alpha=0.8)
    ax3.plot(p[:,0], p[:,2], "gray", alpha=0.3)
    ax3.set_title("XZ View")
    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Z (m)")
    ax3.grid(True, alpha=0.3)

    ax4 = fig3.add_subplot(2, 2, 4)
    t_rel = t - t[0]
    ax4.plot(t_rel, p[:,0], label="X", linewidth=1)
    ax4.plot(t_rel, p[:,1], label="Y", linewidth=1)
    ax4.plot(t_rel, p[:,2], label="Z", linewidth=1)
    ax4.set_title("Position vs Time")
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Position (m)")
    ax4.legend(); ax4.grid(True, alpha=0.3)

    plt.tight_layout()

print(f"\n  All images saved to: {base}")
print(f"  Close plot windows to exit...")
plt.show()
