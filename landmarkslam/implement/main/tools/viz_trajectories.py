#!/usr/bin/env python3
"""可视化 map_03 和 map_19 的轨迹文件"""
import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def load_traj(path, name=""):
    ts, poses = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) < 8: continue
            t = float(parts[0])
            poses.append([float(parts[1]), float(parts[2]), float(parts[3])])
            ts.append(t)
    p = np.array(poses)
    dist = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
    dur = ts[-1] - ts[0] if len(ts) > 1 else 0
    print(f"  {name}: {len(ts)} KFs, {dist:.1f}m, {dur:.1f}s, {dist/dur*3.6 if dur>0 else 0:.1f}km/h")
    return p, ts

base = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/test"

p1, t1 = load_traj(f"{base}/map_3_trajectory.txt", "Map 03")
p2, t2 = load_traj(f"{base}/map_19_trajectory.txt", "Map 19")

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Trajectories: Map 03 & Map 19", fontsize=14, fontweight="bold")

# 3D
ax1 = fig.add_subplot(2, 3, 1, projection="3d")
tn1 = (np.array(t1) - t1[0]) / (t1[-1] - t1[0] + 1e-9)
ax1.scatter(p1[:,0], p1[:,1], p1[:,2], c=tn1, cmap="plasma", s=5, alpha=0.8)
ax1.plot(p1[:,0], p1[:,1], p1[:,2], "gray", alpha=0.3)
ax1.set_title(f"Map 03 ({len(p1)} KFs)"); ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

ax2 = fig.add_subplot(2, 3, 2, projection="3d")
tn2 = (np.array(t2) - t2[0]) / (t2[-1] - t2[0] + 1e-9)
ax2.scatter(p2[:,0], p2[:,1], p2[:,2], c=tn2, cmap="plasma", s=5, alpha=0.8)
ax2.plot(p2[:,0], p2[:,1], p2[:,2], "gray", alpha=0.3)
ax2.set_title(f"Map 19 ({len(p2)} KFs)"); ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

# XY overlaid
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(p1[:,0], p1[:,1], "b-", linewidth=1.5, label="Map 03", alpha=0.7)
ax3.scatter(p1[0,0], p1[0,1], c="green", s=80, marker="o")
ax3.scatter(p1[-1,0], p1[-1,1], c="green", s=80, marker="x")
ax3.plot(p2[:,0], p2[:,1], "r-", linewidth=1.5, label="Map 19", alpha=0.7)
ax3.scatter(p2[0,0], p2[0,1], c="orange", s=80, marker="o")
ax3.scatter(p2[-1,0], p2[-1,1], c="orange", s=80, marker="x")
ax3.set_title("XY Overlay"); ax3.set_xlabel("X (m)"); ax3.set_ylabel("Y (m)")
ax3.legend(); ax3.set_aspect("equal"); ax3.grid(True, alpha=0.3)

# XZ
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(p1[:,0], p1[:,2], "b-", label="Map 03", alpha=0.7)
ax4.scatter(p1[0,0], p1[0,2], c="green", s=50, marker="o")
ax4.scatter(p1[-1,0], p1[-1,2], c="green", s=50, marker="x")
ax4.plot(p2[:,0], p2[:,2], "r-", label="Map 19", alpha=0.7)
ax4.scatter(p2[0,0], p2[0,2], c="orange", s=50, marker="o")
ax4.scatter(p2[-1,0], p2[-1,2], c="orange", s=50, marker="x")
ax4.set_title("XZ Side"); ax4.set_xlabel("X (m)"); ax4.set_ylabel("Z (m)")
ax4.legend(); ax4.grid(True, alpha=0.3)

# Components
ax5 = fig.add_subplot(2, 3, 5)
tr1 = np.array(t1) - t1[0]
ax5.plot(tr1, p1[:,0], "b-", label="X", linewidth=0.8)
ax5.plot(tr1, p1[:,1], "b--", label="Y", linewidth=0.8)
ax5.plot(tr1, p1[:,2], "b:", label="Z", linewidth=0.8)
ax5.set_title("Map 03 Components"); ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Position (m)")
ax5.legend(); ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(2, 3, 6)
tr2 = np.array(t2) - t2[0]
ax6.plot(tr2, p2[:,0], "r-", label="X", linewidth=0.8)
ax6.plot(tr2, p2[:,1], "r--", label="Y", linewidth=0.8)
ax6.plot(tr2, p2[:,2], "r:", label="Z", linewidth=0.8)
ax6.set_title("Map 19 Components"); ax6.set_xlabel("Time (s)"); ax6.set_ylabel("Position (m)")
ax6.legend(); ax6.grid(True, alpha=0.3)

plt.tight_layout()
out = f"{base}/trajectories_viz.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.show()
