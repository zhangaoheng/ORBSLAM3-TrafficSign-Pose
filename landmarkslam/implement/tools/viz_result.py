#!/home/zah/ORB_SLAM3-master/landmarkslam/yolo_venv/bin/python3
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data/runs/2026-06-01_15-58-17_rgbd"

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

map_files = sorted([f for f in os.listdir(base) if f.startswith("map_") and f.endswith("_trajectory.txt")],
                   key=lambda x: int(x.split("_")[1]))

colors = plt.cm.tab20(np.linspace(0, 1, 20))
fig, ax = plt.subplots(figsize=(16, 10))

for i, mf in enumerate(map_files):
    if i >= 60: break
    t, p = load_traj(os.path.join(base, mf))
    mid = int(mf.split("_")[1])
    c = colors[i % 20]
    ax.plot(p[:,0], p[:,1], color=c, linewidth=0.8, alpha=0.6)

main_traj = os.path.join(base, "trajectory.txt")
if os.path.exists(main_traj):
    _, mp = load_traj(main_traj)
    ax.plot(mp[:,0], mp[:,1], "k-", linewidth=2, alpha=0.4, label="Main")

ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
ax.set_aspect("equal", adjustable="datalim")
ax.grid(True, alpha=0.3)
plt.suptitle(f"RGB-D: {len(map_files)} Maps, 13.6km", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(base, "all_maps_overlay.png"), dpi=150, bbox_inches="tight")
print(f"Saved: all_maps_overlay.png")
