#!/home/zah/ORB_SLAM3-master/landmarkslam/yolo_venv/bin/python3
"""为每段地图生成独立的轨迹可视化图"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
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

# ===== 配置 =====
run_dir = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data/runs/2026-06-01_15-58-17_rgbd"
# ================

# 读取 maps_summary.txt
summary = {}
summary_path = os.path.join(run_dir, "maps_summary.txt")
if os.path.exists(summary_path):
    with open(summary_path) as f:
        for line in f:
            if line.startswith("map_id") or not line.strip(): continue
            parts = line.split()
            if len(parts) >= 5:
                mid = int(parts[0])
                summary[mid] = {"start": int(parts[1]), "end": int(parts[2]),
                                "kfs": int(parts[3]), "dist": float(parts[4])}

# 创建输出目录
out_dir = os.path.join(run_dir, "map_viz")
os.makedirs(out_dir, exist_ok=True)

# 获取所有地图文件
map_files = sorted([f for f in os.listdir(run_dir) if f.startswith("map_") and f.endswith("_trajectory.txt")],
                   key=lambda x: int(x.split("_")[1]))

print(f"Generating {len(map_files)} map visualizations...")

for i, mf in enumerate(map_files):
    mid = int(mf.split("_")[1])
    t, p = load_traj(os.path.join(run_dir, mf))
    if len(p) < 3:
        continue
    
    info = summary.get(mid, {})
    dist = info.get("dist", np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    start_f = info.get("start", "?")
    end_f = info.get("end", "?")
    kfs = info.get("kfs", len(p))
    
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"Map {mid}: Frames {start_f} - {end_f}  ({kfs} KFs, {dist:.1f}m)",
                 fontsize=13, fontweight="bold")
    
    # 3D
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    tn = (t - t[0]) / (t[-1] - t[0] + 1e-9)
    ax1.scatter(p[:,0], p[:,1], p[:,2], c=tn, cmap="plasma", s=5, alpha=0.8)
    ax1.plot(p[:,0], p[:,1], p[:,2], "gray", alpha=0.3)
    ax1.set_title("3D View")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    
    # XY
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(p[:,0], p[:,1], c=tn, cmap="plasma", s=8, alpha=0.7)
    ax2.plot(p[:,0], p[:,1], "gray", alpha=0.3)
    ax2.scatter(p[0,0], p[0,1], c="green", s=50, marker="o", label="Start")
    ax2.scatter(p[-1,0], p[-1,1], c="red", s=50, marker="x", label="End")
    ax2.set_title("XY View"); ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    ax2.legend(); ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)
    
    # XZ
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(p[:,0], p[:,2], c=tn, cmap="plasma", s=8, alpha=0.7)
    ax3.plot(p[:,0], p[:,2], "gray", alpha=0.3)
    ax3.set_title("XZ Side View"); ax3.set_xlabel("X (m)"); ax3.set_ylabel("Z (m)")
    ax3.grid(True, alpha=0.3)
    
    # 信息
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    lines = [
        f"Map ID: {mid}",
        f"Frame range: {start_f} - {end_f}",
        f"Total frames in range: {end_f - start_f if isinstance(end_f, int) else '?'}",
        f"Keyframes: {kfs}",
        f"Distance: {dist:.1f} m",
        f"Duration: {t[-1]-t[0]:.1f} s",
        f"Avg speed: {dist/(t[-1]-t[0]+0.01)*3.6:.1f} km/h",
    ]
    ax4.text(0.1, 0.9, "\n".join(lines), fontsize=11, fontfamily="monospace",
             verticalalignment="top", transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"map_{mid:02d}.png"), dpi=120, bbox_inches="tight")
    plt.close()
    
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(map_files)}...")

print(f"\n✅ Done! {len(map_files)} images saved to: {out_dir}")
