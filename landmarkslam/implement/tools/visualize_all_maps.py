#!/home/zah/ORB_SLAM3-master/landmarkslam/yolo_venv/bin/python3
"""可视化所有地图轨迹 + 拼接轨迹"""
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

# 加载所有地图
map_files = sorted([f for f in os.listdir(base) if f.startswith("map_") and f.endswith("_trajectory.txt")])
maps = []
for mf in map_files:
    t, p = load_traj(os.path.join(base, mf))
    mid = int(mf.split("_")[1])
    maps.append((mid, t, p))
    dist = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
    print(f"  地图{mid}: {len(t)} KFs, 路程{dist:.1f}m")

# 加载拼接结果
stitch_file = os.path.join(base, "stitched_trajectory.txt")
if os.path.exists(stitch_file):
    st, sp = load_traj(stitch_file)
    print(f"  拼接后: {len(sp)} KFs, 路程{np.sum(np.linalg.norm(np.diff(sp,axis=0),axis=1)):.1f}m")

# ========================
# 图1: 7个地图独立显示 + 拼接图
# ========================
fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw={"projection": "3d"})
fig.suptitle("7 个地图的独立轨迹 (mono 模式)", fontsize=14)

colors = plt.cm.plasma(np.linspace(0, 1, len(maps)))

for i, (mid, t, p) in enumerate(maps):
    ax = axes[i // 4][i % 4]
    t_norm = (t - t[0]) / (t[-1] - t[0] + 1e-9)
    ax.scatter(p[:,0], p[:,1], p[:,2], c=t_norm, cmap="plasma", s=5, alpha=0.8)
    ax.plot(p[:,0], p[:,1], p[:,2], "gray", alpha=0.3, linewidth=0.5)
    ax.set_title(f"地图 {mid} ({len(t)} KFs)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

# 第8个子图：拼接轨迹
ax = axes[1][3]
if os.path.exists(stitch_file):
    st_norm = (st - st[0]) / (st[-1] - st[0] + 1e-9)
    ax.scatter(sp[:,0], sp[:,1], sp[:,2], c=st_norm, cmap="plasma", s=3, alpha=0.8)
    ax.plot(sp[:,0], sp[:,1], sp[:,2], "gray", alpha=0.3, linewidth=0.5)
    ax.set_title(f"拼接后 ({len(sp)} KFs)")
else:
    ax.text(0.5, 0.5, "无拼接数据", transform=ax.transAxes)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

plt.tight_layout()
plt.savefig(os.path.join(base, "all_maps_3d.png"), dpi=150, bbox_inches="tight")
print(f"\n✅ 已保存: all_maps_3d.png")

# ========================
# 图2: XY 俯视图 - 所有地图叠加 + 拼接
# ========================
fig2, ax2 = plt.subplots(figsize=(14, 10))
fig2.suptitle("所有地图 XY 俯视图 + 拼接结果", fontsize=14)

for i, (mid, t, p) in enumerate(maps):
    ax2.scatter(p[:,0], p[:,1], c=[colors[i]], s=8, alpha=0.6, label=f"地图 {mid}")
    ax2.plot(p[:,0], p[:,1], color=colors[i], alpha=0.3, linewidth=0.5)

if os.path.exists(stitch_file):
    ax2.plot(sp[:,0], sp[:,1], "k-", linewidth=1.5, alpha=0.8, label="拼接轨迹")
    ax2.scatter(sp[0,0], sp[0,1], c="green", s=100, marker="o", label="起点")
    ax2.scatter(sp[-1,0], sp[-1,1], c="red", s=100, marker="x", label="终点")

ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.legend(fontsize=8)
ax2.set_aspect("equal", adjustable="datalim")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base, "all_maps_xy.png"), dpi=150, bbox_inches="tight")
print(f"✅ 已保存: all_maps_xy.png")

# ========================
# 图3: 拼接轨迹详细分析
# ========================
if os.path.exists(stitch_file) and len(sp) > 10:
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10))
    fig3.suptitle("拼接轨迹详细分析", fontsize=14)
    
    # 3D
    ax = axes3[0][0]
    ax = fig3.add_subplot(2, 2, 1, projection="3d")
    st_norm = (st - st[0]) / (st[-1] - st[0] + 1e-9)
    ax.scatter(sp[:,0], sp[:,1], sp[:,2], c=st_norm, cmap="plasma", s=3, alpha=0.8)
    ax.plot(sp[:,0], sp[:,1], sp[:,2], "gray", alpha=0.3)
    ax.set_title("拼接 3D 轨迹"); ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    
    # XY
    ax = axes3[0][1]
    ax.scatter(sp[:,0], sp[:,1], c=st_norm, cmap="plasma", s=5, alpha=0.8)
    ax.plot(sp[:,0], sp[:,1], "gray", alpha=0.3)
    ax.set_title("XY 俯视图"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(True, alpha=0.3)
    
    # XZ
    ax = axes3[1][0]
    ax.scatter(sp[:,0], sp[:,2], c=st_norm, cmap="plasma", s=5, alpha=0.8)
    ax.plot(sp[:,0], sp[:,2], "gray", alpha=0.3)
    ax.set_title("XZ 侧视图"); ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.grid(True, alpha=0.3)
    
    # 分量
    ax = axes3[1][1]
    t_rel = st - st[0]
    ax.plot(t_rel, sp[:,0], label="X", linewidth=0.8)
    ax.plot(t_rel, sp[:,1], label="Y", linewidth=0.8)
    ax.plot(t_rel, sp[:,2], label="Z", linewidth=0.8)
    ax.set_title("位置分量 vs 时间"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (m)")
    ax.legend(); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base, "stitched_analysis.png"), dpi=150, bbox_inches="tight")
    print(f"✅ 已保存: stitched_analysis.png")

print(f"\n🎨 正在显示交互窗口（关闭窗口退出）...")
plt.show()
