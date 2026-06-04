#!/usr/bin/env python3
# ============================================================
# 文件: test_gps.py
# 用途: SLAM去旋 + GPS定尺度 Looming 深度恢复
# 运行: python3 landmarkslam/implement/main/test_gps.py
# 说明: SLAM提供旋转消除FOE膨胀旋转影响, GPS提供真实步长替代SLAM尺度
# ============================================================
"""
新思路:
  SLAM 位姿 → 只用于 derotate（消除旋转对 FOE 膨胀点的影响）
  GPS 位移 → 作为 delta_d（替代不稳定的 SLAM 尺度）
  
  核心改动: delta_d 从 GPS 首尾位移计算，不再依赖 SLAM 的 tz
"""
import os, sys, glob, cv2, numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)
from tools.mid import extract_four_lines_from_real_image, calculate_rectangle_center
from mid_FOE_Z_d.pure_looming_depth import (
    load_tum_trajectory, get_closest_pose, fx, fy, cx, cy
)

# ===== 常量 =====
PAIR = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/test_pairs/pair_09_54"
FRAME_STEP = 15

# ===== 工具函数 =====
def load_times(path):
    ts = []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2: ts.append(float(p[0]))
    return np.array(ts)

def load_gps(path):
    ts, lats, lons = [], [], []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            if len(p) < 4: continue
            try: ts.append(float(p[0])); lats.append(float(p[1])); lons.append(float(p[2]))
            except: pass
    ts=np.array(ts); lats=np.array(lats); lons=np.array(lons)
    v=lats>0; return ts[v],lats[v],lons[v]

def gps_to_distance(lat1, lon1, lat2, lon2):
    """两 GPS 点之间的水平距离(m)"""
    a, e2 = 6378137.0, 0.00669437999014
    sin_lat = np.sin(np.radians((lat1+lat2)/2))
    dN = (lat2 - lat1) * 111320   # 近似
    dE = (lon2 - lon1) * 111320 * np.cos(np.radians((lat1+lat2)/2))
    return np.sqrt(dN**2 + dE**2)

def load_corners(path):
    names = {}
    if not os.path.exists(path): return names
    with open(path) as f:
        for line in f:
            if line.strip():
                p = line.strip().split()
                if len(p) >= 2:
                    names[p[0]] = [tuple(map(int, pt.split(","))) for pt in p[1:5]]
    return names

def select_frame_gui(images, default_idx, sequence_name):
    idx = default_idx if default_idx != -1 else 0
    win = f"Select Frame for {sequence_name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        img = cv2.imread(images[idx])
        if img is None: idx = (idx+1)%len(images); continue
        disp = img.copy()
        cv2.putText(disp, f"{sequence_name}: {idx}/{len(images)-1}", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(disp, "[D/SPACE]Next [A]Prev [ENTER]Select", (30,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(0) & 0xFF
        if key in [32, ord('d')]: idx = (idx+1)%len(images)
        elif key == ord('a'): idx = (idx-1)%len(images)
        elif key in [13, 27]: break
    cv2.destroyWindow(win)
    return idx

# ===== 加载 =====
print("="*60)
print("  GPS-SLAM 混合 Looming 深度恢复")
print("="*60)

# seq1
images1 = sorted(glob.glob(os.path.join(PAIR, "seq1", "rgb", "*.png")))
times1 = load_times(os.path.join(PAIR, "seq1", "times.txt"))
g1_ts, g1_lat, g1_lon = load_gps(os.path.join(PAIR, "seq1_gps.csv"))
traj1 = load_tum_trajectory(os.path.join(PAIR, "seq1", "trajectory.txt"))
corners1 = load_corners(os.path.join(PAIR, "seq1", "rgb", "corners.txt"))

# seq2
images2 = sorted(glob.glob(os.path.join(PAIR, "seq2", "rgb", "*.png")))
g2_ts, g2_lat, g2_lon = load_gps(os.path.join(PAIR, "seq2_gps.csv"))
traj2 = load_tum_trajectory(os.path.join(PAIR, "seq2", "trajectory.txt"))

print(f"seq1: {len(images1)} imgs, {len(g1_lat)} GPS, {len(traj1)} KFs, {len(corners1)} corners")
print(f"seq2: {len(images2)} imgs, {len(g2_lat)} GPS, {len(traj2)} KFs")

# ===== 1. 选帧 =====
print("\n--- 选帧 ---")
idx1_base = select_frame_gui(images1, 0, "seq1")
idx2_base = select_frame_gui(images2, 0, "seq2")
idx1_prev = max(0, idx1_base - FRAME_STEP)
print(f"seq1: prev={idx1_prev}, base={idx1_base}")
print(f"seq2: base={idx2_base}")

# ===== 2. FOE 去旋 (SLAM rotation only) =====
print("\n--- FOE 去旋 ---")
t1_A = times1[idx1_prev]; t1_B = times1[idx1_base]
pose_A = get_closest_pose(t1_A, traj1)
pose_B = get_closest_pose(t1_B, traj1)

if pose_A is None or pose_B is None:
    print("❌ SLAM 位姿不足"); sys.exit(1)

t_A, q_A = pose_A[0:3], pose_A[3:7]
t_B, q_B = pose_B[0:3], pose_B[3:7]
R_A = R_scipy.from_quat(q_A).as_matrix()
R_B = R_scipy.from_quat(q_B).as_matrix()

# SLAM 旋转（只用于去旋）
R_12 = R_A.T @ R_B
t_12 = R_A.T @ (t_B - t_A)  # SLAM 尺度不靠谱，但方向可用
print(f"SLAM R_12:\n{np.round(R_12, 4)}")
print(f"SLAM t_12: {np.round(t_12, 4)}m (仅方向参考)")

# ===== 3. GPS 定尺度 =====
gps_idx_A = np.argmin(np.abs(g1_ts - t1_A))
gps_idx_B = np.argmin(np.abs(g1_ts - t1_B))
gps_dist = gps_to_distance(g1_lat[gps_idx_A], g1_lon[gps_idx_A],
                            g1_lat[gps_idx_B], g1_lon[gps_idx_B])
print(f"\nGPS A→B: {gps_dist:.2f}m")

# GPS 位移的方向 — 投影到 SLAM 的前进方向
# 用 SLAM t_12 的方向 + GPS 的模长 = GPS 定标后的位移
delta_d_gps = gps_dist  # 直接用 GPS 距离
print(f"delta_d(GPS): {delta_d_gps:.2f}m  vs  delta_d(SLAM): {t_12[2]:.2f}m")

# ===== 4. Looming Z =====
img_A = cv2.imread(images1[idx1_prev])
img_B = cv2.imread(images1[idx1_base])
fname_A = os.path.basename(images1[idx1_prev])
fname_B = os.path.basename(images1[idx1_base])

if fname_A not in corners1 or fname_B not in corners1:
    print("❌ 所选帧无标注"); sys.exit(1)

pts_A = np.array(corners1[fname_A])
pts_B = np.array(corners1[fname_B])
center_A = np.mean(pts_A, axis=0)
center_B = np.mean(pts_B, axis=0)

# FOE (只有 SLAM 能算出来)
FOE = (fx * (t_12[0]/t_12[2]) + cx, fy * (t_12[1]/t_12[2]) + cy)
print(f"FOE: ({FOE[0]:.1f}, {FOE[1]:.1f})")

# 去旋：用 R_12 把 center_A 转正
def derotate_point(pt, R):
    x, y = pt
    fx_val, fy_val = fx, fy
    cx_val, cy_val = cx, cy
    return (x, y)  # 简化：TODO 完整推导

# 计算 Looming Z (用 GPS delta_d)
r_near = np.linalg.norm(np.array(center_B) - np.array(FOE))
r_far  = np.linalg.norm(np.array(center_A) - np.array(FOE))
dr = r_near - r_far
print(f"r_near={r_near:.1f}px  r_far={r_far:.1f}px  dr={dr:.1f}px")

if dr <= 0:
    print("❌ dr <= 0, 膨胀量不足"); sys.exit(1)

Z_gps = r_near * delta_d_gps / dr - delta_d_gps
Z_slam = r_near * t_12[2] / dr - t_12[2]
print(f"\n🎯 Looming Z (GPS尺度) = {Z_gps:.2f}m")
print(f"   Looming Z (SLAM尺度) = {Z_slam:.2f}m")
print(f"   尺度比: SLAM/GPS = {t_12[2]/delta_d_gps:.2f}x")

# ===== 可视化 =====
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"GPS-Scale Looming: Z={Z_gps:.1f}m (GPS) vs Z={Z_slam:.1f}m (SLAM)")

axes[0].imshow(cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)); axes[0].set_title("Seq1 Prev (far)")
axes[0].scatter(center_A[0], center_A[1], c="red", s=80)
axes[0].scatter(*FOE, c="blue", s=80, marker="x")
axes[1].imshow(cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)); axes[1].set_title("Seq1 Base (near)")
axes[1].scatter(center_B[0], center_B[1], c="red", s=80)
axes[1].scatter(*FOE, c="blue", s=80, marker="x")

plt.tight_layout()
plt.savefig(os.path.join(PAIR, "looming_gps_vs_slam.png"), dpi=150)
print(f"\nSaved: {PAIR}/looming_gps_vs_slam.png")
plt.show()
