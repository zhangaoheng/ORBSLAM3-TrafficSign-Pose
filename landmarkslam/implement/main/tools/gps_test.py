#!/usr/bin/env python3
# ============================================================
# 文件: gps_test.py
# 用途: GPS-SLAM 对齐可视化，标注帧 & 特征匹配帧显示
# 运行: python3 landmarkslam/implement/main/tools/gps_test.py
# 示例: python3 landmarkslam/implement/main/tools/gps_test.py
# ============================================================
import os, sys, glob, numpy as np

BASE = "/home/zah/ORB_SLAM3-master"
PAIRS_DIR = os.path.join(BASE, "landmarkslam/implement/data/test_pairs")

# ===== 参数（改这里） =====
PAIR_NAME = "pair_09_54"
MANUAL_SCALE = (6.0, 2.2)   # (seq1倍率, seq2倍率), (0,0)=自动
MATCH_FRAMES = "auto"  # "auto"=自动检测标注帧, 或 [634,649,811] 手动指定
MIN_STEP = 20                # 方向计算最小有效位移(m)
# =========================

def load_times(path):
    ts, names = [], []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2: ts.append(float(p[0])); names.append(p[1].replace("rgb/",""))
    return np.array(ts), names

def load_gps(path):
    if not os.path.exists(path): return None, None, None
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

def llh_to_enu(lats, lons):
    lat0,lon0=lats[0],lons[0]; a=6378137.0; e2=0.00669437999014
    sin_lat0=np.sin(np.radians(lat0)); N0=a/np.sqrt(1-e2*sin_lat0**2)
    x0=N0*np.cos(np.radians(lat0))*np.cos(np.radians(lon0))
    y0=N0*np.cos(np.radians(lat0))*np.sin(np.radians(lon0))
    z0=N0*(1-e2)*np.sin(np.radians(lat0))
    sin_lat=np.sin(np.radians(lats)); N=a/np.sqrt(1-e2*sin_lat**2)
    x=N*np.cos(np.radians(lats))*np.cos(np.radians(lons))
    y=N*np.cos(np.radians(lats))*np.sin(np.radians(lons))
    z=N*(1-e2)*np.sin(np.radians(lats))
    dx,dy,dz=x-x0,y-y0,z-z0
    sl,cl=np.sin(np.radians(lat0)),np.cos(np.radians(lat0))
    so,co=np.sin(np.radians(lon0)),np.cos(np.radians(lon0))
    e=-so*dx+co*dy; n=-sl*co*dx-sl*so*dy+cl*dz
    return e,n

def load_traj(path):
    if not os.path.exists(path): return None, None
    ts, xy = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            p = line.split()
            if len(p) >= 8: ts.append(float(p[0])); xy.append([float(p[1]), float(p[2])])
    return np.array(ts), np.array(xy)

def load_corners(path):
    if not os.path.exists(path): return set()
    names = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                names.add(line.strip().split()[0])
    return names

def process_seq(times_path, gps_path, traj_path, corners_path, title, match_frame=None, manual_scale=0):
    seq_ts, img_names = load_times(times_path)
    g1_ts, g1_lat, g1_lon = load_gps(gps_path)
    traj_ts, traj_xy = load_traj(traj_path)
    corners = load_corners(corners_path)
    
    if any(x is None for x in [g1_ts, traj_ts]):
        return None
    
    g1_e, g1_n = llh_to_enu(g1_lat, g1_lon)
    
    # 找有效步长
    gps_i0 = np.argmin(np.abs(g1_ts - seq_ts[0]))
    trj_i0 = np.argmin(np.abs(traj_ts - seq_ts[0]))
    trj_iN = trj_i0 + 1
    while trj_iN < len(traj_xy):
        step = np.sqrt((traj_xy[trj_iN,0]-traj_xy[trj_i0,0])**2 +
                       (traj_xy[trj_iN,1]-traj_xy[trj_i0,1])**2)
        if step >= MIN_STEP: break
        trj_iN += 1
    if trj_iN >= len(traj_xy): return None
    gps_iN = np.argmin(np.abs(g1_ts - traj_ts[trj_iN]))
    
    # 算旋转
    gps_dE = g1_e[gps_iN]-g1_e[gps_i0]; gps_dN = g1_n[gps_iN]-g1_n[gps_i0]
    slam_dX = traj_xy[trj_iN,0]-traj_xy[trj_i0,0]; slam_dZ = traj_xy[trj_iN,1]-traj_xy[trj_i0,1]
    gps_head = np.degrees(np.arctan2(gps_dN, gps_dE))
    slam_head = np.degrees(np.arctan2(slam_dZ, slam_dX))
    offset = gps_head - slam_head
    while offset > 180: offset -= 360
    while offset < -180: offset += 360
    theta = np.radians(offset)
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    
    gps_aligned = np.column_stack([g1_e - g1_e[gps_i0], g1_n - g1_n[gps_i0]])
    slam_aligned = (R @ (traj_xy - traj_xy[trj_i0]).T).T
    gps_len = np.sqrt(gps_aligned[-1,0]**2 + gps_aligned[-1,1]**2)
    slam_len = np.sqrt(slam_aligned[-1,0]**2 + slam_aligned[-1,1]**2)
    scale = manual_scale if manual_scale > 0 else gps_len / max(slam_len, 1)
    
    # 标注帧索引（在图片序列中）
    corners_idx = sorted([i for i, n in enumerate(img_names) if n in corners])
    # 标注帧对应的 GPS 索引
    corners_gps = [np.argmin(np.abs(g1_ts - seq_ts[i])) for i in corners_idx]
    # 标注帧对应的 SLAM 索引
    corners_slam = [np.argmin(np.abs(traj_ts - seq_ts[i])) for i in corners_idx]
    
    # 匹配帧：auto=取标注帧中间作为 base
    if match_frame is None and corners_idx:
        mid = corners_idx[len(corners_idx)//2]
        match_frame = [max(0, mid - 15), mid]  # prev + base
    
    match_info = []
    if match_frame is not None:
        mf_list = match_frame if isinstance(match_frame, list) else [match_frame]
        for mf in mf_list:
            if mf is not None:
                mf_ts = seq_ts[mf]
                mf_gps = np.argmin(np.abs(g1_ts - mf_ts))
                mf_slam = np.argmin(np.abs(traj_ts - mf_ts))
                match_info.append({"frame": mf, "gps": mf_gps, "slam": mf_slam})
    
    return {
        "title": title, "gps": gps_aligned, "slam": slam_aligned * scale,
        "gps_len": gps_len, "slam_len": slam_len*scale, "scale": scale,
        "offset": offset, "step": step,
        "gps_i0": gps_i0, "trj_i0": trj_i0,
        "corners_idx": corners_idx, "corners_gps": corners_gps, "corners_slam": corners_slam,
        "match": match_info
    }

# ===== main =====
pairs = sorted(glob.glob(os.path.join(PAIRS_DIR, PAIR_NAME)))
if not pairs:
    print(f"❌ 找不到: {PAIR_NAME}")
    sys.exit(1)

# 匹配帧："auto" 或手动指定
if MATCH_FRAMES == "auto":
    match_frames = [None, None]  # 后面 process_seq 内处理
else:
    match_frames = [[MATCH_FRAMES[0], MATCH_FRAMES[1]],
                     [MATCH_FRAMES[2]]]

results = []
for pair_dir in pairs:
    name = os.path.basename(pair_dir)
    for seq_name, mf, ms in [("seq1", match_frames[0], MANUAL_SCALE[0]),
                              ("seq2", match_frames[1], MANUAL_SCALE[1])]:
        r = process_seq(
            os.path.join(pair_dir, "seq1" if seq_name=="seq1" else "seq2", "times.txt"),
            os.path.join(pair_dir, f"{seq_name}_gps.csv"),
            os.path.join(pair_dir, "seq1" if seq_name=="seq1" else "seq2", "trajectory.txt"),
            os.path.join(pair_dir, "seq1" if seq_name=="seq1" else "seq2", "rgb", "corners.txt"),
            f"{name}/{seq_name}",
            match_frame=mf, manual_scale=ms
        )
        if r:
            results.append(r)
            n_corners = len(r["corners_idx"])
            m = r["match"]
            if m:
                m_str = "match_fr=" + ",".join(str(x["frame"]) for x in m)
            else:
                m_str = ""
            print(f"  {r['title']:20s} offset={r['offset']:6.1f}°  scale={r['scale']:.1f}x  "
                  f"corners={n_corners}  {m_str}")

import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

for r in results:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(r["gps"][:,0], r["gps"][:,1], "b-", linewidth=2, alpha=0.6, label=f"GPS ({r['gps_len']:.0f}m)")
    ax.plot(r["slam"][:,0], r["slam"][:,1], "r-", linewidth=2, alpha=0.6, label=f"SLAM x{r['scale']:.1f}")
    
    # 标注帧（绿色小点）
    if r["corners_gps"]:
        for gi in r["corners_gps"]:
            ax.scatter(r["gps"][gi,0], r["gps"][gi,1], c="green", s=8, alpha=0.5, zorder=5)
        for si in r["corners_slam"]:
            ax.scatter(r["slam"][si,0], r["slam"][si,1], c="lime", s=8, alpha=0.5, zorder=5)
    
    # 匹配帧（紫色菱形）
    for i, m in enumerate(r["match"]):
        label = f"MATCH {['prev','base','base2'][i] if i<3 else i}"
        ax.scatter(r["gps"][m["gps"],0], r["gps"][m["gps"],1], c="magenta", s=250, marker="D",
                   edgecolors="black", linewidth=2, zorder=10)
        ax.annotate(f"{label} fr{m['frame']}", r["gps"][m["gps"]],
                    textcoords="offset points", xytext=(10, -20), fontsize=9,
                    color="magenta", fontweight="bold")
        ax.scatter(r["slam"][m["slam"],0], r["slam"][m["slam"],1], c="magenta", s=200, marker="D",
                   edgecolors="black", linewidth=2, zorder=10)
    
    # 起终点
    for idx, label, color in [(0, "S", "green"), (len(r["gps"])-1, "E", "red")]:
        ax.scatter(r["gps"][idx,0], r["gps"][idx,1], c=color, s=150, marker="o",
                   edgecolors="black", linewidth=2, zorder=10)
        ax.annotate(label, r["gps"][idx], textcoords="offset points",
                    xytext=(10, -15), fontsize=10, color=color, fontweight="bold")
    
    title = f"{r['title']}  offset={r['offset']:.1f}°"
    if r["match"]: title += "  match_fr=" + ",".join(str(x["frame"]) for x in r["match"])
    ax.set_title(title)
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_aspect("equal"); ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)

print("\n✅ Done")
plt.show()
