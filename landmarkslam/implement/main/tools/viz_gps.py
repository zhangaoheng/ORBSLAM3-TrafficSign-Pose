#!/usr/bin/env python3
"""GPS 轨迹可视化 — 经纬度 → 局部 ENU 坐标"""
import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def llh_to_enu(lat, lon, alt, lat0, lon0, alt0=0):
    """经纬度转 ENU（以第一个点为原点）"""
    a = 6378137.0; f = 1/298.257223563
    e2 = 2*f - f*f
    sin_lat0 = np.sin(np.radians(lat0))
    N0 = a / np.sqrt(1 - e2 * sin_lat0**2)
    x0 = (N0 + alt0) * np.cos(np.radians(lat0)) * np.cos(np.radians(lon0))
    y0 = (N0 + alt0) * np.cos(np.radians(lat0)) * np.sin(np.radians(lon0))
    z0 = (N0*(1-e2) + alt0) * np.sin(np.radians(lat0))

    sin_lat = np.sin(np.radians(lat))
    N = a / np.sqrt(1 - e2 * sin_lat**2)
    x = (N + alt) * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    y = (N + alt) * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    z = (N*(1-e2) + alt) * np.sin(np.radians(lat))

    dx, dy, dz = x - x0, y - y0, z - z0
    # 旋转到 ENU
    slat = np.sin(np.radians(lat0))
    clat = np.cos(np.radians(lat0))
    slon = np.sin(np.radians(lon0))
    clon = np.cos(np.radians(lon0))
    e = -slon * dx + clon * dy
    n = -slat * clon * dx - slat * slon * dy + clat * dz
    u =  clat * clon * dx + clat * slon * dy + slat * dz
    return e, n, u

# 读取 CSV
path = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/test/gps_record_20260529_113616.csv"
ts, lats, lons, alts = [], [], [], []
with open(path) as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < 5: continue
        try:
            ts.append(float(parts[0]))
            lats.append(float(parts[1]))
            lons.append(float(parts[2]))
            alts.append(float(parts[3]))
        except: pass

ts = np.array(ts); lats = np.array(lats); lons = np.array(lons); alts = np.array(alts)

# 只保留有 GPS fix 的点 (fix_quality > 0)
# 这里简化：用高度 > 0 的点（粗过滤）
valid = alts > 0
print(f"Total: {len(ts)}, with alt>0: {np.sum(valid)}")

# 转 ENU
lat0, lon0, alt0 = lats[valid][0], lons[valid][0], alts[valid][0]
e, n, u = llh_to_enu(lats[valid], lons[valid], alts[valid], lat0, lon0, alt0)
dist = np.sum(np.sqrt(np.diff(e)**2 + np.diff(n)**2))
dur = ts[valid][-1] - ts[valid][0]
print(f"Dist: {dist/1000:.2f} km, Duration: {dur:.1f}s, Speed: {dist/dur*3.6:.1f} km/h")

# 可视化
fig = plt.figure(figsize=(16, 10))
fig.suptitle("GPS Trajectory (ENU Coordinates)", fontsize=14, fontweight="bold")

ax1 = fig.add_subplot(2, 2, 1, projection="3d")
tn = (ts[valid] - ts[valid][0]) / (ts[valid][-1] - ts[valid][0] + 1e-9)
ax1.scatter(e, n, u, c=tn, cmap="plasma", s=5, alpha=0.8)
ax1.plot(e, n, u, "gray", alpha=0.3)
ax1.set_title("3D GPS Track"); ax1.set_xlabel("East (m)"); ax1.set_ylabel("North (m)"); ax1.set_zlabel("Up (m)")

ax2 = fig.add_subplot(2, 2, 2)
ax2.scatter(e, n, c=tn, cmap="plasma", s=8, alpha=0.7)
ax2.plot(e, n, "gray", alpha=0.3)
ax2.scatter(e[0], n[0], c="green", s=100, marker="o", label="Start")
ax2.scatter(e[-1], n[-1], c="red", s=100, marker="x", label="End")
ax2.set_title("Top View (EN)"); ax2.set_xlabel("East (m)"); ax2.set_ylabel("North (m)")
ax2.legend(); ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(2, 2, 3)
tr = ts[valid] - ts[valid][0]
ax3.plot(tr, e, label="East", linewidth=0.8)
ax3.plot(tr, n, label="North", linewidth=0.8)
ax3.set_title("Position vs Time"); ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Position (m)")
ax3.legend(); ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(tr, u, "g-", label="Altitude", linewidth=0.8)
ax4.set_title("Altitude vs Time"); ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Height (m)")
ax4.legend(); ax4.grid(True, alpha=0.3)

plt.tight_layout()
out = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/test/gps_trajectory.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.show()
