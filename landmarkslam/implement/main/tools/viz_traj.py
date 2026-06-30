#!/usr/bin/env python3
"""轨迹可视化 — 用法: python3 viz_traj.py <traj.txt>"""
import sys, numpy as np, matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else 'CameraTrajectory.txt'
data = np.loadtxt(path)
x, y, z = data[:,1], data[:,2], data[:,3]
dist = np.sum(np.linalg.norm(np.diff(data[:,1:4], axis=0), axis=1))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(x, y, 'b-', linewidth=0.5)
ax1.set_title(f'{path}\n{len(data)} KFs, {dist:.0f}m')
ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.axis('equal')
ax2.plot(range(len(x)), z, 'g-', linewidth=0.5)
ax2.set_title('Height'); ax2.set_xlabel('KF index'); ax2.set_ylabel('Z (m)')
plt.tight_layout(); plt.show()
