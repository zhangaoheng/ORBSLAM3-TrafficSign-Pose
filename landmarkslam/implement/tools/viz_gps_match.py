#!/usr/bin/env python3
"""GPS trajectory + 2 matched points on same plot"""
import numpy as np, csv
from math import cos, radians, hypot
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

with open('test1.csv') as f:
    rows = [r for r in csv.reader(f)]
gt  = np.array([float(r[0]) for r in rows[1:]])
gla = np.array([float(r[1]) for r in rows[1:]])
glo = np.array([float(r[2]) for r in rows[1:]])

t1 = 1782374465850840320 / 1e9
t2 = 1782375843313394432 / 1e9
i1 = np.argmin(np.abs(gt - t1))
i2 = np.argmin(np.abs(gt - t2))

# 两帧附近 ±60s 的 GPS
w = int(60 / 0.063)
i1s, i1e = max(i1-w,0), min(i1+w, len(gt)-1)
i2s, i2e = max(i2-w,0), min(i2+w, len(gt)-1)

fig, ax = plt.subplots(figsize=(12, 10))
ax.plot(glo, gla, 'lightgray', linewidth=0.3, alpha=0.5, label='Full GPS track')
ax.plot(glo[i1s:i1e], gla[i1s:i1e], 'blue', linewidth=0.8, label='Frame 158 vicinity')
ax.plot(glo[i2s:i2e], gla[i2s:i2e], 'green', linewidth=0.8, label='Frame 254 vicinity')
ax.scatter([glo[i1], glo[i2]], [gla[i1], gla[i2]], c=['red','red'], s=200, zorder=10, edgecolors='k')
ax.text(glo[i1], gla[i1], f'  F158 GPS[{i1+2}]', fontsize=10, color='red', fontweight='bold')
ax.text(glo[i2], gla[i2], f'  F254 GPS[{i2+2}]', fontsize=10, color='red', fontweight='bold')

d = hypot((gla[i2]-gla[i1])*111320, (glo[i2]-glo[i1])*111320*cos(radians(gla[i1])))
ax.set_title(f'GPS Track  |  Frame 158 vs 254  |  dt={gt[i2]-gt[i1]:.0f}s  dist={d:.2f}m')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
ax.legend(); ax.axis('equal'); plt.tight_layout(); plt.show()
