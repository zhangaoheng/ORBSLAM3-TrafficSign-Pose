#!/usr/bin/env python3
"""GPS 真值修正：手动调整 seq2 匹配点，使 GPS 距离匹配视觉结果"""
import numpy as np, csv, os, sys
from math import cos, radians, hypot
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else 'test1.csv'
T1 = float(sys.argv[2]) if len(sys.argv) > 2 else None  # seq1 图片时间戳(ns)
T2 = float(sys.argv[3]) if len(sys.argv) > 3 else None  # seq2 图片时间戳(ns)

with open(CSV_PATH) as f:
    rows = [r for r in csv.reader(f)]
gt  = np.array([float(r[0]) for r in rows[1:]])
gla = np.array([float(r[1]) for r in rows[1:]])
glo = np.array([float(r[2]) for r in rows[1:]])

# 默认用已知的两帧
if T1 is None: T1 = 1782374465850840320 / 1e9
if T2 is None: T2 = 1782375843313394432 / 1e9

i1 = np.argmin(np.abs(gt - T1))
i2 = np.argmin(np.abs(gt - T2))

fig, ax = plt.subplots(figsize=(14, 10))
w = int(60 / 0.063)
i1s, i1e = max(i1-w,0), min(i1+w, len(gt)-1)
i2s, i2e = max(i2-w,0), min(i2+w, len(gt)-1)

ax.plot(glo, gla, 'lightgray', linewidth=0.3, alpha=0.5, label='Full GPS')
ax.plot(glo[i1s:i1e], gla[i1s:i1e], 'blue', linewidth=0.8, label='Seq1 vicinity')
line_seq2, = ax.plot(glo[i2s:i2e], gla[i2s:i2e], 'green', linewidth=0.8, label='Seq2 vicinity')
pt1 = ax.scatter([glo[i1]], [gla[i1]], c='red', s=200, zorder=10, edgecolors='k')
pt2 = ax.scatter([glo[i2]], [gla[i2]], c='orange', s=200, zorder=10, edgecolors='k', marker='s')
txt1 = ax.text(glo[i1], gla[i1], '  Seq1 (fixed)', fontsize=10, color='red', fontweight='bold')
txt2 = ax.text(glo[i2], gla[i2], '  Seq2 (adjustable)', fontsize=10, color='orange', fontweight='bold')

def update():
    d = hypot((gla[i2]-gla[i1])*111320, (glo[i2]-glo[i1])*111320*cos(radians(gla[i1])))
    i2s2, i2e2 = max(i2-w,0), min(i2+w, len(gt)-1)
    line_seq2.set_data(glo[i2s2:i2e2], gla[i2s2:i2e2])
    pt2.set_offsets([[glo[i2], gla[i2]]])
    txt2.set_position((glo[i2], gla[i2]))
    ax.set_title(f'GPS Correction | Seq1=fixed Seq2=adjustable | dist={d:.2f}m | ←→ move  | Enter=confirm  | ESC=quit')
    fig.canvas.draw_idle()

def on_key(event):
    global i2
    if event.key == 'right':
        i2 = min(i2 + 1, len(gt) - 1)
        update()
    elif event.key == 'left':
        i2 = max(i2 - 1, 0)
        update()
    elif event.key in ('enter', 'return'):
        d = hypot((gla[i2]-gla[i1])*111320, (glo[i2]-glo[i1])*111320*cos(radians(gla[i1])))
        print(f'\n=== Corrected ===')
        print(f'Seq1 GPS[{i1+2}]: {gla[i1]:.8f}, {glo[i1]:.8f}  ts={gt[i1]:.3f}s')
        print(f'Seq2 GPS[{i2+2}]: {gla[i2]:.8f}, {glo[i2]:.8f}  ts={gt[i2]:.3f}s')
        print(f'GPS truth: {d:.2f}m')
        print(f'copy this to config: g2i_corrected = {i2}')
    elif event.key == 'escape':
        plt.close()

fig.canvas.mpl_connect('key_press_event', on_key)
update()
print(f'Seq1 GPS[{i1+2}]: ({gla[i1]:.8f}, {glo[i1]:.8f})  dt={(gt[i1]-T1)*1000:.1f}ms')
print(f'Seq2 GPS[{i2+2}]: ({gla[i2]:.8f}, {glo[i2]:.8f})  dt={(gt[i2]-T2)*1000:.1f}ms')
print(f'←→ adjust seq2 | Enter confirm | ESC quit')
plt.show()
