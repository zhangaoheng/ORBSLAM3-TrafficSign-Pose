#!/usr/bin/env python3
"""逐段 3D 查看 split_output 轨迹——空格切下一段，ESC 退出"""
import sys, os, glob, numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

split_dir = sys.argv[1] if len(sys.argv) > 1 else 'split_output'
seg_dirs = sorted(glob.glob(os.path.join(split_dir, 'seg_*')),
                  key=lambda x: int(x.split('_')[-1]))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
idx = [0]

def show_seg():
    ax.clear()
    sd = seg_dirs[idx[0]]
    data = np.loadtxt(os.path.join(sd, 'trajectory.txt'))
    x, y, z = data[:, 1], data[:, 2], data[:, 3]
    dist = np.sum(np.linalg.norm(np.diff(data[:, 1:4], axis=0), axis=1))
    t_norm = np.linspace(0, 1, len(x))
    ax.scatter(x, y, z, c=t_norm, cmap='plasma', s=5, alpha=0.8)
    ax.plot(x, y, z, color='gray', alpha=0.3, linewidth=0.5)
    ax.set_title(f'{os.path.basename(sd)} — {len(data)}KFs, {dist:.0f}m  [{idx[0]+1}/{len(seg_dirs)}]')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.invert_zaxis()
    fig.canvas.draw()

def on_key(event):
    if event.key in ('right', ' '):
        idx[0] = min(idx[0] + 1, len(seg_dirs) - 1)
        show_seg()
    elif event.key == 'left':
        idx[0] = max(idx[0] - 1, 0)
        show_seg()
    elif event.key == 'escape':
        plt.close()

fig.canvas.mpl_connect('key_press_event', on_key)
show_seg()
print(f"共 {len(seg_dirs)} 段 | 空格/→ 下一段 | ← 上一段 | ESC 退出")
plt.show()
