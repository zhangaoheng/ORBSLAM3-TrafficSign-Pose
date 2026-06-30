#!/usr/bin/env python3
"""
Interactive trajectory trimmer: image (left) + trajectory (right) in sync.
Keys: ←→ move | S mark start | E mark end | Enter save | ESC quit
Usage: python3 interactive_split.py <traj.txt> <image_dir> [out_dir]
"""
import sys, os, re, numpy as np, cv2
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ── Load data ──
traj_path = sys.argv[1]
img_root  = sys.argv[2]
out_dir   = sys.argv[3] if len(sys.argv) > 3 else 'manual_split'

img_dir = img_root
euroc_cam = os.path.join(img_root, 'mav0', 'cam0', 'data')
if os.path.isdir(euroc_cam):
    img_dir = euroc_cam

img_files = {}
for fn in os.listdir(img_dir):
    m = re.match(r'(\d+(?:\.\d+)?)\.png', fn)
    if m:
        img_files[float(m.group(1))] = os.path.join(img_dir, fn)
img_ts = sorted(img_files.keys())
print(f"Images: {len(img_ts)}")

traj = np.loadtxt(traj_path)
traj_ts = traj[:, 0]
traj_xyz = traj[:, 1:4]
if traj_ts[0] < 1e12:
    traj_ts = traj_ts * 1e9
print(f"Traj: {len(traj)} rows")

# Vectorized matching
img_arr = np.array(img_ts)
pos = np.searchsorted(img_arr, traj_ts)
valid = pos < len(img_arr)
matched = [(i, int(pos[i])) for i in np.where(valid)[0]]
print(f"Matched: {len(matched)} pairs")

os.makedirs(out_dir, exist_ok=True)
seg_count = 0
frame_idx = 0
mark_start = -1
mark_end = -1

# ── Pre-render trajectory background as image ──
x, y = traj_xyz[:, 0], traj_xyz[:, 1]
fig_traj, ax_bg = plt.subplots(figsize=(8, 6))
ax_bg.plot(x, y, 'gray', linewidth=0.2, alpha=0.5)
ax_bg.set_xlabel('X (m)'); ax_bg.set_ylabel('Y (m)')
ax_bg.axis('equal')
fig_traj.tight_layout(pad=0)
fig_traj.canvas.draw()
bg = np.asarray(fig_traj.canvas.buffer_rgba())[:, :, :3]
plt.close(fig_traj)

# ── Interactive window ──
fig = plt.figure(figsize=(16, 6))
ax_img = fig.add_subplot(1, 2, 1)
ax_traj = fig.add_subplot(1, 2, 2)
ax_traj.imshow(bg, extent=[x.min(), x.max(), y.min(), y.max()])
traj_line, = ax_traj.plot([], [], 'orange', linewidth=1.5)
cur_pt = ax_traj.scatter([], [], c='cyan', s=80, zorder=5, edgecolors='k')
start_pt = ax_traj.scatter([], [], c='lime', s=120, marker='s', zorder=6, edgecolors='k')
end_pt = ax_traj.scatter([], [], c='red', s=120, marker='s', zorder=6, edgecolors='k')
title_txt = ax_traj.set_title('')

def update():
    ti, pi = matched[frame_idx]
    # Image
    img = cv2.imread(img_files[img_ts[pi]])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax_img.clear()
    ax_img.imshow(img)
    ax_img.set_title(f'Frame {frame_idx+1}/{len(matched)}  ts={img_ts[pi]:.0f}ns', fontsize=10)
    ax_img.axis('off')

    # Trajectory markers (fast — only update points)
    cur_pt.set_offsets([[x[ti], y[ti]]])

    if mark_start >= 0:
        ts, _ = matched[mark_start]
        start_pt.set_offsets([[x[ts], y[ts]]])
    else:
        start_pt.set_offsets(np.empty((0, 2)))

    if mark_end >= 0:
        te, _ = matched[mark_end]
        end_pt.set_offsets([[x[te], y[te]]])
        if mark_start >= 0:
            ts, _ = matched[mark_start]
            traj_line.set_data(x[ts:te+1], y[ts:te+1])
        else:
            traj_line.set_data([], [])
    else:
        end_pt.set_offsets(np.empty((0, 2)))
        traj_line.set_data([], [])

    info = f'[{frame_idx+1}/{len(matched)}]  S=start  E=end  Enter=save  ESC=quit'
    if mark_start >= 0 and mark_end >= 0:
        info += f'  |  selected: {mark_end-mark_start+1} frames'
    title_txt.set_text(info)
    fig.canvas.draw_idle()

def save_segment():
    global seg_count, mark_start, mark_end
    if mark_start < 0 or mark_end < 0 or mark_start >= mark_end:
        print("  Mark start (S) and end (E) first")
        return
    ts, _ = matched[mark_start]
    te, _ = matched[mark_end]
    seg_name = f"seg_manual_{seg_count}"
    seg_dir = os.path.join(out_dir, seg_name)
    os.makedirs(seg_dir, exist_ok=True)

    ti_s, _ = matched[mark_start]
    ti_e, _ = matched[mark_end]
    with open(os.path.join(seg_dir, 'trajectory.txt'), 'w') as f:
        for i in range(ti_s, ti_e + 1):
            f.write(f"{traj_ts[i]:.0f} {traj_xyz[i,0]:.6f} {traj_xyz[i,1]:.6f} {traj_xyz[i,2]:.6f} 0 0 0 1\n")

    rgb_dir = os.path.join(seg_dir, 'rgb')
    os.makedirs(rgb_dir, exist_ok=True)
    with open(os.path.join(seg_dir, 'times.txt'), 'w') as f:
        for j in range(mark_start, mark_end + 1):
            _, pi = matched[j]
            ts_img = img_ts[pi]
            src = img_files[ts_img]
            fn = os.path.basename(src)
            f.write(f"{ts_img:.0f} rgb/{fn}\n")
            dst = os.path.join(rgb_dir, fn)
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)
                except OSError:
                    import shutil; shutil.copy2(src, dst)

    print(f"  Saved: {seg_name} ({mark_end - mark_start + 1} frames)")
    seg_count += 1
    mark_start = -1
    mark_end = -1

def on_key(event):
    global frame_idx, mark_start, mark_end
    if event.key == 'right':
        frame_idx = min(frame_idx + 1, len(matched) - 1)
        update()
    elif event.key == 'left':
        frame_idx = max(frame_idx - 1, 0)
        update()
    elif event.key == '6':
        frame_idx = min(frame_idx + 200, len(matched) - 1)
        print(f"  Jump: frame {frame_idx+1}")
        update()
    elif event.key == '4':
        frame_idx = max(frame_idx - 200, 0)
        print(f"  Jump: frame {frame_idx+1}")
        update()
    elif event.key == 's':
        mark_start = frame_idx
        mark_end = -1
        print(f"  Start: frame {frame_idx+1}")
        update()
    elif event.key == 'e':
        if mark_start < 0:
            print("  Mark start first (S)")
            return
        if frame_idx <= mark_start:
            print("  End must be after start")
            return
        mark_end = frame_idx
        print(f"  End: frame {frame_idx+1}  ({mark_end - mark_start + 1} frames)")
        update()
    elif event.key in ('enter', 'return'):
        print("  Enter pressed, saving...")
        save_segment()
        update()
    elif event.key == 'escape':
        plt.close()

# 禁用 matplotlib 默认快捷键 (s=save figure 等)
for k in plt.rcParams.keys():
    if k.startswith('keymap.'):
        plt.rcParams[k] = ''
fig.canvas.mpl_connect('key_press_event', on_key)
update()
print("←→ move | 4/6 skip 200 | S start | E end | Enter(Return) save | ESC quit")
plt.show()
