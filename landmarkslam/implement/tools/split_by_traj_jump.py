#!/usr/bin/env python3
"""按轨迹断点分割图片 + 轨迹"""
import os, sys, re, numpy as np, glob, argparse

JUMP_THRESHOLD = 50.0

def split_traj(traj_path):
    data = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            data.append((float(parts[0]), [float(x) for x in parts[1:4]]))
    segments, seg_start = [], 0
    for i in range(1, len(data)):
        p0, p1 = np.array(data[i-1][1]), np.array(data[i][1])
        if np.linalg.norm(p1 - p0) > JUMP_THRESHOLD:
            segments.append((seg_start, i-1))
            seg_start = i
    segments.append((seg_start, len(data)-1))
    return data, segments

def load_times(times_path):
    entries = []
    with open(times_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2: entries.append((float(parts[0]), os.path.basename(parts[1])))
    return sorted(entries, key=lambda x: x[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("traj"); parser.add_argument("out_dir")
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--threshold", type=float, default=50)
    args = parser.parse_args()

    data, segments = split_traj(args.traj)
    print(f"Traj: {len(data)} rows, {len(segments)} segments")

    # 找到实际的图片目录 (支持 EuRoC mav0/cam0/data/ 子目录)
    img_dir = args.image_dir
    if img_dir and os.path.isdir(img_dir):
        # 如果直接目录没有 png, 尝试 EuRoC 结构
        euroc_cam = os.path.join(img_dir, 'mav0', 'cam0', 'data')
        if not any(fn.endswith('.png') for fn in os.listdir(img_dir)[:10]):
            if os.path.isdir(euroc_cam):
                img_dir = euroc_cam

    # 从图片文件名解析时间戳 (支持整数纳秒: 1782373946559183104.png 和浮点秒: 1782567496.970000.png)
    times = []
    if img_dir and os.path.isdir(img_dir):
        for fn in os.listdir(img_dir):
            m = re.match(r'(\d+(?:\.\d+)?)\.png', fn)
            if m: times.append((float(m.group(1)), fn))
    times.sort(key=lambda x: x[0])
    if times: print(f"Image timestamps (from filenames @ {img_dir}): {len(times)} entries")

    os.makedirs(args.out_dir, exist_ok=True)
    for seg_id, (s, e) in enumerate(segments):
        sd = os.path.join(args.out_dir, f"seg_{seg_id}")
        os.makedirs(sd, exist_ok=True)
        with open(os.path.join(sd, "trajectory.txt"), "w") as f:
            for i in range(s, e+1):
                ts, xyz = data[i]
                f.write(f"{ts} {xyz[0]} {xyz[1]} {xyz[2]} 0 0 0 1\n")
        ts_s, ts_e = data[s][0], data[e][0]
        matched = [(ts, fn) for ts, fn in times if ts_s <= ts <= ts_e]
        if matched:
            with open(os.path.join(sd, "times.txt"), "w") as f:
                for ts, fn in matched: f.write(f"{ts} rgb/{fn}\n")
            rd = os.path.join(sd, "rgb"); os.makedirs(rd, exist_ok=True)
            for _, fn in matched:
                src = os.path.join(img_dir, fn)
                dst = os.path.join(rd, fn)
                if os.path.exists(src) and not os.path.exists(dst):
                    try:
                        os.link(src, dst)
                    except OSError:
                        import shutil
                        shutil.copy2(src, dst)
        print(f"  seg_{seg_id}: traj {s}-{e} imgs {len(matched)}")
    print(f"Done: {args.out_dir}/seg_0 ~ seg_{len(segments)-1}")
