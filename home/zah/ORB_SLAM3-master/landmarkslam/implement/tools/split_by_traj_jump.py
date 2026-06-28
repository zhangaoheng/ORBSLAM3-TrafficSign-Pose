#!/usr/bin/env python3
"""按轨迹断点(跳回原点)分割图片 + 轨迹"""
import os, sys, numpy as np, glob, argparse

JUMP_THRESHOLD = 50.0


def split_traj(traj_path):
    data = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            data.append((float(parts[0]), [float(x) for x in parts[1:4]]))

    segments = []
    seg_start = 0
    for i in range(1, len(data)):
        p0, p1 = np.array(data[i - 1][1]), np.array(data[i][1])
        if np.linalg.norm(p1 - p0) > JUMP_THRESHOLD:
            segments.append((seg_start, i - 1))
            seg_start = i
    segments.append((seg_start, len(data) - 1))
    return data, segments


def load_times(times_path):
    entries = []
    with open(times_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                entries.append((float(parts[0]), os.path.basename(parts[1])))
    return sorted(entries, key=lambda x: x[0])


def find_times_near_traj(traj_path):
    for candidate in [
        os.path.join(os.path.dirname(traj_path), "times.txt"),
        os.path.join(os.path.dirname(os.path.dirname(traj_path)), "times.txt"),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("traj")
    parser.add_argument("out_dir")
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--threshold", type=float, default=50)
    parser.add_argument("--times", default="")
    args = parser.parse_args()

    data, segments = split_traj(args.traj)
    print(f"Trajectory: {len(data)} rows, {len(segments)} segments detected")

    times = []
    tpath = args.times or find_times_near_traj(args.traj)
    if tpath and os.path.exists(tpath):
        times = load_times(tpath)
        print(f"Image timestamps: {len(times)} entries")

    os.makedirs(args.out_dir, exist_ok=True)
    img_files = sorted(glob.glob(os.path.join(args.image_dir, "*.png"))) if args.image_dir else []

    for seg_id, (s, e) in enumerate(segments):
        seg_dir = os.path.join(args.out_dir, f"seg_{seg_id}")
        os.makedirs(seg_dir, exist_ok=True)

        # Write trajectory segment
        with open(os.path.join(seg_dir, "trajectory.txt"), "w") as f:
            for i in range(s, e + 1):
                ts, xyz = data[i]
                f.write(f"{ts} {xyz[0]} {xyz[1]} {xyz[2]} 0 0 0 1\n")

        t_start, t_end = data[s][0], data[e][0]
        matched = [(ts, fn) for ts, fn in times if t_start <= ts <= t_end]

        if matched:
            with open(os.path.join(seg_dir, "times.txt"), "w") as f:
                for ts, fn in matched:
                    f.write(f"{ts} rgb/{fn}\n")
            rgb_dir = os.path.join(seg_dir, "rgb")
            os.makedirs(rgb_dir, exist_ok=True)
            for _, fn in matched:
                src = os.path.join(args.image_dir, fn)
                dst = os.path.join(rgb_dir, fn)
                if os.path.exists(src) and not os.path.exists(dst):
                    os.symlink(os.path.relpath(src, rgb_dir), dst)

        print(f"  seg_{seg_id}: traj {s}-{e} ({e - s + 1} lines), "
              f"images {len(matched)}, time {t_start:.1f}-{t_end:.1f}")

    print(f"\nDone: {args.out_dir}/seg_0 ~ seg_{len(segments) - 1}")
