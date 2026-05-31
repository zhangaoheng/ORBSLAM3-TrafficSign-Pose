#!/usr/bin/env python3
"""
从已保存的 rgb/ + depth/ 目录补全 associations.txt
图片按顺序一一对应（000000.png ↔ depth/000000.png），
从已有的部分 associations.txt 推算帧间隔，然后补全全部。
"""

import os
import sys


def estimate_frame_interval(assoc_path):
    """从已有的 associations.txt 估算帧间隔（秒）"""
    if not os.path.exists(assoc_path):
        return 1.0 / 30.0  # 默认 30fps

    with open(assoc_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    if len(lines) < 2:
        return 1.0 / 30.0

    # 取前几帧算平均间隔
    ts_list = [float(l.split()[0]) for l in lines[:10]]
    diffs = [ts_list[i+1] - ts_list[i] for i in range(len(ts_list)-1)]
    return sum(diffs) / len(diffs)


def gen_assoc(data_dir):
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    assoc_path = os.path.join(data_dir, "associations.txt")

    rgb_count = len([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    depth_count = len([f for f in os.listdir(depth_dir) if f.endswith('.png')])
    n = min(rgb_count, depth_count)

    print(f"  RGB:   {rgb_count} 帧")
    print(f"  Depth: {depth_count} 帧")
    print(f"  对齐:  {n} 帧")

    if n == 0:
        print("❌ 无图片")
        return

    # 估算帧间隔
    interval = estimate_frame_interval(assoc_path)

    # 读已有的 associations.txt 获取第一帧时间戳
    first_ts = None
    if os.path.exists(assoc_path):
        with open(assoc_path) as f:
            for line in f:
                if line.strip():
                    first_ts = float(line.strip().split()[0])
                    break

    if first_ts is None:
        # 没有已知时间戳，用文件修改时间近似
        first_file = os.path.join(rgb_dir, "000000.png")
        first_ts = os.path.getmtime(first_file)
        print(f"  ⚠ 无时间戳信息，使用文件 mtime 近似: {first_ts:.6f}")

    print(f"  帧间隔: {interval*1000:.2f} ms")

    with open(assoc_path, 'w') as f:
        for i in range(n):
            ts = first_ts + i * interval
            frame = f"{i:06d}.png"
            f.write(f"{ts:.6f} rgb/{frame} {ts:.6f} depth/{frame}\n")

    print(f"  ✅ associations.txt 生成完毕：{n} 帧")
    print(f"     第一帧: {first_ts:.6f}")
    print(f"     最后一帧: {first_ts + (n-1)*interval:.6f}")


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) >= 2 else \
        "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data"
    gen_assoc(data_dir)
