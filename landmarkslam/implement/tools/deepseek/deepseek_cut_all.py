#!/usr/bin/env python3
"""
交互式序列分割工具：
  在 lines_all 中，通过 GUI 手动为 lines1 和 lines2 选定帧区间，
  自动复制 rgb/depth 图像，生成对应的 trajectory.txt、associations.txt 等。
"""

import os
import shutil
import glob
import numpy as np
import sys
import cv2

# ================== 路径配置 ==================
LINES_ALL_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/deepseek/lines_all"
OUT_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/deepseek"  # lines1, lines2 将创建在此
TRAJECTORY_FILE = "AllFrames_trajectory.txt"  # 在 lines_all 目录下
# ==============================================

def load_trajectory(traj_path):
    """读取 TUM 轨迹文件，返回 {timestamp: [tx, ty, tz, qx, qy, qz, qw]}"""
    traj = {}
    if not os.path.exists(traj_path):
        print(f"❌ 轨迹文件不存在: {traj_path}")
        return {}  # 不退出，可能没有轨迹
    with open(traj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            ts = float(parts[0])
            traj[ts] = [float(x) for x in parts[1:8]]
    return traj

def save_trajectory(traj_dict, timestamps, output_path):
    """为给定的时间戳列表生成轨迹文件"""
    with open(output_path, 'w') as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts in timestamps:
            # 最近邻匹配，容许0.01秒误差
            closest_ts = min(traj_dict.keys(), key=lambda k: abs(k - ts)) if traj_dict else None
            if closest_ts and abs(closest_ts - ts) < 0.01:
                p = traj_dict[closest_ts]
                f.write(f"{ts:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                        f"{p[3]:.6f} {p[4]:.6f} {p[5]:.6f} {p[6]:.6f}\n")

def select_interval(img_list, timestamps, window_title):
    """
    交互式选择区间，返回起始索引和结束索引（包含）
    操作说明：
      - 左右方向键/D/A 前后翻帧
      - 按 S 标记当前帧为起点
      - 按 E 标记当前帧为终点（若起点未标记则自动标记起点）
      - 按 Enter 确认当前区间（如果已选起点和终点）
      - 按 Esc 退出程序
    """
    if not img_list:
        print("❌ 没有图像！")
        return None, None

    idx = 0
    start_idx, end_idx = -1, -1
    window_name = f"Select {window_title}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        img = cv2.imread(img_list[idx])
        if img is None:
            print(f"无法读取图片: {img_list[idx]}")
            idx = (idx + 1) % len(img_list)
            continue

        disp = img.copy()
        ts = timestamps[idx]
        # 状态提示
        status_lines = [
            f"Seq: {window_title} | Frame: {idx}/{len(img_list)-1} | Time: {ts:.3f}s",
            "[A/Left]: Prev | [D/Right]: Next",
            "[S]: Set START | [E]: Set END",
            "[Enter]: Confirm interval | [ESC]: Quit",
        ]
        if start_idx >= 0:
            status_lines.append(f"START: frame {start_idx}")
        if end_idx >= 0:
            status_lines.append(f"END: frame {end_idx}")
        if start_idx >= 0 and end_idx >= 0 and start_idx > end_idx:
            status_lines.append("⚠️ START after END (invalid)")

        y0 = 30
        for line in status_lines:
            cv2.putText(disp, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y0 += 25

        cv2.imshow(window_name, disp)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(window_name)
            sys.exit(0)

        if key == ord('a') or key == 81:  # 左箭头 (81 是 Linux 下方向键编码)
            idx = max(0, idx - 1)
        elif key == ord('d') or key == 83:  # 右箭头
            idx = min(len(img_list) - 1, idx + 1)
        elif key == ord('s'):
            start_idx = idx
            print(f"  起点设为帧 {idx}, 时间戳 {timestamps[idx]:.3f}")
        elif key == ord('e'):
            end_idx = idx
            if start_idx < 0:
                start_idx = idx  # 如果没设起点，自动设为当前帧
                print(f"  自动设起点为帧 {idx}")
            print(f"  终点设为帧 {idx}, 时间戳 {timestamps[idx]:.3f}")
        elif key == 13:  # Enter
            if start_idx >= 0 and end_idx >= 0 and start_idx <= end_idx:
                cv2.destroyWindow(window_name)
                return start_idx, end_idx
            else:
                print("⚠️ 请先设置有效的起点和终点 (起点≤终点)")

    cv2.destroyWindow(window_name)
    return start_idx, end_idx

def main():
    rgb_dir = os.path.join(LINES_ALL_DIR, "rgb")
    if not os.path.isdir(rgb_dir):
        print(f"❌ RGB 目录不存在: {rgb_dir}")
        sys.exit(1)

    images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    if len(images) < 10:
        print("❌ 图像数量太少，无法分割")
        sys.exit(1)

    # 提取时间戳（秒）
    timestamps = []
    for img_path in images:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        ts = int(fname) / 1e9  # 纳秒转秒
        timestamps.append(ts)

    # 加载整体轨迹
    traj_path = os.path.join(LINES_ALL_DIR, TRAJECTORY_FILE)
    traj_dict = load_trajectory(traj_path)

    # ---------- 交互式选择两个区间 ----------
    intervals = []
    seq_names = ["lines1", "lines2"]
    for name in seq_names:
        print(f"\n🔍 请选择 {name} 的帧区间：")
        start, end = select_interval(images, timestamps, name)
        if start is None or end is None:
            print("❌ 未正确选择区间，退出。")
            sys.exit(1)
        intervals.append((name, start, end))

    # ---------- 复制文件并生成关联 ----------
    for (name, start, end) in intervals:
        seq_dir = os.path.join(OUT_DIR, name)
        rgb_out = os.path.join(seq_dir, "rgb")
        depth_out = os.path.join(seq_dir, "depth")
        os.makedirs(rgb_out, exist_ok=True)
        os.makedirs(depth_out, exist_ok=True)

        selected_imgs = images[start:end+1]
        selected_ts = timestamps[start:end+1]

        # 复制 rgb 和 depth
        for img_path in selected_imgs:
            fname = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(rgb_out, fname))
            depth_src = os.path.join(LINES_ALL_DIR, "depth", fname)
            if os.path.exists(depth_src):
                shutil.copy(depth_src, os.path.join(depth_out, fname))
            else:
                print(f"⚠️ 缺少深度图: {fname}")

        # 内参
        intrinsics_src = os.path.join(LINES_ALL_DIR, "camera_intrinsics.json")
        if os.path.exists(intrinsics_src):
            shutil.copy(intrinsics_src, os.path.join(seq_dir, "camera_intrinsics.json"))

        # 生成 associations.txt
        assoc_file = os.path.join(seq_dir, "associations.txt")
        with open(assoc_file, 'w') as f:
            for ts, img_path in zip(selected_ts, selected_imgs):
                fname = os.path.basename(img_path)
                f.write(f"{ts:.6f} rgb/{fname} {ts:.6f} depth/{fname}\n")

        # 生成分割后的轨迹
        if traj_dict:
            save_trajectory(traj_dict, selected_ts, os.path.join(seq_dir, "trajectory.txt"))
        else:
            print(f"⚠️ 未找到整体轨迹，{name} 将不包含 trajectory.txt")

        print(f"✅ {name}: {end - start + 1} 帧已生成至 {seq_dir}")

    print("\n🎉 分割完成！现在可以在 config.yaml 中分别配置 lines1 和 lines2 的路径了。")

if __name__ == "__main__":
    main()