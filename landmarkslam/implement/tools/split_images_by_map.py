#!/home/zah/ORB_SLAM3-master/landmarkslam/yolo_venv/bin/python3
"""
根据 maps_summary.txt 的帧范围，将原始图片分割到每个地图的独立子目录。

用法:
  python3 split_images_by_map.py <run_dir> <image_dir> <output_dir>

示例:
  python3 split_images_by_map.py \\
      runs/2026-06-01_15-58-17_rgbd \\
      data/extracted_data \\
      data/extracted_data/split_by_map
"""
import os, sys, shutil, glob

def load_summary(path):
    """读取 maps_summary.txt，返回 map_id -> (start, end, kfs, dist)"""
    maps = []
    with open(path) as f:
        for line in f:
            if line.startswith("map_id") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 5:
                mid = int(parts[0])
                start = int(parts[1])
                end = int(parts[2])
                kfs = int(parts[3])
                dist = float(parts[4])
                maps.append((mid, start, end, kfs, dist))
    return maps

def main():
    if len(sys.argv) < 4:
        print("用法: python3 split_images_by_map.py <run_dir> <image_root> <output_dir>")
        print("示例: python3 split_images_by_map.py \\")
        print("    runs/2026-06-01_15-58-17_rgbd \\")
        print("    data/extracted_data \\")
        print("    data/extracted_data/split_by_map")
        sys.exit(1)

    run_dir = sys.argv[1]
    img_root = sys.argv[2]
    out_root = sys.argv[3]

    summary_path = os.path.join(run_dir, "maps_summary.txt")
    if not os.path.exists(summary_path):
        print(f"❌ 找不到 maps_summary.txt: {summary_path}")
        sys.exit(1)

    maps = load_summary(summary_path)
    if not maps:
        print("❌ maps_summary.txt 为空")
        sys.exit(1)

    print(f"📂 图片目录: {img_root}")
    print(f"📊 共 {len(maps)} 段地图")
    print(f"📁 输出目录: {out_root}")
    print()

    # 获取所有图片的文件名列表（按序号排序）
    rgb_dir = os.path.join(img_root, "rgb")
    depth_dir = os.path.join(img_root, "depth")

    if not os.path.isdir(rgb_dir):
        print(f"❌ 找不到 rgb 目录: {rgb_dir}")
        sys.exit(1)

    # 获取文件名列表（000000.png, 000001.png, ...）
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    # 提取序号
    rgb_indices = []
    for f in rgb_files:
        basename = os.path.basename(f)
        idx = int(basename.replace(".png", ""))
        rgb_indices.append((idx, basename))

    has_depth = os.path.isdir(depth_dir)

    print(f"  共 {len(rgb_indices)} 张图片")
    if has_depth:
        depth_files_count = len(glob.glob(os.path.join(depth_dir, "*.png")))
        print(f"  共 {depth_files_count} 张深度图")
    print()

    total_copied = 0
    for mid, start, end, kfs, dist in maps:
        # 创建输出子目录
        map_dir = os.path.join(out_root, f"map_{mid:02d}")
        map_rgb_dir = os.path.join(map_dir, "rgb")
        os.makedirs(map_rgb_dir, exist_ok=True)

        if has_depth:
            map_depth_dir = os.path.join(map_dir, "depth")
            os.makedirs(map_depth_dir, exist_ok=True)

        # 遍历帧范围，复制图片
        count = 0
        for idx, basename in rgb_indices:
            if start <= idx <= end:
                # 链接 RGB（用相对路径节省空间）
                src = os.path.relpath(rgb_dir, map_rgb_dir)
                dst = os.path.join(map_rgb_dir, basename)
                if not os.path.exists(dst):
                    os.symlink(os.path.join(src, basename), dst)
                count += 1

                # 链接 Depth
                if has_depth:
                    depth_src_rel = os.path.relpath(depth_dir, map_depth_dir)
                    depth_dst = os.path.join(map_depth_dir, basename)
                    if os.path.exists(os.path.join(depth_dir, basename)) and not os.path.exists(depth_dst):
                        os.symlink(os.path.join(depth_src_rel, basename), depth_dst)

        total_copied += count

        # 生成 times.txt（mono 用）
        times_path = os.path.join(img_root, "times.txt")
        if os.path.exists(times_path):
            map_times = os.path.join(map_dir, "times.txt")
            with open(times_path) as ft, open(map_times, "w") as fo:
                for line in ft:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            frame_idx = int(parts[1].replace("rgb/", "").replace(".png", ""))
                            if start <= frame_idx <= end:
                                fo.write(line)
                        except:
                            pass

        # 生成 associations.txt（RGB-D 用）
        assoc_path = os.path.join(img_root, "associations.txt")
        if os.path.exists(assoc_path):
            map_assoc = os.path.join(map_dir, "associations.txt")
            with open(assoc_path) as fa, open(map_assoc, "w") as fo:
                for line in fa:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        try:
                            frame_idx = int(parts[1].replace("rgb/", "").replace(".png", ""))
                            if start <= frame_idx <= end:
                                fo.write(line)
                        except:
                            pass

        if count > 0:
            dur = end - start
            print(f"  Map {mid:2d}: 帧 {start:5d}-{end:5d} ({count:5d} 张, {dist:6.1f}m) → {map_dir}")

    print(f"\n✅ 共分割 {total_copied} 张图片到 {len(maps)} 个地图目录")
    print(f"📁 输出根目录: {out_root}")

if __name__ == "__main__":
    main()
