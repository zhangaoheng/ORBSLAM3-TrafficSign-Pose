#!/usr/bin/env python3
"""
数据整理脚本 — 按序列对整理 split_by_map 数据到 test.py 格式

用法:
  python3 prepare_test_data.py <数字对>...
  
  每对数字: A,B 表示 map_A 和 map_B 是同一路牌的两段
  示例: python3 prepare_test_data.py 1,45 2,46 3,47

输出结构:
  test_pairs/
  ├── pair_03_47/
  │   ├── seq1/           ← map_03 数据
  │   │   ├── rgb/
  │   │   ├── depth/
  │   │   ├── times.txt
  │   │   ├── associations.txt
  │   │   └── trajectory.txt
  │   ├── seq2/           ← map_47 数据
  │   │   └── ...
  │   └── gps_segment.csv (共用)
  └── ...
"""
import os, sys, shutil, glob

SPLIT_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data/split_by_map"
RUNS_DIR  = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data/runs"
OUT_DIR   = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/test_pairs"

def find_trajectory(t_start, t_end):
    """从 runs 目录找到时间戳与图片重叠最多的轨迹文件"""
    best_path, best_overlap = None, 0
    for root, dirs, files in os.walk(RUNS_DIR):
        for f in files:
            if not f.startswith("map_") or not f.endswith("_trajectory.txt"):
                continue
            traj_path = os.path.join(root, f)
            with open(traj_path) as tf:
                lines = [l for l in tf if not l.startswith("#") and l.strip()]
                if not lines: continue
                try:
                    tf_first = float(lines[0].split()[0])
                    tf_last  = float(lines[-1].split()[0])
                    overlap = min(tf_last, t_end) - max(tf_first, t_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_path = traj_path
                except: pass
    return best_path

def copy_map(map_id, dest_seq_dir):
    """从 split_by_map 复制一个 map 到目标目录"""
    src = os.path.join(SPLIT_DIR, f"map_{map_id:02d}")
    if not os.path.isdir(src):
        print(f"   ❌ map_{map_id:02d} 不存在")
        return False
    
    os.makedirs(dest_seq_dir, exist_ok=True)
    
    # 复制 times.txt, associations.txt
    for fname in ["times.txt", "associations.txt"]:
        sf = os.path.join(src, fname)
        if os.path.exists(sf):
            shutil.copy2(sf, dest_seq_dir)
    
    # 创建 rgb/ 和 depth/ 符号链接
    for subdir in ["rgb", "depth"]:
        src_sub = os.path.join(src, subdir)
        dst_sub = os.path.join(dest_seq_dir, subdir)
        if os.path.isdir(src_sub) and not os.path.exists(dst_sub):
            # 用绝对路径符号链接（避免相对路径断了）
            os.symlink(os.path.abspath(src_sub), dst_sub)
    
    # 读取图片时间范围，按时间戳匹配轨迹
    times_path = os.path.join(src, "times.txt")
    t_start, t_end = 0, 0
    if os.path.exists(times_path):
        with open(times_path) as f:
            lines = [l for l in f if l.strip()]
            if lines:
                t_start = float(lines[0].split()[0])
                t_end   = float(lines[-1].split()[0])
    traj_path = find_trajectory(t_start, t_end)
    if traj_path:
        shutil.copy2(traj_path, os.path.join(dest_seq_dir, "trajectory.txt"))
        print(f"   轨迹: map_{map_id}_trajectory.txt")
    else:
        print(f"   ⚠️ 未找到 map_{map_id}_trajectory.txt")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("用法: python3 prepare_test_data.py A1,B1 A2,B2 ...")
        print("示例: python3 prepare_test_data.py 3,47 4,49")
        sys.exit(1)
    
    if not os.path.isdir(SPLIT_DIR):
        print(f"❌ split_by_map 不存在: {SPLIT_DIR}")
        sys.exit(1)
    
    pairs = []
    for arg in sys.argv[1:]:
        try:
            a, b = arg.replace("，", ",").split(",")
            pairs.append((int(a), int(b)))
        except:
            print(f"⚠️ 跳过无效参数: {arg}")
    
    print(f"📋 {len(pairs)} 对数据")
    for a, b in pairs:
        print(f"   map_{a:02d} ←→ map_{b:02d}")
    
    for a, b in pairs:
        pair_name = f"pair_{a:02d}_{b:02d}"
        pair_dir = os.path.join(OUT_DIR, pair_name)
        
        # 如果已存在，跳过
        if os.path.isdir(pair_dir):
            print(f"\n⏭️  {pair_name} 已存在，跳过")
            continue
        
        print(f"\n📁 {pair_name}")
        os.makedirs(pair_dir, exist_ok=True)
        
        ok_a = copy_map(a, os.path.join(pair_dir, "seq1"))
        ok_b = copy_map(b, os.path.join(pair_dir, "seq2"))
        
        if ok_a and ok_b:
            # 生成 config.yaml
            config = f"""Camera:
  fx: 426.372
  fy: 425.671
  cx: 435.525
  cy: 244.974

Algorithm:
  frame_step: 15
  cache_file: "cache.json"

Sequence1:
  image_dir: "{os.path.join(pair_dir, 'seq1', 'rgb')}"
  depth_dir: "{os.path.join(pair_dir, 'seq1', 'depth')}"
  trajectory_path: "{os.path.join(pair_dir, 'seq1', 'trajectory.txt')}"
  roi_path: "{os.path.join(pair_dir, 'seq1', 'rgb', 'corners.txt')}"

Sequence2:
  image_dir: "{os.path.join(pair_dir, 'seq2', 'rgb')}"
  trajectory_path: "{os.path.join(pair_dir, 'seq2', 'trajectory.txt')}"
"""
            with open(os.path.join(pair_dir, "config.yaml"), "w") as f:
                f.write(config)
            print(f"   ✅ config.yaml 已生成")
            
            # 生成 GPS 数据
            gps_csv = os.path.join(os.path.dirname(SPLIT_DIR), "..", "test", "gps_record_20260529_113616.csv")
            gps_csv = os.path.normpath(gps_csv)
            for seq_name, map_id in [("seq1", a), ("seq2", b)]:
                seq_dir = os.path.join(pair_dir, seq_name)
                times_path = os.path.join(seq_dir, "times.txt")
                if os.path.exists(gps_csv) and os.path.exists(times_path):
                    with open(times_path) as f:
                        t_lines = [float(l.strip().split()[0]) for l in f if l.strip()]
                    if t_lines:
                        t1, t2 = t_lines[0], t_lines[-1]
                        gps_lines = []
                        with open(gps_csv) as f:
                            header = f.readline()
                            for line in f:
                                try:
                                    ts = float(line.strip().split(",")[0])
                                    if t1 <= ts <= t2:
                                        gps_lines.append(line)
                                except: pass
                        if gps_lines:
                            gps_out = os.path.join(pair_dir, f"{seq_name}_gps.csv")
                            with open(gps_out, "w") as f:
                                f.write(header)
                                for line in gps_lines:
                                    f.write(line)
                            print(f"   {seq_name}: {len(gps_lines)} GPS → {seq_name}_gps.csv")
    
    print(f"\n✅ 完成！输出: {OUT_DIR}")

if __name__ == "__main__":
    main()
