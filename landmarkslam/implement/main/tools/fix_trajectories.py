#!/usr/bin/env python3
import os, shutil, glob

RUNS_DIR = "landmarkslam/implement/data/extracted_data/runs"
PAIRS_DIR = "landmarkslam/implement/data/test_pairs"

for pair_dir in sorted(glob.glob(PAIRS_DIR + "/pair_*")):
    pair_name = os.path.basename(pair_dir)
    parts = pair_name.replace("pair_", "").split("_")
    a, b = int(parts[0]), int(parts[1])
    
    for map_id, seq in [(a, "seq1"), (b, "seq2")]:
        seq_dir = os.path.join(pair_dir, seq)
        times_path = os.path.join(seq_dir, "times.txt")
        if not os.path.exists(times_path):
            print(f"  {pair_name}/{seq}: 无 times.txt")
            continue
        
        with open(times_path) as f:
            lines = [l for l in f if l.strip()]
        t1, t2 = float(lines[0].split()[0]), float(lines[-1].split()[0])
        
        found = None
        for root, dirs, files in os.walk(RUNS_DIR):
            for f in files:
                if f == f"map_{map_id}_trajectory.txt" or f == f"map_{map_id:02d}_trajectory.txt":
                    traj = os.path.join(root, f)
                    with open(traj) as tf:
                        lines = [l for l in tf if not l.startswith("#") and l.strip()]
                        if lines:
                            t_first = float(lines[0].split()[0])
                            t_last  = float(lines[-1].split()[0])
                            # 时间范围有重叠即可
                            if t_first <= t2 and t_last >= t1:
                                found = traj
                                break
            if found: break
        
        dst = os.path.join(seq_dir, "trajectory.txt")
        if found:
            shutil.copy2(found, dst)
            print(f"  {pair_name}/{seq}: ✅ {os.path.basename(found)}")
        else:
            print(f"  {pair_name}/{seq}: ❌ 无匹配轨迹 (map {map_id})")

print("\n✅ Done")
