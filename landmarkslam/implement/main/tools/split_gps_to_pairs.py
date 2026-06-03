#!/usr/bin/env python3
"""给 test_pairs 每个数据对补充 GPS 数据"""
import os, sys, glob

GPS_CSV  = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/test/gps_record_20260529_113616.csv"
PAIRS_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/test_pairs"

def main():
    if not os.path.exists(GPS_CSV):
        print(f"❌ GPS 文件不存在: {GPS_CSV}")
        sys.exit(1)

    # 读取所有 GPS 数据
    gps_data = []
    with open(GPS_CSV) as f:
        header = f.readline()
        for line in f:
            try:
                ts = float(line.split(",")[0])
                gps_data.append((ts, line))
            except: pass
    gps_data.sort()
    print(f"📡 GPS: {len(gps_data)} 条")

    pairs = sorted(glob.glob(os.path.join(PAIRS_DIR, "pair_*")))
    if not pairs:
        print("❌ 无 test_pairs 目录")
        return

    for pair_dir in pairs:
        name = os.path.basename(pair_dir)
        for seq in ["seq1", "seq2"]:
            times_path = os.path.join(pair_dir, seq, "times.txt")
            gps_out    = os.path.join(pair_dir, f"{seq}_gps.csv")
            
            if not os.path.exists(times_path):
                continue
            if os.path.exists(gps_out):
                cnt = sum(1 for _ in open(gps_out)) - 1  # minus header
                if cnt > 0:
                    continue  # 已有数据，跳过

            with open(times_path) as f:
                t_lines = [float(l.strip().split()[0]) for l in f if l.strip()]
            if not t_lines:
                continue
            t1, t2 = t_lines[0], t_lines[-1]

            filtered = [line for ts, line in gps_data if t1 <= ts <= t2]
            if filtered:
                with open(gps_out, "w") as f:
                    f.write(header)
                    f.writelines(filtered)
                print(f"  {name}/{seq}: {len(filtered)} GPS")
            else:
                print(f"  {name}/{seq}: 无匹配 GPS")

    print("\n✅ 完成")

if __name__ == "__main__":
    main()
