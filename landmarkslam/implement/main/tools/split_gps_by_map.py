#!/usr/bin/env python3
"""
GPS 数据分割 — 按每个地图目录的 times.txt 时间范围，过滤出对应 GPS 段

用法: python3 split_gps_by_map.py <test_dir> <gps_csv>
输出: 每个 map_XX 目录下生成 gps_segment.csv
"""
import os, sys, glob

def main():
    if len(sys.argv) < 2:
        print("用法: python3 split_gps_by_map.py <test_dir> [gps_csv]")
        sys.exit(1)

    test_dir = sys.argv[1]
    gps_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(test_dir, "gps_record_20260529_113616.csv")

    if not os.path.exists(gps_csv):
        print(f"❌ GPS 文件不存在: {gps_csv}")
        sys.exit(1)

    # 读取所有 GPS 数据
    gps_data = []
    with open(gps_csv) as f:
        header = f.readline().strip()
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(",")
            if len(parts) < 4: continue
            try:
                ts = float(parts[0])
                gps_data.append((ts, line))
            except:
                pass
    gps_data.sort()
    print(f"📡 GPS: {len(gps_data)} 条记录")
    print(f"   时间: {gps_data[0][0]} → {gps_data[-1][0]}")

    # 找所有 map_XX 目录
    map_dirs = sorted(glob.glob(os.path.join(test_dir, "map_*")))
    if not map_dirs:
        print("❌ 找不到 map_XX 目录")
        sys.exit(1)

    print(f"\n📁 共 {len(map_dirs)} 个地图目录\n")

    for map_dir in map_dirs:
        map_name = os.path.basename(map_dir)
        times_path = os.path.join(map_dir, "times.txt")
        if not os.path.exists(times_path):
            print(f"  {map_name}: 无 times.txt，跳过")
            continue

        # 读时间范围
        times = []
        with open(times_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    times.append(float(parts[0]))
        if not times:
            print(f"  {map_name}: times.txt 为空")
            continue

        t_start, t_end = times[0], times[-1]

        # 过滤 GPS 数据
        filtered = [line for ts, line in gps_data if t_start <= ts <= t_end]

        if filtered:
            out_path = os.path.join(map_dir, "gps_segment.csv")
            with open(out_path, "w") as f:
                f.write(header + "\n")
                for line in filtered:
                    f.write(line + "\n")
            dur = t_end - t_start
            print(f"  {map_name}: {t_start:.3f} → {t_end:.3f} ({dur:.1f}s) → {len(filtered)} GPS → gps_segment.csv")
        else:
            print(f"  {map_name}: {t_start:.3f} → {t_end:.3f} (无匹配GPS)")

    print("\n✅ 完成")

if __name__ == "__main__":
    main()
