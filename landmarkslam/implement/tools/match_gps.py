#!/usr/bin/env python3
"""为 manual_split 每个段匹配 GPS 数据"""
import os, sys, csv, numpy as np
from pathlib import Path

GPS_CSV = sys.argv[1] if len(sys.argv) > 1 else 'test1.csv'
SPLIT_DIR = sys.argv[2] if len(sys.argv) > 2 else 'manual_split'

# 加载 GPS
gps = []
with open(GPS_CSV) as f:
    for line in f:
        if line.startswith('timestamp'):
            continue
        p = line.strip().split(',')
        if len(p) < 4:
            continue
        try:
            gps.append((float(p[0]), float(p[1]), float(p[2]), float(p[3]), int(p[4])))
        except ValueError:
            continue
gps_arr = np.array([g[0] for g in gps])
print(f"GPS: {len(gps)} 条")

for seg in sorted(Path(SPLIT_DIR).glob('seg_manual_*'),
                  key=lambda x: int(x.name.split('_')[-1])):
    times_path = seg / 'times.txt'
    if not times_path.exists():
        print(f"  {seg.name}: 无 times.txt, 跳过")
        continue

    # 读取图片时间戳范围
    img_ts = []
    with open(times_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = line.split()[0]
            try:
                ts = float(t)
                if ts > 1e14:
                    ts /= 1e9
                img_ts.append(ts)
            except ValueError:
                continue
    if not img_ts:
        continue
    t_min, t_max = img_ts[0], img_ts[-1]

    # 匹配 GPS
    idx = np.where((gps_arr >= t_min - 1) & (gps_arr <= t_max + 1))[0]
    if len(idx) == 0:
        print(f"  {seg.name}: 无 GPS 匹配 (t={t_min:.1f}~{t_max:.1f})")
        continue

    # 写 GPS 到段目录（seq1_gps.csv + seq2_gps.csv）
    for name in ('seq1_gps.csv', 'seq2_gps.csv'):
        out = seg / name
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'latitude', 'longitude', 'altitude', 'status'])
            for i in idx:
                w.writerow(gps[i])
    print(f"  {seg.name}: {len(idx)} GPS 条 → seq1/seq2")
