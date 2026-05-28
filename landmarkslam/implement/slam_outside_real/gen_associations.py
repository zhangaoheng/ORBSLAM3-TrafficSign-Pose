#!/usr/bin/env python3
"""从 camera_timestamps.csv 生成 associations.txt（TUM 格式）"""

import csv
import os

base = os.path.dirname(__file__)
csv_path = os.path.join(base, "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data/camera_timestamps.csv")
assoc_path = os.path.join(base, "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data/associations.txt")

count = 0
with open(assoc_path, "w") as fout:
    with open(csv_path) as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            frame_id = row["frame_id"]
            ts_ms = float(row["timestamp_ms"])
            ts_s = ts_ms / 1000.0
            fout.write(f"{ts_s:.6f} rgb/{frame_id} {ts_s:.6f} depth/{frame_id}\n")
            count += 1

print(f"✅ associations.txt 生成完毕，共 {count} 帧")