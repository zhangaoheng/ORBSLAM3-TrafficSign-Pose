#!/home/zah/ORB_SLAM3-master/landmarkslam/yolo_venv/bin/python3
"""检查深度图的有效距离分布"""
import cv2
import numpy as np
import os, random

depth_dir = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data_new/depth"
files = sorted(os.listdir(depth_dir))

random.seed(42)
samples = random.sample(range(len(files)), 30)

print("=== 深度图有效距离分析 ===")
print("RealSense D456 有效量程: 0.6 ~ 6m")
print()

all_depths = []
for idx in samples:
    f = files[idx]
    img = cv2.imread(os.path.join(depth_dir, f), cv2.IMREAD_UNCHANGED)
    if img is None: continue
    vals = img[img > 0].astype(np.float32) / 1000.0  # mm -> m
    all_depths.extend(vals.tolist())
    
    pct_near = np.sum((vals > 0) & (vals < 3)) / len(vals) * 100 if len(vals) > 0 else 0
    pct_mid = np.sum((vals >= 3) & (vals < 6)) / len(vals) * 100 if len(vals) > 0 else 0
    pct_far = np.sum(vals >= 6) / len(vals) * 100 if len(vals) > 0 else 0
    
    print(f"帧{idx:5d}: 有效点数={len(vals):6d}  0-3m={pct_near:5.1f}%  3-6m={pct_mid:5.1f}%  >6m={pct_far:5.1f}%")

all_depths = np.array(all_depths)
print(f"\n{'='*50}")
print(f"总有效点: {len(all_depths)}")
print(f"深度范围: {all_depths.min():.2f}m ~ {all_depths.max():.2f}m")
print(f"平均深度: {all_depths.mean():.2f}m")
print(f"中位深度: {np.median(all_depths):.2f}m")
print(f"0-3m:   {np.sum(all_depths < 3)/len(all_depths)*100:.1f}%")
print(f"3-6m:   {np.sum((all_depths >= 3) & (all_depths < 6))/len(all_depths)*100:.1f}%")
print(f"6-10m:  {np.sum((all_depths >= 6) & (all_depths < 10))/len(all_depths)*100:.1f}%")
print(f">10m:   {np.sum(all_depths >= 10)/len(all_depths)*100:.1f}%")
