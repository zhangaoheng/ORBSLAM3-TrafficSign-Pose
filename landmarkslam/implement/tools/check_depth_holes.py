#!/home/zah/ORB_SLAM3-master/landmarkslam/yolo_venv/bin/python3
"""检查深度图中的黑洞（大面积零值区域）"""
import cv2
import numpy as np
import os, random

depth_dir = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data_new/depth"
files = sorted(os.listdir(depth_dir))

random.seed(42)
samples = random.sample(range(len(files)), 50)

print("=== 深度黑洞分析 ===")
print("黑洞定义: 连续>6m 的大面积零值/噪声区域")
print()

black_hole_frames = []
for idx in samples:
    f = files[idx]
    img = cv2.imread(os.path.join(depth_dir, f), cv2.IMREAD_UNCHANGED)
    if img is None: continue
    
    h, w = img.shape
    # 零值区域 = 深度黑洞（传感器没读到）
    zero_mask = (img == 0)
    zero_pct = np.sum(zero_mask) / (h * w) * 100
    
    # 检查中心区域是否有黑洞（中心 50% 区域）
    ch, cw = h // 4, w // 4
    center_region = zero_mask[ch:3*ch, cw:3*cw]
    center_zero = np.sum(center_region) / center_region.size * 100
    
    # 无效深度（>6m 的噪声值）
    invalid_mask = (img > 6000)  # mm > 6m
    invalid_pct = np.sum(invalid_mask) / (h * w) * 100
    
    flag = ""
    if zero_pct > 30 or center_zero > 20:
        flag = " ⚠️ 大黑洞"
    if zero_pct > 50:
        flag = " 🚨 严重黑洞"
    if invalid_pct > 60:
        flag += " +深度噪声>60%"
    
    black_hole_frames.append((idx, zero_pct, center_zero, invalid_pct, flag))

print(f"{'帧号':>6} {'零值%':>7} {'中心零值%':>9} {'>6m噪声%':>8}  标记")
print(f"{'-'*55}")
for idx, z, cz, inv, flag in black_hole_frames:
    print(f"{idx:6d}  {z:6.1f}%  {cz:8.1f}%  {inv:7.1f}%  {flag}")

# 统计
zeros = [z for _,z,_,_,_ in black_hole_frames]
invalids = [inv for _,_,_,inv,_ in black_hole_frames]
print(f"\n{'='*55}")
print(f"平均零值比例: {np.mean(zeros):.1f}%")
print(f"零值 >30% 的帧: {sum(1 for z in zeros if z > 30)}/{len(zeros)}")
print(f"零值 >50% 的帧: {sum(1 for z in zeros if z > 50)}/{len(zeros)}")
print(f"平均>6m噪声比例: {np.mean(invalids):.1f}%")
print(f"有效深度(0.6-6m)平均: {100 - np.mean(zeros) - np.mean(invalids):.1f}%")
