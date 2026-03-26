#!/usr/bin/env python3
"""
快速测试脚本：验证NPY文件读取功能
"""

import numpy as np
import os
import sys

npy_dir = "/home/zah/ORB_SLAM3-master/test/videos/0123232/Dataset_Final_165123_npy"
num_frames = 5

print("=" * 60)
print("NPY 文件读取测试")
print("=" * 60)

if not os.path.exists(npy_dir):
    print(f"错误: NPY 目录不存在: {npy_dir}")
    sys.exit(1)

for frame_idx in range(1, num_frames + 1):
    npy_file = os.path.join(npy_dir, f"frame_{frame_idx:06d}.npy")
    
    if not os.path.exists(npy_file):
        print(f"Frame {frame_idx}: 文件不存在")
        continue
    
    try:
        data = np.load(npy_file, allow_pickle=True).item()
        refined_kpts = data.get('refined_kpts', data.get('raw_kpts'))
        
        print(f"\nFrame {frame_idx}:")
        print(f"  Frame Index: {data.get('frame_idx')}")
        print(f"  Corner Points (refined_kpts):")
        for i, pt in enumerate(refined_kpts):
            print(f"    Corner {i}: ({pt[0]:.2f}, {pt[1]:.2f})")
        print(f"  Confidence: {data.get('conf', 'N/A')}")
        print(f"  Has Detection: {data.get('has_detection')}")
        
    except Exception as e:
        print(f"Frame {frame_idx}: 错误 - {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
