#!/usr/bin/env python3
"""
将 extract_bag.py 提取的数据转换为 ORB-SLAM3 EuRoC 格式。

运行方式:
  python3 convert_to_euroc.py <extracted_data_dir>

示例:
  python3 convert_to_euroc.py ~/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data/
"""
import os, sys, shutil
from pathlib import Path

data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

euroc_dir = data_dir / "euroc_format"
mav0 = euroc_dir / "mav0"
cam0_dir = mav0 / "cam0" / "data"
imu0_dir = mav0 / "imu0"

cam0_dir.mkdir(parents=True, exist_ok=True)
imu0_dir.mkdir(parents=True, exist_ok=True)

# 1. Convert times.txt → EuRoC times file (timestamp_ns per line) + symlink images
times_in = data_dir / "times.txt"
times_out = mav0 / "cam0" / "data.csv"
rgb_dir = data_dir / "rgb"

print(f"📂 读取 times: {times_in}")
print(f"📂 图像源目录: {rgb_dir}")

count = 0
with open(times_in) as fin, open(times_out, 'w') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        t_sec = float(parts[0])        # seconds
        img_name = parts[1]            # e.g. rgb/000000.png
        base_name = img_name.split('/')[-1]  # 000000.png

        t_ns = int(t_sec * 1e9)
        euRoC_name = f"{t_ns}.png"

        fout.write(f"{t_ns}\n")

        # Link image (don't copy — save disk)
        src = rgb_dir / base_name
        dst = cam0_dir / euRoC_name
        if not dst.exists():
            os.symlink(os.path.relpath(src, cam0_dir), dst)
        count += 1

print(f"✅ 已创建 {count} 个图像链接 → {cam0_dir}")
print(f"✅ times 文件 → {times_out}")

# 2. Convert imu.txt (space-sep) → data.csv (comma-sep)
imu_in = data_dir / "imu.txt"
imu_out = imu0_dir / "data.csv"

count_imu = 0
with open(imu_in) as fin, open(imu_out, 'w') as fout:
    for line in fin:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # format: timestamp_ns w_x w_y w_z a_x a_y a_z
        parts = line.split()
        fout.write(','.join(parts) + '\n')
        count_imu += 1

print(f"✅ IMU 数据 → {imu_out} ({count_imu} 条)")

print(f"\n🎉 EuRoC 格式数据就绪: {euroc_dir}")
print(f"   运行命令:")
print(f"   cd ~/ORB_SLAM3-master")
print(f"   ./Examples/Monocular-Inertial/mono_inertial_euroc \\")
print(f"     Vocabulary/ORBvoc.txt \\")
print(f"     Examples/Monocular-Inertial/RealSense_D456i.yaml \\")
print(f"     {euroc_dir} \\")
print(f"     {euroc_dir}/mav0/cam0/data.csv")
