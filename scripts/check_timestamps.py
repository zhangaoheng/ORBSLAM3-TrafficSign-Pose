#!/usr/bin/env python3
"""离线 EuRoC 数据集时间戳质量检测：图像 Delta、IMU Delta、首帧对齐"""
import sys
from pathlib import Path

def load_times_txt(path):
    ts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ts.append(int(line) / 1e9)  # ns -> s
    return ts

def load_imu_csv(path):
    ts = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            ts.append(int(line.split(',')[0]) / 1e9)
    return ts

def check_deltas(name, ts_list, expected_hz):
    """检查时间戳间隔的异常"""
    if len(ts_list) < 2:
        print(f"  ⚠️ {name}: 数据不足")
        return

    deltas = [ts_list[i+1] - ts_list[i] for i in range(len(ts_list)-1)]
    expected_dt = 1.0 / expected_hz
    max_delta = max(deltas)
    min_delta = min(deltas)
    avg = sum(deltas) / len(deltas)

    jumps = [(i, d) for i, d in enumerate(deltas) if d > expected_dt * 3]
    gaps  = [(i, d) for i, d in enumerate(deltas) if d < 0]

    print(f"  {name}: {len(ts_list)} 条, Delta avg={avg*1000:.2f}ms, "
          f"min={min_delta*1000:.2f}ms, max={max_delta*1000:.2f}ms")
    if jumps:
        print(f"  ⚠️ 大跳变 ({len(jumps)} 处):")
        for i, d in jumps[:5]:
            print(f"      #{i}: {d*1000:.0f}ms (正常={expected_dt*1000:.0f}ms)")
    if gaps:
        print(f"  🔴 时间戳倒退 ({len(gaps)} 处)!")
        for i, d in gaps[:3]:
            print(f"      #{i}: {d*1000:.0f}ms")
    if not jumps and not gaps:
        print(f"  ✅ 时间戳正常")

def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('test1')

    print(f"=== 数据集: {data_dir} ===\n")

    # 1. 图像时间戳
    times_txt = data_dir / 'times.txt'
    if times_txt.exists():
        img_ts = load_times_txt(times_txt)
        check_deltas("图像", img_ts, expected_hz=30)
    else:
        print("  ❌ times.txt 不存在")
        return

    # 2. IMU 时间戳
    imu_csv = data_dir / 'mav0' / 'imu0' / 'data.csv'
    if imu_csv.exists():
        imu_ts = load_imu_csv(imu_csv)
        check_deltas("IMU", imu_ts, expected_hz=200)
    else:
        print("  ❌ IMU data.csv 不存在")
        return

    # 3. 首帧对齐
    print(f"\n--- 首帧对齐 ---")
    print(f"  第一帧图像: {img_ts[0]:.6f}s")
    print(f"  第一条 IMU: {imu_ts[0]:.6f}s")
    diff = img_ts[0] - imu_ts[0]
    print(f"  时间差: {diff*1000:.1f}ms {'✅' if diff > 0 else '🔴 (IMU 在图像之后！)'}")

    # 4. 末帧对齐
    print(f"\n--- 末帧对齐 ---")
    print(f"  最后一帧图像: {img_ts[-1]:.6f}s")
    print(f"  最后一条 IMU: {imu_ts[-1]:.6f}s")
    tail_diff = imu_ts[-1] - img_ts[-1]
    print(f"  尾部余量: {tail_diff*1000:.1f}ms {'✅' if tail_diff > 0 else '🔴 (IMU 数据不足!)'}")

    # 5. 总时长
    print(f"\n--- 总览 ---")
    print(f"  图像: {len(img_ts)} 帧, 时长 {img_ts[-1]-img_ts[0]:.1f}s")
    print(f"  IMU:  {len(imu_ts)} 条, 时长 {imu_ts[-1]-imu_ts[0]:.1f}s")

if __name__ == '__main__':
    main()
