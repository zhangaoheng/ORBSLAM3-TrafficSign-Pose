#!/usr/bin/env python3
"""
批量运行脚本 — 遍历 test_pairs 所有数据对

模式:
  --annotate   打开四角点标注（seq1 + seq2）
  --run        运行 test.py 核心流程（需要已标注）
  --all        标注 + 运行全部流程

用法: python3 batch_test.py --all
"""
import os, sys, glob, shutil, subprocess

BASE = "/home/zah/ORB_SLAM3-master"
TEST_PAIRS = os.path.join(BASE, "landmarkslam/implement/data/test_pairs")
ANNOTATOR  = os.path.join(BASE, "landmarkslam/implement/tools/annotate_corners.py")
TEST_PY    = os.path.join(BASE, "landmarkslam/implement/main/test.py")
VENV_PY    = os.path.join(BASE, "landmarkslam/yolo_venv/bin/python3")

def find_pairs():
    """扫描 test_pairs 目录"""
    if not os.path.isdir(TEST_PAIRS):
        print(f"❌ 找不到: {TEST_PAIRS}")
        return []
    pairs = sorted([d for d in os.listdir(TEST_PAIRS) if d.startswith("pair_")])
    return pairs

def annotate_pair(pair_name):
    """标注 seq1（seq2 不需要标注，只需选帧）"""
    pair_dir = os.path.join(TEST_PAIRS, pair_name)
    rgb_dir = os.path.join(pair_dir, "seq1", "rgb")
    corners_file = os.path.join(rgb_dir, "corners.txt")
    if os.path.exists(corners_file) and os.path.getsize(corners_file) > 0:
        cnt = sum(1 for _ in open(corners_file))
        print(f"   seq1: 已有 {cnt} 个标注，跳过")
        return
    print(f"\n🎯 标注 {pair_name}/seq1 (只需要标序列1)")
    print(f"   [空格]开始  [鼠标]点4角点  [A/D]翻页  [Q]保存退出")
    subprocess.run([VENV_PY, ANNOTATOR, rgb_dir], cwd=BASE)

def run_pair(pair_name):
    """运行一对数据的计算流程"""
    pair_dir = os.path.join(TEST_PAIRS, pair_name)
    config = os.path.join(pair_dir, "config.yaml")
    if not os.path.exists(config):
        print(f"   ❌ 无 config.yaml，跳过")
        return False

    # 只检查 seq1 是否有标注（seq2 不需要）
    corners1 = os.path.join(pair_dir, "seq1", "rgb", "corners.txt")
    if not os.path.exists(corners1) or os.path.getsize(corners1) == 0:
        print(f"   ❌ seq1 未标注，跳过")
        return False
    cnt1 = sum(1 for _ in open(corners1))
    print(f"   seq1: {cnt1} 标注")

    # 替换 config，清除缓存
    main_dir = os.path.dirname(TEST_PY)
    orig_config = os.path.join(main_dir, "config.yaml")
    
    # 备份原 config（只在第一次运行前备份）
    backup_path = os.path.join(main_dir, "config.yaml.bak")
    if not os.path.exists(backup_path) and os.path.exists(orig_config):
        shutil.copy2(orig_config, backup_path)
    
    # 写入当前 pair 的 config
    with open(config) as f:
        with open(orig_config, "w") as out:
            out.write(f.read())
    
    # 清除所有缓存
    for fname in ["cache.json", "selected_frames_cache.json", "experiment_results.txt"]:
        fp = os.path.join(main_dir, fname)
        if os.path.exists(fp):
            os.remove(fp)
            print(f"   清除: {fname}")
    
    print(f"   运行 test.py ...")
    ret = subprocess.run([VENV_PY, TEST_PY], cwd=BASE)
    
    # 恢复 config
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, orig_config)

    # 移动结果到 pair 目录
    runs_dir = os.path.join(os.path.dirname(TEST_PY), "runs")
    if os.path.isdir(runs_dir):
        latest = sorted(os.listdir(runs_dir))[-1] if os.listdir(runs_dir) else None
        if latest:
            src = os.path.join(runs_dir, latest)
            dst = os.path.join(pair_dir, "result")
            if not os.path.exists(dst):
                os.rename(src, dst)
                print(f"   结果已保存: {dst}")

    return ret.returncode == 0

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate", action="store_true", help="只标注")
    parser.add_argument("--run", action="store_true", help="只运行")
    parser.add_argument("--all", action="store_true", help="标注+运行")
    parser.add_argument("--pair", type=str, help="指定数据对 (如 pair_03_47)")
    args = parser.parse_args()

    pairs = find_pairs()
    if not pairs:
        print("❌ 未找到数据对")
        return

    if args.pair:
        if args.pair in pairs:
            pairs = [args.pair]
        else:
            print(f"❌ 找不到: {args.pair}")
            return

    print(f"\n📋 共 {len(pairs)} 对:")
    for p in pairs:
        print(f"   {p}")

    do_annotate = args.annotate or args.all
    do_run = args.run or args.all

    if not do_annotate and not do_run:
        print("\n用法: --annotate | --run | --all")
        return

    for i, pair_name in enumerate(pairs):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(pairs)}] {pair_name}")
        print("="*60)

        if do_annotate:
            annotate_pair(pair_name)

        if do_run:
            ok = run_pair(pair_name)
            if not ok:
                print(f"   ❌ {pair_name} 失败")

    print(f"\n✅ 完成！结果: {TEST_PAIRS}")

if __name__ == "__main__":
    main()
