#!/usr/bin/env python3
"""
紧急恢复脚本：将 saved_rois.txt 中被意外调换顺序的坐标，
恢复成你最初的标注格式（x1,y1,x2,y2 的原始点击顺序）。
脚本假设正确的原始顺序特点是 x1 > x2 且 y1 > y2，
对于不满足该条件的行或误转的8数字行，都会自动纠正。
"""

import re

ROI_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/deepseek/lines1/rgb/saved_rois.txt"

def restore_line(line):
    line = line.strip()
    if not line:
        return ''
    
    nums = [int(float(x)) for x in re.split(r'[,\s]+', line) if x]
    
    if len(nums) == 4:
        x1, y1, x2, y2 = nums
        # 如果是被错误排序成 x1<x2 且 y1<y2，则调换回原始顺序
        if x1 < x2 and y1 < y2:
            return f"{x2},{y2},{x1},{y1}"
        else:
            # 保持原样
            return line
    elif len(nums) == 8:
        # 8数字四边形：提取外接矩形，然后按原始顺序习惯输出（右下,左下,左上,右上？不，还是用x2,y2,x1,y1）
        # 取外接矩形左右上下
        xs = [nums[i] for i in range(0, 8, 2)]
        ys = [nums[i] for i in range(1, 8, 2)]
        left = min(xs)
        right = max(xs)
        top = min(ys)
        bottom = max(ys)
        # 按照你原始标注习惯：x1=right, y1=bottom, x2=left, y2=top
        return f"{right},{bottom},{left},{top}"
    else:
        # 格式不对，保留原样并警告
        print(f"⚠️ 跳过无法识别的行: {line}")
        return line

def main():
    print("📂 读取文件...")
    try:
        with open(ROI_FILE, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ 找不到文件 {ROI_FILE}")
        return
    
    restored = []
    fixed_count = 0
    for line in lines:
        new_line = restore_line(line)
        restored.append(new_line)
        if new_line != line.strip():
            fixed_count += 1
    
    with open(ROI_FILE, 'w') as f:
        for l in restored:
            f.write(l + '\n')
    
    print(f"✅ 修复完成！共处理 {len(lines)} 行，修正了 {fixed_count} 行。")
    print(f"   文件已保存到 {ROI_FILE}")
    print("   现在所有行的格式都已恢复为你最初的标注习惯。")

if __name__ == "__main__":
    main()