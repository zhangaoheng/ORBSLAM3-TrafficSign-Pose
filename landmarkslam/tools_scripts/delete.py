import os
import shutil
import re

# ================= 配置区 =================
# 1. 图片所在的文件夹路径
IMAGE_FOLDER = r'/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243'

# 2. 定义要删除的【帧数范围】列表
# 格式为 (起始帧, 结束帧)，包含起始和结束。
# 例如: [(0, 1000), (4414, 5000)] 表示删除 0-1000 帧以及 4414-5000 帧。
DELETE_RANGES = [
    (0, 10943)  # 比如剔除中间一段被手挡住的
]

# 3. 处理模式：'move' (移动到垃圾桶，推荐) 或 'delete' (彻底删除)
MODE = 'move' 
# ==========================================

def clean_dataset():
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ 找不到文件夹: {IMAGE_FOLDER}")
        return

    # 创建垃圾桶文件夹
    trash_dir = os.path.join(IMAGE_FOLDER, 'trash_bin')
    if MODE == 'move' and not os.path.exists(trash_dir):
        os.makedirs(trash_dir)

    # 获取所有图片列表
    files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png'))]
    print(f"📊 文件夹中共有 {len(files)} 张图片。")

    count = 0
    # 正则表达式匹配 capture_数字_
    pattern = re.compile(r'capture_(\d+)_')

    for filename in files:
        match = pattern.search(filename)
        if not match:
            continue
        
        # 提取帧数序号
        frame_idx = int(match.group(1))

        # 检查该帧是否在任何一个删除范围内
        should_delete = False
        for start, end in DELETE_RANGES:
            if start <= frame_idx <= end:
                should_delete = True
                break
        
        if should_delete:
            src_path = os.path.join(IMAGE_FOLDER, filename)
            
            if MODE == 'move':
                # 移动到 trash_bin
                shutil.move(src_path, os.path.join(trash_dir, filename))
            else:
                # 彻底删除
                os.remove(src_path)
            
            count += 1

    print(f"\n✅ 清理完成！")
    if MODE == 'move':
        print(f"🗑️  已将 {count} 张无用图片移至: {trash_dir}")
    else:
        print(f"🔥 已永久删除 {count} 张图片。")
    print(f"✨ 剩余可用图片: {len(files) - count} 张。")

if __name__ == "__main__":
    clean_dataset()