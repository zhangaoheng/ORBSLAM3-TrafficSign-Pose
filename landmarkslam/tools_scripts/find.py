import os
import shutil

# ================= 1. 配置参数 =================
# 原始包含所有单目图片的文件夹路径
source_dir = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/trash_bin" # ⚠️ 请替换为你真实的文件夹名
# 切片后保存的新文件夹名
target_dir = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data"

# 你手动挑出来的背光路段起止帧号 (包含)
start_idx = 6000  
end_idx = 6500    

os.makedirs(target_dir, exist_ok=True)

# ================= 2. 核心提取与重命名逻辑 =================
print(f">>> 正在从 {source_dir} 提取第 {start_idx} 到 {end_idx} 帧的单目图片...")

# 读取所有文件并按名字排序
files = sorted(os.listdir(source_dir))

new_idx = 0
processed_count = 0

for filename in files:
    # 只处理图片文件
    if not (filename.endswith('.jpg') or filename.endswith('.png')):
        continue

    # 按照下划线分割文件名
    # 例如 "capture_114_1774599289367080100.jpg" 会被分割成:
    # parts[0] = "capture"
    # parts[1] = "114"
    # parts[2] = "1774599289367080100.jpg"
    parts = filename.split('_')
    
    # 确保是我们要处理的格式
    if len(parts) >= 3 and parts[0] == 'capture':
        try:
            orig_idx = int(parts[1])
            timestamp_with_ext = parts[2] # 这个变量直接打包了"时间戳.后缀"
            
            # 判断是否在我们要提取的区间内
            if start_idx <= orig_idx <= end_idx:
                # 重新拼接新的文件名：capture_新序号_原时间戳.后缀
                new_filename = f"capture_{new_idx}_{timestamp_with_ext}"
                
                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(target_dir, new_filename)
                
                # copy2 会保留图片的原始创建时间属性
                shutil.copy2(src_path, dst_path)
                print(f"✅ {filename}  --->  {new_filename}")
                
                new_idx += 1
                processed_count += 1
                
        except ValueError:
            # 如果中间的不是数字，跳过该文件
            continue

print("==================================================")
print(f"🎉 提取完成！共提取并从 0 重新排列了 {processed_count} 张单目图片。")
print(f"📁 新的干净测试序列已保存在: {target_dir}")