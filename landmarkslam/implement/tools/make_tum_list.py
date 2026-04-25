import os
import glob

# 替换为你的 lines2 路径
DATA_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/deepseek/lines1"

def write_list(folder_name, output_file):
    folder_path = os.path.join(DATA_DIR, folder_name)
    files = glob.glob(os.path.join(folder_path, "*.png"))
    files.sort() # 确保按时间顺序排列
    
    with open(os.path.join(DATA_DIR, output_file), 'w') as f:
        for filepath in files:
            filename = os.path.basename(filepath)
            # 提取数字部分（纳秒），转换为秒
            ns_str = filename.split('.')[0]
            # 假设 timestamp_ns 是 16 位数字，将其除以 1e9 转为秒
            sec = float(ns_str) / 1e9 
            # 写入：时间戳(秒) 相对路径
            f.write(f"{sec:.6f} {folder_name}/{filename}\n")

if __name__ == "__main__":
    write_list("rgb", "rgb.txt")
    write_list("depth", "depth.txt")
    print("✅ rgb.txt 和 depth.txt 生成完毕！")