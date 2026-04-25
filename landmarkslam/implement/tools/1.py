import os
import glob

# 替换为你的 lines2 路径
DATA_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/deepseek/lines1"

def make_associations():
    # 找 rgb 文件夹下的所有图片
    files = glob.glob(os.path.join(DATA_DIR, "rgb", "*.png"))
    files.sort() # 按时间戳排序
    
    assoc_file = os.path.join(DATA_DIR, "associations.txt")
    with open(assoc_file, 'w') as f:
        for filepath in files:
            filename = os.path.basename(filepath)
            # 提取数字部分（纳秒），转换为秒
            ns_str = filename.split('.')[0]
            sec = float(ns_str) / 1e9 
            
            # 严格按照 TUM 格式：时间戳 rgb路径 时间戳 depth路径
            # 注意路径是相对 DATA_DIR 的
            f.write(f"{sec:.6f} rgb/{filename} {sec:.6f} depth/{filename}\n")
            
    print(f"✅ 成功生成 associations.txt，共包含 {len(files)} 帧对应关系！")

if __name__ == "__main__":
    make_associations()