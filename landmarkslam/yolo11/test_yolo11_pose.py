from ultralytics import YOLO
import cv2
import time
import time
import os
import matplotlib.pyplot as plt

# ================= 配置区域 =================

# 训练好的模型路径
MODEL_PATH = r"/home/zah/ORB_SLAM3-master/landmarkslam/yolo11/best.pt"


# 测试图片路径 (可以是单张图片，也可以是文件夹)
# 这里建议您放几张真实拍摄的路牌图片，或者使用验证集中的图片
TEST_SOURCE = r"/home/zah/ORB_SLAM3-master/landmarkslam/data/20260321_111801" 
# 如果想测试单张图片，可以改为: r"path/to/your/image.jpg"

# 输出保存路径
OUTPUT_DIR = r"/home/zah/ORB_SLAM3-master/landmarkslam/yolo11_pose_results"

# ===========================================

def main():
    # 1. 加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        return
    
    print(f"正在加载模型: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    # 2. 运行推理
    # save=False 手动保存以控制绘图样式
    # conf=0.25 设置置信度阈值
    print(f"开始推理: {TEST_SOURCE} ...")
    results = model.predict(source=TEST_SOURCE, save=False, project=OUTPUT_DIR, name='predict_run', conf=0.6)

    # 手动保存结果：以时间戳命名建立独立的输出文件夹
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"正在保存绘制结果到 {save_dir} ...")
    
    # +++ 新增：将关键点保存为 txt 文件供 C++ 读取 +++
    txt_save_path = os.path.join(save_dir, "yolo_keypoints.txt")
    with open(txt_save_path, "w") as f_txt:
        for result in results:
            # 绘制图像
            im_array = result.plot(line_width=1, kpt_radius=2)
            
            # 获取文件名
            file_name = os.path.basename(result.path)
            
            # 保存图像
            save_path = os.path.join(save_dir, file_name)
            cv2.imwrite(save_path, im_array)
            
            # 提取关键点
            if result.keypoints is not None and len(result.keypoints.xy) > 0 and len(result.keypoints.xy[0]) > 0:
                # 假设每个图最多检测到一个路牌
                kpts = result.keypoints.xy[0].cpu().numpy() # [N, 2]
                if len(kpts) == 4:
                    # 写入 txt，格式: image_name.png x1 y1 x2 y2 x3 y3 x4 y4
                    line = f"{file_name}"
                    for pt in kpts:
                        line += f" {pt[0]:.2f} {pt[1]:.2f}"
                    f_txt.write(line + "\n")
    # +++ 结束新增 +++

    print(f"\n推理完成！结果已保存到: {save_dir}")

    

if __name__ == "__main__":
    main()
