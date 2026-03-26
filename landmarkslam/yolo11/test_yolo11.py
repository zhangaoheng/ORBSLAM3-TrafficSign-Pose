import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import random

# ================= 配置区域 =================

# 训练好的模型路径 (请确认路径是否正确)
MODEL_PATH = r"d:\zah\Latex\yolo11\road_sign_project\yolo11n_run1\weights\best.pt"

# 测试集图片文件夹
TEST_IMAGES_DIR = r"d:\zah\Latex\yolo_dataset\images\realval"

# 输出结果文件夹
OUTPUT_DIR = r"d:\zah\Latex\yolo11\inference_results"

# 随机抽取的测试数量
SAMPLE_COUNT = 10

# ===========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_inference():
    ensure_dir(OUTPUT_DIR)
    
    # 1. 加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        return
    
    print(f"正在加载模型: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    
    # 2. 获取测试图片
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"错误: 测试集目录不存在 {TEST_IMAGES_DIR}")
        return

    all_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not all_images:
        print("测试集中没有找到图片。")
        return
        
    # 随机抽取
    selected_images = random.sample(all_images, min(SAMPLE_COUNT, len(all_images)))
    print(f"将对 {len(selected_images)} 张图片进行推理测试...")

    # 3. 执行推理
    for img_name in selected_images:
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        
        # conf=0.5: 置信度阈值
        # save=True: 自动保存结果到 runs/detect/... (我们这里手动处理以便保存到指定位置)
        results = model.predict(source=img_path, conf=0.1, save=False)
        
        for result in results:
            # 绘制结果 (返回 numpy 数组)
            im_array = result.plot()
            
            # 保存图片
            save_path = os.path.join(OUTPUT_DIR, f"pred_{img_name}")
            cv2.imwrite(save_path, im_array)
            print(f"已保存结果: {save_path}")
            
            # 打印检测到的框信息
            boxes = result.boxes
            if len(boxes) > 0:
                for box in boxes:
                    conf = box.conf.item()
                    cls = box.cls.item()
                    xyxy = box.xyxy.tolist()[0]
                    print(f"  - 检测到目标: 类别={int(cls)}, 置信度={conf:.4f}, 坐标={xyxy}")
            else:
                print("  - 未检测到目标")

    print(f"\n所有推理完成！结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()
