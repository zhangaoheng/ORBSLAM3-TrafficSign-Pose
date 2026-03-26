from ultralytics import YOLO
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 =================

# 训练好的模型路径
MODEL_PATH = r"D:\zah\Latex\road_sign_pose_project\yolo11n_pose_run2\weights\best.pt"

# 测试图片路径 (可以是单张图片，也可以是文件夹)
# 这里建议您放几张真实拍摄的路牌图片，或者使用验证集中的图片
TEST_SOURCE = r"D:\zah\Latex\unityDataset\VideoFrames\base" 
# 如果想测试单张图片，可以改为: r"path/to/your/image.jpg"

# 输出保存路径
OUTPUT_DIR = r"D:\zah\Latex\yolo11\pose_inference_results_with_roi"

# 带框绘制结果目录
DRAW_DIR = os.path.join(OUTPUT_DIR, 'predictions_with_boxes')

# 关键点抠图结果目录（四个点包围的区域）
ROI_DIR = os.path.join(OUTPUT_DIR, 'roi_extracted')

# ===========================================

def extract_roi_with_mask(image, keypoints):
    """
    根据前4个关键点生成掩码，提取ROI区域
    保持原始位置不变，背景为黑色
    
    Args:
        image: 输入图像
        keypoints: 检测到的关键点数组（Tensor或numpy）
    
    Returns:
        roi_image: 抠图后的图像
    """
    if keypoints is None or len(keypoints) < 4:
        return image
    
    # 创建掩码
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 提取前4个关键点，转换为numpy数组
    if hasattr(keypoints, 'cpu'):
        # 如果是PyTorch Tensor
        poly_points = keypoints[:4].cpu().numpy().astype(np.int32)
    else:
        # 如果已经是numpy数组
        poly_points = keypoints[:4].astype(np.int32)
    
    # 用这4个点绘制多边形并填充
    cv2.fillPoly(mask, [poly_points], 255)
    
    # 应用掩码提取ROI（保持原位置不变）
    roi_image = cv2.bitwise_and(image, image, mask=mask)
    
    return roi_image

def main():
    # 1. 加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        return
    
    print(f"正在加载模型: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    # 2. 运行推理
    print(f"开始推理: {TEST_SOURCE} ...")
    results = model.predict(source=TEST_SOURCE, save=False, project=OUTPUT_DIR, name='predict_run', conf=0.1)

    # 创建输出目录
    if not os.path.exists(DRAW_DIR):
        os.makedirs(DRAW_DIR)
    
    if not os.path.exists(ROI_DIR):
        os.makedirs(ROI_DIR)

    print(f"正在处理结果并保存...")
    for result in results:
        # 1. 绘制带框的结果
        im_array = result.plot(line_width=1, kpt_radius=2)
        
        file_name = os.path.basename(result.path)
        
        # 保存绘制结果（带框）
        draw_save_path = os.path.join(DRAW_DIR, file_name)
        cv2.imwrite(draw_save_path, im_array)
        print(f"已保存绘制结果: {draw_save_path}")
        
        # 2. 根据关键点抠图
        if result.keypoints is not None and hasattr(result.keypoints, 'xy'):
            kpts = result.keypoints.xy
            if kpts is not None and len(kpts) > 0:
                # 需要从原始图像中进行抠图
                original_img = cv2.imread(result.path)
                if original_img is None:
                    print(f"警告: 无法读取原始图像 {result.path}")
                    continue
                
                # 为每个检测的结果单独抠图
                for idx, det_kpts in enumerate(kpts):
                    roi_img = extract_roi_with_mask(original_img, det_kpts)
                    
                    # 构造输出文件名（添加检测序号）
                    name_without_ext = os.path.splitext(file_name)[0]
                    ext = os.path.splitext(file_name)[1]
                    roi_file_name = f"{name_without_ext}_roi_det{idx}{ext}"
                    
                    roi_save_path = os.path.join(ROI_DIR, roi_file_name)
                    cv2.imwrite(roi_save_path, roi_img)
                    print(f"已保存抠图结果: {roi_save_path}")

    print(f"\n推理完成！")
    print(f"- 绘制结果已保存到: {DRAW_DIR}")
    print(f"- 抠图结果已保存到: {ROI_DIR}")

if __name__ == "__main__":
    main()
