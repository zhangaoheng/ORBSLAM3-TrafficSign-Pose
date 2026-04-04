import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def extract_subpixel_orthogonal_lines(image_path, gamma=2.5, clahe_clip=4.0, scale_factor=2.5, angle_tol=15, min_length=3.0):
    """
    亚像素级双主轴正交线段提取器 (融合升维提取与逆向降维映射)
    
    :param image_path: 图像路径
    :param gamma: Gamma 提亮系数
    :param clahe_clip: CLAHE 局部对比度增强系数
    :param scale_factor: 空间升维倍数 (在更高分辨率下提取连续梯度)
    :param angle_tol: 正交聚类容差度数 (水平: 0±tol, 垂直: 90±tol)
    :param min_length: 映射回原图后的最小物理长度阈值 (过滤噪点)
    :return: horiz_lines (水平线段集), vert_lines (垂直线段集)
    """
    print(f"==================================================")
    print(f"🚀 启动亚像素级双主轴特征提取管道")
    print(f"==================================================")
    
    img = cv2.imread(image_path)
    if img is None: raise ValueError("图像读取失败，请检查路径！")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ---------------------------------------------------------
    # 第一阶段：光度增强 (Gamma + CLAHE 寻找最佳梯度断层)
    # ---------------------------------------------------------
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(gray, table)
    
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gamma)

    # ---------------------------------------------------------
    # 第二阶段：空间升维 (Bicubic Upsampling)
    # ---------------------------------------------------------
    h, w = img_clahe.shape
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    img_upscaled = cv2.resize(img_clahe, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 轻微平滑以消除插值锯齿，使 LSD 梯度计算更顺滑
    img_upscaled = cv2.GaussianBlur(img_upscaled, (3, 3), 0)

    # ---------------------------------------------------------
    # 第三阶段：高维空间 LSD 提取
    # ---------------------------------------------------------
    lsd = cv2.createLineSegmentDetector(0)
    lines_up, _, _, _ = lsd.detect(img_upscaled)
    
    # ---------------------------------------------------------
    # 第四阶段：坐标逆映射与结构化聚类 (核心理论落地！)
    # ---------------------------------------------------------
    horiz_lines = []
    vert_lines = []
    discarded_count = 0
    
    if lines_up is not None:
        for line in lines_up:
            # 🛡️【核心保护机制】：数学坐标逆映射，退回真实相机物理模型空间
            x1, y1, x2, y2 = line[0] / scale_factor
            
            # 计算映射回原图后的真实物理像素长度
            length = math.hypot(x2 - x1, y2 - y1)
            if length < min_length:
                discarded_count += 1
                continue # 太短的废线丢弃
                
            # 计算亚像素坐标下的线段绝对角度 (0~180度)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angle = abs(angle)
            if angle > 180: angle -= 180
                
            # 🌟【双主轴大浪淘沙】：剥离非正交笔画 (撇、捺、圆圈)
            if angle <= angle_tol or angle >= (180 - angle_tol):
                # 记录格式: [x1, y1, x2, y2, length置信度权重]
                horiz_lines.append((x1, y1, x2, y2, length))
            elif abs(angle - 90) <= angle_tol:
                vert_lines.append((x1, y1, x2, y2, length))
            else:
                discarded_count += 1
                
    print(f"✅ 提取完成！")
    print(f"   - 原始图像分辨率: {w}x{h}")
    print(f"   - 升维图像分辨率: {new_w}x{new_h} (Scale: {scale_factor}x)")
    print(f"   - 提取有效水平线段 (Red):   {len(horiz_lines)} 条")
    print(f"   - 提取有效垂直线段 (Green): {len(vert_lines)} 条")
    print(f"   - 过滤非正交干扰 (废弃):  {discarded_count} 条")

    # ---------------------------------------------------------
    # 第五阶段：在原分辨率图像上验证可视化结果
    # ---------------------------------------------------------
    # 将 CLAHE 图像转为彩图作为底图，在此之上画线
    vis_img = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)
    
    # 降低底图亮度，让红绿线更加刺眼醒目
    vis_img = (vis_img * 0.4).astype(np.uint8)

    # 绘制水平线段 (红色)
    for x1, y1, x2, y2, _ in horiz_lines:
        cv2.line(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        
    # 绘制垂直线段 (绿色)
    for x1, y1, x2, y2, _ in vert_lines:
        cv2.line(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Sub-pixel Orthogonal Line Clustering (H:{len(horiz_lines)} | V:{len(vert_lines)})", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return horiz_lines, vert_lines

if __name__ == "__main__":
    # 替换为你那张黑漆漆的标志牌测试图路径
    TEST_IMAGE_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/capture_11204_1774600247430027000.jpg" 
    
    try:
        h_lines, v_lines = extract_subpixel_orthogonal_lines(
            TEST_IMAGE_PATH, 
            scale_factor=2.5,   # 放大2.5倍提线
            angle_tol=15,       # 容忍15度的字体倾斜
            min_length=3.0      # 回归原图后，长度必须大于3个像素才算有效笔画
        )
    except Exception as e:
        print(f"Error: {e}")