import cv2
import numpy as np

# ==========================================
# 1. 鼠标交互工具：防抖 4点选区
# ==========================================
def select_four_points(img, window_name):
    points = []
    display_img = img.copy()
    print(f"\n👉 请在 '{window_name}' 中顺时针或逆时针点击区域外围的 4 个点。")

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                if len(points) > 0:
                    last_x, last_y = points[-1]
                    if np.sqrt((x - last_x)**2 + (y - last_y)**2) < 10:
                        return
                points.append((x, y))
                cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
                if len(points) > 1:
                    cv2.line(display_img, points[-2], points[-1], (0, 255, 0), 2)
                if len(points) == 4:
                    cv2.line(display_img, points[3], points[0], (0, 255, 0), 2)
                cv2.imshow(window_name, display_img)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display_img)
    cv2.setMouseCallback(window_name, mouse_callback)
    while len(points) < 4: cv2.waitKey(50) 
    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.int32)

# ==========================================
# 2. 核心功能：激进提取 + 质心距离过滤
# ==========================================
def detect_and_filter_lines_plslam(image_path, window_name):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return [], [], (0, 0, 0, 0)

    pts_poly = select_four_points(img, f"Select 4 Points: {window_name}")
    x, y, w, h = cv2.boundingRect(pts_poly)
    if w == 0 or h == 0: return [], [], (0, 0, 0, 0)
    img_roi = img[y:y+h, x:x+w]
    
    # ----------------------------------------
    # 预处理与提取
    # ----------------------------------------
    gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray_roi, 5, 50, 50)
    gaussian = cv2.GaussianBlur(smooth, (0, 0), 2.0)
    sharpened = cv2.addWeighted(smooth, 3.0, gaussian, -2.0, 0)
    
    lsd = cv2.createLineSegmentDetector(0, scale=2.0)
    lines, _, _, _ = lsd.detect(sharpened)

    result_img = img.copy()
    cv2.polylines(result_img, [pts_poly], isClosed=True, color=(0, 255, 0), thickness=2)

    # 临时存放通过了长度和角度测试的线段
    temp_horizontal = []
    temp_vertical = []
    
    MIN_LENGTH = 4.0 
    ANGLE_TOLERANCE = 12.0

    # ----------------------------------------
    # 第一阶段：基础的长度与角度过滤
    # ----------------------------------------
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            gx1, gy1 = float(x1 + x), float(y1 + y)
            gx2, gy2 = float(x2 + x), float(y2 + y)
            
            line_length = np.sqrt((gx2 - gx1)**2 + (gy2 - gy1)**2)
            if line_length < MIN_LENGTH: continue
            
            if cv2.pointPolygonTest(pts_poly, (gx1, gy1), False) >= 0 and \
               cv2.pointPolygonTest(pts_poly, (gx2, gy2), False) >= 0:
                
                dx, dy = gx2 - gx1, gy2 - gy1
                angle = np.degrees(np.arctan2(dy, dx))
                if angle < 0: angle += 180.0
                
                if angle < ANGLE_TOLERANCE or angle > (180.0 - ANGLE_TOLERANCE):
                    temp_horizontal.append([gx1, gy1, gx2, gy2])
                elif (90.0 - ANGLE_TOLERANCE) < angle < (90.0 + ANGLE_TOLERANCE):
                    temp_vertical.append([gx1, gy1, gx2, gy2])

    all_valid_lines = temp_horizontal + temp_vertical

    # ----------------------------------------
    # 第二阶段：🌟 终极绝杀 - 基于全局质心的距离过滤 🌟
    # ----------------------------------------
    final_horizontal, final_vertical = [], []
    
    if len(all_valid_lines) > 0:
        # 1. 提取所有线段的中心点坐标
        centers = np.array([[(l[0]+l[2])/2.0, (l[1]+l[3])/2.0] for l in all_valid_lines])
        
        # 2. 计算大部队的“全局质心”
        centroid = np.mean(centers, axis=0)
        
        # 3. 计算每个线段中心点到全局质心的距离
        distances = np.linalg.norm(centers - centroid, axis=1)
        
        # 4. 计算标准差，设定剔除阈值 (均值 + 2倍标准差)
        # 为了防止方差极小误伤真笔画，保底允许半径设为 80 像素
        mean_dist = np.mean(distances)
        std_dev = np.std(distances)
        radius_threshold = max(mean_dist + 2.0 * std_dev, 80.0)

        # 5. 执行剔除并绘制最终结果
        for l in temp_horizontal:
            cx, cy = (l[0]+l[2])/2.0, (l[1]+l[3])/2.0
            dist = np.sqrt((cx - centroid[0])**2 + (cy - centroid[1])**2)
            if dist <= radius_threshold:  # 只有在安全圈内的才保留
                final_horizontal.append(l)
                cv2.line(result_img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 0, 0), 2)
                
        for l in temp_vertical:
            cx, cy = (l[0]+l[2])/2.0, (l[1]+l[3])/2.0
            dist = np.sqrt((cx - centroid[0])**2 + (cy - centroid[1])**2)
            if dist <= radius_threshold:
                final_vertical.append(l)
                cv2.line(result_img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 0, 255), 2)

        # 【可视化呈现】画出质心（黄点）和剔除边界（黄圈）
        cv2.circle(result_img, (int(centroid[0]), int(centroid[1])), 6, (0, 255, 255), -1)
        cv2.circle(result_img, (int(centroid[0]), int(centroid[1])), int(radius_threshold), (0, 255, 255), 2)
        
        discard_count = len(all_valid_lines) - (len(final_horizontal) + len(final_vertical))
        print(f"🎯 质心过滤完毕 [{window_name}]:")
        print(f"   - 成功剔除边框杂线: {discard_count} 条")
        print(f"   - 最终保留高质量骨架: 横向 {len(final_horizontal)} 条, 竖向 {len(final_vertical)} 条。")
    else:
        print(f"⚠️ 预处理后未检测到有效线段。")

    res_window = f"Centroid Filtered Result - {window_name}"
    cv2.namedWindow(res_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(res_window, result_img)
    
    return final_horizontal, final_vertical, pts_poly

# ==========================================
if __name__ == "__main__":
    IMG1_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/Homography/1.png" 
    IMG2_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/Homography/2.png" 

    detect_and_filter_lines_plslam(IMG1_PATH, "Image 1")
    detect_and_filter_lines_plslam(IMG2_PATH, "Image 2")

    print("\n✅ 请查看带“安全圈”的过滤结果！按【任意键】关闭。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()