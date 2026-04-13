import numpy as np
import cv2
import glob
import os

# ================= 配置区 =================
DATA_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/data/test_data" 
DEPTH_SCALE = 0.0010000000474974513 
PATTERN_SIZE = (8, 11) 

def get_depth_robust(depth_matrix, x, y):
    x, y = int(x), int(y)
    h, w = depth_matrix.shape
    x1, x2 = max(0, x-2), min(w, x+3)
    y1, y2 = max(0, y-2), min(h, y+3)
    patch = depth_matrix[y1:y2, x1:x2]
    valid_depths = patch[patch > 0]
    if len(valid_depths) == 0: return 0
    return np.median(valid_depths) * DEPTH_SCALE * 100 

def line_equation(x1, y1, x2, y2):
    """ 两点转直线方程 Ax + By + C = 0 """
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C

def intersect_lines(l1, l2):
    """ 两直线求交点 """
    if l1 is None or l2 is None: return None
    A1, B1, C1 = l1
    A2, B2, C2 = l2
    D = A1 * B2 - A2 * B1
    if D == 0: return None
    x = (-C1 * B2 + C2 * B1) / D
    y = (-A1 * C2 + A2 * C1) / D
    return (x, y)

def segment_intersect(p1, p2, p3, p4):
    """ 对角线求交点 """
    xdiff = (p1[0] - p2[0], p3[0] - p4[0])
    ydiff = (p1[1] - p2[1], p3[1] - p4[1])
    def det(a, b): return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    if div == 0: return (0, 0)
    d = (det(p1, p2), det(p3, p4))
    return det(d, xdiff) / div, det(d, ydiff) / div

def extract_four_longest_edges(img_roi):
    """ 
    你的核心思想实现：线检测 -> 过滤 -> 取最长 
    返回: 4条最长直线的参数 (Top, Bottom, Left, Right)，以及这4条线的端点(用于可视化)
    """
    h, w = img_roi.shape[:2]
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    
    # Canny 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 霍夫直线检测
    # 你的核心要求：内部线短，外部线长。所以 minLineLength 设为高度或宽度的 40% 以上！
    min_len = min(w, h) * 0.4
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=min_len, maxLineGap=15)
    
    if lines is None: return None, None

    # 初始化 4 个方向的最长线记录器: [最大长度, x1, y1, x2, y2]
    best_top = [0, 0, 0, 0, 0]
    best_bottom = [0, 0, 0, 0, 0]
    best_left = [0, 0, 0, 0, 0]
    best_right = [0, 0, 0, 0, 0]

    cx, cy = w / 2, h / 2  # ROI 中心

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

        # 按照角度和相对位置，将线段分发到 4 个桶，只保留最长的！
        if angle < 45 or angle > 135: # 水平方向的线
            if mid_y < cy: # 在上半部分
                if length > best_top[0]: best_top = [length, x1, y1, x2, y2]
            else:          # 在下半部分
                if length > best_bottom[0]: best_bottom = [length, x1, y1, x2, y2]
        else: # 垂直方向的线
            if mid_x < cx: # 在左半部分
                if length > best_left[0]: best_left = [length, x1, y1, x2, y2]
            else:          # 在右半部分
                if length > best_right[0]: best_right = [length, x1, y1, x2, y2]

    # 检查是否 4 个边都找到了长线
    if best_top[0]==0 or best_bottom[0]==0 or best_left[0]==0 or best_right[0]==0:
        return None, None

    # 转换为 Ax+By+C=0 的方程
    eq_top = line_equation(*best_top[1:])
    eq_bottom = line_equation(*best_bottom[1:])
    eq_left = line_equation(*best_left[1:])
    eq_right = line_equation(*best_right[1:])

    equations = (eq_top, eq_bottom, eq_left, eq_right)
    raw_lines = (best_top[1:], best_bottom[1:], best_left[1:], best_right[1:])
    
    return equations, raw_lines


def process_image(color_path, depth_path):
    print(f"\n{'='*40}")
    print(f"🎯 正在分析: {color_path.split('/')[-1]}")
    
    img = cv2.imread(color_path)
    depth_matrix = np.load(depth_path)
    vis_img = img.copy()

    # 1. 拿内部绝对真值
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_full, PATTERN_SIZE, None)
    if not ret: return
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray_full, corners, (11, 11), (-1, -1), criteria).reshape(-1, 2)
    gt_cx, gt_cy = np.mean(corners[:, 0]), np.mean(corners[:, 1])
    gt_z_cm = get_depth_robust(depth_matrix, gt_cx, gt_cy)

    # 2. 画框定 ROI
    print("👉 请画框包围标定板，按空格确认。")
    roi = cv2.selectROI("Length Filter Detection", vis_img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Length Filter Detection")
    if roi[2] == 0 or roi[3] == 0: return

    rx, ry, rw, rh = roi
    img_roi = img[ry:ry+rh, rx:rx+rw]

    # 3. 执行你的【长度过滤线检测】
    equations, raw_lines = extract_four_longest_edges(img_roi)
    
    if equations is None:
        print("❌ 长度过滤未找齐 4 条最长外边，请重新画框！")
        return

    eq_top, eq_bottom, eq_left, eq_right = equations

    # 4. 求 4 条直线的绝对交点 (需把局部坐标加回全局坐标)
    def to_global(pt): return (pt[0] + rx, pt[1] + ry)
    
    p_tl = to_global(intersect_lines(eq_top, eq_left))
    p_tr = to_global(intersect_lines(eq_top, eq_right))
    p_bl = to_global(intersect_lines(eq_bottom, eq_left))
    p_br = to_global(intersect_lines(eq_bottom, eq_right))

    # 5. 算法对抗
    bbox_cx, bbox_cy = rx + rw / 2, ry + rh / 2
    bbox_z_cm = get_depth_robust(depth_matrix, bbox_cx, bbox_cy)

    algo_cx, algo_cy = segment_intersect(p_tl, p_br, p_tr, p_bl)
    algo_z_cm = get_depth_robust(depth_matrix, algo_cx, algo_cy)

    # 6. 打印结果
    err_bbox = abs(bbox_z_cm - gt_z_cm)
    err_algo = abs(algo_z_cm - gt_z_cm)
    
    print(f"\n📊 【基于长度过滤线检测 的 端到端误差】")
    print(f"⭐ 绝对真值深度:     {gt_z_cm:.2f} cm")
    print(f"❌ YOLO 框几何中心: {bbox_z_cm:.2f} cm  (误差: {err_bbox:.2f} cm)")
    print(f"✅ 线检测对角线:     {algo_z_cm:.2f} cm  (误差: {err_algo:.2f} cm)")

    # ================= 可视化 =================
    cv2.circle(vis_img, (int(gt_cx), int(gt_cy)), 7, (255, 0, 0), -1) # 真值蓝点
    
    cv2.rectangle(vis_img, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 2)
    cv2.circle(vis_img, (int(bbox_cx), int(bbox_cy)), 5, (0, 0, 255), -1)

    # 【重要展示】：画出被你长度过滤选出的那 4 条“最长物理边” (亮蓝色加粗)
    for line in raw_lines:
        x1, y1, x2, y2 = line
        cv2.line(vis_img, (int(x1+rx), int(y1+ry)), (int(x2+rx), int(y2+ry)), (255, 255, 0), 4)

    # 画出延长线产生的 4 个角点和交叉对角线 (绿线)
    pts = [p_tl, p_tr, p_br, p_bl]
    for p in pts: cv2.circle(vis_img, (int(p[0]), int(p[1])), 5, (0, 255, 0), -1)
    cv2.line(vis_img, (int(p_tl[0]), int(p_tl[1])), (int(p_br[0]), int(p_br[1])), (0, 255, 0), 2)
    cv2.line(vis_img, (int(p_tr[0]), int(p_tr[1])), (int(p_bl[0]), int(p_bl[1])), (0, 255, 0), 2)
    cv2.circle(vis_img, (int(algo_cx), int(algo_cy)), 5, (0, 255, 0), -1)

    cv2.putText(vis_img, f"BBox Err: {err_bbox:.2f}cm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(vis_img, f"Algo Err: {err_algo:.2f}cm", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Length Filter Detection Pipeline", vis_img)
    save_path = color_path.replace(".png", "_len_filter_cmp.png")
    cv2.imwrite(save_path, vis_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    all_files = sorted(glob.glob(f"{DATA_DIR}/*color*.png"))
    color_files = [f for f in all_files if "_cmp" not in f and "_result" not in f and "_manual" not in f]
    for color_path in color_files:
        depth_path = color_path.replace("color", "depth").replace(".png", ".npy")
        if os.path.exists(depth_path):
            process_image(color_path, depth_path)
    cv2.destroyAllWindows()