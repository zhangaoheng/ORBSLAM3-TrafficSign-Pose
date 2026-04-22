import cv2
import numpy as np

# ==========================================
# 核心纯几何数学逻辑 (保持你要求的逻辑不变)
# ==========================================
def line_intersection(l1, l2):
    """计算两条直线的交点 (直线相互延长求交)"""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-6:
        return None

    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    return (int(x), int(y))

def calculate_rectangle_center(line_top, line_bottom, line_left, line_right):
    """纯几何推演：四线求四角 -> 四角连对角线 -> 对角线求中心"""
    corner_tl = line_intersection(line_top, line_left)     
    corner_tr = line_intersection(line_top, line_right)    
    corner_bl = line_intersection(line_bottom, line_left)  
    corner_br = line_intersection(line_bottom, line_right) 

    if None in (corner_tl, corner_tr, corner_bl, corner_br):
        return None, None

    diag1 = [corner_tl[0], corner_tl[1], corner_br[0], corner_br[1]]
    diag2 = [corner_tr[0], corner_tr[1], corner_bl[0], corner_bl[1]]

    center = line_intersection(diag1, diag2)
    return center, (corner_tl, corner_tr, corner_bl, corner_br)

# ==========================================
# 真实图像处理：从真实图片中提取 4 条物理边缘
# ==========================================
def extract_four_lines_from_real_image(image, roi_bbox, pad_ratio=0.15):
    """使用线检测从真实图片中提取上下左右 4 条主边缘，返回格式 [x1, y1, x2, y2]"""
    x, y, w, h = roi_bbox
    
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    
    x1_p = max(0, x - pad_x)
    y1_p = max(0, y - pad_y)
    x2_p = min(image.shape[1], x + w + pad_x)
    y2_p = min(image.shape[0], y + h + pad_y)
    
    roi = image[y1_p:y2_p, x1_p:x2_p]
    if roi.size == 0: return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(gray)[0]
    
    if lines is None: return None
    
    rw = x2_p - x1_p
    rh = y2_p - y1_p

    lines_top, lines_bottom, lines_left, lines_right = [], [], [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        # 🚨 放宽初筛：左边缘如果是模糊的，LSD会把它切成很短的线段
        if length < min(rw, rh) * 0.05: continue
            
        angle = np.abs(np.arctan2(dy, dx) * 180.0 / np.pi)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if angle < 30 or angle > 150: # 横线
            if cy < rh / 2.0: lines_top.append(line[0])
            else: lines_bottom.append(line[0])
        elif 60 < angle < 120: # 竖线
            if cx < rw / 2.0: lines_left.append(line[0])
            else: lines_right.append(line[0])

    def get_outermost_points(lines_list, is_horizontal, is_min):
        """极值截胡策略：强行从外向内推，不管它是不是比内部文字边框短！"""
        if not lines_list: return []
        
        def line_len(l): return np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2)
        def line_pos(l): return (l[1] + l[3]) / 2.0 if is_horizontal else (l[0] + l[2]) / 2.0
        
        # 1. 约束物理合理范围
        expected_coord = pad_y if is_horizontal and is_min else (rh - pad_y if is_horizontal and not is_min else (pad_x if is_min else rw - pad_x))
        max_dist_to_roi = (rh if is_horizontal else rw) * 0.35 # 放宽到 35% 偏移容忍范围
        
        valid_lines = [l for l in lines_list if abs(line_pos(l) - expected_coord) < max_dist_to_roi]
        if not valid_lines: valid_lines = lines_list
            
        # 2. 强制从外向内排序 (离边界最近的最先考虑)
        if is_min:
            valid_lines.sort(key=line_pos)
        else:
            valid_lines.sort(key=line_pos, reverse=True)
            
        # 3. 从最外侧往里扫描，只要遇到长度达到总体 8% 的有效短线，我们就认为它是物理边沿
        best_line = valid_lines[0]
        len_threshold = (rw if is_horizontal else rh) * 0.08
        for l in valid_lines:
            if line_len(l) > len_threshold:
                best_line = l
                break
                
        # 4. 把与 best_line 共线的碎线段收拢缝合计算
        ref_coord = line_pos(best_line)
        margin = (rh if is_horizontal else rw) * 0.08 # 8% 坐标容差聚合
        
        pts = []
        for l in lines_list:
            if abs(line_pos(l) - ref_coord) < margin:
                pts.extend([[l[0], l[1]], [l[2], l[3]]])
        return pts

    pts_top = get_outermost_points(lines_top, is_horizontal=True, is_min=True)
    pts_bottom = get_outermost_points(lines_bottom, is_horizontal=True, is_min=False)
    pts_left = get_outermost_points(lines_left, is_horizontal=False, is_min=True)
    pts_right = get_outermost_points(lines_right, is_horizontal=False, is_min=False)

    def fit_and_get_endpoints(pts):
        if len(pts) < 2: return None
        vx, vy, x0, y0 = cv2.fitLine(np.array(pts, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        length = max(w, h) * 2 
        pt1 = (int(x0[0] - vx[0] * length + x1_p), int(y0[0] - vy[0] * length + y1_p))
        pt2 = (int(x0[0] + vx[0] * length + x1_p), int(y0[0] + vy[0] * length + y1_p))
        return [pt1[0], pt1[1], pt2[0], pt2[1]]

    line_t = fit_and_get_endpoints(pts_top)
    line_b = fit_and_get_endpoints(pts_bottom)
    line_l = fit_and_get_endpoints(pts_left)
    line_r = fit_and_get_endpoints(pts_right)

    if None in (line_t, line_b, line_l, line_r): return None
    return line_t, line_b, line_l, line_r

# ==========================================
# 主程序：在真实图片上运行与可视化
# ==========================================
def test_on_real_image(image_path):
    print(f">>> 正在读取真实图片: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("❌ 错误：找不到图片，请检查路径。")
        return

    clone = img.copy()
    
    # 1. 框选路牌区域
    print(">>> 请用鼠标框选路牌区域，按 [SPACE] 或 [ENTER] 确认。")
    roi_bbox = cv2.selectROI("Select Sign", clone, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Sign")
    if roi_bbox == (0, 0, 0, 0): return

    # 2. 从真实图片提取 4 条物理线
    lines = extract_four_lines_from_real_image(img, roi_bbox)
    if lines is None:
        print("❌ 提取失败：图片中未能检测出完整的四条边。")
        return
    line_top, line_bottom, line_left, line_right = lines

    # 3. 喂给纯几何算法算中心
    center, corners = calculate_rectangle_center(line_top, line_bottom, line_left, line_right)

    if center is not None:
        tl, tr, bl, br = corners

        # [画图] 1. 画出四条无限延长的物理边缘（蓝色线）
        cv2.line(clone, (line_top[0], line_top[1]), (line_top[2], line_top[3]), (255, 0, 0), 1)
        cv2.line(clone, (line_bottom[0], line_bottom[1]), (line_bottom[2], line_bottom[3]), (255, 0, 0), 1)
        cv2.line(clone, (line_left[0], line_left[1]), (line_left[2], line_left[3]), (255, 0, 0), 1)
        cv2.line(clone, (line_right[0], line_right[1]), (line_right[2], line_right[3]), (255, 0, 0), 1)

        # [画图] 2. 画出互相延长的 4 个角点（黄色圆点）
        for pt in corners:
            cv2.circle(clone, pt, 5, (0, 255, 255), -1)

        # [画图] 3. 角点交叉连线（绿色对角线）
        cv2.line(clone, tl, br, (0, 255, 0), 1)
        cv2.line(clone, tr, bl, (0, 255, 0), 1)

        # [画图] 4. 标出算出的最终中心点（红色十字准星）
        # cx, cy = center
        # cv2.line(clone, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 2)
        # cv2.line(clone, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 2)
        # cv2.circle(clone, center, 4, (0, 0, 255), -1)

        # print(f"✅ 成功! 真实坐标中心点 X={cx}, Y={cy}")
        
    cv2.imshow("Real Image Geometry Result", clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 使用你自己的绝对路径
    IMAGE_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/mid.png"
    test_on_real_image(IMAGE_PATH)