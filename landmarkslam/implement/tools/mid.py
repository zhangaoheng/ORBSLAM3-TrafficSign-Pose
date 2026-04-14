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
def extract_four_lines_from_real_image(image, roi_bbox):
    """使用线检测从真实图片中提取上下左右 4 条主边缘，返回格式 [x1, y1, x2, y2]"""
    x, y, w, h = roi_bbox
    roi = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(gray)[0]
    
    if lines is None: return None

    pts_top, pts_bottom, pts_left, pts_right = [], [], [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length < min(w, h) * 0.1: continue
            
        angle = np.abs(np.arctan2(dy, dx) * 180.0 / np.pi)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if angle < 30 or angle > 150:
            if cy < h / 2.0: pts_top.extend([[x1, y1], [x2, y2]])
            else: pts_bottom.extend([[x1, y1], [x2, y2]])
        elif 60 < angle < 120:
            if cx < w / 2.0: pts_left.extend([[x1, y1], [x2, y2]])
            else: pts_right.extend([[x1, y1], [x2, y2]])

    def fit_and_get_endpoints(pts):
        if len(pts) < 2: return None
        vx, vy, x0, y0 = cv2.fitLine(np.array(pts, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        # 算出一个足够长的线段以便后续求交
        length = max(w, h) * 2 
        pt1 = (int(x0[0] - vx[0] * length + x), int(y0[0] - vy[0] * length + y))
        pt2 = (int(x0[0] + vx[0] * length + x), int(y0[0] + vy[0] * length + y))
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