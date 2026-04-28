import sys
import os
import cv2
import numpy as np
import glob
import math
import json
import re
import matplotlib.pyplot as plt
import torch
import yaml
import datetime
from kornia.feature import LoFTR
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R_scipy

# ==============================================================================
# 配置环境变量以导入子文件夹中的模块
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from tools.mid import extract_four_lines_from_real_image, calculate_rectangle_center
from mid_FOE_Z_d.pure_looming_depth import (
    load_tum_trajectory, get_closest_pose, 
    calculate_relative_motion, calculate_pure_looming_Z, load_saved_rois, fx, fy, cx, cy
)
from mid_FOE_Z_d.imgs_mid import process_sequence_with_cached_rois

# ==============================================================================
# 🌟 全局日志记录器
# ==============================================================================
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "experiment_results.txt")
_log_file_initialized = False
QUIET_MODE = False

def log_print(*args, **kwargs):
    global _log_file_initialized
    msg = " ".join(map(str, args))
    if not QUIET_MODE:
        print(msg, **kwargs)
    if QUIET_MODE:
        mode = "a"
    else:
        mode = "w" if not _log_file_initialized else "a"
    with open(LOG_FILE_PATH, mode, encoding="utf-8") as f:
        f.write(msg + "\n")
    _log_file_initialized = True

# ==============================================================================
# 🌟 全局加载 YAML 配置文件
# ==============================================================================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
if not os.path.exists(CONFIG_PATH):
    log_print(f"❌ 找不到配置文件: {CONFIG_PATH}")
    sys.exit(1)

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

fx = config['Camera']['fx']
fy = config['Camera']['fy']
cx = config['Camera']['cx']
cy = config['Camera']['cy']

K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]], dtype=np.float64)
K_inv = np.linalg.inv(K)

FRAME_STEP = config['Algorithm']['frame_step']

# ==============================================================================
# 🌟 模块 0：平面参数估计（从角点+深度推算 n 和 d）
# ==============================================================================
def estimate_plane_from_corners(corners_2d, Z_center, K_mat):
    """
    从 4 个角点（像素坐标）和中心深度 Z 估计平面参数。
    corners_2d: (tl, tr, br, bl) 四角点像素坐标
    Z_center: 目标中心深度（米）
    K_mat: 相机内参矩阵 (3x3)
    返回: n (单位法向量, 指向相机), d (平面距离, n·X = d)
    """
    K_inv_mat = np.linalg.inv(K_mat)
    points_3d = []
    for corner in corners_2d:
        ray = K_inv_mat @ np.array([corner[0], corner[1], 1.0], dtype=np.float64)
        X = ray * (Z_center / ray[2])
        points_3d.append(X)
    pts = np.array(points_3d)
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    _, _, vh = np.linalg.svd(pts_centered)
    n = vh[2]
    if n[2] < 0:
        n = -n
    d_plane = float(np.dot(n, centroid))
    return n, d_plane

def rotation_angle_deg(R):
    """从旋转矩阵计算旋转角（度）"""
    cos_theta = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    return float(math.degrees(math.acos(cos_theta)))

# ==============================================================================
# 🌟 模块 1：Looming 核心数学与去旋
# ==============================================================================
def derotate_point(P_raw, R_12):
    H = K @ R_12 @ K_inv
    P_homo = np.array([P_raw[0], P_raw[1], 1.0])
    P_pure_homo = H @ P_homo
    u_pure = P_pure_homo[0] / P_pure_homo[2]
    v_pure = P_pure_homo[1] / P_pure_homo[2]
    return (u_pure, v_pure)

def calculate_pure_looming_Z_v2(P1, P2_pure, FOE, delta_d):
    """
    Looming 测距公式（修正版）。
    P1: 近帧 ROI 中心（原始坐标），对应 r_near
    P2_pure: 远帧 ROI 中心（去旋后），对应 r_far
    公式：Z = r_near * Δd / dr - Δd （或等价 Z = r_far * Δd / dr）
    """
    r_near = math.sqrt((P1[0] - FOE[0])**2 + (P1[1] - FOE[1])**2)
    r_far = math.sqrt((P2_pure[0] - FOE[0])**2 + (P2_pure[1] - FOE[1])**2)
    dr = r_near - r_far
    if dr <= 0.2:
        return None, r_near, r_far, dr
    Z = r_near * delta_d / dr - delta_d
    return Z, r_near, r_far, dr

def calculate_motion_rt(pose1, pose2):
    t1, q1 = pose1[0:3], pose1[3:7]
    t2, q2 = pose2[0:3], pose2[3:7]
    R1 = R_scipy.from_quat(q1).as_matrix()
    R2 = R_scipy.from_quat(q2).as_matrix()
    R_12 = R1.T @ R2               
    t_12 = R1.T @ (t2 - t1)        
    return R_12, t_12

# ==============================================================================
# 🌟 模块 2：GUI 选帧与交互工具
# ==============================================================================
def select_frame_gui(images, default_idx, sequence_name):
    idx = default_idx if default_idx != -1 else 0
    window_name = f"Select Frame for {sequence_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while True:
        img = cv2.imread(images[idx])
        if img is None: break
        display_img = img.copy()
        cv2.putText(display_img, f"Seq: {sequence_name} | Frame: {idx}/{len(images)-1}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display_img, "[SPACE] or [D]: Next | [A]: Prev | [ENTER]: Select", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(0) & 0xFF
        if key == 32 or key == ord('d'): idx = (idx + 1) % len(images)
        elif key == ord('a'): idx = (idx - 1) % len(images)
        elif key == 13 or key == 27: break
    cv2.destroyWindow(window_name)
    return idx

def select_four_points(img, window_name):
    points = []
    display_img = img.copy()
    log_print(f"\n👉 请在 '{window_name}' 中顺时针或逆时针点击区域外围的 4 个点。")

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
    cv2.waitKey(500)
    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.int32)

# ==============================================================================
# 🌟 增强的 ROI 编辑工具（显示线检测、角点、中心）
# ==============================================================================
def edit_existing_rois(images, roi_path):
    rois = []
    if os.path.exists(roi_path):
        with open(roi_path, 'r') as f:
            for line in f:
                line = line.strip()
                rois.append(line)
    while len(rois) < len(images):
        rois.append('')

    idx = 0
    win_name = "Fix ROI | A/D: prev/next | E: re-annotate | Q: save & continue"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    def parse_rect(line_str):
        if not line_str:
            return None
        parts = line_str.split(',')
        if len(parts) != 4:
            return None
        try:
            x, y, w, h = map(int, parts)
            return (x, y, w, h)
        except:
            return None

    while True:
        img = cv2.imread(images[idx])
        if img is None:
            idx = (idx + 1) % len(images)
            continue

        disp = img.copy()
        rect = parse_rect(rois[idx])

        if rect:
            x, y, w, h = rect
            lines = extract_four_lines_from_real_image(img, rect)
            if lines is not None:
                line_top, line_bottom, line_left, line_right = lines
                center, corners = calculate_rectangle_center(line_top, line_bottom, line_left, line_right)

                if center is not None and corners is not None:
                    cx_int, cy_int = center
                    tl, tr, bl, br = corners
                    cv2.polylines(disp, [np.array([tl, tr, br, bl])], isClosed=True, color=(0, 255, 255), thickness=1)
                    cv2.line(disp, (line_top[0], line_top[1]), (line_top[2], line_top[3]), (255, 0, 0), 1)
                    cv2.line(disp, (line_bottom[0], line_bottom[1]), (line_bottom[2], line_bottom[3]), (255, 0, 0), 1)
                    cv2.line(disp, (line_left[0], line_left[1]), (line_left[2], line_left[3]), (255, 0, 0), 1)
                    cv2.line(disp, (line_right[0], line_right[1]), (line_right[2], line_right[3]), (255, 0, 0), 1)
                    for pt in [tl, tr, br, bl]:
                        cv2.circle(disp, pt, 5, (0, 255, 255), -1)
                    cv2.line(disp, tl, br, (0, 255, 0), 1)
                    cv2.line(disp, tr, bl, (0, 255, 0), 1)
                    cv2.line(disp, (cx_int - 20, cy_int), (cx_int + 20, cy_int), (0, 0, 255), 2)
                    cv2.line(disp, (cx_int, cy_int - 20), (cx_int, cy_int + 20), (0, 0, 255), 2)
                    cv2.circle(disp, center, 4, (0, 0, 255), -1)
                    mode_text = "Geometry View"
                else:
                    mode_text = "Center lost"
            else:
                mode_text = "Lines lost"
        else:
            mode_text = "No ROI"

        info = f"Frame {idx}/{len(images)-1} | {mode_text}"
        cv2.putText(disp, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(disp, "[A/Left]:Prev [D/Right]:Next [E]:Re-annotate [Q]:Save&Continue",
                    (10, disp.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win_name, disp)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            with open(roi_path, 'w') as f:
                for r in rois:
                    f.write(r + '\n')
            log_print(f"✅ ROI 修正已保存至 {roi_path}")
            break
        elif key == ord('d') or key == 83:
            idx = (idx + 1) % len(images)
        elif key == ord('a') or key == 81:
            idx = (idx - 1) % len(images)
        elif key == ord('e'):
            new_roi = cv2.selectROI(f"Re-annotate Frame {idx}", img, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(f"Re-annotate Frame {idx}")
            if new_roi[2] > 0 and new_roi[3] > 0:
                x, y, w, h = new_roi
                rois[idx] = f"{x},{y},{w},{h}"
                log_print(f"   ✅ 帧 {idx} 已更新为: {rois[idx]}")
            else:
                log_print(f"   ⚠️ 帧 {idx} 取消框选")

    cv2.destroyWindow(win_name)

# ==============================================================================
# 🌟 模块 3：汉字特征提取、LoFTR 匹配及检验
# ==============================================================================
def detect_and_filter_lines_plslam(image_path, window_name, cached_pts=None):
    img = cv2.imread(image_path)
    if img is None: return [], [], (0, 0, 0, 0)

    if cached_pts is not None:
        pts_poly = np.array(cached_pts, dtype=np.int32)
        log_print(f"📁 命中缓存，跳过 '{window_name}' 的四选点 GUI")
    else:
        pts_poly = select_four_points(img, f"Select 4 Points: {window_name}")
    x, y, w, h = cv2.boundingRect(pts_poly)
    if w == 0 or h == 0: return [], [], (0, 0, 0, 0)
    img_roi = img[y:y+h, x:x+w]
    
    gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray_roi, 5, 50, 50)
    gaussian = cv2.GaussianBlur(smooth, (0, 0), 2.0)
    sharpened = cv2.addWeighted(smooth, 3.0, gaussian, -2.0, 0)
    
    lsd = cv2.createLineSegmentDetector(0, scale=2.0)
    lines, _, _, _ = lsd.detect(sharpened)

    result_img = img.copy()
    cv2.polylines(result_img, [pts_poly], isClosed=True, color=(0, 255, 0), thickness=2)

    temp_horizontal, temp_vertical = [], []
    MIN_LENGTH, ANGLE_TOLERANCE = 4.0, 12.0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            gx1, gy1 = float(x1 + x), float(y1 + y)
            gx2, gy2 = float(x2 + x), float(y2 + y)
            line_length = np.sqrt((gx2 - gx1)**2 + (gy2 - gy1)**2)
            if line_length < MIN_LENGTH: continue
            
            if cv2.pointPolygonTest(pts_poly, (gx1, gy1), False) >= 0 and cv2.pointPolygonTest(pts_poly, (gx2, gy2), False) >= 0:
                dx, dy = gx2 - gx1, gy2 - gy1
                angle = np.degrees(np.arctan2(dy, dx))
                if angle < 0: angle += 180.0
                if angle < ANGLE_TOLERANCE or angle > (180.0 - ANGLE_TOLERANCE): temp_horizontal.append([gx1, gy1, gx2, gy2])
                elif (90.0 - ANGLE_TOLERANCE) < angle < (90.0 + ANGLE_TOLERANCE): temp_vertical.append([gx1, gy1, gx2, gy2])

    all_valid_lines = temp_horizontal + temp_vertical
    final_horizontal, final_vertical = [], []
    
    if len(all_valid_lines) > 0:
        centers_arr = np.array([[(l[0]+l[2])/2.0, (l[1]+l[3])/2.0] for l in all_valid_lines])
        centroid = np.mean(centers_arr, axis=0)
        distances = np.linalg.norm(centers_arr - centroid, axis=1)
        mean_dist, std_dev = np.mean(distances), np.std(distances)
        radius_threshold = max(mean_dist + 2.0 * std_dev, 80.0)

        for l in temp_horizontal:
            cx_line, cy_line = (l[0]+l[2])/2.0, (l[1]+l[3])/2.0
            if np.sqrt((cx_line - centroid[0])**2 + (cy_line - centroid[1])**2) <= radius_threshold:
                final_horizontal.append(l)
                cv2.line(result_img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 0, 0), 2)
                
        for l in temp_vertical:
            cx_line, cy_line = (l[0]+l[2])/2.0, (l[1]+l[3])/2.0
            if np.sqrt((cx_line - centroid[0])**2 + (cy_line - centroid[1])**2) <= radius_threshold:
                final_vertical.append(l)
                cv2.line(result_img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 0, 255), 2)

        cv2.circle(result_img, (int(centroid[0]), int(centroid[1])), 6, (0, 255, 255), -1)
        cv2.circle(result_img, (int(centroid[0]), int(centroid[1])), int(radius_threshold), (0, 255, 255), 2)
        
    res_window = f"Centroid Filtered Result - {window_name}"
    cv2.namedWindow(res_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(res_window, result_img)
    
    return final_horizontal, final_vertical, pts_poly

def match_images_with_loftr_roi(img1_color, img2_color, pts1_roi=None, cached_pts2_roi=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"👉 正在使用设备: {device} 运行 LoFTR")

    if pts1_roi is None:
        pts1_roi = select_four_points(img1_color, "Image 1: Select ROI for LoFTR")
    else:
        log_print(f"✅ 复用汉字骨架 ROI 作为 Image 1 LoFTR 区域，免去重复标注")
    
    if cached_pts2_roi is not None:
        pts2_roi = np.array(cached_pts2_roi, dtype=np.int32)
        log_print(f"📁 命中缓存，跳过 'Image 2: Select ROI for LoFTR' 的四选点 GUI")
    else:
        pts2_roi = select_four_points(img2_color, "Image 2: Select ROI for LoFTR")

    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    mask1, mask2 = np.zeros_like(img1_gray), np.zeros_like(img2_gray)
    cv2.fillPoly(mask1, [pts1_roi], 255)
    cv2.fillPoly(mask2, [pts2_roi], 255)

    img1_masked = cv2.bitwise_and(img1_gray, img1_gray, mask=mask1)
    img2_masked = cv2.bitwise_and(img2_gray, img2_gray, mask=mask2)

    img1_tensor = torch.from_numpy(img1_masked)[None, None].float() / 255.0
    img2_tensor = torch.from_numpy(img2_masked)[None, None].float() / 255.0
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)

    log_print("⏳ 正在加载 LoFTR 模型进行局部精准匹配...")
    matcher = LoFTR(pretrained='outdoor').to(device)
    matcher.eval()

    with torch.no_grad():
        correspondences = matcher({"image0": img1_tensor, "image1": img2_tensor})

    mkpts1 = correspondences['keypoints0'].cpu().numpy()
    mkpts2 = correspondences['keypoints1'].cpu().numpy()
    confidence = correspondences['confidence'].cpu().numpy()
    
    conf_thresh = 0.5
    good_pts1, good_pts2 = [], []
    
    for p1, p2, conf in zip(mkpts1, mkpts2, confidence):
        if conf > conf_thresh:
            in_poly1 = cv2.pointPolygonTest(pts1_roi, (float(p1[0]), float(p1[1])), False) >= 0
            in_poly2 = cv2.pointPolygonTest(pts2_roi, (float(p2[0]), float(p2[1])), False) >= 0
            if in_poly1 and in_poly2:
                good_pts1.append(p1)
                good_pts2.append(p2)

    log_print(f"🔪 经过严格 ROI 过滤，获得 {len(good_pts1)} 对纯净特征匹配点！")
    return np.array(good_pts1), np.array(good_pts2), pts2_roi

def visualize_matches_one_by_one(img1_color, img2_color, pts1, pts2, mask):
    keypoints1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts1]
    keypoints2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts2]
    
    valid_indices = [i for i, is_inlier in enumerate(mask.ravel()) if is_inlier]
    if not valid_indices: return

    log_print("\n🔬 开启显微镜模式：准备逐个审查匹配点！按 D 下一个，A 上一个，Q 退出。")
    current_idx = 0
    window_name = "Microscope Mode: Inspect Matches (Press Q to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        match_idx = valid_indices[current_idx]
        single_match = [cv2.DMatch(match_idx, match_idx, 0)]
        matched_img = cv2.drawMatches(
            img1_color, keypoints1, img2_color, keypoints2, 
            single_match, None, matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), 
            flags=cv2.DrawMatchesFlags_DEFAULT
        )

        cv2.putText(matched_img, f"Match: {current_idx + 1} / {len(valid_indices)} | ID: {match_idx}", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow(window_name, matched_img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27: break
        elif key == ord('d'): current_idx = (current_idx + 1) % len(valid_indices)
        elif key == ord('a'): current_idx = (current_idx - 1) % len(valid_indices)

    cv2.destroyAllWindows()

# ==============================================================================
# 🌟 模块 4：3D 正交验证与可视化引擎
# ==============================================================================
def backproject_line_to_plane(line_2d, n, K_inv_local):
    ray1 = K_inv_local @ np.array([line_2d[0], line_2d[1], 1.0])
    ray2 = K_inv_local @ np.array([line_2d[2], line_2d[3], 1.0])
    dot1, dot2 = np.dot(n, ray1), np.dot(n, ray2)
    if abs(dot1) < 1e-6 or abs(dot2) < 1e-6: return None
    P1_3d = (1.0 / dot1) * ray1
    P2_3d = (1.0 / dot2) * ray2
    vec_3d = P2_3d - P1_3d
    norm = np.linalg.norm(vec_3d)
    if norm < 1e-6: return None
    return vec_3d / norm

def evaluate_orthogonality(h_lines, v_lines, n, K_local):
    K_inv_local = np.linalg.inv(K_local)
    h_lines_sorted = sorted(h_lines, key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2, reverse=True)[:15]
    v_lines_sorted = sorted(v_lines, key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2, reverse=True)[:15]
    h_vecs_3d = [v for v in [backproject_line_to_plane(l, n, K_inv_local) for l in h_lines_sorted] if v is not None]
    v_vecs_3d = [v for v in [backproject_line_to_plane(l, n, K_inv_local) for l in v_lines_sorted] if v is not None]
    if not h_vecs_3d or not v_vecs_3d: return float('inf')
    ortho_error, count = 0.0, 0
    for vh in h_vecs_3d:
        for vv in v_vecs_3d:
            ortho_error += abs(np.dot(vh, vv))
            count += 1
    return ortho_error / count if count > 0 else float('inf')

def draw_camera(ax, R_local, t, label, scale=0.4, color='black'):
    t_flat = t.flatten()
    ax.scatter(*t_flat, color=color, s=50)
    ax.text(t_flat[0], t_flat[1], t_flat[2], f'  {label}', fontsize=10, weight='bold', color=color)
    x_ax = t_flat + R_local @ np.array([scale, 0, 0])
    y_ax = t_flat + R_local @ np.array([0, scale, 0])
    z_ax = t_flat + R_local @ np.array([0, 0, scale])
    ax.plot([t_flat[0], x_ax[0]], [t_flat[1], x_ax[1]], [t_flat[2], x_ax[2]], color='r', linewidth=2)
    ax.plot([t_flat[0], y_ax[0]], [t_flat[1], y_ax[1]], [t_flat[2], y_ax[2]], color='g', linewidth=2)
    ax.plot([t_flat[0], z_ax[0]], [t_flat[1], z_ax[1]], [t_flat[2], z_ax[2]], color='b', linewidth=2)

def visualize_dashboard(img1, img2, R_rel, t_real, X_3d, n_cv):
    fig = plt.figure(figsize=(18, 6))
    fig.canvas.manager.set_window_title("Metric Pose Dashboard")
    ax1 = fig.add_subplot(1, 3, 1); ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)); ax1.set_title("Seq 1"); ax1.axis('off')
    ax2 = fig.add_subplot(1, 3, 2); ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)); ax2.set_title("Seq 2"); ax2.axis('off')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    origin1, R1 = np.array([0, 0, 0]), np.eye(3)
    draw_camera(ax3, R1, origin1, "Cam1 (Ref)", color='black')
    origin2, R2 = (-R_rel.T @ t_real).flatten(), R_rel.T
    draw_camera(ax3, R2, origin2, "Cam2", color='dodgerblue')
    target = X_3d.flatten()
    ax3.scatter(*target, color='orange', s=100, marker='*')
    normal_end = target + n_cv.flatten() * 0.5 
    ax3.plot([target[0], normal_end[0]], [target[1], normal_end[1]], [target[2], normal_end[2]], color='purple', linewidth=2)
    ax3.plot([origin1[0], target[0]], [origin1[1], target[1]], [origin1[2], target[2]], 'k:', alpha=0.4)
    ax3.plot([origin2[0], target[0]], [origin2[1], target[1]], [origin2[2], target[2]], 'b:', alpha=0.4)
    all_pts = np.vstack([origin1, origin2, target, normal_end])
    max_range = np.array([all_pts[:,0].max()-all_pts[:,0].min(), all_pts[:,1].max()-all_pts[:,1].min(), all_pts[:,2].max()-all_pts[:,2].min()]).max() / 2.0
    mid_x = (all_pts[:,0].max()+all_pts[:,0].min()) * 0.5
    mid_y = (all_pts[:,1].max()+all_pts[:,1].min()) * 0.5
    mid_z = (all_pts[:,2].max()+all_pts[:,2].min()) * 0.5
    ax3.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3.set_zlim(mid_z - max_range, mid_z + max_range)
    ax3.invert_yaxis(); ax3.view_init(elev=-25, azim=-45)
    plt.tight_layout()
    plt.show()

def visualize_3d_scene(R, t, n, d, K_local, roi):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("SLAM 3D Viewer: Relative Pose & True Planar Target", fontsize=14)
    C1 = np.array([0.0, 0.0, 0.0])
    ax.scatter(*C1, color='black', s=50, marker='s'); ax.text(*C1, " Cam 1 (Ref)", color='black')
    R_inv = R.T
    C2 = (-R_inv @ t).flatten()
    ax.scatter(*C2, color='black', s=50, marker='^'); ax.text(*C2, " Cam 2", color='black')
    ax.plot([C1[0], C2[0]], [C1[1], C2[1]], [C1[2], C2[2]], color='gray', linestyle='--')
    
    pts = np.array(roi, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    
    K_inv_local = np.linalg.inv(K_local)
    def get_3d_point(p_2d):
        ray = K_inv_local @ np.array([p_2d[0], p_2d[1], 1.0])
        return (d / np.dot(n, ray)) * ray
        
    P_TL, P_TR, P_BR, P_BL = get_3d_point(rect[0]), get_3d_point(rect[1]), get_3d_point(rect[2]), get_3d_point(rect[3])
    plane_center, vec_x, vec_y = P_TL, P_TR - P_TL, P_BL - P_TL
    Z_axis = -n if n[2] > 0 else n 
    ax.quiver(*plane_center, *vec_x, color='red', length=1.0, linewidth=2.5)
    ax.quiver(*plane_center, *vec_y, color='green', length=1.0, linewidth=2.5)
    ax.quiver(*plane_center, *Z_axis, color='blue', length=np.linalg.norm(vec_x)*0.5, linewidth=2.5)
    poly = [[P_TL, P_TR, P_BR, P_BL]]
    ax.add_collection3d(Poly3DCollection(poly, alpha=0.4, facecolor='cyan', edgecolor='k'))
    
    all_pts = np.array([C1, C2, P_TL, P_TR, P_BL, P_BR])
    max_range = np.array([all_pts[:,0].max()-all_pts[:,0].min(), all_pts[:,1].max()-all_pts[:,1].min(), all_pts[:,2].max()-all_pts[:,2].min()]).max() / 2.0
    mid_x, mid_y, mid_z = (all_pts[:,0].max()+all_pts[:,0].min())*0.5, (all_pts[:,1].max()+all_pts[:,1].min())*0.5, (all_pts[:,2].max()+all_pts[:,2].min())*0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range); ax.set_ylim(mid_y - max_range, mid_y + max_range); ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.invert_yaxis(); ax.invert_zaxis(); ax.view_init(elev=-25, azim=-45)
    plt.show()

# ==============================================================================
# 🌟 模块 5：四选点缓存管理
# ==============================================================================
def load_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except (json.JSONDecodeError, TypeError):
        return {}

def save_cache(cache_path, data):
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def make_pair_key(idx1, idx2):
    return f"{idx1}_{idx2}"

# ==============================================================================
# 🌟 模块 6：精度评估引擎（对比深度真值、法向量、绝对位姿）
# ==============================================================================
def _load_tum_list(filename):
    traj = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            ts = float(parts[0])
            data = [float(x) for x in parts[1:8]]
            traj.append((ts, data))
    return traj

def get_gt_depth(depth_dir, img_idx, center_x, center_y):
    depth_files = sorted(os.listdir(depth_dir))
    if img_idx >= len(depth_files):
        return None, None
    fname = depth_files[img_idx]
    depth_path = os.path.join(depth_dir, fname)
    depth_mm = cv2.imread(depth_path, -1)
    if depth_mm is None:
        return None, None
    x, y = int(center_x), int(center_y)
    patch = depth_mm[max(0, y-1):y+2, max(0, x-1):x+2]
    patch = patch[patch > 0]
    if len(patch) == 0:
        return None, None
    Z_gt = np.median(patch) / 1000.0
    return Z_gt, fname

def fit_plane_from_depth(depth_dir, img_idx, roi_polygon):
    depth_files = sorted(os.listdir(depth_dir))
    if img_idx >= len(depth_files):
        return None, None, None
    fname = depth_files[img_idx]
    depth_path = os.path.join(depth_dir, fname)
    depth_mm = cv2.imread(depth_path, -1)
    if depth_mm is None:
        return None, None, None
    mask = np.zeros(depth_mm.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi_polygon, dtype=np.int32)], 255)
    ys, xs = np.where(mask > 0)
    points_3d = []
    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        Z_mm = depth_mm[y, x]
        if Z_mm <= 0:
            continue
        Z_m = Z_mm / 1000.0
        pt = np.array([x, y, 1.0])
        ray = K_inv @ pt
        X_3d = ray * (Z_m / ray[2])
        points_3d.append(X_3d)
    if len(points_3d) < 3:
        return None, None, None
    points = np.array(points_3d)
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    b = -points[:, 2]
    coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_coeff, d = coeff
    n_raw = np.array([a, b_coeff, 1.0])
    n = n_raw / np.linalg.norm(n_raw)
    if n[2] < 0:
        n = -n
    centroid = np.mean(points, axis=0)
    d_plane = float(np.dot(n, centroid))
    return n, d_plane, fname

def evaluate_accuracy(idx1_prev, idx1_base, idx2_base, Z_looming, dr, delta_d,
                      best_R, t_real, best_n, roi_target,
                      traj1_path, traj2_path, depth_dir, roi_path_seq1):
    log_print("\n\n" + "="*60)
    log_print("🎯 精度评估开始")
    log_print("="*60)

    traj1 = _load_tum_list(traj1_path)
    traj2 = _load_tum_list(traj2_path)

    if idx1_prev >= len(traj1) or idx2_base >= len(traj2):
        log_print("❌ 轨迹索引超出范围")
        return

    pose_A_arr = np.array(traj1[idx1_prev][1])
    pose_C_arr = np.array(traj2[idx2_base][1])
    R_gt, t_gt = calculate_motion_rt(pose_A_arr, pose_C_arr)

    roi_polygon = None
    center_x, center_y = None, None
    if os.path.exists(roi_path_seq1):
        with open(roi_path_seq1, 'r') as f:
            lines = f.readlines()
        if idx1_base < len(lines):
            roi_line = lines[idx1_base].strip()
            coords = [int(float(x)) for x in re.split(r'[ ,\t]+', roi_line) if x != '']
            if len(coords) >= 4:
                if len(coords) == 8:
                    pts = np.array([[coords[i], coords[i+1]] for i in range(0, 8, 2)])
                elif len(coords) == 4:
                    cx_r = coords[0] + coords[2] / 2.0
                    cy_r = coords[1] + coords[3] / 2.0
                    pts = np.array([[coords[0], coords[1]],
                                    [coords[0] + coords[2], coords[1]],
                                    [coords[0] + coords[2], coords[1] + coords[3]],
                                    [coords[0], coords[1] + coords[3]]])
                else:
                    pts = None
                if pts is not None:
                    center = np.mean(pts, axis=0)
                    center_x, center_y = center[0], center[1]
                    roi_polygon = pts.astype(int).tolist()

    if center_x is None:
        center_x, center_y = np.mean(roi_target, axis=0)
        roi_polygon = roi_target.tolist()

    Z_gt, depth_fname = get_gt_depth(depth_dir, idx1_base, center_x, center_y)
    if Z_gt is not None:
        log_print(f"📏 深度真值 (帧{idx1_base}): {Z_gt:.4f} m   (深度图: {depth_fname})")
        err_abs = abs(Z_looming - Z_gt)
        err_rel = err_abs / Z_gt * 100
        log_print(f"✅ Looming 深度误差: {err_abs:.4f} m, 相对误差: {err_rel:.2f}%")

    n_gt = None
    if roi_polygon is not None and depth_dir:
        n_gt, _, _ = fit_plane_from_depth(depth_dir, idx1_base, roi_polygon)

    t_est_vec = t_real.flatten()
    t_err = np.linalg.norm(t_est_vec - t_gt)
    gt_len = np.linalg.norm(t_gt)
    t_err_rel = t_err / gt_len * 100 if gt_len > 0 else 0
    R_diff = best_R.T @ R_gt
    rot_err_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2.0, -1, 1))
    rot_err_deg = math.degrees(rot_err_rad)

    log_print(f"\n--- 跨序列绝对位姿精度 ---")
    log_print(f"真实平移 t_gt: {t_gt.flatten()} m")
    log_print(f"估计平移 t_est: {t_est_vec}")
    log_print(f"平移误差 (欧式距离): {t_err:.4f} m, 相对错误: {t_err_rel:.2f}%")
    log_print(f"真实旋转矩阵 R_gt:\n{np.round(R_gt, 4)}")
    log_print(f"估计旋转矩阵 R_est:\n{np.round(best_R, 4)}")
    log_print(f"旋转误差: {rot_err_deg:.4f}°")

    if n_gt is not None:
        angle = math.degrees(math.acos(min(np.abs(np.dot(best_n, n_gt)), 1.0)))
        log_print(f"\n--- 正交筛选验证 ---")
        log_print(f"真实法向量 n_gt: {np.round(n_gt, 4)}")
        log_print(f"估计法向量 n_est (best_n): {np.round(best_n, 4)}")
        log_print(f"法向量夹角: {angle:.2f}°")
        if angle < 10.0:
            log_print("🎯 正交筛选成功命中真实解！")

    log_print("\n" + "=" * 60)
    log_print("精度评估结束")
    log_print("=" * 60)

# ==============================================================================
# 🌟 模块 7：批量评估（深度 Z、平面参数 n/d、帧间位姿）
# ==============================================================================
def batch_evaluate_looming():
    """
    批量安静模式：遍历序列1所有连续帧对，计算 Looming Z 并与深度真值对比。
    同时计算平面参数 n/d、帧间位姿变换，全部与 GT 对比。
    自动从 ROI (x,y,w,h) 提取四个角点，无需手动标注。
    不弹出任何 GUI 窗口。
    """
    log_print("\n\n" + "="*80)
    log_print(f"🚀 [BATCH EVALUATION START] Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("="*80)
    
    FOLDER_PATH_1 = config['Sequence1']['image_dir']
    DEPTH_DIR_1 = config['Sequence1'].get('depth_dir', '')
    TRAJ_PATH_1 = config['Sequence1']['trajectory_path']
    ROI_PATH_1 = config['Sequence1']['roi_path']
    
    if not DEPTH_DIR_1 or not os.path.isdir(DEPTH_DIR_1):
        log_print("❌ 深度图目录未配置或不存在，无法进行批量评估。")
        return
    
    images1 = sorted(glob.glob(os.path.join(FOLDER_PATH_1, "*.png")))
    if len(images1) < 2:
        log_print("❌ 图片数量不足")
        return
    
    slam_poses1 = load_tum_trajectory(TRAJ_PATH_1)
    if len(slam_poses1) < 2:
        log_print("❌ 轨迹数据不足")
        return
    
    if os.path.exists(ROI_PATH_1):
        log_print(f"✅ 发现已有 ROI 文件 ({ROI_PATH_1})，直接加载")
        saved_rois1 = load_saved_rois(ROI_PATH_1)
    else:
        log_print("⏳ ROI 文件不存在，尝试生成（可能弹出 GUI）...")
        process_sequence_with_cached_rois(FOLDER_PATH_1)
        saved_rois1 = load_saved_rois(ROI_PATH_1)
    
    if not saved_rois1:
        log_print("❌ saved_rois 为空，请先运行 process_sequence_with_cached_rois")
        return
    
    # 从 saved_rois 构建 ROI 多边形列表（用于 fit_plane_from_depth）
    roi_polygons = []
    for roi in saved_rois1:
        if roi is None:
            roi_polygons.append(None)
            continue
        try:
            x_r, y_r, w_r, h_r = roi
            poly = [[x_r, y_r], [x_r+w_r, y_r], [x_r+w_r, y_r+h_r], [x_r, y_r+h_r]]
            roi_polygons.append(poly)
        except:
            roi_polygons.append(None)
    
    log_print(f"\n📊 开始批量评估：共 {len(images1)} 帧，步长 {FRAME_STEP}")
    log_print(f"\n{'i_far':>6} {'i_near':>6} {'Z_loom':>8} {'Z_gt':>8} {'Err_Z':>8} {'ErrZ%':>7} "
              f"{'d_est':>8} {'d_gt':>8} {'Err_d':>8} {'Ang_n':>7} {'Rot_deg':>7} {'|t|':>7} {'dr_px':>7} {'status'}")
    log_print("-" * 135)
    
    DR_THRESHOLD = 5.0
    MAX_Z_THRESHOLD = 30.0
    MIN_DELTA_D = 0.02

    errors_abs = []
    errors_rel = []
    errors_d = []
    angles_n = []
    errors_all = []
    valid_count = 0
    skip_dr = 0
    skip_roi = 0
    skip_depth = 0
    skip_pose = 0
    skip_other = 0
    
    for i_near in range(FRAME_STEP, len(images1)):
        i_far = i_near - FRAME_STEP
        
        if i_near >= len(saved_rois1) or i_far >= len(saved_rois1):
            skip_roi += 1
            continue
        
        roi_near = saved_rois1[i_near] if i_near < len(saved_rois1) else None
        roi_far = saved_rois1[i_far] if i_far < len(saved_rois1) else None
        
        if roi_near is None or roi_far is None:
            skip_roi += 1
            continue
        
        try:
            img_near = cv2.imread(images1[i_near])
            img_far = cv2.imread(images1[i_far])
            if img_near is None or img_far is None:
                skip_other += 1
                continue
            
            # 提取四条骨架线并计算中心 + 角点
            lines_near = extract_four_lines_from_real_image(img_near, roi_near)
            lines_far = extract_four_lines_from_real_image(img_far, roi_far)
            
            if not lines_near or not lines_far:
                skip_roi += 1
                continue
            
            center_near_P1, corners_near = calculate_rectangle_center(*lines_near)
            center_far_raw, corners_far = calculate_rectangle_center(*lines_far)
            
            if center_near_P1 is None or center_far_raw is None:
                skip_roi += 1
                continue
            
            # 获取位姿
            time_near = float(os.path.basename(images1[i_near]).replace(".png", "")) / 1e9
            time_far = float(os.path.basename(images1[i_far]).replace(".png", "")) / 1e9
            
            pose_near = get_closest_pose(time_near, slam_poses1)
            pose_far = get_closest_pose(time_far, slam_poses1)
            
            if pose_near is None or pose_far is None:
                skip_pose += 1
                continue
            
            R_12, t_12 = calculate_motion_rt(pose_far, pose_near)
            t_near_frame = R_12.T @ t_12
            delta_d = t_near_frame[2]
            t_norm = float(np.linalg.norm(t_near_frame))
            rot_angle = rotation_angle_deg(R_12)
            
            if abs(delta_d) < 0.01:
                skip_pose += 1
                continue
            
            FOE = (fx * (t_near_frame[0] / t_near_frame[2]) + cx,
                   fy * (t_near_frame[1] / t_near_frame[2]) + cy)
            center_far_pure = derotate_point(center_far_raw, R_12.T)
            
            Z_result = calculate_pure_looming_Z_v2(center_near_P1, center_far_pure, FOE, delta_d)
            if Z_result is None or Z_result[0] is None:
                skip_dr += 1
                continue
            
            Z_looming, r_near, r_far, dr = Z_result
            
            # 低信噪比过滤
            if dr < DR_THRESHOLD:
                skip_dr += 1
                continue
            if abs(delta_d) < MIN_DELTA_D:
                skip_dr += 1
                continue
            if Z_looming > MAX_Z_THRESHOLD or Z_looming <= 0:
                skip_dr += 1
                continue
            
            # GT 深度
            center_x, center_y = center_near_P1[0], center_near_P1[1]
            Z_gt, depth_fname = get_gt_depth(DEPTH_DIR_1, i_near, center_x, center_y)
            
            if Z_gt is None or Z_gt <= 0:
                skip_depth += 1
                continue
            
            err_Z = abs(Z_looming - Z_gt)
            err_Z_pct = err_Z / Z_gt * 100 if Z_gt > 0 else 0
            
            # 平面参数：估计 n/d (from corners + Z_looming)
            n_est, d_est = None, None
            if corners_near is not None:
                n_est, d_est = estimate_plane_from_corners(corners_near, Z_looming, K)
            
            # GT 平面参数 (n_gt, d_gt) from depth map
            n_gt, d_gt = None, None
            if i_near < len(roi_polygons) and roi_polygons[i_near] is not None:
                n_gt, d_gt, _ = fit_plane_from_depth(DEPTH_DIR_1, i_near, roi_polygons[i_near])
            
            # d 误差
            err_d = None
            if d_est is not None and d_gt is not None:
                err_d = abs(d_est - d_gt)
            
            # n 角度误差
            angle_n = None
            if n_est is not None and n_gt is not None:
                cos_ang = np.clip(np.abs(np.dot(n_est, n_gt)), 0.0, 1.0)
                angle_n = float(math.degrees(math.acos(cos_ang)))
            
            # 记录
            errors_abs.append(err_Z)
            errors_rel.append(err_Z_pct)
            if err_d is not None:
                errors_d.append(err_d)
            if angle_n is not None:
                angles_n.append(angle_n)
            
            detail = {
                'i_far': i_far, 'i_near': i_near,
                'Z_looming': float(Z_looming), 'Z_gt': float(Z_gt),
                'err_Z': float(err_Z), 'err_Z_pct': float(err_Z_pct),
                'dr_px': float(dr), 'delta_d': float(delta_d),
                't_norm': t_norm, 'rot_deg': rot_angle,
            }
            if n_est is not None:
                detail['n_est'] = n_est.tolist()
                detail['d_est'] = float(d_est)
            if n_gt is not None:
                detail['n_gt'] = n_gt.tolist()
                detail['d_gt'] = float(d_gt)
            if err_d is not None:
                detail['err_d'] = float(err_d)
            if angle_n is not None:
                detail['angle_n_deg'] = float(angle_n)
            
            errors_all.append(detail)
            valid_count += 1
            
            # 列式输出
            d_est_str = f"{d_est:.3f}" if d_est is not None else "---"
            d_gt_str = f"{d_gt:.3f}" if d_gt is not None else "---"
            err_d_str = f"{err_d:.3f}" if err_d is not None else "---"
            ang_str = f"{angle_n:.1f}" if angle_n is not None else "---"
            
            log_print(f"{i_far:>6} {i_near:>6} {Z_looming:>8.3f} {Z_gt:>8.3f} {err_Z:>8.3f} {err_Z_pct:>6.1f} "
                      f"{d_est_str:>8} {d_gt_str:>8} {err_d_str:>8} {ang_str:>7} {rot_angle:>7.1f} {t_norm:>7.4f} {dr:>7.2f} ✓")
            
        except Exception as e:
            skip_other += 1
            log_print(f"{i_far:>6} {i_near:>6} {'--':>8} {'--':>8} {'--':>8} {'--':>7} "
                      f"{'--':>8} {'--':>8} {'--':>8} {'--':>7} {'--':>7} {'--':>7} {'--':>7} ✗ {str(e)[:30]}")
            continue
    
    # ============ 统计汇总 ============
    total_skip = skip_dr + skip_roi + skip_depth + skip_pose + skip_other
    log_print("\n" + "="*80)
    log_print("📊 批量评估统计")
    log_print("="*80)
    log_print(f"总帧对数: {len(images1) - FRAME_STEP}")
    log_print(f"有效帧对数: {valid_count}")
    log_print(f"跳过帧对数: {total_skip} (dr={skip_dr}, roi={skip_roi}, depth={skip_depth}, pose={skip_pose}, other={skip_other})")
    
    if valid_count > 0 and len(errors_abs) > 0:
        errors_abs_arr = np.array(errors_abs)
        errors_rel_arr = np.array(errors_rel)
        
        mae_z = np.mean(errors_abs_arr)
        rmse_z = np.sqrt(np.mean(errors_abs_arr**2))
        median_z = np.median(errors_abs_arr)
        std_z = np.std(errors_abs_arr)
        mean_rel = np.mean(errors_rel_arr)
        median_rel = np.median(errors_rel_arr)
        
        log_print(f"\n--- 深度 Z 误差 ---")
        log_print(f"  MAE:       {mae_z:.4f} m")
        log_print(f"  RMSE:      {rmse_z:.4f} m")
        log_print(f"  Median:    {median_z:.4f} m")
        log_print(f"  Std:       {std_z:.4f} m")
        log_print(f"  Min/Max:   {np.min(errors_abs_arr):.4f} / {np.max(errors_abs_arr):.4f} m")
        log_print(f"  Mean Rel:  {mean_rel:.2f}%")
        log_print(f"  Median Rel:{median_rel:.2f}%")
        
        within_5cm = int(np.sum(errors_abs_arr < 0.05))
        within_10cm = int(np.sum(errors_abs_arr < 0.10))
        within_20cm = int(np.sum(errors_abs_arr < 0.20))
        log_print(f"\n深度误差分布:")
        log_print(f"  < 5cm:     {within_5cm}/{valid_count} ({within_5cm/valid_count*100:.1f}%)")
        log_print(f"  < 10cm:    {within_10cm}/{valid_count} ({within_10cm/valid_count*100:.1f}%)")
        log_print(f"  < 20cm:    {within_20cm}/{valid_count} ({within_20cm/valid_count*100:.1f}%)")
        
        # ---- d 误差统计 ----
        if len(errors_d) > 0:
            d_arr = np.array(errors_d)
            log_print(f"\n--- 平面距离 d 误差 (n={len(errors_d)}) ---")
            log_print(f"  MAE:       {np.mean(d_arr):.4f} m")
            log_print(f"  RMSE:      {np.sqrt(np.mean(d_arr**2)):.4f} m")
            log_print(f"  Median:    {np.median(d_arr):.4f} m")
            log_print(f"  Min/Max:   {np.min(d_arr):.4f} / {np.max(d_arr):.4f} m")
        
        # ---- n 角度误差统计 ----
        if len(angles_n) > 0:
            ang_arr = np.array(angles_n)
            log_print(f"\n--- 法向量 n 角度误差 (n={len(angles_n)}) ---")
            log_print(f"  Mean:      {np.mean(ang_arr):.2f}°")
            log_print(f"  Median:    {np.median(ang_arr):.2f}°")
            log_print(f"  Std:       {np.std(ang_arr):.2f}°")
            log_print(f"  Min/Max:   {np.min(ang_arr):.2f}° / {np.max(ang_arr):.2f}°")
            within_5deg = int(np.sum(ang_arr < 5.0))
            within_10deg = int(np.sum(ang_arr < 10.0))
            log_print(f"  < 5°:      {within_5deg}/{len(angles_n)} ({within_5deg/len(angles_n)*100:.1f}%)")
            log_print(f"  < 10°:     {within_10deg}/{len(angles_n)} ({within_10deg/len(angles_n)*100:.1f}%)")
        
        # ---- n 一致性分析（理想情况下所有帧的 n 应该差不多）----
        n_est_list = []
        for d in errors_all:
            if 'n_est' in d:
                n_est_list.append(np.array(d['n_est']))
        if len(n_est_list) > 1:
            n_stack = np.stack(n_est_list)
            n_mean = np.mean(n_stack, axis=0)
            n_deviations = []
            for n_i in n_est_list:
                cos_a = np.clip(np.abs(np.dot(n_i, n_mean)), 0.0, 1.0)
                n_deviations.append(float(math.degrees(math.acos(cos_a))))
            nd_arr = np.array(n_deviations)
            log_print(f"\n--- 法向量 n 一致性（各帧偏离均值角度）---")
            log_print(f"  n_mean:    [{n_mean[0]:.4f}, {n_mean[1]:.4f}, {n_mean[2]:.4f}]")
            log_print(f"  Mean 偏离: {np.mean(nd_arr):.2f}°")
            log_print(f"  Median:    {np.median(nd_arr):.2f}°")
            log_print(f"  Std:       {np.std(nd_arr):.2f}°")
            log_print(f"  Max 偏离:  {np.max(nd_arr):.2f}°")
        
        # ---- 位姿统计 ----
        t_norms_all = [d.get('t_norm', 0) for d in errors_all if 't_norm' in d]
        rot_angles_all = [d.get('rot_deg', 0) for d in errors_all if 'rot_deg' in d]
        if len(t_norms_all) > 0:
            t_arr = np.array(t_norms_all)
            r_arr = np.array(rot_angles_all)
            log_print(f"\n--- 帧间位姿变换 (n={len(t_arr)}) ---")
            log_print(f"  |t| Mean/Median/Min/Max: {np.mean(t_arr):.4f} / {np.median(t_arr):.4f} / {np.min(t_arr):.4f} / {np.max(t_arr):.4f} m")
            log_print(f"  R_angle Mean/Median/Min/Max: {np.mean(r_arr):.2f}° / {np.median(r_arr):.2f}° / {np.min(r_arr):.2f}° / {np.max(r_arr):.2f}°")
        
        # ---- 保存 JSON ----
        results_json = os.path.join(os.path.dirname(__file__), "batch_results.json")
        summary = {
            'total_pairs': len(images1) - FRAME_STEP,
            'valid_count': valid_count,
            'skip_dr': skip_dr, 'skip_roi': skip_roi, 'skip_depth': skip_depth,
            'skip_pose': skip_pose, 'skip_other': skip_other,
            'Z_error': {
                'mae_m': float(mae_z), 'rmse_m': float(rmse_z),
                'median_m': float(median_z), 'std_m': float(std_z),
                'mean_rel_pct': float(mean_rel), 'median_rel_pct': float(median_rel),
                'within_5cm': within_5cm, 'within_10cm': within_10cm, 'within_20cm': within_20cm
            }
        }
        if len(errors_d) > 0:
            d_arr = np.array(errors_d)
            summary['d_error'] = {
                'mae_m': float(np.mean(d_arr)), 'rmse_m': float(np.sqrt(np.mean(d_arr**2))),
                'median_m': float(np.median(d_arr)), 'min_m': float(np.min(d_arr)), 'max_m': float(np.max(d_arr))
            }
        if len(angles_n) > 0:
            ang_arr = np.array(angles_n)
            summary['n_angle_error'] = {
                'mean_deg': float(np.mean(ang_arr)), 'median_deg': float(np.median(ang_arr)),
                'std_deg': float(np.std(ang_arr)), 'min_deg': float(np.min(ang_arr)), 'max_deg': float(np.max(ang_arr)),
                'within_5deg': within_5deg, 'within_10deg': within_10deg
            }
        if len(n_est_list) > 1:
            summary['n_consistency'] = {
                'n_mean': n_mean.tolist(),
                'mean_deviation_deg': float(np.mean(nd_arr)),
                'median_deviation_deg': float(np.median(nd_arr)),
                'std_deviation_deg': float(np.std(nd_arr)),
                'max_deviation_deg': float(np.max(nd_arr))
            }
        if len(t_norms_all) > 0:
            t_arr = np.array(t_norms_all)
            r_arr = np.array(rot_angles_all)
            summary['pose_stats'] = {
                't_norm_mean': float(np.mean(t_arr)), 't_norm_median': float(np.median(t_arr)),
                't_norm_min': float(np.min(t_arr)), 't_norm_max': float(np.max(t_arr)),
                'rot_deg_mean': float(np.mean(r_arr)), 'rot_deg_median': float(np.median(r_arr)),
                'rot_deg_min': float(np.min(r_arr)), 'rot_deg_max': float(np.max(r_arr))
            }
        
        with open(results_json, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'details': errors_all}, f, indent=2, ensure_ascii=False)
        log_print(f"\n💾 详细结果已保存至: {results_json}")
    else:
        log_print("\n⚠️  无有效帧对可用于统计。")
    
    log_print(f"\n✅ [BATCH EVALUATION COMPLETE] Results saved to {LOG_FILE_PATH}")
    return valid_count

# ==============================================================================
# 🚀 最终主流水线引擎
# ==============================================================================
def integrate_and_solve_metric_pose():
    log_print("\n\n" + "="*80)
    log_print(f"🚀 [EXPERIMENT START] Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("="*80)

    cache_file = config['Algorithm']['cache_file']
    if not os.path.isabs(cache_file):
        cache_file = os.path.join(os.path.dirname(__file__), cache_file)

    FOLDER_PATH_1 = config['Sequence1']['image_dir']
    DEPTH_DIR_1 = config['Sequence1'].get('depth_dir', '')
    TRAJ_PATH_1 = config['Sequence1']['trajectory_path']
    ROI_PATH_1 = config['Sequence1']['roi_path']

    FOLDER_PATH_2 = config['Sequence2']['image_dir']
    TRAJ_PATH_2 = config['Sequence2']['trajectory_path']

    images1 = sorted(glob.glob(os.path.join(FOLDER_PATH_1, "*.png")))
    images2 = sorted(glob.glob(os.path.join(FOLDER_PATH_2, "*.png")))
    
    slam_poses1 = load_tum_trajectory(TRAJ_PATH_1)

    log_print(f">>> 🚀 启动 [物理度量恢复] 跨序列双剑合璧！")
    log_print(f">>> 逻辑树：GUI选帧 -> Looming去旋测距 -> 画框提取汉字骨架 -> LoFTR特征匹配算H -> 正交筛选 -> 绝对平移 -> 精度评估")
    
    cache_data = load_cache(cache_file)
    
    # ==== 1. GUI 选帧 ====
    idx1_base, idx2_base = 0, 0 
    if cache_data:
        pairs = cache_data.get('pairs', {})
        loaded = False
        for pair_key, entry in pairs.items():
            if isinstance(entry, dict) and 'idx1' in entry and 'idx2' in entry:
                idx1_base = entry['idx1']
                idx2_base = entry['idx2']
                loaded = True
                break
        if loaded:
            log_print(f"\n📁 发现统一缓存，自动加载选帧：Lines1[{idx1_base}] <---> Lines2[{idx2_base}]")
        else:
            log_print(f"\n⚠️ 缓存文件存在但无可用的帧号记录，启动 GUI 选帧...")
            idx1_base = select_frame_gui(images1, idx1_base, "Sequence 1 (Lines1)")
            idx2_base = select_frame_gui(images2, idx2_base, "Sequence 2 (Lines2)")
            log_print(f"👉 最终选定计算帧：Lines1_img[{idx1_base}] <---> Lines2_img[{idx2_base}]")
    else:
        log_print("\n🔔 [GUI 选帧] 请在窗口中操作...")
        idx1_base = select_frame_gui(images1, idx1_base, "Sequence 1 (Lines1)")
        idx2_base = select_frame_gui(images2, idx2_base, "Sequence 2 (Lines2)")
        log_print(f"👉 最终选定计算帧：Lines1_img[{idx1_base}] <---> Lines2_img[{idx2_base}]")
    
    pair_key = make_pair_key(idx1_base, idx2_base)
    cached_data = cache_data.get('pairs', {}).get(pair_key, None)

    # ==== 2. Looming 测距 ====
    log_print("\n👉 检验序列 1 (基准) 的连续测距掩码框...")
    process_sequence_with_cached_rois(FOLDER_PATH_1)
    
    log_print("\n🔧 是否需要修正 ROI 标注？在窗口中按 E 编辑错误帧，按 Q 保存并继续。")
    edit_existing_rois(images1, ROI_PATH_1)
    
    saved_rois1 = load_saved_rois(ROI_PATH_1)

    idx1_prev = idx1_base - FRAME_STEP
    if idx1_prev < 0 or idx1_base >= len(saved_rois1):
        log_print("❌ Lines1 序列前序帧不足以计算 Looming，请调整 FRAME_STEP 或选取靠后的帧。")
        sys.exit(1)

    time1_A = float(os.path.basename(images1[idx1_prev]).replace(".png", "")) / 1e9
    time1_B = float(os.path.basename(images1[idx1_base]).replace(".png", "")) / 1e9

    pose1_A = get_closest_pose(time1_A, slam_poses1)
    pose1_B = get_closest_pose(time1_B, slam_poses1)

    R_12, t_12 = calculate_motion_rt(pose1_A, pose1_B)
    
    t_near_frame = R_12.T @ t_12
    tx, ty, tz = t_near_frame
    delta_d = tz
    log_print(f"    [Motion Info] tx: {tx:.4f}, ty: {ty:.4f}, tz(delta_d): {tz:.4f}")
    if delta_d < 0.01: 
        log_print(f"⚠️ 序列 1 运动太小 (delta_d={delta_d:.4f})，无法计算 Looming Z！")
        sys.exit(1)

    FOE = (fx * (tx / tz) + cx, fy * (ty / tz) + cy)
    img1_A = cv2.imread(images1[idx1_prev])
    img1_B = cv2.imread(images1[idx1_base])
    
    lines1_A = extract_four_lines_from_real_image(img1_A, saved_rois1[idx1_prev])
    lines1_B = extract_four_lines_from_real_image(img1_B, saved_rois1[idx1_base])
    if not lines1_A or not lines1_B: 
        log_print("❌ 序列 1 提取 ROI 框失败")
        sys.exit(1)

    center1_B_looming, _ = calculate_rectangle_center(*lines1_B)
    center1_A_raw, _ = calculate_rectangle_center(*lines1_A)

    center1_A_pure = derotate_point(center1_A_raw, R_12.T)
    
    Z_looming, r1, r2, dr = calculate_pure_looming_Z_v2(center1_B_looming, center1_A_pure, FOE, delta_d)
    if Z_looming is None: 
        log_print(f"❌ 序列 1 Looming 计算失败 (膨胀量 dr 太小或为负数)")
        sys.exit(1)
    
    log_print(f"[Lines1 内部] ✅ (被动测距模块) FOE 物理深度计算成功: Z = {Z_looming:.3f} m (dr = {dr:.2f}px)")

    # ==== 3. 汉字骨架 ====
    log_print("\n" + "="*40)
    log_print(" 🛠️ 阶段 1：提取汉字骨架 (提供正交先验数据)")
    log_print("="*40)

    cached_lines_pts = cached_data.get('lines_roi', None) if cached_data else None
    h_lines, v_lines, roi_target = detect_and_filter_lines_plslam(
        images1[idx1_base], "Image 1: Select Lines ROI", cached_pts=cached_lines_pts
    )
    
    if len(h_lines) < 2 or len(v_lines) < 2: 
        log_print("❌ 有效线段太少，无法进行正交验证，程序终止。")
        sys.exit(1)
    log_print(f"✅ 成功提取汉字骨架：横向 {len(h_lines)} 条，竖向 {len(v_lines)} 条。")

    # ==== 4. LoFTR 匹配 ====
    log_print("\n" + "="*40)
    log_print(" 🛠️ 阶段 2：启动 LoFTR 引擎解算数学位姿")
    log_print("="*40)
    img2_base = cv2.imread(images2[idx2_base])
    
    cached_loftr_pts2 = cached_data.get('loftr_roi2', None) if cached_data else None
    pts1, pts2, pts2_roi = match_images_with_loftr_roi(
        img1_B, img2_base, pts1_roi=roi_target, cached_pts2_roi=cached_loftr_pts2
    )
    
    if 'pairs' not in cache_data:
        cache_data['pairs'] = {}
    cache_data['pairs'][pair_key] = {
        'idx1': idx1_base,
        'idx2': idx2_base,
        'lines_roi': roi_target.tolist(),
        'loftr_roi2': pts2_roi.tolist(),
        'saved_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    save_cache(cache_file, cache_data)
    log_print(f"💾 统一缓存已保存至 {cache_file} (键: {pair_key})")
    
    if pts1 is None or len(pts1) < 4:
        log_print("❌ LoFTR 有效匹配点不足，程序终止。")
        sys.exit(1)

    log_print(f"\n👉 正在基于 LoFTR 特征点计算单应性矩阵 H ...")
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H is None: 
        log_print("❌ 单应性矩阵 H 计算失败！")
        sys.exit(1)

    log_print("--- 高精度单应性矩阵 H ---")
    log_print(np.round(H, 4))

    visualize_matches_one_by_one(img1_B, img2_base, pts1, pts2, mask)

    # ==== 5. 正交筛选 ====
    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
    log_print(f"\n✅ 成功分解出 {num_solutions} 组位姿解。")

    log_print("\n" + "="*40)
    log_print(" 🕵️‍♂️ 阶段 3：真理验算 (3D 正交性误差比对)")
    log_print("="*40)

    best_idx, min_error = -1, float('inf')
    ortho_errors_list = []
    normals_list = []

    for i in range(num_solutions):
        n_cv = normals[i].flatten()
        if n_cv[2] > 0: 
            error = evaluate_orthogonality(h_lines, v_lines, n_cv, K)
            log_print(f"🟢 [候选解 {i+1}] 3D正交误差: {error:.4f}  |  数学解 n_cv: {np.round(n_cv, 4)}")
            ortho_errors_list.append(error)
            normals_list.append(n_cv.tolist())
            if error < min_error:
                min_error = error; best_idx = i

    if best_idx == -1: 
        log_print("❌ 所有的解都不符合物理规律！")
        sys.exit(1)

    best_R = rotations[best_idx]
    best_t = translations[best_idx]
    best_n = normals[best_idx].flatten()
    log_print(f"🎉 验算完毕！【解 {best_idx+1}】完美通过汉字 3D 正交检验，是唯一真实姿态！")

    # ==== 6. 绝对平移还原 ====
    p_2d = np.array([center1_B_looming[0], center1_B_looming[1], 1.0]).reshape(3, 1)
    ray = K_inv @ p_2d
    
    X_3d = ray * (Z_looming / ray[2][0])
    d = float((best_n.T @ X_3d)[0])
    t_real = best_t * abs(d)

    log_print("\n🏆 (两极反转) 双序列跨界绝对姿态提取结果！")
    log_print("="*40)
    log_print(f" -> 当下帧专属深度 Z_Looming: {Z_looming:.3f} m")
    log_print(f" -> 摄像机到平面绝对垂直距离 d: {abs(d):.3f} m")
    log_print(f" -> 跨序列真实的绝对平移 t_real (m):\n{t_real}")
    log_print(f" -> 跨序列的绝对旋转矩阵 R :\n{np.round(best_R, 4)}")
    log_print("="*40)

    log_print("\n🎨 正在启动 3D 姿态与图像联合可视化窗口 (可拖拽旋转)...")
    visualize_dashboard(img1_B, img2_base, best_R, t_real, X_3d, best_n)
    
    log_print("\n🎨 正在启动带有真实纸张物理形变的 3D 窗口...")
    visualize_3d_scene(best_R, t_real, best_n, d, K, roi_target)

    # ==== 7. 精度评估 ====
    if DEPTH_DIR_1 and os.path.isdir(DEPTH_DIR_1):
        log_print("\n🔍 深度图目录存在，启动自动精度评估...")
        evaluate_accuracy(
            idx1_prev=idx1_prev,
            idx1_base=idx1_base,
            idx2_base=idx2_base,
            Z_looming=Z_looming,
            dr=dr,
            delta_d=delta_d,
            best_R=best_R,
            t_real=t_real,
            best_n=best_n,
            roi_target=roi_target,
            traj1_path=TRAJ_PATH_1,
            traj2_path=TRAJ_PATH_2,
            depth_dir=DEPTH_DIR_1,
            roi_path_seq1=ROI_PATH_1
        )
    else:
        log_print("\n⚠️  深度图目录未配置或不存在，跳过精度评估。")

    log_print(f"\n✅ [EXPERIMENT COMPLETE] Results saved to {LOG_FILE_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ORB-SLAM3 Looming Metric Solver")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="安静批量模式：遍历序列1所有连续帧对，计算 Looming Z 与深度真值误差，不弹出 GUI")
    args = parser.parse_args()
    
    if args.quiet:
        QUIET_MODE = True
        import matplotlib
        matplotlib.use('Agg')
        batch_evaluate_looming()
    else:
        integrate_and_solve_metric_pose()