import sys
import os
import cv2
import numpy as np
import glob
import math
import matplotlib.pyplot as plt
import torch
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
    calculate_relative_motion, calculate_pure_looming_Z, load_saved_rois, FRAME_STEP, fx, fy, cx, cy
)
from mid_FOE_Z_d.imgs_mid import process_sequence_with_cached_rois

# ==============================================================================
# 🌟 全局参数与相机内参 (D456)
# ==============================================================================
fx, fy = 429.78045654, 429.78045654
cx, cy = 429.94277954, 241.57313537

K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]], dtype=np.float64)
K_inv = np.linalg.inv(K)

FRAME_STEP = 15  # 跨帧步长 (拉大基线，建议 5 到 15)

# ==============================================================================
# 🌟 模块 1：Looming 核心数学与去旋
# ==============================================================================
def derotate_point(P_raw, R_12):
    """使用纯旋转单应性矩阵 H 消除相机颠簸"""
    H = K @ R_12 @ K_inv
    P_homo = np.array([P_raw[0], P_raw[1], 1.0])
    P_pure_homo = H @ P_homo
    u_pure = P_pure_homo[0] / P_pure_homo[2]
    v_pure = P_pure_homo[1] / P_pure_homo[2]
    return (u_pure, v_pure)

def calculate_pure_looming_Z_v2(P1, P2_pure, FOE, delta_d):
    """纯平移宇宙下的 Looming 绝杀公式"""
    r1 = math.sqrt((P1[0] - FOE[0])**2 + (P1[1] - FOE[1])**2)
    r2 = math.sqrt((P2_pure[0] - FOE[0])**2 + (P2_pure[1] - FOE[1])**2)
    dr = r2 - r1
    if dr <= 0.2: 
        return None, r1, r2, dr
    Z = (r1 * delta_d) / dr
    return Z, r1, r2, dr

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
    """防抖版 4点选区：划定目标 ROI 掩码边界"""
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
    cv2.waitKey(500)
    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.int32)

# ==============================================================================
# 🌟 模块 3：汉字特征提取、LoFTR 匹配及检验 (一字不删)
# ==============================================================================
def detect_and_filter_lines_plslam(image_path, window_name):
    img = cv2.imread(image_path)
    if img is None: return [], [], (0, 0, 0, 0)

    # 这里的 4 个点只用来框选汉字区域 ROI
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
        centers = np.array([[(l[0]+l[2])/2.0, (l[1]+l[3])/2.0] for l in all_valid_lines])
        centroid = np.mean(centers, axis=0)
        distances = np.linalg.norm(centers - centroid, axis=1)
        mean_dist, std_dev = np.mean(distances), np.std(distances)
        radius_threshold = max(mean_dist + 2.0 * std_dev, 80.0)

        for l in temp_horizontal:
            cx, cy = (l[0]+l[2])/2.0, (l[1]+l[3])/2.0
            if np.sqrt((cx - centroid[0])**2 + (cy - centroid[1])**2) <= radius_threshold:
                final_horizontal.append(l)
                cv2.line(result_img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 0, 0), 2)
                
        for l in temp_vertical:
            cx, cy = (l[0]+l[2])/2.0, (l[1]+l[3])/2.0
            if np.sqrt((cx - centroid[0])**2 + (cy - centroid[1])**2) <= radius_threshold:
                final_vertical.append(l)
                cv2.line(result_img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 0, 255), 2)

        cv2.circle(result_img, (int(centroid[0]), int(centroid[1])), 6, (0, 255, 255), -1)
        cv2.circle(result_img, (int(centroid[0]), int(centroid[1])), int(radius_threshold), (0, 255, 255), 2)
        
    res_window = f"Centroid Filtered Result - {window_name}"
    cv2.namedWindow(res_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(res_window, result_img)
    
    return final_horizontal, final_vertical, pts_poly

def match_images_with_loftr_roi(img1_color, img2_color):
    """
    使用点击的 4 个点生成硬掩码，在此掩码范围内运行 LoFTR 特征匹配
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"👉 正在使用设备: {device} 运行 LoFTR")

    pts1_roi = select_four_points(img1_color, "Image 1: Select ROI for LoFTR")
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

    print("⏳ 正在加载 LoFTR 模型进行局部精准匹配...")
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

    print(f"🔪 经过严格 ROI 过滤，获得 {len(good_pts1)} 对纯净特征匹配点！")
    return np.array(good_pts1), np.array(good_pts2)

def visualize_matches_one_by_one(img1_color, img2_color, pts1, pts2, mask):
    keypoints1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts1]
    keypoints2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts2]
    
    valid_indices = [i for i, is_inlier in enumerate(mask.ravel()) if is_inlier]
    if not valid_indices: return

    print("\n🔬 开启显微镜模式：准备逐个审查匹配点！按 D 下一个，A 上一个，Q 退出。")
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
def backproject_line_to_plane(line_2d, n, K_inv):
    ray1 = K_inv @ np.array([line_2d[0], line_2d[1], 1.0])
    ray2 = K_inv @ np.array([line_2d[2], line_2d[3], 1.0])
    dot1, dot2 = np.dot(n, ray1), np.dot(n, ray2)
    if abs(dot1) < 1e-6 or abs(dot2) < 1e-6: return None
    P1_3d = (1.0 / dot1) * ray1
    P2_3d = (1.0 / dot2) * ray2
    vec_3d = P2_3d - P1_3d
    norm = np.linalg.norm(vec_3d)
    if norm < 1e-6: return None
    return vec_3d / norm

def evaluate_orthogonality(h_lines, v_lines, n, K):
    K_inv = np.linalg.inv(K)
    h_lines_sorted = sorted(h_lines, key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2, reverse=True)[:15]
    v_lines_sorted = sorted(v_lines, key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2, reverse=True)[:15]
    h_vecs_3d = [v for v in [backproject_line_to_plane(l, n, K_inv) for l in h_lines_sorted] if v is not None]
    v_vecs_3d = [v for v in [backproject_line_to_plane(l, n, K_inv) for l in v_lines_sorted] if v is not None]
    if not h_vecs_3d or not v_vecs_3d: return float('inf')
    ortho_error, count = 0.0, 0
    for vh in h_vecs_3d:
        for vv in v_vecs_3d:
            ortho_error += abs(np.dot(vh, vv))
            count += 1
    return ortho_error / count if count > 0 else float('inf')

def draw_camera(ax, R, t, label, scale=0.4, color='black'):
    t = t.flatten()
    ax.scatter(*t, color=color, s=50)
    ax.text(t[0], t[1], t[2], f'  {label}', fontsize=10, weight='bold', color=color)
    x_ax = t + R @ np.array([scale, 0, 0])
    y_ax = t + R @ np.array([0, scale, 0])
    z_ax = t + R @ np.array([0, 0, scale])
    ax.plot([t[0], x_ax[0]], [t[1], x_ax[1]], [t[2], x_ax[2]], color='r', linewidth=2)
    ax.plot([t[0], y_ax[0]], [t[1], y_ax[1]], [t[2], y_ax[2]], color='g', linewidth=2)
    ax.plot([t[0], z_ax[0]], [t[1], z_ax[1]], [t[2], z_ax[2]], color='b', linewidth=2)

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

def visualize_3d_scene(R, t, n, K, roi):
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
    
    def get_3d_point(p_2d):
        ray = K_inv @ np.array([p_2d[0], p_2d[1], 1.0])
        return (1.0 / np.dot(n, ray)) * ray
        
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
# 🚀 最终主流水线引擎：严格融合，绝对保留所有功能！
# ==============================================================================
def integrate_and_solve_metric_pose():
    # ==== 路径配置 ====
    FOLDER_PATH_1 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/lines2/cam0/data"
    TRAJ_PATH_1 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/output/all_frames_lines2_slam_traj.txt"
    ROI_PATH_1 = os.path.join(FOLDER_PATH_1, "saved_rois.txt")

    FOLDER_PATH_2 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/lines1/cam0/data"
    TRAJ_PATH_2 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/output/all_frames_lines1_slam_traj.txt"

    # ==== 加载数据 ====
    images1 = sorted(glob.glob(os.path.join(FOLDER_PATH_1, "*.png")))
    images2 = sorted(glob.glob(os.path.join(FOLDER_PATH_2, "*.png")))
    
    # 注意：这里的 slam_poses1 是从 TUM 格式轨迹文件中加载的绝对位姿列表，包含时间戳、平移和四元数
    slam_poses1 = load_tum_trajectory(TRAJ_PATH_1)

    print(f">>> 物理度量恢复")
    print(f">>> (GUI选帧 -> Looming去旋测距 -> 提取汉字骨架 -> LoFTR特征匹配算H -> 正交筛选 -> 绝对平移)")
    
    # ==== 1. GUI 手动挑选计算帧 ====
    idx1_base, idx2_base = 0, 0 
    cache_file = os.path.join(os.path.dirname(__file__), "selected_frames_cache.txt")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            lines = f.read().strip().split()
            if len(lines) >= 2:
                idx1_base, idx2_base = int(lines[0]), int(lines[1])
        print(f"\n📁 发现缓存文件，自动加载选帧：Lines1[{idx1_base}] <---> Lines2[{idx2_base}]")
    else:
        print("\n🔔 [GUI 选帧] 请在窗口中操作...")
        idx1_base = select_frame_gui(images1, idx1_base, "Sequence 1 (Lines1)")
        idx2_base = select_frame_gui(images2, idx2_base, "Sequence 2 (Lines2)")
        with open(cache_file, "w") as f: f.write(f"{idx1_base}\n{idx2_base}")

    # ==== 2. 高精度去旋 Looming 测距 ====
    print("\n👉 检验序列 1 (基准) 的连续测距掩码框...")
    process_sequence_with_cached_rois(FOLDER_PATH_1)
    saved_rois1 = load_saved_rois(ROI_PATH_1)

    idx1_next = idx1_base + FRAME_STEP
    if idx1_next >= len(images1) or idx1_next >= len(saved_rois1) or idx1_base >= len(saved_rois1):
        exit("❌ Lines1 序列后续帧不足以计算 Looming，请调整 FRAME_STEP 或选取靠前的帧。")

    # 从 SLAM 轨迹中找到与选帧时间戳最接近的位姿
    time1_A = float(os.path.basename(images1[idx1_base]).replace(".png", "")) / 1e9
    time1_B = float(os.path.basename(images1[idx1_next]).replace(".png", "")) / 1e9

    
    pose1_A = get_closest_pose(time1_A, slam_poses1)
    pose1_B = get_closest_pose(time1_B, slam_poses1)

    R_12, t_12 = calculate_motion_rt(pose1_A, pose1_B)
    tx, ty, tz = t_12
    delta_d = tz
    if delta_d < 0.01: exit(f"⚠️ 序列 1 运动太小 (delta_d={delta_d:.4f})，无法计算 Looming Z！")

    FOE = (fx * (tx / tz) + cx, fy * (ty / tz) + cy)
    img1_A = cv2.imread(images1[idx1_base])
    img1_B = cv2.imread(images1[idx1_next])
    
    lines1_A = extract_four_lines_from_real_image(img1_A, saved_rois1[idx1_base])
    lines1_B = extract_four_lines_from_real_image(img1_B, saved_rois1[idx1_next])
    if not lines1_A or not lines1_B: exit("❌ 序列 1 提取 ROI 框失败")

    center1_A_looming, _ = calculate_rectangle_center(*lines1_A)
    center1_B_raw, _ = calculate_rectangle_center(*lines1_B)

    # 去旋洗掉颠簸
    center1_B_pure = derotate_point(center1_B_raw, R_12)
    # 计算 Looming 深度
    Z_looming, _, _, dr = calculate_pure_looming_Z_v2(center1_A_looming, center1_B_pure, FOE, delta_d)
    if Z_looming is None: exit("❌ 序列 1 Looming 计算失败")
    print(f"[Lines1 内部] ✅ (被动测距模块) FOE 物理深度计算成功: Z = {Z_looming:.3f} m")

    # ==== 3. 提取汉字骨架 (包含1次 4点选区弹窗) ====
    print("\n" + "="*40)
    print(" 🛠️ 阶段 1：提取汉字骨架 (提供正交先验数据)")
    print("="*40)
    # 将会弹出窗口，让你点 4 个点生成汉字区域的 ROI 掩码
    h_lines, v_lines, roi_target = detect_and_filter_lines_plslam(images1[idx1_base], "Image 1: Select Lines ROI")
    
    if len(h_lines) < 2 or len(v_lines) < 2: exit("❌ 有效线段太少，无法进行正交验证，程序终止。")
    print(f"✅ 成功提取汉字骨架：横向 {len(h_lines)} 条，竖向 {len(v_lines)} 条。")

    # ==== 4. 启动 LoFTR 匹配引擎 (包含2次 4点选区弹窗) ====
    print("\n" + "="*40)
    print(" 🛠️ 阶段 2：启动 LoFTR 引擎解算数学位姿")
    print("="*40)
    img2_base = cv2.imread(images2[idx2_base])
    
    # 将会弹出 2 次窗口，让你分别在两张图上点 4 个点生成 LoFTR 匹配范围的掩码
    pts1, pts2 = match_images_with_loftr_roi(img1_A, img2_base)
    
    if pts1 is None or len(pts1) < 4:
        exit("❌ LoFTR 有效匹配点不足，程序终止。")

    print(f"\n👉 正在基于 LoFTR 特征点计算单应性矩阵 H ...")
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H is None: exit("❌ 单应性矩阵 H 计算失败！")

    visualize_matches_one_by_one(img1_A, img2_base, pts1, pts2, mask)

    # ==== 5. 正交验证与姿态筛选 ====
    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
    
    print("\n" + "="*40)
    print(" 🕵️‍♂️ 阶段 3：真理验算 (3D 正交性误差比对)")
    print("="*40)

    best_idx, min_error = -1, float('inf')

    for i in range(num_solutions):
        n_cv = normals[i].flatten()
        if n_cv[2] > 0: 
            error = evaluate_orthogonality(h_lines, v_lines, n_cv, K)
            print(f"🟢 [候选解 {i+1}] 3D正交误差: {error:.4f}  |  数学解 n_cv: {np.round(n_cv, 4)}")
            if error < min_error:
                min_error = error; best_idx = i

    if best_idx == -1: exit("❌ 所有的解都不符合物理规律！")

    best_R = rotations[best_idx]
    best_t = translations[best_idx]
    best_n = normals[best_idx].flatten()
    print(f"🎉 验算完毕！【解 {best_idx+1}】完美通过汉字 3D 正交检验，是唯一真实姿态！")

    # ==== 6. 还原绝对平移与可视化 ====
    # 将第一步 Looming 的中心点用来投影打出射线算距离
    p_2d = np.array([center1_A_looming[0], center1_A_looming[1], 1.0]).reshape(3, 1)
    ray = K_inv @ p_2d
    
    # 通过 Looming 深度将这个射线投影到 3D 空间中，得到一个 3D 点 X_3d，这个点应该在纸张平面上
    X_3d = ray * (Z_looming / ray[2][0])
    # 计算这个 3D 点到平面的距离 d，理论上应该等于 Looming 测得的 Z_looming
    d = float((best_n.T @ X_3d)[0])
    # 通过这个距离 d 和数学解的 t 的方向关系，来还原出跨序列的绝对平移 t_real
    t_real = best_t * abs(d)

    print("\n🏆 (两极反转) 双序列跨界绝对姿态提取结果！")
    print("="*40)
    print(f" -> 当下帧专属深度 Z_Looming: {Z_looming:.3f} m")
    print(f" -> 摄像机到平面绝对垂直距离 d: {abs(d):.3f} m")
    print(f" -> 跨序列真实的绝对平移 t_real (m):\n{t_real}")
    print(f" -> 跨序列的绝对旋转矩阵 R :\n{np.round(best_R, 4)}")
    print("="*40)

    print("\n🎨 正在启动 3D 姿态与图像联合可视化窗口 (可拖拽旋转)...")
    visualize_dashboard(img1_A, img2_base, best_R, t_real, X_3d, best_n)
    
    print("\n🎨 正在启动带有真实纸张物理形变的 3D 窗口...")
    visualize_3d_scene(best_R, t_real, best_n, K, roi_target)

if __name__ == "__main__":
    integrate_and_solve_metric_pose()