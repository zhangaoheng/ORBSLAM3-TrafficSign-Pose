import sys
import os
import cv2
import numpy as np
import glob
import math

# 配置环境变量以导入子文件夹中的模块
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# 动态引入你编写的神奇算法模块：
# --- 1. 从 lines 提取位姿/法向单应性约束 ---
from lines.lines_tool import detect_and_filter_lines_plslam
from lines.Homography_loftr_tool import match_images_with_loftr_roi, match_images_with_auto_roi
from lines.Homography_lines_n import evaluate_orthogonality

# --- 2. 从 mid_FOE_Z_d 提取膨胀深度测距与 SLAM 轨迹关联 ---
from mid_FOE_Z_d.pure_looming_depth import (
    load_tum_trajectory, get_closest_pose, 
    calculate_relative_motion, calculate_pure_looming_Z, load_saved_rois, FRAME_STEP, fx, fy, cx, cy
)
from tools.mid import extract_four_lines_from_real_image, calculate_rectangle_center
from mid_FOE_Z_d.imgs_mid import process_sequence_with_cached_rois

# ======= 相机内参 (以你的 D456 为例) =======
K = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float64)
K_inv = np.linalg.inv(K)

def get_closest_inter_sequence_image_pair(images1, images2, slam_poses1, slam_poses2):
    """寻找两个序列中空间距离最近的两个拍摄点"""
    min_dist = float('inf')
    best_img1 = None
    best_img2 = None
    best_idx1 = -1
    best_idx2 = -1
    
    # 假设图像按时间戳命名
    times1 = {i: float(os.path.basename(p).replace(".png", ""))/1e9 for i, p in enumerate(images1)}
    times2 = {i: float(os.path.basename(p).replace(".png", ""))/1e9 for i, p in enumerate(images2)}
    
    for i1, t1 in times1.items():
        pose1 = get_closest_pose(t1, slam_poses1)
        if pose1 is None: continue
        for i2, t2 in times2.items():
            pose2 = get_closest_pose(t2, slam_poses2)
            if pose2 is None: continue
            
            dist = np.linalg.norm(pose1[0:3] - pose2[0:3])
            if dist < min_dist:
                min_dist = dist
                best_img1 = images1[i1]
                best_img2 = images2[i2]
                best_idx1 = i1
                best_idx2 = i2
                
    return best_idx1, best_idx2, min_dist

def integrate_and_solve_metric_pose():
    # ==== 序列 1 (基准建图) 路径配置 ====
    FOLDER_PATH_1 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/lines2/cam0/data"
    TRAJ_PATH_1 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/output/all_frames_lines2_slam_traj.txt"
    ROI_PATH_1 = os.path.join(FOLDER_PATH_1, "saved_rois.txt")

    # ==== 序列 2 (查询/定位) 路径配置 ====
    FOLDER_PATH_2 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/lines1/cam0/data"
    TRAJ_PATH_2 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/output/all_frames_lines1_slam_traj.txt"
    # ROI_PATH_2 不再需要全序列提取

    images1 = sorted(glob.glob(os.path.join(FOLDER_PATH_1, "*.png")))
    images2 = sorted(glob.glob(os.path.join(FOLDER_PATH_2, "*.png")))
    
    slam_poses1 = load_tum_trajectory(TRAJ_PATH_1)
    slam_poses2 = load_tum_trajectory(TRAJ_PATH_2)

    print(f">>> 🚀 启动 [物理度量恢复] 跨序列双剑合璧！ (Lines1 测 Z, d + Lines1/Lines2 算 H)...")
    
    # 1. 首先！！无特征、无框寻找两个序列的空间最近点
    print(f"正在仅依靠轨迹数据扫描空间最近姿态对...")
    idx1_base, idx2_base, min_dist = get_closest_inter_sequence_image_pair(images1, images2, slam_poses1, slam_poses2)
    if idx1_base == -1 or idx2_base == -1:
        print("未找到匹配的空间最近点，请检查轨迹！")
        return
        
    print(f"✅ 找到最近关联！Lines1_img[{idx1_base}] 与 Lines2_img[{idx2_base}], 空间距离: {min_dist:.3f} m")

    # 2. 序列 1 (Lines1) 负责提供深度测距 -> 它需要依赖连续的两帧框 (所以需保持全序列标注)
    print("👉 检验序列 1 (基准) 的连续测距掩码框...")
    process_sequence_with_cached_rois(FOLDER_PATH_1)
    saved_rois1 = load_saved_rois(ROI_PATH_1)

    # [逻辑流 ①] 在序列1内部：使用 idx1_base 与 idx1_base+FRAME_STEP 计算 Z 和 d
    idx1_next = idx1_base + FRAME_STEP
    if idx1_next >= len(images1) or idx1_next >= len(saved_rois1) or idx1_base >= len(saved_rois1):
        print("Lines1 序列后续帧不足以计算 Looming，请调整 FRAME_STEP 或选取靠前的帧。")
        return

    time1_A = float(os.path.basename(images1[idx1_base]).replace(".png", "")) / 1e9
    time1_B = float(os.path.basename(images1[idx1_next]).replace(".png", "")) / 1e9

    pose1_A = get_closest_pose(time1_A, slam_poses1)
    pose1_B = get_closest_pose(time1_B, slam_poses1)

    tx, ty, tz = calculate_relative_motion(pose1_A, pose1_B)
    delta_d = tz
    if delta_d < 0.02:
        print("⚠️ 序列 1 (Lines1) 的运动 delta_d 太小，无法计算 Looming Z！")
        return

    FOE = (fx * (tx / tz) + cx, fy * (ty / tz) + cy)
    
    img1_A = cv2.imread(images1[idx1_base])
    img1_B = cv2.imread(images1[idx1_next])
    
    lines1_A = extract_four_lines_from_real_image(img1_A, saved_rois1[idx1_base])
    lines1_B = extract_four_lines_from_real_image(img1_B, saved_rois1[idx1_next])
    
    if not lines1_A or not lines1_B:
        print("⚠️ 序列 1 提取 ROI 框失败")
        return

    center1_A, corners1_A = calculate_rectangle_center(*lines1_A)
    center1_B, _ = calculate_rectangle_center(*lines1_B)

    Z_looming, _, _, _ = calculate_pure_looming_Z(center1_A, center1_B, FOE, delta_d)
    if Z_looming is None:
        print("⚠️ 序列 1 Looming 计算失败")
        return

    print("\n"); print(f"[Lines1 内部] ✅ (被动测距模块) FOE 物理深度计算成功: Z = {Z_looming:.3f} m")


    # ============================================================
    # [逻辑流 ②] 跨序列运算 (Lines1的A图 与 Lines2唯独的那一张定位基准图)
    # ============================================================
    print("\n"); print(f"🔄 (跨界寻迹模块) 启动序列间提取 (Lines1 <--> Lines2)...")
    img2_base = cv2.imread(images2[idx2_base])
    
    # 【亮点更新】不再提取序列2的全部狂！只针对刚刚算出来的最近帧手动标注！
    print(f"🔔 [简化操作] 用户仅需针对序列 2 的当前匹配帧，框选一次 ROI:")
    roi_bbox_2 = cv2.selectROI(f"Select Target on Lines2 Frame {idx2_base}", img2_base, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    
    if roi_bbox_2[2] == 0 or roi_bbox_2[3] == 0:
        print("⚠️ 序列 2 手动取消框选，退出！")
        return

    lines2_base = extract_four_lines_from_real_image(img2_base, roi_bbox_2)
    if not lines2_base:
        print("⚠️ 序列 2 基准图像提取 ROI 失败")
        return
        
    center2_base, corners2_base = calculate_rectangle_center(*lines2_base)
    
    # 提取两张图像外置完全共面的绝对区域，作为 LoFTR 的硬掩盖边界
    pts1_roi = np.array(corners1_A, dtype=np.int32)
    pts2_roi = np.array(corners2_base, dtype=np.int32)
    
    pts1, pts2 = match_images_with_auto_roi(img1_A, img2_base, pts1_roi, pts2_roi)
    if pts1 is None or len(pts1) < 4:
        print("  ⚠️ 特征点不足，跨序列匹配失败！")
        return
        
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    num_solutions, Rs, Ts, normals = cv2.decomposeHomographyMat(H, K)
    
    best_idx = -1
    for j in range(num_solutions):
        if normals[j].flatten()[2] > 0: # 简单面向相机粗筛 (后续可替换正交验证)
            best_idx = j
            break
            
    if best_idx == -1:
        print("  ⚠️ 无合法相机朝向解，跳过！")
        return

    R_rel = Rs[best_idx]
    t_norm = Ts[best_idx]
    n_cv = normals[best_idx].flatten()

    print(f"✅ 无尺度跨序列 H(R,t) 位姿计算完毕! 法向: {n_cv}")

    # ============================================================
    # [逻辑流 ③] 双序列完美闭环：用 Lines1 算出的 Z_looming 恢复真实 t_real
    # ============================================================
    p_2d = np.array([center1_A[0], center1_A[1], 1.0]).reshape(3, 1)
    ray = K_inv @ p_2d
    
    # 利用序列 1 的深度恢复透视投影常数量
    X_3d = ray * (Z_looming / ray[2][0])
    d = float((n_cv.T @ X_3d)[0])
    
    # 把常数量投射到跨度平移之上
    t_real = t_norm * abs(d)

    print("\n"); print(f"🏆 (两极反转) 双序列跨界绝对姿态提取结果！")
    print("="*40)
    print(f" -> 摄像机到平面绝对垂直距离: {abs(d):.3f} m (代数 d={d:.3f})")
    print(f" -> Lines1至Lines2 的真实相对绝对平移 t_real (m):\n{t_real}")
    print(f" -> Lines1至Lines2 的旋转矩阵 R 变换:\n{R_rel}")
    print("="*40)

if __name__ == "__main__":
    integrate_and_solve_metric_pose()
