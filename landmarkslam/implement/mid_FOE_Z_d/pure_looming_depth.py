import cv2
import os
import glob
import numpy as np
import math
from scipy.spatial.transform import Rotation as R_scipy

# ==========================================
# 1. 导入你的核心模块
# ==========================================
from tools.mid import extract_four_lines_from_real_image, calculate_rectangle_center

# ==========================================
# 2. 相机内参 (deepseek D456) 与超参数
# ==========================================
fx, fy = 426.372, 425.671
cx, cy = 435.525, 244.974

K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]], dtype=np.float64)
K_inv = np.linalg.inv(K)

FRAME_STEP = 15

# ==========================================
# 3. 基础加载函数
# ==========================================
def load_saved_rois(txt_path):
    rois = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                if line.strip():
                    x, y, w, h = map(int, line.strip().split(','))
                    rois.append((x, y, w, h))
    return rois

def load_tum_trajectory(traj_path):
    poses = {}
    with open(traj_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            data = list(map(float, line.strip().split()))
            poses[data[0]] = np.array(data[1:]) 
    return poses

def get_closest_pose(target_time, poses_dict, time_thresh=0.03):
    times = np.array(list(poses_dict.keys()))
    idx = np.argmin(np.abs(times - target_time))
    closest_time = times[idx]
    if np.abs(closest_time - target_time) < time_thresh:
        return poses_dict[closest_time]
    return None

def calculate_relative_motion(pose1, pose2):
    """返回 R_12 和 t_12，R_12 用于去旋，t_12 用于 FOE 和 delta_d"""
    t1, q1 = pose1[0:3], pose1[3:7]
    t2, q2 = pose2[0:3], pose2[3:7]
    R1 = R_scipy.from_quat(q1).as_matrix()
    R2 = R_scipy.from_quat(q2).as_matrix()
    R_12 = R1.T @ R2
    t_12 = R1.T @ (t2 - t1)
    return R_12, t_12

# ==========================================
# 4. 单应去旋 + Looming 测距引擎
# ==========================================
def derotate_point(P_raw, R_mat):
    """将帧A的像素点去旋到纯平移宇宙（对齐帧B的朝向）
    H = K @ R_mat @ K_inv，其中 R_mat = R_12.T 把帧B旋转坐标系拉回帧A"""
    H = K @ R_mat @ K_inv
    P_homo = np.array([P_raw[0], P_raw[1], 1.0])
    P_pure_homo = H @ P_homo
    return (P_pure_homo[0] / P_pure_homo[2], P_pure_homo[1] / P_pure_homo[2])

def calculate_pure_looming_Z(P_center_far, P_center_near, FOE, delta_d, R_12=None):
    """去旋后计算 Looming Z。
    P_center_far:  远帧（帧A）的LSD中心（原始像素坐标）
    P_center_near: 近帧（帧B）的LSD中心（作为纯平移参考，不去旋）
    R_12:          R_A->B，用于将远帧方向旋转对齐到近帧
    若 R_12=None 则使用原始坐标（兼容旧调用，但不再推荐）"""
    if R_12 is not None:
        # 将帧A中心去旋到帧B的相机方向：H = K @ R_12.T @ K_inv
        P_far_pure = derotate_point(P_center_far, R_12.T)
    else:
        P_far_pure = P_center_far

    r_far = math.sqrt((P_far_pure[0] - FOE[0])**2 + (P_far_pure[1] - FOE[1])**2)
    r_near = math.sqrt((P_center_near[0] - FOE[0])**2 + (P_center_near[1] - FOE[1])**2)

    dr = r_near - r_far

    # 保护机制：膨胀量必须为正且足够大
    if dr <= 0.5:
        return None, r_far, r_near, dr

    # Z = r_near * delta_d / dr - delta_d
    # 推导：r_far / r_near = Z / (Z + delta_d)，展开得 Z = r_near * delta_d / dr - delta_d
    Z = (r_near * delta_d) / dr - delta_d
    return Z, r_far, r_near, dr

# ==========================================
# 5. 主流水线
# ==========================================
if __name__ == "__main__":
    FOLDER_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/mid"
    TRAJ_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/slam_frames/all_frames_my_mid_test.txt"
    ROI_PATH = os.path.join(FOLDER_PATH, "saved_rois.txt")

    images = sorted(glob.glob(os.path.join(FOLDER_PATH, "*.png")))
    slam_poses = load_tum_trajectory(TRAJ_PATH)
    saved_rois = load_saved_rois(ROI_PATH)

    print(f">>> 启动 [去旋修正版] Looming 测距流水线！跨帧步长: {FRAME_STEP} ...\n")

    for i in range(len(images) - FRAME_STEP):
        idx1 = i
        idx2 = i + FRAME_STEP

        if idx1 >= len(saved_rois) or idx2 >= len(saved_rois):
            break

        # [步骤 1] 匹配 SLAM 姿态
        time1 = float(os.path.basename(images[idx1]).replace(".png", "")) / 1e9
        time2 = float(os.path.basename(images[idx2]).replace(".png", "")) / 1e9

        pose1 = get_closest_pose(time1, slam_poses)
        pose2 = get_closest_pose(time2, slam_poses)
        if pose1 is None or pose2 is None: continue

        # [步骤 2] 获取 R_12、t_12 与 FOE
        R_12, t_12 = calculate_relative_motion(pose1, pose2)
        tx, ty, tz = t_12
        delta_d = tz
        if delta_d < 0.02: continue

        FOE = (fx * (tx / tz) + cx, fy * (ty / tz) + cy)

        # [步骤 3] 获取图像中的真实 2D 中心点
        img1 = cv2.imread(images[idx1])
        img2 = cv2.imread(images[idx2])

        lines1 = extract_four_lines_from_real_image(img1, saved_rois[idx1])
        lines2 = extract_four_lines_from_real_image(img2, saved_rois[idx2])
        if not lines1 or not lines2: continue

        center1, _ = calculate_rectangle_center(*lines1)
        center2, _ = calculate_rectangle_center(*lines2)
        if not center1 or not center2: continue

        # [步骤 4] 去旋 + 测距
        Z, r_far, r_near, dr = calculate_pure_looming_Z(
            center1, center2, FOE, delta_d, R_12
        )

        # [步骤 5] 可视化输出
        display = img1.copy()

        # 画 FOE (黄色)
        cv2.drawMarker(display, (int(FOE[0]), int(FOE[1])), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
        cv2.putText(display, "FOE", (int(FOE[0])+10, int(FOE[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 画 P1 (绿色)
        cv2.circle(display, (int(center1[0]), int(center1[1])), 5, (0, 255, 0), -1)
        cv2.putText(display, "P1", (int(center1[0])-20, int(center1[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 画 P2 (红色)
        cv2.circle(display, (int(center2[0]), int(center2[1])), 5, (0, 0, 255), -1)
        cv2.putText(display, "P2", (int(center2[0])+10, int(center2[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if Z is not None:
            print(f"[{idx1:03d}->{idx2:03d}] 测距成功! Z = {Z:.2f} m | dZ = {delta_d*100:.1f} cm | dr = {dr:.2f} px")
            cv2.line(display, (int(center1[0]), int(center1[1])), (int(center2[0]), int(center2[1])), (255, 255, 255), 1)
            cv2.putText(display, f"Depth Z: {Z:.2f} m", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            print(f"[{idx1:03d}->{idx2:03d}] 膨胀异常或太小 (dr={dr:.2f}px)")
            cv2.putText(display, f"Depth Z: N/A (dr={dr:.1f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.putText(display, f"delta_d: {delta_d*100:.1f} cm", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Looming Depth (Derotated)", display)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(">>> 测距结束！")