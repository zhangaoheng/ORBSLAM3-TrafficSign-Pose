import cv2
import os
import glob
import numpy as np
import math
from scipy.spatial.transform import Rotation as R_scipy

# ==========================================
# 模块 1：导入你的自定义 2D 几何提取函数
# ==========================================
from tools.mid import extract_four_lines_from_real_image, calculate_rectangle_center

# ==========================================
# 模块 2：全局参数与相机内参 (D456)
# ==========================================
fx, fy = 429.78045654, 429.78045654
cx, cy = 429.94277954, 241.57313537

K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]], dtype=np.float64)
K_inv = np.linalg.inv(K)   

FRAME_STEP = 10  # 跨帧步长 (拉大基线，建议 5 到 15)

# ==========================================
# 模块 3：基础数据读取函数
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
    t1, q1 = pose1[0:3], pose1[3:7]
    t2, q2 = pose2[0:3], pose2[3:7]
    R1 = R_scipy.from_quat(q1).as_matrix()
    R2 = R_scipy.from_quat(q2).as_matrix()
    R_12 = R1.T @ R2               
    t_12 = R1.T @ (t2 - t1)        
    return R_12, t_12

# ==========================================
# 模块 4：单应去旋与 Looming 核心数学
# ==========================================
def derotate_point(P_raw, R_12):
    """使用纯旋转单应性矩阵 H 消除相机颠簸"""
    H = K @ R_12 @ K_inv
    # 转为齐次坐标 [u, v, 1]
    P_homo = np.array([P_raw[0], P_raw[1], 1.0])
    # 映射回第一帧姿态
    P_pure_homo = H @ P_homo
    # 归一化回 2D 像素坐标
    u_pure = P_pure_homo[0] / P_pure_homo[2]
    v_pure = P_pure_homo[1] / P_pure_homo[2]
    return (u_pure, v_pure)

def calculate_looming_Z(P1, P2_pure, FOE, delta_d):
    """纯平移宇宙下的 Looming 绝杀公式"""
    r1 = math.sqrt((P1[0] - FOE[0])**2 + (P1[1] - FOE[1])**2)
    r2 = math.sqrt((P2_pure[0] - FOE[0])**2 + (P2_pure[1] - FOE[1])**2)
    dr = r2 - r1
    
    if dr <= 0.2: # 像素膨胀量太小，无意义
        return None, r1, r2, dr
        
    Z = (r1 * delta_d) / dr
    return Z, r1, r2, dr

# ==========================================
# 模块 5：主循环流水线 (模块化组装)
# ==========================================
if __name__ == "__main__":
    FOLDER_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/mid"
    TRAJ_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/slam_frames/all_frames_my_mid_test.txt"
    ROI_PATH = os.path.join(FOLDER_PATH, "saved_rois.txt")

    images = sorted(glob.glob(os.path.join(FOLDER_PATH, "*.png")))
    slam_poses = load_tum_trajectory(TRAJ_PATH)
    saved_rois = load_saved_rois(ROI_PATH)

    print(f">>> 启动 Looming 测距流水线！跨帧步长: {FRAME_STEP} ...\n")

    for i in range(len(images) - FRAME_STEP):
        idx1 = i
        idx2 = i + FRAME_STEP
        
        # 必须确保这两帧都有保存好的 ROI 框
        if idx1 >= len(saved_rois) or idx2 >= len(saved_rois):
            break

        # 【步骤 1】获取时间戳与 SLAM 位姿
        time1 = float(os.path.basename(images[idx1]).replace(".png", "")) / 1e9
        time2 = float(os.path.basename(images[idx2]).replace(".png", "")) / 1e9
        
        pose1 = get_closest_pose(time1, slam_poses)
        pose2 = get_closest_pose(time2, slam_poses)
        if pose1 is None or pose2 is None: continue

        # 算出运动参数 R 和 t
        R_12, t_12 = calculate_relative_motion(pose1, pose2)
        tx, ty, tz = t_12
        delta_d = tz
        if delta_d < 0.01: continue # 没往前走

        # 算出 FOE
        FOE = (fx * (tx / tz) + cx, fy * (ty / tz) + cy)

        # 【步骤 2】调用你的模块，提取 2D 几何中心
        img1 = cv2.imread(images[idx1])
        img2 = cv2.imread(images[idx2])
        
        lines1 = extract_four_lines_from_real_image(img1, saved_rois[idx1])
        lines2 = extract_four_lines_from_real_image(img2, saved_rois[idx2])
        if not lines1 or not lines2: continue

        center1, _ = calculate_rectangle_center(*lines1)
        center2_raw, _ = calculate_rectangle_center(*lines2)
        if not center1 or not center2_raw: continue

        # 【步骤 3】单应去旋：洗掉颠簸！
        center2_pure = derotate_point(center2_raw, R_12)

        # 【步骤 4】代入 Looming 计算深度 Z
        Z, r1, r2, dr = calculate_looming_Z(center1, center2_pure, FOE, delta_d)

        if Z is not None:
            print(f"[{idx1:03d}->{idx2:03d}] 🌟 测距成功! Z = {Z:.2f} 米 | dZ = {delta_d*100:.1f}cm | dr = {dr:.2f}px")
            
            # ==============================
            # 可视化：看清单应去旋的威力
            # ==============================
            display = img1.copy()
            
            # 1. 画 FOE (黄色十字)
            cv2.drawMarker(display, (int(FOE[0]), int(FOE[1])), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
            cv2.putText(display, "FOE", (int(FOE[0])+10, int(FOE[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 2. 画 第 1 帧的中心点 P1 (绿色圆点)
            cv2.circle(display, (int(center1[0]), int(center1[1])), 5, (0, 255, 0), -1)
            cv2.putText(display, "P1", (int(center1[0])-20, int(center1[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 3. 画 第 2 帧的受污染点 P2_raw (红色空心圆)
            cv2.circle(display, (int(center2_raw[0]), int(center2_raw[1])), 5, (0, 0, 255), 2)
            
            # 4. 画 第 2 帧的纯净点 P2_pure (蓝色实心点) -> 这才是真正的物理膨胀点！
            cv2.circle(display, (int(center2_pure[0]), int(center2_pure[1])), 5, (255, 0, 0), -1)
            cv2.putText(display, "P2_pure", (int(center2_pure[0])+10, int(center2_pure[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 画线：P1 连向 P2_pure，它应该完美指向（或背离） FOE！
            cv2.line(display, (int(center1[0]), int(center1[1])), (int(center2_pure[0]), int(center2_pure[1])), (255, 255, 255), 1)

            # 数据看板
            cv2.putText(display, f"Depth Z: {Z:.2f} m", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display, f"delta_d: {delta_d*100:.1f} cm", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"delta_r: {dr:.2f} px", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Looming Depth Final", display)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            print(f"[{idx1:03d}->{idx2:03d}] ⚠️ 膨胀太小跳过 (dr={dr:.2f}px)")

    cv2.destroyAllWindows()
    print(">>> 测距结束！")