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
# 2. 相机内参 (D456) 与超参数
# ==========================================
fx, fy = 429.78045654, 429.78045654
cx, cy = 429.94277954, 241.57313537

# ⭐ 核心：因为我们放弃了去旋转，必须把步长拉大，让目标充分膨胀，淹没颠簸噪声！
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
    """只提取平移量 t_12，放弃 R_12 的像素级应用"""
    t1, q1 = pose1[0:3], pose1[3:7]
    t2, q2 = pose2[0:3], pose2[3:7]
    R1 = R_scipy.from_quat(q1).as_matrix()
    R2 = R_scipy.from_quat(q2).as_matrix()
    
    # R_12 仅用于把世界坐标系的平移转换到相机坐标系下
    t_12 = R1.T @ (t2 - t1)        
    return t_12

# ==========================================
# 4. 纯净 Looming 测距引擎
# ==========================================
def calculate_pure_looming_Z(P1, P2, FOE, delta_d):
    """没有任何矩阵变换，只有最纯粹的点到点几何距离"""
    r1 = math.sqrt((P1[0] - FOE[0])**2 + (P1[1] - FOE[1])**2)
    r2 = math.sqrt((P2[0] - FOE[0])**2 + (P2[1] - FOE[1])**2)
    
    dr = r2 - r1
    
    # 保护机制：如果膨胀量极小，或者由于颠簸导致算出了负数（目标缩小），直接抛弃
    if dr <= 0.5: 
        return None, r1, r2, dr
        
    Z = (r1 * delta_d) / dr
    return Z, r1, r2, dr

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

    print(f">>> 🚀 启动 [无旋转纯净版] Looming 测距流水线！跨帧步长: {FRAME_STEP} ...\n")

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

        # [步骤 2] 获取平移向量与 FOE
        t_12 = calculate_relative_motion(pose1, pose2)
        tx, ty, tz = t_12
        delta_d = tz
        if delta_d < 0.02: continue # 没怎么往前走，放弃

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

        # [步骤 4] 极限测距
        Z, r1, r2, dr = calculate_pure_looming_Z(center1, center2, FOE, delta_d)

        # [步骤 5] 可视化输出
        display = img1.copy()
        
        # 画 FOE (黄色)
        cv2.drawMarker(display, (int(FOE[0]), int(FOE[1])), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
        cv2.putText(display, "FOE", (int(FOE[0])+10, int(FOE[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 画 P1 (绿色)
        cv2.circle(display, (int(center1[0]), int(center1[1])), 5, (0, 255, 0), -1)
        cv2.putText(display, "P1", (int(center1[0])-20, int(center1[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 画 P2 (原汁原味的红色点)
        cv2.circle(display, (int(center2[0]), int(center2[1])), 5, (0, 0, 255), -1)
        cv2.putText(display, "P2", (int(center2[0])+10, int(center2[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if Z is not None:
            print(f"[{idx1:03d}->{idx2:03d}] ✅ 测距成功! Z = {Z:.2f} m | dZ = {delta_d*100:.1f} cm | dr = {dr:.2f} px")
            # 连线
            cv2.line(display, (int(center1[0]), int(center1[1])), (int(center2[0]), int(center2[1])), (255, 255, 255), 1)
            # 数据看板
            cv2.putText(display, f"Depth Z: {Z:.2f} m", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            print(f"[{idx1:03d}->{idx2:03d}] ⚠️ 膨胀异常或太小 (dr={dr:.2f}px)")
            cv2.putText(display, f"Depth Z: N/A (dr={dr:.1f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.putText(display, f"delta_d: {delta_d*100:.1f} cm", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Pure Looming Depth", display)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(">>> 测距结束！")