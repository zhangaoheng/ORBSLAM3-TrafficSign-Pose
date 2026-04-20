import os
import glob
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# ==========================================
# 1. 相机内参 (基于 D456.yaml)
# ==========================================
fx, fy = 429.78045654, 429.78045654
cx, cy = 429.94277954, 241.57313537

K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]])
K_inv = np.linalg.inv(K)

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
        return poses_dict[closest_time], closest_time
    return None, None

def calculate_relative_motion(pose1, pose2):
    t1, q1 = pose1[0:3], pose1[3:7]
    R1 = R_scipy.from_quat(q1).as_matrix()
    
    t2, q2 = pose2[0:3], pose2[3:7]
    R2 = R_scipy.from_quat(q2).as_matrix()
    
    R_12 = R1.T @ R2               
    t_12 = R1.T @ (t2 - t1)        
    return R_12, t_12

if __name__ == "__main__":
    TRAJ_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/slam_frames/all_frames_my_mid_test.txt"
    IMG_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/mid"
    
    slam_poses = load_tum_trajectory(TRAJ_PATH)
    images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    
    # 诊断
    slam_times = np.array(list(slam_poses.keys()))
    img_times = np.array([float(os.path.basename(f).replace(".png", "")) / 1e9 for f in images])
    if np.max(img_times) < np.min(slam_times):
        print("❌ 完蛋！你的路牌图片全在 SLAM 初始化成功之前！")
        exit()

    # ========================================================
    # ⭐ 核心修复：设置跨帧步长，压制 FOE 乱飞
    # ========================================================
    
    # 对于步长的设置
    FRAME_STEP = 10  # 跨 5 帧计算一次，拉大物理前进距离 (Delta d)
    
    print(f">>> 准备对齐图片，当前跨帧步长: {FRAME_STEP} 帧。启动可视化窗口...\n")
    success_count = 0
    
    # 注意这里循环范围变成了 len(images) - FRAME_STEP
    for i in range(len(images) - FRAME_STEP):
        # 取第 i 帧
        time1 = float(os.path.basename(images[i]).replace(".png", "")) / 1e9
        # 取第 i + FRAME_STEP 帧！
        time2 = float(os.path.basename(images[i+FRAME_STEP]).replace(".png", "")) / 1e9
        
        pose1, t_real1 = get_closest_pose(time1, slam_poses)
        pose2, t_real2 = get_closest_pose(time2, slam_poses)
        
        if pose1 is not None and pose2 is not None:
            R_12, t_12 = calculate_relative_motion(pose1, pose2)
            delta_d = t_12[2] 
            
            if t_12[2] > 0.001:
                u_foe = fx * (t_12[0] / t_12[2]) + cx
                v_foe = fy * (t_12[1] / t_12[2]) + cy
            else:
                u_foe, v_foe = cx, cy
            
            print(f"[{i:03d} -> {i+FRAME_STEP:03d}] 匹配成功! Δd: {delta_d * 100:.2f} cm | FOE: ({u_foe:.1f}, {v_foe:.1f})")
            success_count += 1

            # ========================================================
            # 👁️ 可视化模块
            # ========================================================
            img = cv2.imread(images[i])
            if img is None: continue

            # 画出相机光心 (图像物理中心点 cx, cy) - 用蓝色十字标出
            cv2.drawMarker(img, (int(cx), int(cy)), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.putText(img, "Optical Center", (int(cx) - 50, int(cy) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # 画出 FOE 膨胀焦点 - 用醒目的红色靶心标出
            uf, vf = int(u_foe), int(v_foe)
            if -5000 < uf < 5000 and -5000 < vf < 5000:
                cv2.drawMarker(img, (uf, vf), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)
                cv2.circle(img, (uf, vf), 15, (0, 0, 255), 2)
                cv2.putText(img, "FOE", (uf + 20, vf), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 连线：从光心到 FOE
                cv2.line(img, (int(cx), int(cy)), (uf, vf), (0, 255, 255), 2)

            # 在左上角打上关键数据看板
            cv2.putText(img, f"Frame: {i:03d} -> {i+FRAME_STEP:03d}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Forward dZ: {delta_d*100:.2f} cm", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"FOE: ({u_foe:.1f}, {v_foe:.1f})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 显示图像
            cv2.imshow("FOE Visualizer - Stable", img)
            
            # 等待按键 (0表示无限等待，直到你按任意键)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print(">>> 用户手动退出可视化。")
                break

    cv2.destroyAllWindows()
    print(f"\n🎉 处理完毕！")