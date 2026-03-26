import sys

path = "/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py"
with open(path, 'r', encoding='utf-8') as f:
    code = f.read()

# 1. 增加 3D 坐标模型点
o1 = "obj_pts = np.array([[-sign_w/2, -sign_h/2], [sign_w/2, -sign_h/2], [sign_w/2, sign_h/2], [-sign_w/2, sign_h/2]], dtype=np.float32)"
n1 = o1 + "\n    # 同时定义带有Z轴(Z=0)的3D理论位置，用于 PnP 求解相机位姿\n    obj_pts_3d = np.array([[-sign_w/2, -sign_h/2, 0.0], [sign_w/2, -sign_h/2, 0.0], [sign_w/2, sign_h/2, 0.0], [-sign_w/2, sign_h/2, 0.0]], dtype=np.float32)"
code = code.replace(o1, n1)

# 2. 增加列表变两
o2 = """    all_sign_points = []
    all_sign_colors = []
    all_sign_frames = []

    processed_frames = 0"""
n2 = """    all_sign_points = []
    all_sign_colors = []
    all_sign_frames = []
    
    all_camera_centers = []
    all_camera_frames = []

    processed_frames = 0"""
code = code.replace(o2, n2)

# 3. 增加 PnP 位姿计算
o3 = """        # ================== 核心思路修改：先抠图再提点 =================="""
n3 = """        # ================== 新增：通过 PnP 算法计算相机在局部坐标系中的轨迹点 ==================
        # 用带 Z 轴深度的 obj_pts_3d 与图片 2D 角点计算真实的相机位姿
        success, rvec, tvec = cv2.solvePnP(obj_pts_3d, corners, K, None)
        if success:
            R_mat, _ = cv2.Rodrigues(rvec)
            # R_mat, tvec 是从标志牌(世界)到相机的变换，反求相机中心在标志牌坐标系的位置 C = -R^T * tvec
            camera_center = -R_mat.T @ tvec
            cx, cy, cz = float(camera_center[0][0]), float(camera_center[1][0]), float(camera_center[2][0])
            
            # 记录单独的轨迹点
            all_camera_centers.append([cx, cy, cz])
            all_camera_frames.append(base_name)
            
            # 同时也添加到统一点云中展示，设置亮绿色 [0, 255, 0] (BGR格式=Green) 便于和标志牌区分
            all_sign_points.append([cx, cy, cz])
            all_sign_colors.append(np.array([0, 255, 0]))
            all_sign_frames.append(base_name + "_camera")

        # ================== 核心思路修改：先抠图再提点 =================="""
code = code.replace(o3, n3)

# 4. 增加保存文件逻辑
o4 = """    print(f"Saved point cloud to {args.output}")"""
n4 = """    print(f"Saved point cloud to {args.output}")

    # ================== 新增：保存单独的相机运行轨迹 ==================
    traj_output = args.output.replace(".txt", "_trajectory.txt")
    if len(all_camera_centers) > 0:
        with open(traj_output, 'w') as f:
            f.write("# Format: Frame_Name X Y Z\\n")
            for c, fn in zip(all_camera_centers, all_camera_frames):
                f.write(f"{fn} {c[0]:.5f} {c[1]:.5f} {c[2]:.5f}\\n")
        print(f"Saved camera trajectory to {traj_output}")"""
code = code.replace(o4, n4)

with open(path, 'w', encoding='utf-8') as f:
    f.write(code)

print("Path applied successfully.")
