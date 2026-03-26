import re

file_path = "/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Add prev_rvec, prev_tvec initialization
if "prev_rvec = None" not in text:
    text = text.replace("processed_frames = 0", "processed_frames = 0\n    prev_rvec = None\n    prev_tvec = None\n")

# Modify the PnP block
old_pnp = """        # ================== 新增：通过 PnP 算法计算相机在局部坐标系中的轨迹点 ==================
        # 用带 Z 轴深度的 obj_pts_3d 与图片 2D 角点计算真实的相机位姿
        success, rvec, tvec = cv2.solvePnP(obj_pts_3d, corners, K, None, flags=cv2.SOLVEPNP_IPPE)
        if success:"""

new_pnp = """        # ================== 新增：利用上一帧的结果作为先验进行 PnP 求解，防止平面歧义翻转 ==================
        if prev_rvec is None:
            # 第一帧，使用 IPPE 获取一个稳定的初始解
            success, rvec, tvec = cv2.solvePnP(obj_pts_3d, corners, K, None, flags=cv2.SOLVEPNP_IPPE)
        else:
            # 后续帧，使用上一帧作为初始猜想，强制解算器在当前轨迹附近优化，防止跳跃到镜像解
            success, rvec, tvec = cv2.solvePnP(obj_pts_3d, corners, K, None, prev_rvec.copy(), prev_tvec.copy(), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        
        if success:
            prev_rvec = rvec
            prev_tvec = tvec"""

text = text.replace(old_pnp, new_pnp)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

