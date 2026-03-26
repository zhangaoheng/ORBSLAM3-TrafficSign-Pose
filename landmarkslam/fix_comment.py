with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "w", encoding='utf-8') as f:
    f.write("""import cv2
import numpy as np
import os
import glob
import argparse

def load_yolo_detections(filename):
    \"\"\"
    加载YOLO检测结果的txt文件。
    该txt文件格式假定为每一行为：[图片名称] [x1 y1] [x2 y2] [x3 y3] [x4 y4]
    代表单张图中路牌的4个角点像素坐标。
    返回一个字典，键是图片名，值是4个角点的numpy数组。
    \"\"\"
    detections = {}
    if not os.path.exists(filename):
        return detections
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            # 确保行中至少有图片名加8个坐标数据(一共9个部分)
            if len(parts) >= 9:
                img_name = parts[0]
                pts = []
                for i in range(4):
                    # 将 x, y 坐标转为浮点数并存入列表
                    pts.append([float(parts[1 + i*2]), float(parts[2 + i*2])])
                detections[img_name] = np.array(pts, dtype=np.float32)
    return detections

def load_camera_intrinsics(yaml_file):
    \"\"\"
    加载相机的内参矩阵 (K)。
    如果在传入的 yaml 文件中找到了相应字段，则覆盖默认值；否则使用默认值。
    \"\"\"
    # 设置默认内参 (由 landmarkslam.yaml 提供)
    fx = 935.307
    fy = 935.307
    cx = 960.0
    cy = 540.0
    
    # 尝试从 yaml 配置文件中读取内参
    try:
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r') as f:
                for line in f:
                    if "Camera1.fx:" in line: fx = float(line.split(":")[1].strip())
                    if "Camera1.fy:" in line: fy = float(line.split(":")[1].strip())
                    if "Camera1.cx:" in line: cx = float(line.split(":")[1].strip())
                    if "Camera1.cy:" in line: cy = float(line.split(":")[1].strip())
    except: 
        pass
        
    # 构建 3x3 相机内参矩阵
    K = np.array([[fx, 0, cx], 
                  [0, fy, cy], 
                  [0, 0,  1]], dtype=np.float64)
    return K

def is_point_inside_polygon(pt, polygon):
    \"\"\"
    利用 OpenCV 的 pointPolygonTest 函数判断一个 2D 像素点是否在 YOLO 给出的 4 边形检测框内。
    \"\"\"
    # 返回 >= 0 表示点在多边形内部或边界上，<0 表示在外部
    return cv2.pointPolygonTest(polygon, (float(pt[0]), float(pt[1])), False) >= 0

def backproject_to_plane(pt, K_inv, R, t):
    \"\"\"
    数学核心函数：将 2D 图像特征点反投影到 3D 空间。
    我们假设路牌是一个完美的 2D 平面 (在局部坐标系中 Z=0)。
    通过相机投影原理：深度参数 s * K^-1 * [u, v, 1]^T = P_c (相机坐标系下的 3D 点)
    然后将 P_c 转换到路牌的世界坐标系下相交于 Z=0 求得坐标。
    \"\"\"
    # 1. 2D 像素坐标变成齐次坐标 pt_c = [u, v, 1]^T
    pt_c = np.array([pt[0], pt[1], 1.0], dtype=np.float64).reshape(3, 1)
    
    # 2. 乘以内参矩阵的逆转换为相机归一化坐标射线
    ray_c = K_inv @ pt_c
    
    # 3. 求解尺度 s (光线与 Z=0 平面相交的深度)
    R_T = R.T
    r_row2 = R_T[2, :] # R^T的最后一行，对应 Z 维度的变换
    num = np.dot(r_row2, t)
    den = np.dot(r_row2, ray_c)
    
    # 若分母过小，说明光线平行于平面无交点，舍弃
    if abs(float(den)) < 1e-6: return None
    s = num / den
    
    # 4. 根据求得的尺度 s 还原出相机坐标系下的 3D 点坐标 P_c
    P_c = s * ray_c
    
    # 5. 再将其转换到局部物体坐标系 P_w = R^T * (P_c - t)
    P_w = R_T @ (P_c - t)
    
    # 返回对应的 [X, Y, Z]
    return [float(P_w[0]), float(P_w[1]), float(P_w[2])]

def save_point_cloud_txt(points, colors, frames, filename):
    \"\"\"
    保存计算出的点云数据到 TXT 文件中。
    每行格式为：X Y Z R G B Frame_Name
    \"\"\"
    if not points: return
    with open(filename, 'w') as f:
        # 同时遍历每个点的位置、颜色以及来源帧
        for p, c, fn in zip(points, colors, frames):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[2])} {int(c[1])} {int(c[0])} {fn}\\n")

def main():
    # ==== 0. 接受命令行参数配置 ====
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="../landmarkslam.yaml")               # YAML 配置文件，用于提取内参
    parser.add_argument("--images", default="../data/20260321_111801/")             # 提供原始图像的数据集文件夹
    parser.add_argument("--yolo", required=True)                                    # YOLO 检测角点保存文件
    parser.add_argument("--output", default="/home/zah/ORB_SLAM3-master/landmarkslam/output2/sign_cloud_py.txt") # 输出结果点云文本
    args = parser.parse_args()

    # ==== 1. 加载相机内参 ====
    K = load_camera_intrinsics(args.settings)
    K_inv = np.linalg.inv(K)  # 预先求内参矩阵的逆矩阵以加速计算

    # ==== 2. 获取目标检测结果 ====
    yolo_data = load_yolo_detections(args.yolo)
    if not yolo_data: return

    # ==== 3. 读取本地所有待处理的图片序列 ====
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        image_paths.extend(glob.glob(os.path.join(args.images, ext)))
    image_paths.sort() # 按帧名排序

    # ==== 4. 定义3D空间中标志牌模型先验尺度 ====
    # 将标准路牌认定为中心在原点(0,0)，宽度和高度为 1.0 的正方形。Z=0的局部坐标系
    sign_w, sign_h = 1.0, 1.0
    obj_pts = np.array([
        [-sign_w/2, -sign_h/2], # 左下
        [ sign_w/2, -sign_h/2], # 右下
        [ sign_w/2,  sign_h/2], # 右上
        [-sign_w/2,  sign_h/2]  # 左上
    ], dtype=np.float32)

    # ==== 5. 初始化 ORB 特征提取器预设 ====
    # 从您的 yaml 配置文件同步过来的具体参数，使得角点的提取逻辑与您的 ORB-SLAM3 一致
    orb = cv2.ORB_create(
        nfeatures=1000,         # 最大提取 1000 个特征点
        scaleFactor=1.2,        # 缩放金字塔金字塔系数
        nlevels=8,              # 提取图层数
        edgeThreshold=20,       # 边缘过滤的阈值边界
        firstLevel=0, 
        WTA_K=2, 
        scoreType=cv2.ORB_HARRIS_SCORE, 
        patchSize=20, 
        fastThreshold=7         # FAST 角点响应的最低灵敏度阈值
    )

    # 存储最终的所有 3D点，他们的颜色，和他们分别出自哪张图像的记录列表
    all_sign_points = []
    all_sign_colors = []
    all_sign_frames = []

    # ==== 6. 核心处理循环: 逐帧处理 ====
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        
        # 仅处理被 YOLO 检测出角点的有效图像
        if base_name not in yolo_data: continue
        corners = yolo_data[base_name]  # [4, 2] 包含这帧里路牌的四个像素点
        
        im = cv2.imread(img_path)
        if im is None: continue

        # ---- A. 通过四个平面对应的角点对求解单应性矩阵 (Homography) ----
        # findHomography 是求平面映射的，将我们假定的3D 1x1尺度的 4个角点投影到当前的图像上对应位置
        H, status = cv2.findHomography(obj_pts, corners)
        if H is None: continue

        # ---- B. 矩阵分解，求取当前路牌平面的位姿 (R 旋转，t 平移法向量) ----
        num, Rs, ts, normals = cv2.decomposeHomographyMat(H, K)
        if num == 0: continue

        # 单应性分解会产生多个可能解，通常根据几何常识过滤: 
        # 路牌的法向 Z 是面向相机的，并且离我们处于前方的一个绝对正距离(t_Z > 0)
        best_idx = 0
        for j in range(num):
            if normals[j][2][0] < 0 and ts[j][2][0] > 0:
                best_idx = j
                break
                
        # 拿到筛选出的最优平面位姿
        R = Rs[best_idx]
        t = ts[best_idx]

        # ---- C. ORB特征点提取提取与筛选区域 ----
        keypoints, des = orb.detectAndCompute(im, None)
        if keypoints is None: continue
        
        valid_pts_count = 0
        for kp in keypoints:
            # 只过滤、提取处于YOLO四边形框"内部"的特征点（排除背景）
            if is_point_inside_polygon(kp.pt, corners):
                # 利用上述求得的平面位姿进行反投影计算出真正的 3D 点
                pt3d = backproject_to_plane(kp.pt, K_inv, R, t)
                if pt3d is not None:
                    # 如果有有效的深度，推入列表
                    all_sign_points.append(pt3d)
                    
                    # 同时利用像素坐标提取出原图里这个点对应的 RGB 颜色
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    color = im[y, x] # OpenCV 读取的数据结构为 [B, G, R]
                    
                    # 放入总数组维护
                    all_sign_colors.append(color)
                    all_sign_frames.append(base_name)
                    valid_pts_count += 1

    # ==== 7. 将循环处理获得的所有点位信息保存。 ====
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_point_cloud_txt(all_sign_points, all_sign_colors, all_sign_frames, args.output)

if __name__ == "__main__":
    main()
""")
