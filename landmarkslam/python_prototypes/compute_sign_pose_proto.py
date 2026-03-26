import cv2
import numpy as np
import os
import glob
import argparse

def load_yolo_detections(filename):
    """
    加载 YOLO 目标检测输出的角点结果文件。
    该文件记录了每张图片中目标（标志牌）的 4 个多边形角点像素坐标。
    返回一个字典：键为图片文件名，值为包含4个 [x, y] 坐标的 numpy 数组。
    """
    detections = {}
    if not os.path.exists(filename):
        return detections
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            # 确保行内包含文件名以及至少4个角点的8个浮点数值
            if len(parts) >= 9:
                img_name = parts[0]
                pts = []
                # 提取 4 个角点的 (x, y) 像素坐标
                for i in range(4):
                    pts.append([float(parts[1 + i*2]), float(parts[2 + i*2])])
                detections[img_name] = np.array(pts, dtype=np.float32)
    return detections

def load_camera_intrinsics(yaml_file):
    """
    从 SLAM 的配置文件 (.yaml) 中解析并加载相机的内参矩阵 (K)。
    如果文件不存在或解析失败，则使用硬编码的默认参数。
    内参矩阵主要包含：焦距 (fx, fy) 和 光心偏移主点 (cx, cy)。
    """
    fx, fy, cx, cy = 935.307, 935.307, 960.0, 540.0
    try:
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r') as f:
                for line in f:
                    if "Camera1.fx:" in line: fx = float(line.split(":")[1].strip())
                    if "Camera1.fy:" in line: fy = float(line.split(":")[1].strip())
                    if "Camera1.cx:" in line: cx = float(line.split(":")[1].strip())
                    if "Camera1.cy:" in line: cy = float(line.split(":")[1].strip())
    except: pass
    
    # 构造 3x3 相机内参矩阵
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0,  1]], dtype=np.float64)
    return K


def sort_corners(pts):
    """
    将输入的4个点按照 左上、右上、右下、左下 的顺序稳定排序。
    通过计算中心点，利用 arctan2 求极角排序。
    """
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    idx = np.argsort(angles)
    return pts[idx]

def is_point_inside_polygon(pt, polygon):
    """
    判断特征点是否位于 YOLO 检测输出的 4 个角点形成的多边形框内。
    使用 OpenCV 的 pointPolygonTest 进行计算，如果是正数或 0 表示在形内或边界上。
    """
    return cv2.pointPolygonTest(polygon, (float(pt[0]), float(pt[1])), False) >= 0

def save_point_cloud_txt(points, colors, frames, filename):
    """
    将生成的相对尺度 3D 点云以普通 TXT 格式保存落地。
    每一行格式为: X Y Z (空间坐标), R G B (从原图抓取的颜色), Frame_Name (所属图片帧名称)。
    """
    if not points: return
    with open(filename, 'w') as f:
        for p, c, fn in zip(points, colors, frames):
            # c 是 OpenCV 的 BGR，因此 c[2]是Red, c[1]是Green, c[0]是Blue
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[2])} {int(c[1])} {int(c[0])} {fn}\n")

def main():
    # 1. 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="../landmarkslam.yaml", help="SLAM系统配置参数")
    parser.add_argument("--images", default="../data/20260321_111801/", help="输入的原始图像文件夹")
    parser.add_argument("--yolo", required=True, help="YOLO输出的txt检测框文件路径")
    parser.add_argument("--output", default="/home/zah/ORB_SLAM3-master/landmarkslam/output2/sign_cloud_py.txt", help="最终点云TXT的保存路径")
    args = parser.parse_args()

    print(f"Loading camera intrinsics from {args.settings}...")
    K = load_camera_intrinsics(args.settings)
    K_inv = np.linalg.inv(K)

    print(f"Loading YOLO detected bounding boxes from {args.yolo}...")
    # 2. 读取所有图片帧中的目标检测边界点
    yolo_data = load_yolo_detections(args.yolo)
    if not yolo_data: 
        print("No YOLO detections loaded. Exiting.")
        return
    print(f"Loaded YOLO boxes for {len(yolo_data)} images.")

    # 3. 按名称排序加载所有图像，方便按顺序处理
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        image_paths.extend(glob.glob(os.path.join(args.images, ext)))
    image_paths.sort()
    print(f"Found {len(image_paths)} images in {args.images}.")

    # 4. 定义标志牌在局部坐标系中的二维模型尺寸
    # 根据我们设定的长宽比 270:160，选用相对尺度 2.7 和 1.6
    sign_w, sign_h = 2.7, 1.6
    # 按顺时针或逆时针定义四个角点在局部 2D 坐标系（Z=0）的理论位置 
    # [左上, 右上, 右下, 左下] （假设中心在 0,0）
    obj_pts = np.array([[-sign_w/2, -sign_h/2], [sign_w/2, -sign_h/2], [sign_w/2, sign_h/2], [-sign_w/2, sign_h/2]], dtype=np.float32)
    # 同时定义带有Z轴(Z=0)的3D理论位置，用于 PnP 求解相机位姿
    obj_pts_3d = np.array([[-sign_w/2, -sign_h/2, 0.0], [sign_w/2, -sign_h/2, 0.0], [sign_w/2, sign_h/2, 0.0], [-sign_w/2, sign_h/2, 0.0]], dtype=np.float32)
    # 同时定义带有Z轴(Z=0)的3D理论位置，用于 PnP 求解相机位姿
    obj_pts_3d = np.array([[-sign_w/2, -sign_h/2, 0.0], [sign_w/2, -sign_h/2, 0.0], [sign_w/2, sign_h/2, 0.0], [-sign_w/2, sign_h/2, 0.0]], dtype=np.float32)

    # 5. 初始化 ORB 特征提取器（稍微放宽阈值，以便提取更丰富的文字特征）
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=20, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=20, fastThreshold=5)

    all_sign_points = []
    all_sign_colors = []
    all_sign_frames = []
    
    all_camera_centers = []
    all_camera_frames = []

    processed_frames = 0
    prev_rvec = None
    prev_tvec = None

    # 6. 逐帧处理提取，将 2D 图像像素转换为 3D 相对坐标系点
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        # 如果当前帧没有YOLO检测结果，则跳过
        if base_name not in yolo_data: continue
        
        corners = sort_corners(yolo_data[base_name]) # 强制按方位角排序，稳定为 [左上, 右上, 右下, 左下]
        im = cv2.imread(img_path)
        if im is None: continue

        # 6.1 计算单应性矩阵 (Homography Matrix)
        # H 将我们定义的二维局部平面(obj_pts) 映射到-> 图片上的像素区域(corners)
        H, status = cv2.findHomography(obj_pts, corners)
        if H is None: continue

        # 6.2 计算单应性矩阵的逆 (Inverse Homography)
        # 我们需要反向操作：把图像上的像素映射回->二维局部平面
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            continue

        # ================== 新增：计算并记录4个检测角点 ==================
        for corner in corners:
            pt_img = np.array([corner[0], corner[1], 1.0], dtype=np.float64)
            pt_plane = H_inv @ pt_img
            if abs(pt_plane[2]) > 1e-6: # 避免除零
                pt3d = [float(pt_plane[0]/pt_plane[2]), float(pt_plane[1]/pt_plane[2]), 0.0]
                all_sign_points.append(pt3d)
                # 使用醒目的纯红色 (OpenCV BGR 格式：0, 0, 255) 以便在点云中显著区分角点
                all_sign_colors.append(np.array([0, 0, 255]))
                all_sign_frames.append(base_name + "_corner")


        # ================== 新增：利用上一帧的结果作为先验进行 PnP 求解，防止平面歧义翻转 ==================
        if prev_rvec is None:
            # 第一帧，使用 IPPE 获取一个稳定的初始解
            success, rvec, tvec = cv2.solvePnP(obj_pts_3d, corners, K, None, flags=cv2.SOLVEPNP_IPPE)
        else:
            # 后续帧，使用上一帧作为初始猜想，强制解算器在当前轨迹附近优化，防止跳跃到镜像解
            success, rvec, tvec = cv2.solvePnP(obj_pts_3d, corners, K, None, prev_rvec.copy(), prev_tvec.copy(), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        
        if success:
            prev_rvec = rvec
            prev_tvec = tvec
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

        # ================== 核心思路修改：先抠图再提点 ==================
        # 通过包围盒求出ROI区域
        x, y, w, h = cv2.boundingRect(corners)
        
        # 确保裁剪边界不会超出图像实际大小
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(im.shape[1], x + w)
        y2 = min(im.shape[0], y + h)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        roi = im[y1:y2, x1:x2]
        
        # 6.3 仅在标志牌的 ROI 小图区域提取 ORB 特征点 (不再污染全图名额)
        keypoints, des = orb.detectAndCompute(roi, None)
        if keypoints is None: continue
        
        pts_in_frame = 0
        for kp in keypoints:
            # 还原特征点在原始全图中的真实像素坐标
            # (kp.pt 是在 roi 中的坐标，需要加上外接矩形的偏移量)
            kp_pt_global = (kp.pt[0] + x1, kp.pt[1] + y1)
            
            # 使用还原后的全图坐标做多边形判断
            if is_point_inside_polygon(kp_pt_global, corners):
                # 将特征点的 2D 像素坐标转为齐次坐标格式 (u, v, 1)
                pt_img = np.array([kp_pt_global[0], kp_pt_global[1], 1.0], dtype=np.float64)
                
                # 利用 H_inv 进行相乘变换，得到它在相对尺度平面内的坐标（带缩放因子）
                pt_plane = H_inv @ pt_img
                if abs(pt_plane[2]) < 1e-6: continue # 避免除零

                # 对齐次坐标进行透视除法归一化 (X/W, Y/W)，且因为是平面，所以 Z 轴固定写为 0.0
                pt3d = [float(pt_plane[0]/pt_plane[2]), float(pt_plane[1]/pt_plane[2]), 0.0]
                all_sign_points.append(pt3d)
                
                # 取出它在原图里面真实的颜色（由于 OpenCV 读取，此时为 BGR 格式）
                px, py = int(kp_pt_global[0]), int(kp_pt_global[1])
                color = im[py, px] 
                all_sign_colors.append(color)
                all_sign_frames.append(base_name)
                pts_in_frame += 1
        
        if pts_in_frame > 0:
            print(f"Processed {base_name}: Configured {pts_in_frame} relative 3D points via ROI crop.")
            processed_frames += 1

    # 7. 处理完毕，统计与保存结果
    print(f"\nProcessing complete! Processed {processed_frames} frames.")
    print(f"Total points generated: {len(all_sign_points)}")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 我们将保存分离为三个文件：包含全部(合并)、仅标志牌、仅轨迹
    output_combined = args.output
    output_sign_only = args.output.replace(".txt", "_sign.txt")
    output_traj_only = args.output.replace(".txt", "_trajectory.txt")

    # 1. 组合文件 (包含标志牌点、红色角点、绿色轨迹点) - 适配可视化交互工具
    save_point_cloud_txt(all_sign_points, all_sign_colors, all_sign_frames, output_combined)
    print(f"Saved combined cloud (Sign + Trajectory) to {output_combined}")

    # 2. 仅道路标志牌点云文件 (排除纯绿色的相机点)
    sign_only_pts = []
    sign_only_cols = []
    sign_only_frames = []
    for p, c, fn in zip(all_sign_points, all_sign_colors, all_sign_frames):
        # 如果不是我们硬编码加进去的纯绿色轨迹点 (0, 255, 0)，就认为是标志牌/角点
        if not (c[0] == 0 and c[1] == 255 and c[2] == 0):
            sign_only_pts.append(p)
            sign_only_cols.append(c)
            sign_only_frames.append(fn)
    save_point_cloud_txt(sign_only_pts, sign_only_cols, sign_only_frames, output_sign_only)
    print(f"Saved sign-only point cloud to {output_sign_only}")

    # 3. 仅轨迹文件 (专门标准化的轨迹输出格式 Frame X Y Z)
    if len(all_camera_centers) > 0:
        traj_colors = [np.array([0, 255, 0])] * len(all_camera_centers)
        save_point_cloud_txt(all_camera_centers, traj_colors, all_camera_frames, output_traj_only)
        print(f"Saved camera trajectory-only trace to {output_traj_only}")

if __name__ == "__main__":
    main()
