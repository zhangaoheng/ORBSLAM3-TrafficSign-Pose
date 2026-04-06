import numpy as np               # 矩阵与数学运算
import pandas as pd              # 数据表处理
from scipy.optimize import least_squares  # LM 图优化求解器
from scipy.spatial.transform import Rotation as R # 3D 姿态变换
import os
import cv2
import math
import logging
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =========================================================
# 1. 核心配置与参数
# =========================================================
TRAJ_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/output/FrameTrajectory_TUM_20260331_154838.txt"
MAP_FILE  = "/home/zah/ORB_SLAM3-master/landmarkslam/output/Filename_Mapping_20260331_154838.csv"
BBOX_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/yolo_simulated_bboxes.txt"
IMAGE_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/" 

EXPERIMENT_OUT_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/output/looming_fusion_results/"
LOG_FILE = os.path.join(EXPERIMENT_OUT_DIR, "fusion_experiment.log")
CSV_FILE = os.path.join(EXPERIMENT_OUT_DIR, "looming_results.csv")
VIS_STEPS_DIR = os.path.join(EXPERIMENT_OUT_DIR, "vis_steps/") 

os.makedirs(EXPERIMENT_OUT_DIR, exist_ok=True)
os.makedirs(VIS_STEPS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- 相机内参与标志牌物理模型 ---
FX, FY = 501.00685446, 496.63593447
CX, CY = 316.00266456, 233.80218648
WIDTH, HEIGHT = 1.26, 0.6
HALF_W, HALF_H = WIDTH / 2.0, HEIGHT / 2.0
P_S = np.array([[-HALF_W, -HALF_H, 0.0], [ HALF_W, -HALF_H, 0.0], 
                [ HALF_W,  HALF_H, 0.0], [-HALF_W,  HALF_H, 0.0]])

# --- 图优化惩罚权重 ---
W_REPROJ  = 1.0    # 物理角点重投影权重
W_BBOX    = 0.02   # YOLO框宏观约束权重
W_STRUCT  = 0.5    # 背景墙平面约束权重
W_ORTHO   = 10.0   # 汉字正交线条死锁权重
W_LOOMING = 5.0    # 【新增】：Looming先验深度锁死权重！

# =========================================================
# 2. 特征提取引擎 (几何边缘 + 文字纹理)
# =========================================================
def get_T_WC(pose_tum):
    T_WC = np.eye(4)
    T_WC[:3, :3] = R.from_quat(pose_tum[3:7]).as_matrix()
    T_WC[:3, 3] = pose_tum[0:3]
    return T_WC

def extract_optical_projection_hw(img_path, u1, v1, u2, v2, fname):
    """【模块A】：Looming 亚像素边缘提取 (返回物理长宽和精调后的BoundingBox)"""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: return None, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    
    pad = 2
    crop_u1, crop_v1 = max(0, int(u1) - pad), max(0, int(v1) - pad)
    crop_u2, crop_v2 = min(img_w, int(u2) + pad), min(img_h, int(v2) + pad)
    roi = gray[crop_v1:crop_v2, crop_u1:crop_u2]
    roi_h, roi_w = roi.shape 
    
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(roi)
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(roi_enhanced)
    
    fallback_w, fallback_h = float(u2 - u1), float(v2 - v1)
    fallback_bbox = np.array([u1, v1, u2, v2])
    if lines is None: return fallback_w, fallback_h, fallback_bbox
        
    all_x, all_y = [], []
    min_len = min(roi_w, roi_h) * 0.4 
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.hypot(x2-x1, y2-y1)
        if length > min_len:
            all_x.extend([x1 + crop_u1, x2 + crop_u1])
            all_y.extend([y1 + crop_v1, y2 + crop_v1])

    final_w, final_h = fallback_w, fallback_h
    w_triggered, h_triggered = True, True
    fallback_ratio = fallback_w / fallback_h

    if len(all_x) >= 2:
        ext_w = max(all_x) - min(all_x)
        if 0.85 < (ext_w / fallback_h) / fallback_ratio < 1.15:
            final_w, w_triggered = float(ext_w), False
    if len(all_y) >= 2:
        ext_h = max(all_y) - min(all_y)
        if 0.85 < (fallback_w / ext_h) / fallback_ratio < 1.15:
            final_h, h_triggered = float(ext_h), False

    # 构建基于亚像素重构的精确边界框，供后续 LM 优化器当作角点约束使用！
    xmin, xmax = (min(all_x), max(all_x)) if not w_triggered and all_x else (u1, u2)
    ymin, ymax = (min(all_y), max(all_y)) if not h_triggered and all_y else (v1, v2)
    refined_bbox = np.array([xmin, ymin, xmax, ymax])
    
    return final_w, final_h, refined_bbox

def extract_subpixel_orthogonal_lines(img_path, roi_bbox, gamma=2.5, clahe_clip=4.0, scale_factor=2.5, angle_tol=15):
    """【模块B】：提取汉字内部横平竖直的结构线段，用于强压 Pitch/Yaw 姿态漂移"""
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None: return [], []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(gray, table)
    img_clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(img_gamma) 
    
    h, w = img_clahe.shape 
    img_up = cv2.resize(img_clahe, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
    img_up = cv2.GaussianBlur(img_up, (3, 3), 0)

    lsd = cv2.createLineSegmentDetector(0)
    lines_up, _, _, _ = lsd.detect(img_up)
    
    horiz_lines, vert_lines = [], []
    if lines_up is not None:
        for line in lines_up:
            x1, y1, x2, y2 = line[0] / scale_factor 
            xmin, ymin, xmax, ymax = roi_bbox
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if not (xmin <= cx <= xmax and ymin <= cy <= ymax): continue
            
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 3.0: continue 
                
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180.0
            if angle <= angle_tol or angle >= (180 - angle_tol):
                horiz_lines.append((x1, y1, x2, y2, length)) 
            elif abs(angle - 90) <= angle_tol:
                vert_lines.append((x1, y1, x2, y2, length))  
                
    return horiz_lines, vert_lines

# =========================================================
# 3. LM 图优化数学推导引擎
# =========================================================
def project_to_pixel(x_state, P_model, T_CW):
    t_WS, r_WS = x_state[0:3], R.from_rotvec(x_state[3:6]).as_matrix()
    P_W = np.dot(P_model, r_WS.T) + t_WS
    P_C = np.dot(P_W, T_CW[:3, :3].T) + T_CW[:3, 3]
    Z_C = np.maximum(P_C[:, 2], 1e-6)
    u = FX * (P_C[:, 0] / Z_C) + CX
    v = FY * (P_C[:, 1] / Z_C) + CY
    return np.column_stack((u, v))

def get_inverse_homography(x_state, T_CW):
    t_WS, r_WS = x_state[0:3], R.from_rotvec(x_state[3:6]).as_matrix()
    T_WS = np.eye(4)
    T_WS[:3, :3], T_WS[:3, 3] = r_WS, t_WS
    T_CS = np.dot(T_CW, T_WS) 
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]]) 
    M = np.column_stack((T_CS[:3, 0], T_CS[:3, 1], T_CS[:3, 3]))
    try: return np.linalg.inv(np.dot(K, M))
    except: return np.eye(3)

def pixel_to_sign_plane(u, v, H_inv):
    p_img = np.array([u, v, 1.0])
    p_sign = np.dot(H_inv, p_img)
    return p_sign[0] / p_sign[2], p_sign[1] / p_sign[2]

def joint_residuals(x, clear_frames, blur_frames, plane_prior, looming_z, last_T_WC):
    """
    终极惩罚函数：五大约束联合出击！
    1. 角点重投影误差
    2. 汉字正交形态误差
    3. 语义框包裹误差
    4. 物理墙面共面误差
    5. 【新增】Looming 深度收敛锚点误差
    """
    res = []
    
    # A. 锚点重投影 (用 Looming 精调过的角点)
    for f in clear_frames:
        pts_hat = project_to_pixel(x, P_S, f['T_CW'])
        res.append((f['corners'] - pts_hat).flatten() * W_REPROJ)
        
        # B. 汉字正交死锁
        if 'h_lines' in f and 'v_lines' in f:
            H_inv = get_inverse_homography(x, f['T_CW'])
            for u1, v1, u2, v2, weight in f['h_lines']:
                _, Y1 = pixel_to_sign_plane(u1, v1, H_inv)
                _, Y2 = pixel_to_sign_plane(u2, v2, H_inv)
                res.append(np.array([(Y1 - Y2) * weight * W_ORTHO]))
            for u1, v1, u2, v2, weight in f['v_lines']:
                X1, _ = pixel_to_sign_plane(u1, v1, H_inv)
                X2, _ = pixel_to_sign_plane(u2, v2, H_inv)
                res.append(np.array([(X1 - X2) * weight * W_ORTHO]))
                
    # C. 语义框粗托底
    for f in blur_frames:
        pts_hat = project_to_pixel(x, P_S, f['T_CW'])
        b_hat = np.array([np.min(pts_hat[:,0]), np.min(pts_hat[:,1]), np.max(pts_hat[:,0]), np.max(pts_hat[:,1])])
        res.append((f['bbox'] - b_hat) * W_BBOX)
        
    # D. 背景墙托底
    if plane_prior:
        dist = np.dot(plane_prior['n'], x[0:3]) + plane_prior['d'] 
        res.append(np.array([dist]) * W_STRUCT) 
        
    # E. 【灵魂核心】Looming 深度抛锚锁定
    # 计算当前猜想的标志牌中心点，在最后一帧相机坐标系下的深度 Z
    t_WS = x[0:3] # 标志牌在世界系的坐标
    cam_pos_world = last_T_WC[:3, 3] # 相机在世界系的坐标
    cam_z_axis = last_T_WC[:3, 2] # 相机在世界系的Z轴朝向向量
    # 点乘求投影距离 = 标志牌在相机光轴上的绝对深度
    current_z = np.dot((t_WS - cam_pos_world), cam_z_axis) 
    # 惩罚项：优化器猜出来的深度，必须死死咬住 Looming 算出的最终收敛锚点深度！
    res.append(np.array([current_z - looming_z]) * W_LOOMING)
        
    return np.concatenate(res)

# =========================================================
# 4. 主干管线集成 (Main Pipeline)
# =========================================================
def main():
    logger.info("="*60)
    logger.info(" 🚀 [融合系统启动] Looming Depth + LM 6-DoF 图优化中心")
    logger.info("="*60)

    traj_data = np.loadtxt(TRAJ_FILE)
    traj_dict = {row[0]: row[1:] for row in traj_data}
    all_ts = np.array(list(traj_dict.keys()))
    
    mapping_df = pd.read_csv(MAP_FILE)
    bbox_df = pd.read_csv(BBOX_FILE, header=None, names=['fname', 'u1', 'v1', 'u2', 'v2'], sep=',', comment='#')
    
    frames_data = [] 
    logger.info(">>> [第一阶段] 运行 Looming 亚像素特征抓取...")
    for _, row in bbox_df.iterrows():
        fname = str(row['fname']).strip()
        match_rows = mapping_df[mapping_df['filename'] == fname]
        if match_rows.empty: continue
        ts = match_rows.iloc[0]['timestamp_s']
        closest_ts = all_ts[np.argmin(np.abs(all_ts - ts))]
        T_WC = get_T_WC(traj_dict[closest_ts])
        
        u1, v1, u2, v2 = float(row['u1']), float(row['v1']), float(row['u2']), float(row['v2'])
        img_path = os.path.join(IMAGE_DIR, fname)
        if not os.path.exists(img_path): continue
            
        w, h, refined_bbox = extract_optical_projection_hw(img_path, u1, v1, u2, v2, fname)
        if w is None: continue
        
        frames_data.append({
            'fname': fname, 'img_path': img_path, 'ts': ts,
            'width': w, 'height': h, 'refined_bbox': refined_bbox, 'yolo_bbox': np.array([u1,v1,u2,v2]),
            'pos_w': T_WC[:3, 3], 'T_WC': T_WC, 'T_CW': np.linalg.inv(T_WC)
        })

    logger.info(">>> [第二阶段] 执行时序膨胀解算获取绝对深度 Z_final ...")
    f_start = frames_data[0]
    w1, h1 = f_start['width'], f_start['height']
    final_z_looming = 8.0 # 保底初始值

    for i in range(1, len(frames_data)):
        f_end = frames_data[i]
        w2, h2 = f_end['width'], f_end['height']
        delta_d = np.linalg.norm(f_end['pos_w'] - f_start['pos_w'])
        delta_w, delta_h = w2 - w1, h2 - h1
        
        Z_w = (w2 * delta_d) / delta_w if delta_w > 8.0 else None
        Z_h = (h2 * delta_d) / delta_h if delta_h > 8.0 else None
        
        if Z_w and Z_h:
            diff = abs(Z_w - Z_h) / max(Z_w, Z_h)
            final_z_looming = (Z_w + Z_h) / 2.0 if diff < 0.15 else (Z_h if delta_h > delta_w else Z_w)
        elif Z_w or Z_h:
            final_z_looming = Z_w if Z_w else Z_h
            
    logger.info(f"    📍 Looming 收敛完毕，提供完美的绝对深度先验锚点: {final_z_looming:.4f} units")

    logger.info(">>> [第三阶段] 构建大熔炉：提取汉字结构，准备 6-DoF 图优化...")
    blur_frames, clear_frames = [], []
    for i, f in enumerate(frames_data):
        if i < len(frames_data) - 2:
            blur_frames.append({'T_CW': f['T_CW'], 'bbox': f['yolo_bbox']})
        else:
            # 清晰帧：提取汉字线段
            h_lines, v_lines = extract_subpixel_orthogonal_lines(f['img_path'], roi_bbox=f['refined_bbox'])
            # 极核优化：用 Looming 算出来的亚像素精调框，充当物理角点！
            rb = f['refined_bbox']
            obs_corners = np.array([[rb[0], rb[1]], [rb[2], rb[1]], [rb[2], rb[3]], [rb[0], rb[3]]])
            clear_frames.append({
                'T_CW': f['T_CW'], 'corners': obs_corners, 'h_lines': h_lines, 'v_lines': v_lines
            })

    plane_prior = {'n': np.array([0, 0, 1]), 'd': -12.5} 
    
    # 核心：使用 Looming 给出的 Z_final 直接反推 3D 平移初值
    last_T_WC = frames_data[-1]['T_WC']
    init_t = last_T_WC[:3, 3] + last_T_WC[:3, 2] * final_z_looming 
    x0 = np.concatenate([init_t, [0, 0, 0]])

    logger.info(">>> [第四阶段] 🚀 引擎点火：SciPy Levenberg-Marquardt 多约束联合寻优...")
    res = least_squares(joint_residuals, x0, args=(clear_frames, blur_frames, plane_prior, final_z_looming, last_T_WC),
                        method='lm', xtol=1e-8, ftol=1e-8, verbose=1)

    x_opt = res.x 
    logger.info("="*60)
    logger.info(" 🎉 交通标志牌 6-DoF 位姿最终联合优化成功！")
    logger.info(f" 📍 世界系中心坐标 (X, Y, Z) :  {x_opt[0]:.4f}, {x_opt[1]:.4f}, {x_opt[2]:.4f}")
    logger.info(f" 📐 标志牌朝向 (R_WS) : \n{R.from_rotvec(x_opt[3:6]).as_matrix()}")
    logger.info("="*60)

    # 3D 态势可视化
    logger.info(">>> 正在生成全维 3D 态势图 (关闭弹窗即可退出)...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    cam_xs = [f['pos_w'][0] for f in frames_data]
    cam_ys = [f['pos_w'][1] for f in frames_data]
    cam_zs = [f['pos_w'][2] for f in frames_data]
    ax.plot(cam_xs, cam_ys, cam_zs, label='SLAM Camera Trajectory', color='blue', marker='.', linestyle='dashed', alpha=0.5)
    ax.scatter(cam_xs[0], cam_ys[0], cam_zs[0], color='green', s=100, marker='s', label='Start')
    ax.scatter(cam_xs[-1], cam_ys[-1], cam_zs[-1], color='red', s=100, marker='^', label='End (t_last)')

    sign_corners_world = np.dot(P_S, R.from_rotvec(x_opt[3:6]).as_matrix().T) + x_opt[0:3] 
    verts = [list(zip(sign_corners_world[:, 0], sign_corners_world[:, 1], sign_corners_world[:, 2]))]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.8, facecolor='gold', edgecolor='black', linewidths=2))
    
    # 画一条视线指示最终深度
    ax.plot([cam_xs[-1], x_opt[0]], [cam_ys[-1], x_opt[1]], [cam_zs[-1], x_opt[2]], 
            color='orange', linestyle=':', label=f'Optimized LOS (Z≈{final_z_looming:.2f})')

    max_range = np.array([np.max(cam_xs)-np.min(cam_xs), np.max(cam_ys)-np.min(cam_ys), np.max(cam_zs)-np.min(cam_zs), 5.0]).max() / 2.0
    mid_x, mid_y, mid_z = (np.max(cam_xs)+np.min(cam_xs))*0.5, (np.max(cam_ys)+np.min(cam_ys))*0.5, (np.max(cam_zs)+np.min(cam_zs))*0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('World X (m)'); ax.set_ylabel('World Y (m)'); ax.set_zlabel('World Z (m)')
    ax.set_title('Fusion Model: 6-DoF LM Optimization Anchor via Looming Depth', fontsize=14, fontweight='bold')
    ax.legend()
    ax.view_init(elev=20, azim=-45) 
    plt.show() 

if __name__ == "__main__":
    main()