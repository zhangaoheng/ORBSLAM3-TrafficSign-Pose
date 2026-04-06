import numpy as np               # 核心数学库：负责所有的矩阵乘法、向量加减、求逆矩阵等底层数学运算
import pandas as pd              # 数据处理库：负责读取由逗号分隔的 CSV 和 TXT 日志文件
from scipy.optimize import least_squares  # 核心算法库：SciPy 提供的 LM (Levenberg-Marquardt) 非线性最小二乘求解器
from scipy.spatial.transform import Rotation as R # 核心空间库：专门用于处理 3D 空间中极其容易算错的旋转（四元数、旋转向量、旋转矩阵之间的相互转换）
import os                        # 操作系统接口：用于拼接文件路径、创建文件夹
import cv2                       # OpenCV：工业级计算机视觉库，负责图像读取、色彩空间转换、对比度增强、底层线段提取
import math                      # 基础数学库：提供计算绝对值、平方根、三角函数等操作
import logging                   # 日志库：负责把终端打印的信息同时写入到 log 文件中，方便复盘
import csv                       # CSV 库：负责把最终的实验数据一行行写入本地表格
import matplotlib.pyplot as plt  # 绘图库：负责画出 2D 的收敛曲线和 3D 的空间态势图
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # 3D 绘图扩展：专门用于在 3D 坐标系中画出带颜色的多边形实体面片

# =========================================================
# 1. 核心配置与参数 (全局环境变量)
# =========================================================
# --- 输入数据路径 ---
TRAJ_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/output/FrameTrajectory_TUM_20260331_154838.txt" # SLAM 跑出来的绝对轨迹，TUM 格式 [时间戳 tx ty tz qx qy qz qw]
MAP_FILE  = "/home/zah/ORB_SLAM3-master/landmarkslam/output/Filename_Mapping_20260331_154838.csv"   # 将图片文件名与物理系统时间戳(秒)对应起来的映射表
BBOX_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/yolo_simulated_bboxes.txt" # YOLO 神经网络吐出来的粗糙 2D 识别框
IMAGE_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/"          # 存放真实采集到的环境照片的文件夹

# --- 输出数据路径 ---
EXPERIMENT_OUT_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/output/looming_fusion_results/"
LOG_FILE = os.path.join(EXPERIMENT_OUT_DIR, "fusion_experiment.log")
CSV_FILE = os.path.join(EXPERIMENT_OUT_DIR, "looming_results.csv")
VIS_STEPS_DIR = os.path.join(EXPERIMENT_OUT_DIR, "vis_steps/") 

# 自动创建不存在的输出文件夹
os.makedirs(EXPERIMENT_OUT_DIR, exist_ok=True)
os.makedirs(VIS_STEPS_DIR, exist_ok=True)

# 配置日志记录器，让终端输出和文本文件输出保持一致的格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- 相机内参 (Camera Intrinsics) ---
# 这是针孔相机模型的核心，它决定了真实 3D 世界的光线穿过镜头后，会落在 CMOS 传感器的第几个像素上。
FX, FY = 501.00685446, 496.63593447  # X和Y方向的焦距 (单位: 像素)。表示相机镜头的拉伸程度。
CX, CY = 316.00266456, 233.80218648  # 光心坐标 (单位: 像素)。通常在图像的正中心附近。

# --- 标志牌物理 3D 模型 (Local Sign Coordinate System) ---
WIDTH, HEIGHT = 1.26, 0.6            # 标志牌真实的物理宽度和高度 (单位: 米)
HALF_W, HALF_H = WIDTH / 2.0, HEIGHT / 2.0
# P_S 数组定义了标志牌的“局部坐标系”。
# 我们假设标志牌的绝对正中心是原点 (0,0,0)，整个牌子平铺在 XY 平面上，因此 Z 坐标全为 0。
# 顺序为：[左上角, 右上角, 右下角, 左下角]
P_S = np.array([[-HALF_W, -HALF_H, 0.0], [ HALF_W, -HALF_H, 0.0], 
                [ HALF_W,  HALF_H, 0.0], [-HALF_W,  HALF_H, 0.0]])

# --- LM 图优化目标函数的惩罚权重 (Hyperparameters for Optimization) ---
# 优化器的目标是让所有的误差都变成 0。不同的权重代表我们对该特征的“信任程度”。
W_REPROJ  = 1.0    # 权重A: 角点重投影约束。亚像素角点很准，所以权重定为基准 1.0，用于确定标志牌的基础平移 (X,Y)。
W_BBOX    = 0.02   # 权重C: YOLO语义框包裹约束。因为 YOLO 框是浮动的、粗糙的，权重给很低 (0.02)，仅用于防止优化器发散跑出画面外。
W_STRUCT  = 0.5    # 权重D: 背景平面共面约束。SLAM 提取的墙面先验，用于防止标志牌在 Z 轴上前后乱飘。
W_ORTHO   = 10.0   # 权重B: 汉字正交约束。极其重要！因为 100 米外标志牌微小的旋转根本看不出来，只能通过高权重 (10.0) 逼迫汉字横平竖直，从而解算出极其微弱的 Pitch 和 Yaw。
W_LOOMING = 5.0    # 权重E: 深度抛锚约束。将 Looming 算出的绝对深度作为强力“金锚”，锁定 Z 轴，彻底解决单目尺度模糊的死穴。

# =========================================================
# 2. 前端：特征提取与数据预处理模块 (从 2D 像素里榨取信息)
# =========================================================

def get_T_WC(pose_tum):
    """
    功能：将 TUM 格式的位姿转换为 4x4 齐次变换矩阵。
    输入：pose_tum [tx, ty, tz, qx, qy, qz, qw] (长度为7的一维数组)
    输出：T_WC (4x4 矩阵，代表 相机坐标系 -> 世界坐标系 的位姿变换)
    """
    T_WC = np.eye(4) # 初始化 4x4 对角线为 1 的单位矩阵
    # 利用 scipy.spatial.transform.Rotation 将四元数转为 3x3 旋转矩阵，塞入左上角
    T_WC[:3, :3] = R.from_quat(pose_tum[3:7]).as_matrix()
    # 将平移向量塞入右侧最后一列
    T_WC[:3, 3] = pose_tum[0:3]
    return T_WC

def extract_optical_projection_hw(img_path, u1, v1, u2, v2, fname):
    """
    【模块A】：Looming 亚像素边缘提取器
    作用：突破 YOLO 整数像素和语义抖动的限制，利用底层光影梯度，寻找最贴合物体物理边界的亚像素包围盒。
    数学原理：利用 LSD 提取局部高梯度线段，通过空间极值法(Max/Min)挤压出物理边界。
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: return None, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    
    # 策略 1：ROI 外扩 (Padding)。给 YOLO 框上下左右各留 2 个像素的余量，防止真实的边缘被 YOLO 截断。
    pad = 2
    crop_u1, crop_v1 = max(0, int(u1) - pad), max(0, int(v1) - pad)
    crop_u2, crop_v2 = min(img_w, int(u2) + pad), min(img_h, int(v2) + pad)
    roi = gray[crop_v1:crop_v2, crop_u1:crop_u2]
    roi_h, roi_w = roi.shape 
    
    # 策略 2：抗逆光增强。CLAHE 可以在不过度曝光的情况下，拉开暗部铁皮与文字的对比度。
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(roi)
    
    # 策略 3：LSD (Line Segment Detector) 亚像素线段提取
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(roi_enhanced)
    
    # 设定保底值 (Fallback)，以防极端模糊情况导致特征提取崩溃
    fallback_w, fallback_h = float(u2 - u1), float(v2 - v1)
    fallback_bbox = np.array([u1, v1, u2, v2])
    if lines is None: return fallback_w, fallback_h, fallback_bbox
        
    all_x, all_y = [], []
    # 策略 4：长度海选过滤。只保留长度大于 ROI 短边 40% 的线段（过滤掉噪点和短的汉字笔画，保留标志牌的超长外边缘）。
    min_len = min(roi_w, roi_h) * 0.4 
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.hypot(x2-x1, y2-y1)
        if length > min_len:
            # 注意：LSD 提取的坐标是相对 ROI 左上角的，必须加上 crop_u1/v1 映射回原图的全局坐标系
            all_x.extend([x1 + crop_u1, x2 + crop_u1])
            all_y.extend([y1 + crop_v1, y2 + crop_v1])

    final_w, final_h = fallback_w, fallback_h
    w_triggered, h_triggered = True, True
    fallback_ratio = fallback_w / fallback_h

    # 策略 5：极值挤压与常识熔断 (护城河机制)
    # 利用长线段的极端坐标(最大最小X)来撑起宽度。如果由于电线干扰导致算出的宽度离谱（长宽比偏差 > 15%），则触发护城河拦截，退回保底值。
    if len(all_x) >= 2:
        ext_w = max(all_x) - min(all_x)
        if 0.85 < (ext_w / fallback_h) / fallback_ratio < 1.15:
            final_w, w_triggered = float(ext_w), False
    if len(all_y) >= 2:
        ext_h = max(all_y) - min(all_y)
        if 0.85 < (fallback_w / ext_h) / fallback_ratio < 1.15:
            final_h, h_triggered = float(ext_h), False

    # 输出极其关键的亚像素 BoundingBox，这个框的四个角将取代 YOLO，作为 LM 优化的核心定位角点
    xmin, xmax = (min(all_x), max(all_x)) if not w_triggered and all_x else (u1, u2)
    ymin, ymax = (min(all_y), max(all_y)) if not h_triggered and all_y else (v1, v2)
    refined_bbox = np.array([xmin, ymin, xmax, ymax])
    
    return final_w, final_h, refined_bbox

def extract_subpixel_orthogonal_lines(img_path, roi_bbox, gamma=2.5, clahe_clip=4.0, scale_factor=2.5, angle_tol=15):
    """
    【模块B】：提取汉字内部横平竖直的结构线段 (正交特征)
    作用：捕捉标志牌内部的文字笔画纹理，用于后续的反向单应性映射，以此纠正 3D 姿态偏角。
    数学原理：利用 Gamma 曲线拉升暗部细节，放大图像增加梯度采样点，最后利用几何角度(atan2)筛选横竖直线。
    """
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None: return [], []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 策略 1：非线性 Gamma 提亮，专门针对夜晚/逆光的黑色字迹进行细节榨取
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(gray, table)
    img_clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(img_gamma) 
    
    # 策略 2：超分辨率插值 (Upscaling)。因为远处的汉字太小（可能只有 10x10 像素），LSD 算法算不出连续的梯度。
    # 我们将其放大 2.5 倍，制造平滑的过度带，再用高斯模糊(GaussianBlur)去一下马赛克锯齿。
    h, w = img_clahe.shape 
    img_up = cv2.resize(img_clahe, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
    img_up = cv2.GaussianBlur(img_up, (3, 3), 0)

    lsd = cv2.createLineSegmentDetector(0)
    lines_up, _, _, _ = lsd.detect(img_up)
    
    horiz_lines, vert_lines = [], []
    if lines_up is not None:
        for line in lines_up:
            # 策略 3：坐标降维映射。因为是在放大 2.5 倍的图上找的线，坐标必须除以 scale_factor 还原回真实像素。
            x1, y1, x2, y2 = line[0] / scale_factor 
            
            # 过滤 A：只保留包裹在 BoundingBox 内部的线段，踢掉背景树枝
            xmin, ymin, xmax, ymax = roi_bbox
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if not (xmin <= cx <= xmax and ymin <= cy <= ymax): continue
            
            # 过滤 B：太短的线段大概率是斑点噪声
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 3.0: continue 
                
            # 策略 4：角度过滤。math.atan2 算出线段的倾斜斜率（弧度），转化为角度。
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180.0
            # 0° 或 180° 附近的是横线 (横折钩里的横)
            if angle <= angle_tol or angle >= (180 - angle_tol):
                horiz_lines.append((x1, y1, x2, y2, length)) 
            # 90° 附近的是竖线 (竖心旁里的竖)
            elif abs(angle - 90) <= angle_tol:
                vert_lines.append((x1, y1, x2, y2, length))  
                
    return horiz_lines, vert_lines

# =========================================================
# 3. 中端：投影几何与数学推导引擎 (2D 像素与 3D 物理世界的桥梁)
# =========================================================

def project_to_pixel(x_state, P_model, T_CW):
    """
    功能：正向透视投影。给定一个猜想的 3D 位姿，推算出标志牌 4 个角应该出现在照片的哪几个像素上。
    输入：x_state (优化器猜想的6自由度姿态), P_model (牌子局部3D坐标), T_CW (相机在世界系下的位姿)
    数学公式：P_c = R_wc^T * (R_ws * P_s + t_ws) + t_wc_inv ;  u = fx * (X_c / Z_c) + cx
    """
    # 获取优化器猜想的：标志牌到世界系的平移 t_WS 和 旋转矩阵 r_WS
    t_WS, r_WS = x_state[0:3], R.from_rotvec(x_state[3:6]).as_matrix()
    
    # [变换 1]: 标志牌局部系 -> 世界系
    P_W = np.dot(P_model, r_WS.T) + t_WS
    
    # [变换 2]: 世界系 -> 当前相机坐标系 (利用传入的 T_CW)
    P_C = np.dot(P_W, T_CW[:3, :3].T) + T_CW[:3, 3]
    
    # [变换 3]: 针孔相机模型投影。X,Y 除以 Z (近大远小原理)，再乘以焦距。
    Z_C = np.maximum(P_C[:, 2], 1e-6) # 保护机制：防止除以 0 报错
    u = FX * (P_C[:, 0] / Z_C) + CX
    v = FY * (P_C[:, 1] / Z_C) + CY
    
    # 返回组合好的 Nx2 的像素坐标数组
    return np.column_stack((u, v))

def get_inverse_homography(x_state, T_CW):
    """
    【黑科技模块】：逆向单应矩阵 (Inverse Homography Matrix)
    作用：把照片上 2D 的像素点，像放幻灯片一样，反向“透视投射”回到 3D 的标志牌铁皮物理平面上。
    原理：由于标志牌是一个绝对平整的平面（Z=0），所以完整的 3D 变换可以降维成一个 3x3 的 2D 平面映射矩阵 H。
    """
    t_WS, r_WS = x_state[0:3], R.from_rotvec(x_state[3:6]).as_matrix()
    
    # 构建 标志牌 -> 世界 的 4x4 变换矩阵
    T_WS = np.eye(4)
    T_WS[:3, :3], T_WS[:3, 3] = r_WS, t_WS
    
    # 构建 标志牌 -> 当前相机 的复合变换矩阵 T_CS (先转到世界，再转到相机)
    T_CS = np.dot(T_CW, T_WS) 
    
    # 相机内参矩阵 K (3x3)
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]]) 
    
    # 【降维核心】：因为标志牌上所有点在自己的局部坐标系下 Z=0，所以 T_CS 矩阵负责 Z 轴的那一列（第3列，索引2）变成了废操作。
    # 我们直接把第 1、2、4 列抽出来，拼成一个 3x3 矩阵 M。
    M = np.column_stack((T_CS[:3, 0], T_CS[:3, 1], T_CS[:3, 3]))
    
    # H_CS 就是正向的单应矩阵 (3D平面 变 2D图像)
    try: 
        # 我们要的是 2D 变 3D，所以要求 H_CS 的逆矩阵！
        return np.linalg.inv(np.dot(K, M))
    except np.linalg.LinAlgError: 
        # 万一矩阵奇异（比如相机和牌子完全平行），返回单位矩阵保命
        return np.eye(3)

def pixel_to_sign_plane(u, v, H_inv):
    """
    功能：利用上面算好的逆向单应矩阵 H_inv，把像素 (u,v) 还原为真实标志牌上的物理米数 (X, Y)。
    数学公式：P_physical = H_inv * P_pixel_homogenous
    """
    p_img = np.array([u, v, 1.0]) # 提升维度至齐次坐标
    p_sign = np.dot(H_inv, p_img) # 矩阵乘法，透视变换
    # 齐次归一化，得到真实的物理平面坐标 (单位：米)
    return p_sign[0] / p_sign[2], p_sign[1] / p_sign[2]


# =========================================================
# 4. 后端：LM 优化器核心惩罚函数 (The Judge / Objective Function)
# =========================================================

def joint_residuals(x, clear_frames, blur_frames, plane_prior, looming_z, last_T_WC):
    """
    【终极裁判函数】：这实际上就是我们学术论文里列出的代价函数 (Cost Function) E(x)。
    工作机制：优化器 LM 每改变一次 `x` (它猜的姿态)，就会调用一次这个函数。
    这个函数会根据这个猜测的 `x`，计算出五项物理指标，发现跟真实情况不符，就把误差值塞进 `res` 数组返回。
    LM 的终极目标就是：寻找一个完美的 `x`，使得 `sum(res^2)` 最小。
    """
    res = [] # 这是一个超级长的一维列表，装满了成百上千个微小的误差数值
    
    # ---------------- 约束 A & B：基于极近距离清晰帧的微观死锁 ----------------
    for f in clear_frames:
        # A. 物理角点重投影约束 (Reprojection Error)
        pts_hat = project_to_pixel(x, P_S, f['T_CW'])
        # 误差 = (照片上的真实亚像素角点 - 优化器投影出的假角点)。差值越大，罚得越狠。
        res.append((f['corners'] - pts_hat).flatten() * W_REPROJ)
        
        # B. 汉字正交形态约束 (Orthogonal Texture Error)
        if 'h_lines' in f and 'v_lines' in f:
            H_inv = get_inverse_homography(x, f['T_CW'])
            
            # 横线约束：把照片上的 2D 横线贴回 3D 铁皮上。
            for u1, v1, u2, v2, weight in f['h_lines']:
                _, Y1 = pixel_to_sign_plane(u1, v1, H_inv)
                _, Y2 = pixel_to_sign_plane(u2, v2, H_inv)
                # 物理规律：真正的横线在 3D 铁皮上，左右两端的高度必须一样 (Y1 必须等于 Y2)。
                # 如果 Y1 ≠ Y2，说明优化器猜想的 Pitch 或 Roll 倾斜了。施加 W_ORTHO 高强度惩罚！
                res.append(np.array([(Y1 - Y2) * weight * W_ORTHO]))
                
            # 竖线约束：同理。
            for u1, v1, u2, v2, weight in f['v_lines']:
                X1, _ = pixel_to_sign_plane(u1, v1, H_inv)
                X2, _ = pixel_to_sign_plane(u2, v2, H_inv)
                # 物理规律：真正的竖线在 3D 铁皮上，上下两端的水平位置必须一样 (X1 必须等于 X2)。
                # 如果 X1 ≠ X2，说明优化器猜想的 Yaw 偏航角歪了！
                res.append(np.array([(X1 - X2) * weight * W_ORTHO]))
                
    # ---------------- 约束 C：基于远距离模糊帧的视锥托底 ----------------
    for f in blur_frames:
        pts_hat = project_to_pixel(x, P_S, f['T_CW'])
        # 计算投影出来的四个点所形成的包围盒 (MinMax Box)
        b_hat = np.array([np.min(pts_hat[:,0]), np.min(pts_hat[:,1]), np.max(pts_hat[:,0]), np.max(pts_hat[:,1])])
        # 只要你投影的框别跑出 YOLO 给定的框太远就行，权重极低 (W_BBOX=0.02)
        res.append((f['bbox'] - b_hat) * W_BBOX)
        
    # ---------------- 约束 D：物理大楼平面共面约束 ----------------
    if plane_prior:
        # 数学：点到平面的空间几何距离 = 法向量 n · 坐标点 x + 常数 d
        dist = np.dot(plane_prior['n'], x[0:3]) + plane_prior['d'] 
        # 牌子必须挂在墙上，距离必须尽量为 0
        res.append(np.array([dist]) * W_STRUCT) 
        
    # ---------------- 约束 E：【核心灵魂】Looming 绝对深度锚点约束 ----------------
    # 作用：彻底锁死单目 SLAM 最致命的尺度漂移问题 (Scale Drift)
    
    t_WS = x[0:3] # 获取优化器当前猜想的标志牌 3D 绝对中心点
    cam_pos_world = last_T_WC[:3, 3] # 最后一帧相机的物理坐标
    cam_z_axis = last_T_WC[:3, 2]    # 最后一帧相机光轴 (Z轴) 在世界坐标系下的单位朝向向量
    
    # 向量投影：把 (标志牌位置 - 相机位置) 的向量，投影到相机光轴上，算出严格意义上的物理摄像距离 (深度)
    current_z = np.dot((t_WS - cam_pos_world), cam_z_axis) 
    
    # 霸道法则：无论优化器怎么转动牌子，它算出来的牌子深度 current_z，
    # 必须死死咬住第一阶段利用时序膨胀公式算出来的绝对深度真值 looming_z！
    res.append(np.array([current_z - looming_z]) * W_LOOMING)
        
    # 返回拼接成一长条的一维误差数组给 LM
    return np.concatenate(res)


# =========================================================
# 5. 管线集成与执行控制流 (Main Flow)
# =========================================================
def main():
    logger.info("="*60)
    logger.info(" 🚀 [融合系统启动] Looming Depth + LM 6-DoF 图优化中心")
    logger.info("="*60)

    # 【Step 1】数据对齐与装载 (Data Loading & Association)
    # SLAM 和相机是两个独立的系统，它们的时间戳不一定完全重合，需要找最相近的进行匹配对齐。
    traj_data = np.loadtxt(TRAJ_FILE)
    traj_dict = {row[0]: row[1:] for row in traj_data}
    all_ts = np.array(list(traj_dict.keys()))
    
    mapping_df = pd.read_csv(MAP_FILE)
    bbox_df = pd.read_csv(BBOX_FILE, header=None, names=['fname', 'u1', 'v1', 'u2', 'v2'], sep=',', comment='#')
    
    frames_data = [] 
    logger.info(">>> [第一阶段] 运行 Looming 亚像素特征抓取...")
    
    # 遍历 YOLO 抓到的所有图片
    for _, row in bbox_df.iterrows():
        fname = str(row['fname']).strip()
        match_rows = mapping_df[mapping_df['filename'] == fname]
        if match_rows.empty: continue
        
        # 时空对齐：用相机的曝光时间，去查 SLAM 轨迹，找到那一瞬间相机在空间中的绝对位姿 T_WC
        ts = match_rows.iloc[0]['timestamp_s']
        closest_ts = all_ts[np.argmin(np.abs(all_ts - ts))]
        T_WC = get_T_WC(traj_dict[closest_ts])
        
        u1, v1, u2, v2 = float(row['u1']), float(row['v1']), float(row['u2']), float(row['v2'])
        img_path = os.path.join(IMAGE_DIR, fname)
        if not os.path.exists(img_path): continue
            
        # 呼叫前端模块，榨取亚像素精度的宽高，以及修正过的框
        w, h, refined_bbox = extract_optical_projection_hw(img_path, u1, v1, u2, v2, fname)
        if w is None: continue # 提取失败就跳过这帧
        
        # 将这极其珍贵的一帧数据，打包成大字典放入列表
        frames_data.append({
            'fname': fname, 'img_path': img_path, 'ts': ts,
            'width': w, 'height': h, 'refined_bbox': refined_bbox, 'yolo_bbox': np.array([u1,v1,u2,v2]),
            'pos_w': T_WC[:3, 3], 'T_WC': T_WC, 'T_CW': np.linalg.inv(T_WC)
        })

    # 【Step 2】Looming 时序膨胀测距 (Scale Recovery)
    # 利用近大远小原理：深度 Z = 物理位移 d / (像素放大倍数 - 1)
    logger.info(">>> [第二阶段] 执行时序膨胀解算获取绝对深度 Z_final ...")
    f_start = frames_data[0] # 固定第一帧为远端参照物
    w1, h1 = f_start['width'], f_start['height']
    final_z_looming = 8.0 # 保底初始值，防止后面完全没算出来报错

    # 循环算每一帧，利用交叉验证过滤噪声，逼近出最完美的最终深度锚点
    for i in range(1, len(frames_data)):
        f_end = frames_data[i]
        w2, h2 = f_end['width'], f_end['height']
        delta_d = np.linalg.norm(f_end['pos_w'] - f_start['pos_w']) # 真实的 3D 欧式距离位移
        delta_w, delta_h = w2 - w1, h2 - h1
        
        # Looming 核心数学公式
        Z_w = (w2 * delta_d) / delta_w if delta_w > 8.0 else None
        Z_h = (h2 * delta_d) / delta_h if delta_h > 8.0 else None
        
        # 交叉验证仲裁：如果宽高算的深度差不多，就取平均消除白噪声；如果差太多，就选膨胀量大的（信噪比高）。
        if Z_w and Z_h:
            diff = abs(Z_w - Z_h) / max(Z_w, Z_h)
            final_z_looming = (Z_w + Z_h) / 2.0 if diff < 0.15 else (Z_h if delta_h > delta_w else Z_w)
        elif Z_w or Z_h:
            final_z_looming = Z_w if Z_w else Z_h
            
    logger.info(f"    📍 Looming 收敛完毕，提供完美的绝对深度先验锚点: {final_z_looming:.4f} units")

    # 【Step 3】准备后端优化的燃料 (Feature Classification)
    logger.info(">>> [第三阶段] 构建大熔炉：提取汉字结构，准备 6-DoF 图优化...")
    blur_frames, clear_frames = [], []
    for i, f in enumerate(frames_data):
        # 离得远的图片，看不清字，只剥离出粗糙框给优化器宏观托底
        if i < len(frames_data) - 2:
            blur_frames.append({'T_CW': f['T_CW'], 'bbox': f['yolo_bbox']}) 
        else:
            # 离得极近的最后两张图片，可以清晰看到字，启动汉字特征提取器
            h_lines, v_lines = extract_subpixel_orthogonal_lines(f['img_path'], roi_bbox=f['refined_bbox'])
            rb = f['refined_bbox']
            # 注意：这四个角点是第一阶段 LSD 挤压出来的极致精细的物理角点，比 YOLO 准一百倍。
            obs_corners = np.array([[rb[0], rb[1]], [rb[2], rb[1]], [rb[2], rb[3]], [rb[0], rb[3]]])
            clear_frames.append({
                'T_CW': f['T_CW'], 'corners': obs_corners, 'h_lines': h_lines, 'v_lines': v_lines
            })

    plane_prior = {'n': np.array([0, 0, 1]), 'd': -12.5} 
    
    # 【神来之笔：用 Looming 深度反推初值 x0】
    # 非线性优化极容易掉进局部极小值（Local Minimum）。一个极品的好初值是成功的一半。
    # 我们利用最后一帧的相机位置，顺着它的光轴（Z轴）方向，朝前射出一段距离（就是刚刚算出的 Looming 深度），得到标志牌的初始 3D 位置。
    last_T_WC = frames_data[-1]['T_WC']
    init_t = last_T_WC[:3, 3] + last_T_WC[:3, 2] * final_z_looming 
    x0 = np.concatenate([init_t, [0, 0, 0]]) # 拼接成 6 维数组：平移XYZ + 初始不旋转[0,0,0]

    # 【Step 4】启动 LM 优化器进行多源融合降维打击
    logger.info(">>> [第四阶段] 🚀 引擎点火：SciPy Levenberg-Marquardt 多约束联合寻优...")
    # 调用 scipy 的 least_squares 函数。它会疯狂调用 joint_residuals，疯狂微调 x0，直到误差无法再小。
    res = least_squares(joint_residuals, x0, args=(clear_frames, blur_frames, plane_prior, final_z_looming, last_T_WC),
                        method='lm', xtol=1e-8, ftol=1e-8, verbose=1)

    x_opt = res.x # 这就是千锤百炼出来的唯一真理：标志牌完美的 6-DoF 姿态
    logger.info("="*60)
    logger.info(" 🎉 交通标志牌 6-DoF 位姿最终联合优化成功！")
    logger.info(f" 📍 世界系绝对坐标 (X, Y, Z) :  {x_opt[0]:.4f}, {x_opt[1]:.4f}, {x_opt[2]:.4f}")
    logger.info(f" 📐 标志牌绝对朝向 (R_WS) : \n{R.from_rotvec(x_opt[3:6]).as_matrix()}")
    logger.info("="*60)

    # =========================================================
    # 6. 可视化引擎 (给肉眼看的 3D 态势沙盘)
    # =========================================================
    logger.info(">>> 正在生成全维 3D 态势图 (关闭弹窗即可退出)...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 绘制相机历史运行轨迹 (蓝虚线)
    cam_xs = [f['pos_w'][0] for f in frames_data]
    cam_ys = [f['pos_w'][1] for f in frames_data]
    cam_zs = [f['pos_w'][2] for f in frames_data]
    ax.plot(cam_xs, cam_ys, cam_zs, label='SLAM Camera Trajectory', color='blue', marker='.', linestyle='dashed', alpha=0.5)
    ax.scatter(cam_xs[0], cam_ys[0], cam_zs[0], color='green', s=100, marker='s', label='Start') # 起点绿方块
    ax.scatter(cam_xs[-1], cam_ys[-1], cam_zs[-1], color='red', s=100, marker='^', label='End (t_last)') # 终点红三角

    # 2. 绘制优化算出的金灿灿实体标志牌面板
    # 把标志牌的物理角点模型 P_S，通过算出的终极矩阵，变换到 3D 物理空间去
    sign_corners_world = np.dot(P_S, R.from_rotvec(x_opt[3:6]).as_matrix().T) + x_opt[0:3] 
    # 把四个点连成一个闭合多边形，渲染填充上金色
    verts = [list(zip(sign_corners_world[:, 0], sign_corners_world[:, 1], sign_corners_world[:, 2]))]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.8, facecolor='gold', edgecolor='black', linewidths=2))
    
    # 3. 画一根代表深度探测结果的“镭射线” (从最后相机点直指目标板)
    ax.plot([cam_xs[-1], x_opt[0]], [cam_ys[-1], x_opt[1]], [cam_zs[-1], x_opt[2]], 
            color='orange', linestyle=':', label=f'Optimized LOS (Z≈{final_z_looming:.2f})')

    # 4. 坐标轴基准调平。如果不做这一步，相机的轨迹会被拉扯得像面条，无法体现真实的物理 1:1 视觉比例。
    max_range = np.array([np.max(cam_xs)-np.min(cam_xs), np.max(cam_ys)-np.min(cam_ys), np.max(cam_zs)-np.min(cam_zs), 5.0]).max() / 2.0
    mid_x, mid_y, mid_z = (np.max(cam_xs)+np.min(cam_xs))*0.5, (np.max(cam_ys)+np.min(cam_ys))*0.5, (np.max(cam_zs)+np.min(cam_zs))*0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('World X (m)'); ax.set_ylabel('World Y (m)'); ax.set_zlabel('World Z (m)')
    ax.set_title('Fusion Model: 6-DoF LM Optimization Anchor via Looming Depth', fontsize=14, fontweight='bold')
    ax.legend()
    # 调整到一个上帝右侧俯视的经典沙盘观察视角
    ax.view_init(elev=20, azim=-45) 
    plt.show() 

# 标准 Python 运行入口
if __name__ == "__main__":
    main()