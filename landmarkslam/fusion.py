# ==============================================================================
# 📂 实验数据输出说明书 (Output Documentation for V1.2 Baseline)
# ==============================================================================
# 所有结果均保存在: /landmarkslam/output/looming_fusion_results/
# 
# 1️⃣ 【核心对标数据 - 用于明日双目精度对比】
#    - final_6dof_pose.txt : 
#        最终解算的标志牌 6-DoF 姿态。包含世界坐标 (Tx, Ty, Tz) 和 3x3 旋转矩阵。
#        [用途]: 明天直接拿双目点云算出的中心点和法向量与此文件中的数字做差，计算绝对误差。
#    - looming_convergence.csv : 
#        记录 Looming 深度从第一帧到最后一帧的收敛过程。
#        [字段]: 物理位移(delta_d), 宽/高膨胀像素, 单维深度(Z_w/Z_h), 融合深度(Z_final)。
#        [用途]: 在 Excel 中以 delta_d 为 X 轴画折线图，展示单目深度如何走向稳定。
#    - fusion_experiment.log : 
#        全流程运行日志，包含 ASCII 表格、每一帧提取的线条数、LM 迭代次数和残差代价。
#
# 2️⃣ 【宏观空间轨迹 - 用于论文/报告配图】
#    - traj_view_XY_Top.png : 俯视图 (Top View)，展示相机轨迹和标志牌的 X-Y 平面位置。
#    - traj_view_XZ_Front.png : 正视图 (Front View)，展示相机高度变化与标志牌的纵深关系。
#    - traj_view_YZ_Side.png : 侧视图 (Side View)，展示标志牌在侧面的投影位置。
#    - [交互窗口] : 运行结束弹出的 3D 沙盘，可鼠标旋转，查看相机行驶和目标板的相对关系。
#
# 3️⃣ 【手术级特征诊断 - 保存在 vis_steps/ 目录】 (针对最后几帧清晰图像)
#    - {fname}_looming_1_roi_gray.jpg : 原始灰度 ROI。检查 YOLO 框是否精准切分。
#    - {fname}_looming_2_lsd_red.jpg  : 红色长线图。检查 LSD 算法是否抓取到了干净的物理边缘。
#    - {fname}_looming_3_bbox_green.jpg : 绿色极值框。这是 LM 优化器真正“看”到的物理角点。
#    - {fname}_ortho_1_upscaled.jpg   : 放大提亮图。检查暗光环境下，汉字笔画是否被拉升到清晰可辨。
#    - {fname}_ortho_2_lines.jpg      : 红绿笔画图。红线代表横向，绿线代表纵向。这是姿态修正的能量源。
#
# 4️⃣ 【终极姿态测谎仪 - 保存在 vis_steps/ 目录】
#    - final_looming_diagnostic_warped_residue.jpg : 
#        利用算出的姿态将 2D 照片“反向透视投射”到 3D 物理平板后的效果。
#        [看点]: 观察图中的汉字是否“横平竖直”。若出现粗红/绿框报警，说明在该位置笔画歪斜，
#               姿态角(Pitch/Yaw)仍有修正空间。这是证明算法正交约束有效性的最强证据。
# ==============================================================================

import numpy as np               # 核心数学库：负责所有的矩阵乘法、向量加减、求逆矩阵等底层数学运算
import pandas as pd              # 数据处理库：负责读取由逗号分隔的 CSV 和 TXT 日志文件
from scipy.optimize import least_squares  # 核心算法库：SciPy 提供的 LM (Levenberg-Marquardt) 非线性最小二乘求解器
from scipy.spatial.transform import Rotation as R # 核心空间库：专门用于处理 3D 空间中极其容易算错的旋转（四元数转旋转矩阵等）
import os                        # 操作系统接口：用于拼接文件路径、创建文件夹
import cv2                       # OpenCV：工业级计算机视觉库，负责图像读取、变灰度、提亮、提取底层亚像素线段
import math                      # 基础数学库：提供计算欧氏距离(hypot)和角度(atan2)
import logging                   # 日志库：负责把终端打印的信息同时写入到 log 文件中，方便实验复盘
import csv                       # CSV 库：负责把最终的深度收敛过程写入本地表格
import matplotlib.pyplot as plt  # 绘图库：负责画出 2D 的收敛曲线和 3D 的空间态势图
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # 3D 绘图扩展：专门用于在 3D 坐标系中画出带颜色的多边形实体面片

# =========================================================
# 1. 核心配置与参数 (实验全局环境变量)
# =========================================================
# --- 输入数据路径 (请确保这些路径与你机器上的一致) ---
TRAJ_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/output/FrameTrajectory_TUM_20260331_154838.txt" # SLAM 输出的绝对轨迹 (TUM 格式)
MAP_FILE  = "/home/zah/ORB_SLAM3-master/landmarkslam/output/Filename_Mapping_20260331_154838.csv"   # 图像文件名 -> 物理时间戳映射表
BBOX_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/yolo_simulated_bboxes.txt" # YOLO 吐出的 2D 识别框
IMAGE_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/"          # 真实环境照片的存放目录

# --- 输出数据路径规划 (为了明天的双目对比实验专门设计) ---
EXPERIMENT_OUT_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/output/looming_fusion_results/"
LOG_FILE = os.path.join(EXPERIMENT_OUT_DIR, "fusion_experiment.log")        # 核心日志记录
CSV_FILE = os.path.join(EXPERIMENT_OUT_DIR, "looming_convergence.csv")      # 记录 Looming 收敛过程
POSE_OUT_FILE = os.path.join(EXPERIMENT_OUT_DIR, "final_6dof_pose.txt")     # [对比核心] 保存最终 6-DoF 姿态，明天直接跟双目真值硬碰硬！
VIS_STEPS_DIR = os.path.join(EXPERIMENT_OUT_DIR, "vis_steps/")              # 保存图像提取的中间过程，用于写论文配图

# 自动创建不存在的输出文件夹
os.makedirs(EXPERIMENT_OUT_DIR, exist_ok=True)
os.makedirs(VIS_STEPS_DIR, exist_ok=True)

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- 相机内参 (Camera Intrinsics) ---
# 针孔相机模型的核心，决定了真实 3D 物理世界的光线穿过镜头后，会落在照片的第几个像素上。
FX, FY = 501.00685446, 496.63593447  # X和Y方向的焦距 (像素)。代表相机的视野广度。
CX, CY = 316.00266456, 233.80218648  # 光心坐标 (像素)。通常在图像的正中心附近。

# --- 标志牌物理 3D 模型 (Local Sign Coordinate System) ---
WIDTH, HEIGHT = 1.26, 0.6            # 标志牌真实的物理宽度和高度 (米)
HALF_W, HALF_H = WIDTH / 2.0, HEIGHT / 2.0
# P_S: 以标志牌绝对中心为原点 (0,0,0) 时，四个物理角点在局部坐标系下的 3D 坐标。
# 顺序为：[左上角, 右上角, 右下角, 左下角]。因为牌子是平的，Z 全部为 0.0。
P_S = np.array([[-HALF_W, -HALF_H, 0.0], [ HALF_W, -HALF_H, 0.0], 
                [ HALF_W,  HALF_H, 0.0], [-HALF_W,  HALF_H, 0.0]])

# --- LM 图优化目标函数的惩罚权重 (Hyperparameters) ---
# 这些权重决定了优化器在遇到矛盾时“该听谁的”。
W_REPROJ  = 1.0    # 约束 A: 角点重投影约束。亚像素角点极其精确，作为决定 X, Y 平移的主力基石。
W_BBOX    = 0.02   # 约束 C: YOLO语义框包裹约束。因为 YOLO 框粗糙浮动，权重给极低，仅用于宏观防发散（保底）。
W_STRUCT  = 0.5    # 约束 D: 背景平面共面约束。SLAM 提取的大楼墙面先验，防止单目牌子在 Z 轴前后乱飘。
W_ORTHO   = 10.0   # 约束 B: 汉字正交死锁约束。权重极高！强行逼迫汉字横平竖直，以此解算出远距离极难观测的 Pitch/Yaw 倾斜角。
W_LOOMING = 5.0    # 约束 E: 深度抛锚约束。将 Looming 算出的绝对深度作为强力“金锚”，彻底解决单目尺度模糊的死穴。

# =========================================================
# 2. 前端：特征提取与数据预处理模块 (从 2D 像素里榨取高精特征)
# =========================================================

def get_T_WC(pose_tum):
    """把 SLAM 输出的 TUM 格式的四元数和位置，变成 4x4 的齐次变换矩阵 T_WC"""
    T_WC = np.eye(4)
    T_WC[:3, :3] = R.from_quat(pose_tum[3:7]).as_matrix()
    T_WC[:3, 3] = pose_tum[0:3]
    return T_WC

def extract_optical_projection_hw(img_path, u1, v1, u2, v2, fname):
    """
    【模块A】：Looming 亚像素边缘提取器 (已严格按照用户图片需求重构可视化输出)
    作用：剔除 YOLO 粗糙边框的抖动，通过底层光影梯度，寻找最贴合标志牌物理边界的亚像素包围盒。
    返回值：最终提取出的长、宽，以及精调后的 2D BoundingBox。
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: return None, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    fname_base = os.path.splitext(fname)[0]
    
    # 策略 1：ROI 外扩 (Padding)。给 YOLO 框留 2 个像素余量，防止真实边缘被 YOLO 切断。
    pad = 2
    crop_u1, crop_v1 = max(0, int(u1) - pad), max(0, int(v1) - pad)
    crop_u2, crop_v2 = min(img_w, int(u2) + pad), min(img_h, int(v2) + pad)
    roi = gray[crop_v1:crop_v2, crop_u1:crop_u2]
    roi_h, roi_w = roi.shape 
    
    # 📸 [严格对标展示 1]: 保存纯灰度 ROI 裁剪图 (对应你上传的底图)
    cv2.imwrite(os.path.join(VIS_STEPS_DIR, f"{fname_base}_looming_1_roi_gray.jpg"), roi)
    
    # 策略 2：抗逆光增强 (CLAHE)。
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(roi)
    
    # 策略 3：LSD 亚像素线段提取
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(roi_enhanced)
    
    fallback_w, fallback_h = float(u2 - u1), float(v2 - v1)
    fallback_bbox = np.array([u1, v1, u2, v2])

    # 准备两块与 ROI 大小一致的彩色画板，用于分别绘制红线和绿框
    vis_roi_red = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    vis_roi_green = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    if lines is None: return fallback_w, fallback_h, fallback_bbox
        
    all_x, all_y = [], []
    # 策略 4：长度海选过滤。扔掉短小的噪点，只保留长于 ROI 短边 40% 的极长线段。
    min_len = min(roi_w, roi_h) * 0.4 
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.hypot(x2-x1, y2-y1)
        if length > min_len:
            # 记录全局坐标供算法核心使用
            all_x.extend([x1 + crop_u1, x2 + crop_u1])
            all_y.extend([y1 + crop_v1, y2 + crop_v1])
            # 📸 [严格对标展示 2]: 在画板上用红色绘制提取到的所有长线段 (对应你上传的红线图)
            cv2.line(vis_roi_red, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    cv2.imwrite(os.path.join(VIS_STEPS_DIR, f"{fname_base}_looming_2_lsd_red.jpg"), vis_roi_red)

    # 策略 5：极值挤压与常识熔断
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

    # 提取最终坐标
    xmin, xmax = (min(all_x), max(all_x)) if not w_triggered and all_x else (u1, u2)
    ymin, ymax = (min(all_y), max(all_y)) if not h_triggered and all_y else (v1, v2)
    
    # 📸 [严格对标展示 3]: 用绿色绘制最终挤压出的极值外边框 (对应你上传的绿线图)
    # 将全局极值坐标映射回 ROI 局部坐标系以便画图
    roi_xmin, roi_xmax = int(xmin - crop_u1), int(xmax - crop_u1)
    roi_ymin, roi_ymax = int(ymin - crop_v1), int(ymax - crop_v1)
    cv2.line(vis_roi_green, (roi_xmin, roi_ymin), (roi_xmin, roi_ymax), (0, 255, 0), 2) # 左竖线
    cv2.line(vis_roi_green, (roi_xmax, roi_ymin), (roi_xmax, roi_ymax), (0, 255, 0), 2) # 右竖线
    cv2.line(vis_roi_green, (roi_xmin, roi_ymin), (roi_xmax, roi_ymin), (0, 255, 0), 2) # 上横线
    cv2.line(vis_roi_green, (roi_xmin, roi_ymax), (roi_xmax, roi_ymax), (0, 255, 0), 2) # 下横线
    cv2.imwrite(os.path.join(VIS_STEPS_DIR, f"{fname_base}_looming_3_bbox_green.jpg"), vis_roi_green)

    # 极其关键的输出：基于亚像素重构的精确边界框 (Refined Bbox)。
    refined_bbox = np.array([xmin, ymin, xmax, ymax])
    return final_w, final_h, refined_bbox

def extract_subpixel_orthogonal_lines(img_path, roi_bbox, fname, gamma=2.5, clahe_clip=4.0, scale_factor=2.5, angle_tol=15):
    """
    【模块B】：提取汉字内部横平竖直的结构线段 (正交特征提取)
    作用：捕捉标志牌内部的文字笔画，用于反向单应性映射，以此纠正 3D 姿态偏角。
    """
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None: return [], []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fname_base = os.path.splitext(fname)[0]
    
    # 策略 1：非线性 Gamma 提亮
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(gray, table)
    img_clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(img_gamma) 
    
    # 策略 2：超分辨率插值 (放大 2.5 倍制造平滑梯度)
    h, w = img_clahe.shape 
    img_up = cv2.resize(img_clahe, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
    img_up = cv2.GaussianBlur(img_up, (3, 3), 0) 
    
    cv2.imwrite(os.path.join(VIS_STEPS_DIR, f"{fname_base}_ortho_1_upscaled.jpg"), img_up)

    lsd = cv2.createLineSegmentDetector(0)
    lines_up, _, _, _ = lsd.detect(img_up)
    vis_lines_canvas = cv2.cvtColor(img_up, cv2.COLOR_GRAY2BGR)
    
    horiz_lines, vert_lines = [], []
    if lines_up is not None:
        for line in lines_up:
            x1, y1, x2, y2 = line[0] / scale_factor 
            xmin, ymin, xmax, ymax = roi_bbox
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            
            # 过滤 A：只保留包裹在标志牌 BoundingBox 内部的线段
            if not (xmin <= cx <= xmax and ymin <= cy <= ymax): continue
            
            # 过滤 B：太短的线段视为噪声
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 3.0: continue 
                
            # 策略 3：基于反正切 atan2 的角度过滤
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180.0
            up_x1, up_y1, up_x2, up_y2 = int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[0][3])
            
            if angle <= angle_tol or angle >= (180 - angle_tol):
                horiz_lines.append((x1, y1, x2, y2, length)) 
                cv2.line(vis_lines_canvas, (up_x1, up_y1), (up_x2, up_y2), (0, 0, 255), 2) # 红线为横
            elif abs(angle - 90) <= angle_tol:
                vert_lines.append((x1, y1, x2, y2, length))  
                cv2.line(vis_lines_canvas, (up_x1, up_y1), (up_x2, up_y2), (0, 255, 0), 2) # 绿线为竖

    cv2.imwrite(os.path.join(VIS_STEPS_DIR, f"{fname_base}_ortho_2_lines.jpg"), vis_lines_canvas)
    return horiz_lines, vert_lines

# =========================================================
# 3. 中端：投影几何与数学推导引擎 (2D 像素与 3D 物理的桥梁)
# =========================================================

def project_to_pixel(x_state, P_model, T_CW):
    """正向透视投影：P_c = R_cw * (R_ws * P_s + t_ws) + t_cw"""
    t_WS, r_WS = x_state[0:3], R.from_rotvec(x_state[3:6]).as_matrix()
    P_W = np.dot(P_model, r_WS.T) + t_WS
    P_C = np.dot(P_W, T_CW[:3, :3].T) + T_CW[:3, 3]
    Z_C = np.maximum(P_C[:, 2], 1e-6) # 防止被 0 除
    u = FX * (P_C[:, 0] / Z_C) + CX
    v = FY * (P_C[:, 1] / Z_C) + CY
    return np.column_stack((u, v))

def get_inverse_homography(x_state, T_CW):
    """逆向单应矩阵 (Inverse Homography)：把照片上的 2D 汉字反向贴回 3D 物理平面"""
    t_WS, r_WS = x_state[0:3], R.from_rotvec(x_state[3:6]).as_matrix()
    T_WS = np.eye(4)
    T_WS[:3, :3], T_WS[:3, 3] = r_WS, t_WS
    T_CS = np.dot(T_CW, T_WS) 
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]]) 
    M = np.column_stack((T_CS[:3, 0], T_CS[:3, 1], T_CS[:3, 3])) # 抽离 Z 轴列降维
    try: return np.linalg.inv(np.dot(K, M))
    except: return np.eye(3) 

def pixel_to_sign_plane(u, v, H_inv):
    """利用逆矩阵，把像素转化回真实的物理坐标 (X_米, Y_米)"""
    p_img = np.array([u, v, 1.0])
    p_sign = np.dot(H_inv, p_img)
    return p_sign[0] / p_sign[2], p_sign[1] / p_sign[2] # 齐次归一化

# =========================================================
# 4. 后端：LM 优化器核心惩罚函数 (The Target Function)
# =========================================================

def joint_residuals(x, clear_frames, blur_frames, plane_prior, looming_z, last_T_WC):
    """
    【终极裁判官】：代价函数 (Cost Function)。
    工作机制：优化器 LM 不断微调 x，直到残差数组 res 里所有的数值平方和逼近于 0。
    """
    res = []
    
    # ---------------- 约束 A & B：基于最后两帧的高精微观死锁 ----------------
    for f in clear_frames:
        # A: 物理角点重投影约束
        pts_hat = project_to_pixel(x, P_S, f['T_CW'])
        res.append((f['corners'] - pts_hat).flatten() * W_REPROJ)
        
        # B: 汉字正交形态约束 (强力压制 Pitch/Yaw)
        if 'h_lines' in f and 'v_lines' in f:
            H_inv = get_inverse_homography(x, f['T_CW'])
            # 把照片上的横线贴回 3D 铁皮上，要求它左右两端高度必须相等 (Y1 == Y2)
            for u1, v1, u2, v2, weight in f['h_lines']:
                _, Y1 = pixel_to_sign_plane(u1, v1, H_inv)
                _, Y2 = pixel_to_sign_plane(u2, v2, H_inv)
                res.append(np.array([(Y1 - Y2) * weight * W_ORTHO]))
            # 把照片上的竖线贴回 3D 铁皮上，要求它上下两端水平位置必须相等 (X1 == X2)
            for u1, v1, u2, v2, weight in f['v_lines']:
                X1, _ = pixel_to_sign_plane(u1, v1, H_inv)
                X2, _ = pixel_to_sign_plane(u2, v2, H_inv)
                res.append(np.array([(X1 - X2) * weight * W_ORTHO]))
                
    # ---------------- 约束 C：基于远距离帧的视锥宏观托底 ----------------
    for f in blur_frames:
        pts_hat = project_to_pixel(x, P_S, f['T_CW'])
        b_hat = np.array([np.min(pts_hat[:,0]), np.min(pts_hat[:,1]), np.max(pts_hat[:,0]), np.max(pts_hat[:,1])])
        res.append((f['bbox'] - b_hat) * W_BBOX) 
        
    # ---------------- 约束 D：物理墙面共面托底 ----------------
    if plane_prior:
        dist = np.dot(plane_prior['n'], x[0:3]) + plane_prior['d'] 
        res.append(np.array([dist]) * W_STRUCT) 
        
    # ---------------- 约束 E：【核武器】Looming 绝对深度锚点 ----------------
    t_WS = x[0:3] 
    cam_pos_world = last_T_WC[:3, 3] 
    cam_z_axis = last_T_WC[:3, 2]    
    # 向量投影算出绝对物理景深
    current_z = np.dot((t_WS - cam_pos_world), cam_z_axis) 
    # 霸王条款：当前深度必须死死咬住第一阶段算出的 Looming 真值锚点 (looming_z)！
    res.append(np.array([current_z - looming_z]) * W_LOOMING)
        
    return np.concatenate(res)

# =========================================================
# 5. 可视化存储与呈现模块 (解耦的绘图沙盘)
# =========================================================
def export_and_plot_trajectory(frames_data, x_opt, final_z_looming):
    logger.info(">>> [阶段五] 正在保存空间三视图 PNG 及渲染 3D 态势沙盘...")
    
    cam_xs = [f['pos_w'][0] for f in frames_data]
    cam_ys = [f['pos_w'][1] for f in frames_data]
    cam_zs = [f['pos_w'][2] for f in frames_data]
    
    sign_corners_world = np.dot(P_S, R.from_rotvec(x_opt[3:6]).as_matrix().T) + x_opt[0:3] 
    lm_x, lm_y, lm_z = x_opt[0], x_opt[1], x_opt[2]

    # --- [操作 1]：静默保存三视图，用于写论文配图 ---
    views = [
        ('XY Plane (Top View/俯视图)', cam_xs, cam_ys, lm_x, lm_y, 'X', 'Y', 'traj_view_XY_Top.png'),
        ('XZ Plane (Front View/正视图)', cam_xs, cam_zs, lm_x, lm_z, 'X', 'Z', 'traj_view_XZ_Front.png'),
        ('YZ Plane (Side View/侧视图)', cam_ys, cam_zs, lm_y, lm_z, 'Y', 'Z', 'traj_view_YZ_Side.png')
    ]
    for title, cx, cy, lx, ly, x_label, y_label, filename in views:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(cx, cy, linestyle='-', color='b', alpha=0.5, label='Camera Path')
        ax.scatter(cx[0], cy[0], color='green', s=100, edgecolor='k', label='Cam Start')
        ax.scatter(cx[-1], cy[-1], color='red', s=100, edgecolor='k', label='Cam End')
        ax.scatter(lx, ly, color='gold', marker='*', s=300, edgecolor='k', label='Optimized LM ⭐')
        ax.set_title(title); ax.set_xlabel(f'World {x_label}'); ax.set_ylabel(f'World {y_label}')
        ax.grid(True, linestyle='--'); ax.legend(loc='best'); ax.axis('equal') 
        fig.savefig(os.path.join(EXPERIMENT_OUT_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
    logger.info(f"    🖼️ 三视图(XY, XZ, YZ)已保存至: {EXPERIMENT_OUT_DIR}")

    # --- [操作 2]：弹出支持鼠标拖拽旋转的 3D 交互沙盘 ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(cam_xs, cam_ys, cam_zs, label='SLAM Camera Trajectory', color='blue', marker='.', linestyle='dashed', alpha=0.5)
    ax.scatter(cam_xs[0], cam_ys[0], cam_zs[0], color='green', s=100, marker='s', label='Start') 
    ax.scatter(cam_xs[-1], cam_ys[-1], cam_zs[-1], color='red', s=100, marker='^', label='End (t_last)') 

    verts = [list(zip(sign_corners_world[:, 0], sign_corners_world[:, 1], sign_corners_world[:, 2]))]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.8, facecolor='gold', edgecolor='black', linewidths=2))
    
    # 画一条视差镭射线：从相机的最后一帧光心，直直射向标志牌的中心点
    ax.plot([cam_xs[-1], x_opt[0]], [cam_ys[-1], x_opt[1]], [cam_zs[-1], x_opt[2]], 
            color='orange', linestyle=':', label=f'Optimized LOS (Z≈{final_z_looming:.2f}m)')

    # 【防畸变基准调平】
    max_range = np.array([np.max(cam_xs)-np.min(cam_xs), np.max(cam_ys)-np.min(cam_ys), np.max(cam_zs)-np.min(cam_zs), 5.0]).max() / 2.0
    mid_x, mid_y, mid_z = (np.max(cam_xs)+np.min(cam_xs))*0.5, (np.max(cam_ys)+np.min(cam_ys))*0.5, (np.max(cam_zs)+np.min(cam_zs))*0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('World X (m)'); ax.set_ylabel('World Y (m)'); ax.set_zlabel('World Z (m)')
    ax.set_title('V1.0 Baseline: 6-DoF LM Optimization Anchor via Looming Depth', fontsize=14, fontweight='bold')
    ax.legend()
    ax.view_init(elev=20, azim=-45) 
    
    logger.info(">>> 🚀 正在弹出 3D 轨迹交互窗口，可用鼠标拖拽旋转全方位检查态势 (关掉窗口程序结束)...")
    plt.show() 

# =========================================================
# 6. 管线集成与执行控制流 (The Central Nervous System)
# =========================================================
def main():
    logger.info("="*60)
    logger.info(" 🚀 [融合系统启动] V1.0 Baseline (Looming Depth + LM 6-DoF)")
    logger.info("="*60)

    # 【Step 1】：数据对齐与装载。
    traj_data = np.loadtxt(TRAJ_FILE)
    traj_dict = {row[0]: row[1:] for row in traj_data}
    all_ts = np.array(list(traj_dict.keys()))
    
    mapping_df = pd.read_csv(MAP_FILE)
    bbox_df = pd.read_csv(BBOX_FILE, header=None, names=['fname', 'u1', 'v1', 'u2', 'v2'], sep=',', comment='#')
    
    frames_data = [] 
    logger.info(">>> [第一阶段] 运行 Looming 亚像素特征抓取 (结果保存在 vis_steps 目录)...")
    
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

    # 【Step 2】：Looming 时序膨胀解算。深度 Z = 物理位移 d / (像素放大倍数 - 1)
    logger.info(">>> [第二阶段] 执行时序膨胀解算获取绝对深度 Z_final ...")
    f_start = frames_data[0] 
    w1, h1 = f_start['width'], f_start['height']
    final_z_looming = 8.0 
    csv_results = []

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
            
        csv_results.append({
            't1_fname': f_start['fname'], 't2_fname': f_end['fname'],
            'delta_d': delta_d, 'Z_w': Z_w if Z_w else -1, 'Z_h': Z_h if Z_h else -1, 'Z_final': final_z_looming
        })
            
    logger.info(f"    📍 Looming 收敛完毕，提供完美的绝对深度先验锚点: {final_z_looming:.4f} units")
    
    keys = csv_results[0].keys()
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(csv_results)

    # 【Step 3】：准备后端优化的燃料，提取汉字特征
    logger.info(">>> [第三阶段] 构建大熔炉：提取汉字结构，准备 6-DoF 图优化...")
    blur_frames, clear_frames = [], []
    for i, f in enumerate(frames_data):
        if i < len(frames_data) - 2:
            blur_frames.append({'T_CW': f['T_CW'], 'bbox': f['yolo_bbox']}) 
        else:
            h_lines, v_lines = extract_subpixel_orthogonal_lines(f['img_path'], f['refined_bbox'], f['fname'])
            logger.info(f"    - {f['fname'][-20:]}: 提取到 {len(h_lines)} 条横线, {len(v_lines)} 条竖线")
            rb = f['refined_bbox']
            # 将 Looming 前端提取出的亚像素精确边界框，直接充当 LM 优化器信赖的物理角点！
            obs_corners = np.array([[rb[0], rb[1]], [rb[2], rb[1]], [rb[2], rb[3]], [rb[0], rb[3]]])
            clear_frames.append({'T_CW': f['T_CW'], 'corners': obs_corners, 'h_lines': h_lines, 'v_lines': v_lines})

    plane_prior = {'n': np.array([0, 0, 1]), 'd': -12.5} 
    
    # 【神来之笔】：直接用刚才 Looming 算出来的绝对深度，结合相机光轴射线，反推出极其精准的初始 3D 位置 x0
    last_T_WC = frames_data[-1]['T_WC']
    init_t = last_T_WC[:3, 3] + last_T_WC[:3, 2] * final_z_looming 
    x0 = np.concatenate([init_t, [0, 0, 0]])
    logger.info(f"    - 优化器超强初始姿态 (x0): {x0}")

    # 【Step 4】：启动 LM 优化器进行炼丹
    logger.info(">>> [第四阶段] 🚀 引擎点火：SciPy LM 多约束联合寻优...")
    res = least_squares(joint_residuals, x0, args=(clear_frames, blur_frames, plane_prior, final_z_looming, last_T_WC),
                        method='lm', xtol=1e-8, ftol=1e-8, verbose=2)

    x_opt = res.x 
    logger.info("="*60)
    logger.info(" 🎉 交通标志牌 6-DoF 位姿最终联合优化成功！")
    logger.info(f" 📍 世界系绝对坐标 (X, Y, Z) :  {x_opt[0]:.4f}, {x_opt[1]:.4f}, {x_opt[2]:.4f}")
    logger.info(f" 📐 标志牌绝对朝向 (R_WS) : \n{R.from_rotvec(x_opt[3:6]).as_matrix()}")
    logger.info(f" 📉 优化器迭代次数: {res.nfev}, 最终残差代价: {res.cost:.6f}")
    logger.info("="*60)
    
    # 将算出来的绝对坐标、旋转向量、旋转矩阵保存为纯文本文件。
    # 明天做双目相机实验时，直接拿双目算出的 3D 点云匹配矩阵来跟这个文件里的数字对比计算误差率！
    with open(POSE_OUT_FILE, "w", encoding="utf-8") as f:
        f.write("=== Final 6-DoF Pose (World Coordinate) ===\n")
        f.write(f"Tx, Ty, Tz (meters): {x_opt[0]:.6f}, {x_opt[1]:.6f}, {x_opt[2]:.6f}\n")
        f.write(f"Rotation Vector (Rx, Ry, Rz): {x_opt[3]:.6f}, {x_opt[4]:.6f}, {x_opt[5]:.6f}\n\n")
        f.write("Rotation Matrix (3x3):\n")
        f.write(np.array2string(R.from_rotvec(x_opt[3:6]).as_matrix(), separator=', '))
    logger.info(f">>> 📝 最终姿态对标基准数据已落盘至: {POSE_OUT_FILE}")

    # 【Step 5】：可视化并输出三视图和沙盘
    export_and_plot_trajectory(frames_data, x_opt, final_z_looming)

if __name__ == "__main__":
    main()