import numpy as np               # NumPy: 用于进行高效的矩阵运算和向量计算（如点乘、矩阵乘法、求逆等）
import pandas as pd              # Pandas: 用于读取和处理 CSV/TXT 表格数据（如时间戳映射表、YOLO框数据）
from scipy.optimize import least_squares  # SciPy: LM 非线性最小二乘优化器，算法的“大脑”，负责不断试错寻找最优位姿
from scipy.spatial.transform import Rotation as R # SciPy: 用于处理极其复杂的 3D 旋转（如四元数转旋转矩阵、旋转向量转矩阵）
import os                        # OS: 用于处理文件路径拼接
import cv2                       # OpenCV: 计算机视觉库，负责图像读取、变灰度、提亮、放大、提取线段
import math                      # Math: Python 内置数学库，用于计算线段长度（勾股定理）和角度（反正切）

# =========================================================
# 1. 核心配置区 (你的真实环境数据输入)
# =========================================================
# 【文件路径变量】
# TRAJ_FILE: 底盘 SLAM 系统跑出来的绝对轨迹文件路径（包含时间戳和相机在世界里的XYZ位置及旋转四元数）
TRAJ_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/output/FrameTrajectory_TUM_20260331_154838.txt" 
# MAP_FILE: 图像文件名与真实物理时间戳（秒）的对应关系表
MAP_FILE  = "/home/zah/ORB_SLAM3-master/landmarkslam/output/Filename_Mapping_20260331_154838.csv"     
# BBOX_FILE: YOLO 目标检测算法输出的 2D 标志牌边界框位置（格式为：文件名, 左上角X, 左上角Y, 右下角X, 右下角Y）
BBOX_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/yolo_simulated_bboxes.txt" 
# IMAGE_DIR: 存放真实暗光测试图片的文件夹，用于代码读取图片并提取汉字线段
IMAGE_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/"              

# 【相机内参参数】 (针孔相机模型的核心，决定了 3D 空间的光线如何打在 2D 像素阵列上)
FX = 501.00685446 # 焦距 (Focal Length) 的 X 轴分量，单位是像素
FY = 496.63593447 # 焦距 (Focal Length) 的 Y 轴分量，单位是像素
CX = 316.00266456 # 光心 (Principal Point) 的 X 坐标，即相机镜头正中心对应的像素列标
CY = 233.80218648 # 光心 (Principal Point) 的 Y 坐标，即相机镜头正中心对应的像素行标

# 【标志牌 3D 物理模型参数】 
WIDTH = 1.26   # 标志牌的真实物理宽度，单位：米
HEIGHT = 0.6   # 标志牌的真实物理高度，单位：米
HALF_W = WIDTH / 2.0  # 宽度的一半 (0.63米)
HALF_H = HEIGHT / 2.0 # 高度的一半 (0.30米)

# 【P_S: 标志牌局部坐标系模型 (Points in Sign Coordinate)】
# 假设我们把坐标系的原点(0,0,0)定在标志牌的绝对正中心。
# 那么标志牌 4 个物理尖角的 3D 坐标就可以用加减 HALF_W 和 HALF_H 来表示。
# 第 3 个数字是 0.0，因为标志牌是一个扁平的平面，厚度忽略不计，Z轴坐标为0。
P_S = np.array([
    [-HALF_W, -HALF_H, 0.0], # 索引0: 左上角坐标 (-0.63, -0.3, 0)
    [ HALF_W, -HALF_H, 0.0], # 索引1: 右上角坐标 (+0.63, -0.3, 0)
    [ HALF_W,  HALF_H, 0.0], # 索引2: 右下角坐标 (+0.63, +0.3, 0)
    [-HALF_W,  HALF_H, 0.0]  # 索引3: 左下角坐标 (-0.63, +0.3, 0)
])

# 【图优化权重参数】 (告诉优化器，哪个约束条件最靠谱，最应该听谁的)
W_REPROJ   = 1.0    # 权重 1：角点重投影。这是最精确的物理顶点匹配，权重最高，是决定最终姿态的基石。
W_BBOX     = 0.02   # 权重 2：YOLO检测框。由于 YOLO 框不够贴合真实边缘，存在误差，所以权重给得很低，只用来防止标志牌飘到十万八千里外。
W_STRUCT   = 0.5    # 权重 3：背景墙平面。防止单目相机的深度（Z轴）发生剧烈漂移，权重中等。
W_ORTHO    = 10.0   # 权重 4：汉字正交线段。这是我们的核心创新点，用于强行压制微小的旋转倾斜（Pitch/Yaw），所以给极高的惩罚权重。

# =========================================================
# 2. 亚像素正交线段提取器 (图像前端：负责从图像里抠出特征)
# =========================================================
def extract_subpixel_orthogonal_lines(image, roi_bbox=None, gamma=2.5, clahe_clip=4.0, scale_factor=2.5, angle_tol=15, min_length=3.0):
    """
    image: 输入的 numpy 图像矩阵
    roi_bbox: YOLO 框区域 [xmin, ymin, xmax, ymax]，只在这个框里找线段，防背景树叶干扰
    gamma: Gamma 曲线提亮参数
    clahe_clip: CLAHE 算法的对比度增强上限
    scale_factor: 图像放大倍数（制造平滑梯度）
    angle_tol: 容忍的字体倾斜角度（比如 15，意味着 0±15度的算横线，90±15度的算竖线）
    min_length: 映射回原图后，线段的最短物理像素长度（低于此值的认为是噪点渣子）
    """
    # 图像防空保护：如果没读到图，返回两个空列表
    if image is None: return [], []
    
    # 将彩图转为单通道灰度图（因为提取形状不需要颜色，只要明暗黑白）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    
    # --- 第一阶段：强行提亮暗部，拉开汉字（黑）与铁皮（灰白）的对比度 ---
    inv_gamma = 1.0 / gamma # 计算 gamma 的倒数，用于生成查表
    # 生成一个 0-255 的映射表 (Look Up Table)。利用幂函数，把暗部像素的值强行拉高。
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(gray, table) # 应用查表，完成整体提亮
    
    # 建立 CLAHE 对象（限制对比度自适应直方图均衡化），把图片分成 8x8 的小块，分别拉满局部对比度
    # clahe_clip(4.0) 控制拉伸的猛烈程度，越高字越黑、底越白
    img_clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(img_gamma) 
    
    # --- 第二阶段：空间升维 (解决汉字太小、糊成一团，LSD算不出梯度的问题) ---
    h, w = img_clahe.shape # 获取增强后图像的高 (h) 和宽 (w)
    # 利用 cv2.INTER_CUBIC (双三次插值) 算法，把原图的宽和高分别乘以 scale_factor (2.5)，生成一张高清大图
    img_upscaled = cv2.resize(img_clahe, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
    # 放大后边缘会有一点马赛克锯齿，用 3x3 的高斯滤波器轻轻抹平一下，方便后续提取
    img_upscaled = cv2.GaussianBlur(img_upscaled, (3, 3), 0)

    # --- 第三阶段：利用 LSD 算法提取线段 ---
    lsd = cv2.createLineSegmentDetector(0) # 初始化 OpenCV 自带的 LSD (Line Segment Detector)
    lines_up, _, _, _ = lsd.detect(img_upscaled) # 在放大的图上找直线，返回值 lines_up 是所有线段坐标的集合
    
    horiz_lines, vert_lines = [], [] # 准备两个空列表，用来装筛选后的“横线”和“竖线”
    
    if lines_up is not None: # 如果成功找到了线段
        for line in lines_up: # 遍历每一条找出来的线段
            # 【核心操作】：因为线段是在放大 2.5 倍的图上找的，坐标全变大了。
            # 我们必须把它的左端点 (x1, y1) 和右端点 (x2, y2) 全部除以 2.5，缩小回真实物理图像的坐标。
            x1, y1, x2, y2 = line[0] / scale_factor 
            
            # --- 第四阶段：三重过滤 (扔掉垃圾线段) ---
            # 过滤 1：利用 YOLO 框物理隔离背景
            if roi_bbox is not None:
                xmin, ymin, xmax, ymax = roi_bbox # 解析 YOLO 框的四个边界
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0 # 计算提取出的这条线段的中心点坐标 (cx, cy)
                # 如果线段的中心点不在 YOLO 框的范围内，说明它是树叶或者楼房的边缘，直接丢弃 (continue 跳过本轮循环)
                if not (xmin <= cx <= xmax and ymin <= cy <= ymax):
                    continue
            
            # 过滤 2：扔掉太短的线
            # 用 math.hypot 计算两点之间的欧式距离（即线段长度）
            length = math.hypot(x2 - x1, y2 - y1)
            # 如果缩放回原图后，线段长度不足 3 个像素 (min_length)，说明是个噪点渣子，丢弃
            if length < min_length: continue 
                
            # 过滤 3：扔掉汉字的“撇”、“捺”、“点” (利用角度进行大浪淘沙)
            # math.atan2 算出斜率角，math.degrees 转为度数，abs 取绝对值，% 180 确保角度落在 0~179.99 之间
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180.0
            
            # 判断逻辑：
            # 如果角度在 [0, 15] 或者 [165, 180] 之间，说明它是一根“横线” (Horizontal)
            if angle <= angle_tol or angle >= (180 - angle_tol):
                # 将起点、终点和线段长度 (作为后续优化器的置信度权重) 打包存入 horiz_lines 列表
                horiz_lines.append((x1, y1, x2, y2, length)) 
            # 如果角度在 [75, 105] 之间 (90±15)，说明它是一根“竖线” (Vertical)
            elif abs(angle - 90) <= angle_tol:
                vert_lines.append((x1, y1, x2, y2, length))  
            # 剩下的比如 45度角，全是“撇”和“捺”，代码不作任何处理，直接被抛弃！
                
    return horiz_lines, vert_lines # 将提纯后的两组线段数据还给主程序

# =========================================================
# 3. 数学计算引擎 (处理极其烧脑的 3D/2D 坐标系变换)
# =========================================================
def get_T_CW(pose_tum):
    """
    将 SLAM 输出的 TUM 格式，转为 4x4 的齐次变换矩阵 T_CW (Transform from World to Camera)
    pose_tum: 一个长度为 7 的一维数组 [x, y, z, qx, qy, qz, qw]
    """
    t_WC, q_WC = pose_tum[0:3], pose_tum[3:7] # 前 3 个是世界系下的 XYZ 平移，后 4 个是姿态四元数
    T_WC = np.eye(4) # 初始化一个 4x4 的单位矩阵 (对角线为1，其余为0)
    T_WC[:3, :3] = R.from_quat(q_WC).as_matrix() # 把四元数转换成 3x3 的旋转矩阵，塞进左上角
    T_WC[:3, 3] = t_WC # 把 XYZ 平移塞进右上角的第 4 列
    # SLAM 给出的是“相机在世界里的位姿 (T_WC)”，我们需要的是“世界在相机里的位姿 (T_CW)”，所以用 np.linalg.inv 求逆！
    return np.linalg.inv(T_WC) 

def project_to_pixel(x_state, P_model, T_CW):
    """
    【针孔相机投影模型】 (3D 坐标 -> 2D 像素)
    x_state: 优化器当前正在“瞎猜”的标志牌 6-DoF 位姿 [tx, ty, tz, rx, ry, rz]
    P_model: 标志牌 4 个角点的局部 3D 坐标 (就是上面的 P_S)
    T_CW: 当前这一帧相机在世界里的绝对位姿矩阵
    """
    t_WS = x_state[0:3] # 获取猜测的标志牌平移向量 (Translation World-to-Sign)
    r_WS = R.from_rotvec(x_state[3:6]).as_matrix() # 获取猜测的标志牌旋转向量，并转成 3x3 旋转矩阵
    
    # 步骤1: 把标志牌的 4 个角点，从“局部系”搬到“世界系” (乘以旋转矩阵，加上平移)
    P_W = np.dot(P_model, r_WS.T) + t_WS
    
    # 步骤2: 把这 4 个世界角点，搬到当前的“相机坐标系”里 (应用 T_CW)
    P_C = np.dot(P_W, T_CW[:3, :3].T) + T_CW[:3, 3]
    
    # 步骤3: 针孔相机透视投影。把相机的 3D 点拍成 2D 的照片像素点。
    Z_C = np.maximum(P_C[:, 2], 1e-6) # 获取 3D 点在相机系下的 Z 轴深度。加个 1e-6 是防止被 0 除导致程序崩溃。
    # 公式：像素 u = 焦距 fx * (X / Z) + 光心 cx ; 像素 v = 焦距 fy * (Y / Z) + 光心 cy
    u = FX * (P_C[:, 0] / Z_C) + CX
    v = FY * (P_C[:, 1] / Z_C) + CY
    return np.column_stack((u, v)) # 把算出来的 u 和 v 拼成两列返回 (形状: 4行2列)

def get_inverse_homography(x_state, T_CW):
    """
    【黑科技：逆向单应矩阵 (Inverse Homography) 计算】
    作用：把照片上 2D 的文字线条，像放幻灯片一样，反向“投影”回到 3D 的标志牌铁皮平面上。
    """
    t_WS = x_state[0:3]
    r_WS = R.from_rotvec(x_state[3:6]).as_matrix()
    T_WS = np.eye(4) # 初始化标志牌到世界系的 4x4 变换矩阵
    T_WS[:3, :3] = r_WS
    T_WS[:3, 3] = t_WS

    # T_CS = T_CW 乘以 T_WS。这一步直接算出了“标志牌局部系”到“相机坐标系”的直接变换矩阵。
    T_CS = np.dot(T_CW, T_WS) 
    # K 是相机的内参矩阵 (3x3)
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]]) 
    
    # 【降维魔法】：因为标志牌是个平面，所有点在局部系下的 Z 坐标都是 0！
    # 所以 T_CS 矩阵的第 3 列（负责 Z 轴的列）完全没用。我们把它抽掉，把 3x4 变成 3x3 矩阵 M。
    M = np.column_stack((T_CS[:3, 0], T_CS[:3, 1], T_CS[:3, 3]))
    H_CS = np.dot(K, M) # 内参 K 乘以矩阵 M，得到正向单应矩阵 H_CS (从标志牌 3D 变 2D 图像)
    
    try:
        return np.linalg.inv(H_CS) # 我们要求的是从 2D 变回 3D 标志牌，所以对 H_CS 求逆矩阵！
    except np.linalg.LinAlgError:
        return np.eye(3) # 万一发生极其罕见的矩阵不可逆情况（奇异），返回单位矩阵保命。

def pixel_to_sign_plane(u, v, H_inv):
    """
    接收像素坐标 (u, v) 和上面算出来的逆矩阵 H_inv。
    吐出这个像素点在真实 3D 标志牌物理表面上的 (X, Y) 坐标（单位：米）。
    """
    p_img = np.array([u, v, 1.0]) # 把 2D 像素变成齐次坐标 (补个1)
    p_sign = np.dot(H_inv, p_img) # 乘上逆矩阵，打回 3D 物理空间
    # 齐次坐标归一化（除以第三个元素），得到真实的 X 和 Y 米数。
    return p_sign[0] / p_sign[2], p_sign[1] / p_sign[2]

# =========================================================
# 4. 构造四重联合优化误差函数 (LM 优化器的“惩罚考卷”)
# =========================================================
def joint_residuals(x, clear_frames, blur_frames, plane_prior):
    """
    x: 优化器当前胡乱猜测的标志牌 6-DoF 位姿向量 [tx, ty, tz, rx, ry, rz]
    clear_frames: 最后两帧极度清晰的数据（包含角点和文字红绿线）
    blur_frames: 前面比较模糊的数据（只包含 YOLO 的边界框）
    plane_prior: SLAM 算出来的背景墙方程式
    返回值 res: 是一条长长的一维数组，里面装满了所有惩罚项的误差值。优化器的目标是把这个数组里所有的数字压榨到 0。
    """
    res = [] # 初始化空数组，用来装载所有的误差/惩罚
    
    # --- 约束 A: 几何锚点重投影约束 ---
    for f in clear_frames: # 遍历清晰帧
        pts_hat = project_to_pixel(x, P_S, f['T_CW']) # 根据当前猜的姿态 x，算出标志牌角点应该在照片的什么位置
        # f['corners'] 是真实照片里的角点位置。
        # 误差 = (真实位置 - 猜的位置)。展平后乘以权重 W_REPROJ。如果猜错了，这个差值就会很大。
        res.append((f['corners'] - pts_hat).flatten() * W_REPROJ)
        
        # --- 约束 B: 【核心原创】基于文字线条的正交物理死锁 ---
        if 'h_lines' in f and 'v_lines' in f: # 如果这帧图提取出了文字线条
            H_inv = get_inverse_homography(x, f['T_CW']) # 计算当前猜的姿态下的“反推矩阵”
            
            # 【惩罚横线歪斜】
            for u1, v1, u2, v2, weight in f['h_lines']: # 遍历每一条红色的横线
                _, Y1 = pixel_to_sign_plane(u1, v1, H_inv) # 算出左端点反推到 3D 平板上的 Y(高度) 坐标
                _, Y2 = pixel_to_sign_plane(u2, v2, H_inv) # 算出右端点反推到 3D 平板上的 Y(高度) 坐标
                # 理论上，横线在 3D 物理世界中必须是绝对水平的，即左边和右边的高度必须相等 (Y1 == Y2)。
                # 误差 = (Y1 - Y2) * 线条自身长度权重 * 极端高压惩罚权重(W_ORTHO)。
                res.append(np.array([(Y1 - Y2) * weight * W_ORTHO]))
                
            # 【惩罚竖线歪斜】
            for u1, v1, u2, v2, weight in f['v_lines']: # 遍历每一条绿色的竖线
                X1, _ = pixel_to_sign_plane(u1, v1, H_inv) # 算出上端点反推到 3D 平板上的 X(左右) 坐标
                X2, _ = pixel_to_sign_plane(u2, v2, H_inv) # 算出下端点反推到 3D 平板上的 X(左右) 坐标
                # 竖线的上下端点在 3D 空间必须一样宽 (X1 == X2)。
                res.append(np.array([(X1 - X2) * weight * W_ORTHO]))
        
    # --- 约束 C: 语义视锥约束 (宏观粗定位) ---
    for f in blur_frames: # 遍历模糊帧
        pts_hat = project_to_pixel(x, P_S, f['T_CW']) # 算出 4 个角点的像素位置
        # 根据算出来的 4 个点，求出一个最小包围矩形框 b_hat [xmin, ymin, xmax, ymax]
        b_hat = np.array([np.min(pts_hat[:,0]), np.min(pts_hat[:,1]), 
                          np.max(pts_hat[:,0]), np.max(pts_hat[:,1])])
        # 惩罚：优化器猜出来的这个框，必须跟 YOLO 神经网络识别出来的框 (f['bbox']) 尽量重合。
        res.append((f['bbox'] - b_hat) * W_BBOX)
        
    # --- 约束 D: 背景物理平面托底 (防深度爆炸) ---
    if plane_prior:
        n, d = plane_prior['n'], plane_prior['d'] # n 是背景墙的法向量，d 是偏置常数
        # 点到平面的距离公式：法向量 n 点乘 中心点坐标 x[0:3]，加上 d。
        dist = np.dot(n, x[0:3]) + d 
        # 惩罚：标志牌中心点到背景墙平面的距离，必须尽量为 0 (即贴在墙上)。
        res.append(np.array([dist]) * W_STRUCT) 
        
    return np.concatenate(res) # 把所有几十上百个惩罚项首尾相连拼成一个超级长的一维数组，返回给优化器

# =========================================================
# 5. 主程序：数据对齐与开始求解
# =========================================================
def main():
    print(">>> 正在初始化数据与特征提取器...")
    
    # 1. 杂乱数据读取
    traj_data = np.loadtxt(TRAJ_FILE) # 读轨迹文件
    traj_dict = {row[0]: row[1:] for row in traj_data} # 把轨迹文件变成字典，时间戳为键，位姿为值
    mapping = pd.read_csv(MAP_FILE)   # 读映射表
    bboxes_df = pd.read_csv(BBOX_FILE, comment='#', header=None, 
                            names=['filename', 'u_min', 'v_min', 'u_max', 'v_max']) # 读 YOLO 框
    
    blur_frames, clear_frames = [], [] # 初始化模糊帧和清晰帧容器
    all_ts = np.array(list(traj_dict.keys())) # 提取所有 SLAM 轨迹的时间戳，转成数组方便查询

    # 2. 时空对齐（把独立的数据打包成一个个 Frame 对象）
    for i, row in bboxes_df.iterrows(): # 循环遍历 YOLO 表里的每一行图片
        match = mapping[mapping['filename'] == row['filename']] # 去映射表里找这张图片的物理时间戳
        if match.empty: continue # 找不到就跳过
        
        ts = match.iloc[0]['timestamp_s'] # 拿到相机的曝光时间戳
        # 找 SLAM 轨迹里，和相机曝光时间戳最接近的那一个时刻 (np.abs 算绝对差值，np.argmin 找最小的索引)
        closest_ts = all_ts[np.argmin(np.abs(all_ts - ts))] 
        T_CW = get_T_CW(traj_dict[closest_ts]) # 把找到的四元数位姿，转成 T_CW 矩阵
        bbox = np.array([row['u_min'], row['v_min'], row['u_max'], row['v_max']]) # 解析出 YOLO 框坐标
        
        # 分流逻辑：除了最后两帧，其他都当做远距离的模糊帧
        if i < len(bboxes_df) - 2:
            blur_frames.append({'T_CW': T_CW, 'bbox': bbox}) # 模糊帧只存位姿和粗略的框
        else:
            # 最后两帧当做极度靠近的清晰帧，开始干重活
            img_path = os.path.join(IMAGE_DIR, row['filename']) # 拼出真实图像的电脑硬盘路径
            img = cv2.imread(img_path) # 用 OpenCV 读取这张真实图像
            
            # 【调用核心函数】把图和 YOLO 框传进去，抠出汉字的红绿线。
            h_lines, v_lines = extract_subpixel_orthogonal_lines(img, roi_bbox=bbox)
            
            # 因为没有人工标定，所以暂时借用 YOLO 框的 4 个顶角，冒充标志牌真实的物理尖角
            obs_corners = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], 
                                    [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
            
            # 把这些极其宝贵的精确数据，全部打包塞进清晰帧列表里
            clear_frames.append({
                'T_CW': T_CW, 
                'corners': obs_corners,
                'h_lines': h_lines, 
                'v_lines': v_lines
            })

    # 这是你在另一个模块用 RANSAC 算出来的背景大楼/电线杆平面方程
    plane_prior = {'n': np.array([0, 0, 1]), 'd': -12.5} 

    # 3. 设定 LM 优化器的初始“瞎猜”点 (Initial Guess x0)
    # x0 的格式是 6 个未知数：[平移X, 平移Y, 平移Z, 旋转向量Rx, 旋转向量Ry, 旋转向量Rz]
    last_T_WC = np.linalg.inv(clear_frames[-1]['T_CW']) # 获取最后一帧相机在世界里的绝对位姿
    # 初始化平移猜测：让标志牌乖乖待在相机最后一帧位置的正前方 (Z轴方向) 8 米处。
    init_t = last_T_WC[:3, 3] + last_T_WC[:3, 2] * 8.0 
    x0 = np.concatenate([init_t, [0, 0, 0]]) # 旋转向量初始化为 [0,0,0]，代表完全不旋转的完美正对姿态。

    print(f">>> 初始深度猜测: {init_t[2]:.2f}米. 开始图优化...")
    
    # 4. 🚀 启动火箭：调用 SciPy 的最小二乘优化器
    # method='lm': 使用 Levenberg-Marquardt 算法 (SLAM 界最经典的非线性求解器)
    # joint_residuals: 把我们写的惩罚函数传进去
    # args: 把我们准备好的数据 (清晰帧，模糊帧，背景墙) 全塞进惩罚函数里
    res = least_squares(joint_residuals, x0, args=(clear_frames, blur_frames, plane_prior),
                        method='lm', xtol=1e-8, ftol=1e-8, verbose=1)

    # 5. 收获成果
    x_opt = res.x # .x 属性里存的就是经过成百上千次试错后，误差最小的终极 6-DoF 答案！
    print("\n" + "="*60)
    print("🎉 交通标志牌 6-DoF 位姿联合优化成功！")
    print("="*60)
    print(f"📍 世界系中心坐标 (X, Y, Z) [米]: \n   {x_opt[0]:.4f}, {x_opt[1]:.4f}, {x_opt[2]:.4f}")
    print(f"📐 标志牌朝向 (旋转矩阵 R_WS): \n{R.from_rotvec(x_opt[3:6]).as_matrix()}") # 把输出的 3位旋转向量，变回人类能看懂的 3x3 旋转矩阵
    print("="*60)

    # =========================================================
    # 6. 3D 可视化 (利用 Matplotlib 画出上帝视角的态势图，给导师看的)
    # =========================================================
    print(">>> 正在生成 3D 态势感知图...")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection # 用于在 3D 空间里画实体的多边形色块

    fig = plt.figure(figsize=(10, 8)) # 创建一块 10x8 的画布
    ax = fig.add_subplot(111, projection='3d') # 添加一个支持 3D 绘图的坐标轴

    # --- 6.1 画出 SLAM 提供的车辆（相机）行驶轨迹 ---
    cam_xs, cam_ys, cam_zs = [], [], [] # 准备三个空列表装载轨迹点
    for f in blur_frames + clear_frames: # 遍历所有的帧
        T_WC = np.linalg.inv(f['T_CW']) # 转回世界坐标系
        cam_xs.append(T_WC[0, 3]) # 提取世界系下的 X 位置
        cam_ys.append(T_WC[1, 3]) # 提取世界系下的 Y 位置
        cam_zs.append(T_WC[2, 3]) # 提取世界系下的 Z 位置

    # 画一条蓝色的虚线连起所有轨迹点
    ax.plot(cam_xs, cam_ys, cam_zs, label='Camera Trajectory', color='blue', marker='.', linestyle='dashed', alpha=0.7)
    # 在起点画个大大的绿色正方形
    ax.scatter(cam_xs[0], cam_ys[0], cam_zs[0], color='green', s=100, marker='s', label='Start')
    # 在终点画个大大的橙色三角形
    ax.scatter(cam_xs[-1], cam_ys[-1], cam_zs[-1], color='orange', s=100, marker='^', label='End')

    # --- 6.2 画出算出来的 3D 标志牌实体 ---
    t_opt = x_opt[0:3] # 获取最优解里的 XYZ 位置
    R_opt = R.from_rotvec(x_opt[3:6]).as_matrix() # 获取最优解里的旋转矩阵
    # 利用 P_S 局部坐标系，推算出这块 1.26*0.6 米的铁皮的 4 个角，在真实世界里的三维绝对坐标
    sign_corners_world = np.dot(P_S, R_opt.T) + t_opt 

    # 将 4 个点的三维坐标连起来，生成一个红色的、带黑边的 3D 多边形面片 (Poly3D)
    verts = [list(zip(sign_corners_world[:, 0], sign_corners_world[:, 1], sign_corners_world[:, 2]))]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.8, facecolor='red', edgecolor='black', linewidths=2))

    # --- 6.3 调整坐标轴比例，防止 3D 图像被拉长畸变 ---
    # 算出 XYZ 三个轴里跨度最大的那个距离，然后让所有轴的显示范围都等于这个最大跨度。
    # 这样就能实现 3D 空间里的 1:1:1 绝对物理比例展示，牌子才不会被画扁。
    max_range = np.array([np.max(cam_xs)-np.min(cam_xs), np.max(cam_ys)-np.min(cam_ys), np.max(cam_zs)-np.min(cam_zs), 5.0]).max() / 2.0
    mid_x, mid_y, mid_z = (np.max(cam_xs)+np.min(cam_xs))*0.5, (np.max(cam_ys)+np.min(cam_ys))*0.5, (np.max(cam_zs)+np.min(cam_zs))*0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 打上坐标轴的单位标签 (m 表示米)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.legend(loc='upper left') # 在左上角显示图例
    
    # elev=20表示仰角20度，azim=-45表示侧转45度，这是一种能完美呈现纵深感的老司机御用视角。
    ax.view_init(elev=20, azim=-45) 
    plt.show() # 弹出画布展示图像！

# 如果当前脚本是直接运行的主程序，则触发 main() 函数
if __name__ == "__main__":
    main()