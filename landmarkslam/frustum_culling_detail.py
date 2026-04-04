# ==============================================================================
# 导入底层数学与工具库 (绝对纯净，已彻底剔除 sklearn 等过度封装的黑盒依赖)
# ==============================================================================
import numpy as np               # [核心] 提供矩阵乘法、奇异值分解(SVD)、协方差计算等底层线性代数支持
import pandas as pd              # [辅助] 仅用于高效解析 CSV 格式的时间戳-图片映射表，便于序列对齐
from scipy.optimize import least_squares  # [核心] 提供 Levenberg-Marquardt (LM) 非线性图优化求解器，用于最小二乘逼近
from scipy.spatial.transform import Rotation as R # [核心] 提供四元数(Quaternion)到旋转矩阵(Rotation Matrix)及欧拉角的严密转换
import os                        # [辅助] 用于处理跨平台的文件路径、创建输出文件夹
import cv2                       # [核心] 提供图像 I/O、色彩空间转换、CLAHE 对比度均衡化及 LSD 亚像素直线提取
import math                      # [辅助] 提供高精度的标量数学计算 (如 atan2 求平面上的绝对方位倾角)
import matplotlib.pyplot as plt  # [核心] 提供基于渲染后端的 3D 态势感知图的交互式可视化绘制

# =========================================================
# 1. 核心物理与路径配置区 
# =========================================================
# SLAM 算法运行后输出的相机轨迹文件，记录了每一帧对应的相机世界位姿 (T_WC)
TRAJ_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/output/FrameTrajectory_TUM_20260331_154838.txt" 
# 图片文件名与系统绝对时间戳的对齐表，用于将 YOLO 图片与 SLAM 轨迹进行时间同步
MAP_FILE  = "/home/zah/ORB_SLAM3-master/landmarkslam/output/Filename_Mapping_20260331_154838.csv"     
# YOLO 神经网络输出的 2D 目标检测像素框文件 (u_min, v_min, u_max, v_max)
BBOX_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/yolo_simulated_bboxes.txt" 
# 暗光原始图像序列库的路径，用于后续提取亚像素线段
IMAGE_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/"              
# 算法运行日志与最终 3D 渲染图的落盘目录
DEBUG_OUT_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/output/debug_vis_pipeline/"    

# 【核心解耦点】引入真实的 SLAM 3D 稀疏点云文件 (.ply)。这是打破单目相机尺度不可观 (Scale Ambiguity) 魔咒的物理钥匙
POINT_CLOUD_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/output/PointCloud_20260331_154838.ply"

# 相机内参矩阵 K 的核心参数
FX, FY = 501.00685446, 496.63593447 # 焦距 (Focal Length)，单位为像素，决定了透视畸变的剧烈程度
CX, CY = 316.00266456, 233.80218648 # 光心坐标 (Principal Point)，单位为像素，决定了图像投影的绝对中心偏移

# 交通标志牌在真实物理世界中的绝对出厂尺寸 (单位：米)
WIDTH, HEIGHT = 1.26, 0.6   
# 计算一半的宽高，用于构建以标志牌几何中心为原点的局部坐标系
HALF_W, HALF_H = WIDTH / 2.0, HEIGHT / 2.0 

# 定义标志牌在自身的“局部绝对坐标系 (Local Frame)”下的四个物理顶角坐标。
# 物理先验：标志牌绝对平整，因此局部 Z 轴坐标恒等于 0
P_S = np.array([
    [-HALF_W, -HALF_H, 0.0], # 顶点 0：左上角
    [ HALF_W, -HALF_H, 0.0], # 顶点 1：右上角
    [ HALF_W,  HALF_H, 0.0], # 顶点 2：右下角
    [-HALF_W,  HALF_H, 0.0]  # 顶点 3：左下角
])

# 信息矩阵 (Information Matrix) 对角线权重分配，用于控制 LM 优化器在四种残差冲突时的妥协倾向
W_REPROJ = 1.0   # 角点重投影约束：权重高。利用多视图基线(Baseline)的三角化原理，微调 X/Y 平移
W_BBOX   = 0.02  # 视锥限位约束：权重极低。仅充当软性“引力盆”，在深度猜到几百米外时把它硬拽回来
W_STRUCT = 0.5   # 深度托底约束：权重中。依靠 RANSAC 提取的物理背景墙面方程，斩断单目深度的无穷发散
W_ORTHO  = 10.0  # 微观正交死锁：权重极高。汉字笔画的绝对正交是人类文明真理，对倾斜角(Pitch/Roll)具有最高裁决权

# 全局状态机变量，用于控制在 LM 优化器进行第 0 次盲猜(迭代)时，打印一次极其详细的内部矩阵状态，防止日志刷屏
GLOBAL_EVAL_COUNT = 0 

# =========================================================
# 2. 亚像素正交线段提取器 (专为极其恶劣的暗光/逆光场景定制)
# =========================================================
def extract_subpixel_orthogonal_lines(image, roi_bbox, mprint, frame_name="debug", out_dir=None):
    """
    功能：从暗光图片中，提取标志牌区域内的绝对横线和竖线。
    image: 输入的 BGR 原始图像矩阵
    roi_bbox: YOLO 检测出的边界框 [u_min, v_min, u_max, v_max]，用于过滤背景杂线
    mprint: 日志打印回调函数
    """
    scale_factor = 2.5 # 图像无损放大倍数。通过放大，让 LSD 算法能在更大的像素网格上跑，从而榨取亚像素级别的边缘梯度
    angle_tol = 15     # 几何容差。只要线段的角度偏差超过绝对水平/垂直 15 度，就视为干扰(树枝/撇捺)并抛弃
    min_length = 3.0   # 像素长度阈值。过滤掉高频噪点引起的极短碎线
    
    # 【预处理 1】将三通道彩图转为单通道灰度图，因为边缘提取只看亮度梯度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    
    # 【极暗光抢救 A】Gamma 非线性逆向提亮。通过 1.0/2.5 的指数拉伸，把被黑暗吞噬的低灰度值强行拉亮
    inv_gamma = 1.0 / 2.5 
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(gray, table) # 使用查表法 (Look-Up Table) 极速完成全图亮度映射
    
    # 【极暗光抢救 B】CLAHE (对比度受限的自适应直方图均衡)。在 8x8 的局部网格内增强对比度，让字迹的白色边缘从黑色背景中凸显
    img_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(img_gamma) 
    
    # 【亚像素准备】读取图像的高宽，并使用最高质量的三次样条插值 (INTER_CUBIC) 进行强制放大
    h, w = img_clahe.shape 
    img_upscaled = cv2.resize(img_clahe, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
    # 放大后必然产生锐利的锯齿，使用 3x3 高斯模糊进行轻微平滑，防止 LSD 算法沿着锯齿提取出一万根碎线
    img_upscaled = cv2.GaussianBlur(img_upscaled, (3, 3), 0)

    # 启动 OpenCV 封装的顶级直线段检测器 LSD (Line Segment Detector)
    lsd = cv2.createLineSegmentDetector(0) 
    # lines_up 存放的是放大图上的线段坐标数组 [ [[x1,y1,x2,y2]], [[x1,y1,x2,y2]], ... ]
    lines_up, _, _, _ = lsd.detect(img_upscaled) 
    
    # 分别用来存放合格的横线和竖线
    horiz_lines, vert_lines = [], [] 
    raw_count = len(lines_up) if lines_up is not None else 0
    mprint(f"    ├─ [LSD 原始提取]: 升维(x{scale_factor})后共发现 {raw_count} 条微观梯度线")
    
    if lines_up is not None: 
        for line in lines_up: 
            # 【核心操作】将提取到的坐标除以放大倍数，强行缩放回原图尺寸，此时的 x,y 会变成带小数点的浮点数，实现“亚像素”
            x1, y1, x2, y2 = line[0] / scale_factor 
            
            # 【空间掩码过滤】只有线段的中点落在 YOLO 检测框 roi_bbox 内部时，才被认为是标志牌上的文字线段
            if roi_bbox is not None:
                xmin, ymin, xmax, ymax = roi_bbox 
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0 # 计算线段中点
                # 如果中点不在 YOLO 框内，直接 continue 跳过
                if not (xmin <= cx <= xmax and ymin <= cy <= ymax): continue
            
            # 【长度滤波】使用勾股定理计算线段在原图上的真实像素长度
            length = math.hypot(x2 - x1, y2 - y1)
            if length < min_length: continue # 太短的抛弃
                
            # 【几何角度计算】利用 atan2 计算从 (x1,y1) 到 (x2,y2) 向量相对于图像 X 轴的绝对倾角，并映射到 0~180 度区间
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180.0
            
            # 【双主轴聚类】根据计算出的绝对角度，将线段归类到“横线堆”或“竖线堆”
            # 横线判断：角度接近 0 度，或者接近 180 度
            if angle <= angle_tol or angle >= (180 - angle_tol):
                horiz_lines.append((x1, y1, x2, y2, length)) 
            # 竖线判断：角度接近 90 度
            elif abs(angle - 90) <= angle_tol:
                vert_lines.append((x1, y1, x2, y2, length))  
                
    mprint(f"    └─ [双主轴正交聚类]: 过滤背景与撇捺后，保留有效横线 {len(horiz_lines)} 条，竖线 {len(vert_lines)} 条")
    return horiz_lines, vert_lines

# =========================================================
# 3. 多视图几何推导引擎 
# =========================================================
def get_T_CW(pose_tum):
    """
    功能：坐标系逆变换。
    背景：SLAM 系统的轨迹文件默认给出的是 T_WC (世界系到相机系的变换)，即相机在世界里的位置。
    但在做 3D 投影时，我们需要把世界坐标拉到相机面前算，因此需要获取严格的逆矩阵 T_CW = T_WC^-1。
    """
    # 拆分传入的7维向量：前三维是世界系下的平移 t，后四维是世界系下的旋转四元数 q
    t_WC, q_WC = pose_tum[0:3], pose_tum[3:7] 
    
    # 初始化一个 4x4 的单位齐次变换矩阵
    T_WC = np.eye(4) 
    # 使用 Scipy 的 Rotation 库，将四元数安全地解析为 3x3 的刚体旋转矩阵 R
    T_WC[:3, :3] = R.from_quat(q_WC).as_matrix() 
    # 将平移向量填入第四列
    T_WC[:3, 3] = t_WC
    
    # 矩阵求逆，得到我们需要的 T_CW
    return np.linalg.inv(T_WC) 

def get_inverse_homography(x_state, T_CW, mprint=None):
    """
    【神级推导】生成逆向单应映射矩阵 H_inv。
    原理：这套算法的灵魂！当优化器猜测了一个标志牌的 6-DoF 位姿 (x_state) 后，这就定义了一个 3D 虚拟平面。
    通过计算单应矩阵 H 的逆 H_inv，我们可以把照片上那些带有严重透视畸变的 2D 文字线段，
    像剥洋葱一样，直接逆向“拍扁”回这个纯平的 3D 铁皮面上！
    如果 x_state 猜对了，映射回去的线一定是绝对横平竖直的；如果猜错了，线就是歪的，从而产生巨大误差惩罚！
    """
    # 解析优化器当前试错的 6 维状态向量：前 3 维是标志牌在世界系的平移，后 3 维是旋转向量(李代数)
    t_WS = x_state[0:3]
    # 将 3 维旋转向量解析为 3x3 旋转矩阵
    r_WS = R.from_rotvec(x_state[3:6]).as_matrix()
    
    # 构造 T_WS：代表局部标志牌坐标系 (Sign) 到 全局世界坐标系 (World) 的变换矩阵
    T_WS = np.eye(4) 
    T_WS[:3, :3] = r_WS; T_WS[:3, 3] = t_WS
    
    # 【链式法则】相乘得到 T_CS = T_CW * T_WS。这个矩阵表示：标志牌相对于当前相机光心的相对位姿！
    T_CS = np.dot(T_CW, T_WS) 
    
    # 构造相机的 3x3 内参矩阵 K
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]]) 
    
    # 【平面单应核心定理】因为标志牌是一个纯平铁皮，所有字都在局部的 Z=0 平面上。
    # 根据多视图几何定理，3D投影到2D的公式中，T_CS 的第三列 (Z轴旋转) 的作用被抹零了。
    # 因此，把 T_CS 的第 1、2 列(X,Y轴旋转)和第 4 列(平移)拼接起来，形成一个 3x3 的矩阵 M
    M = np.column_stack((T_CS[:3, 0], T_CS[:3, 1], T_CS[:3, 3]))
    
    try:
        # 单应映射矩阵 H = 相机内参 K * 几何变换 M
        H = np.dot(K, M)
        # 求逆，得到从 2D 像素反推回 3D 铁皮坐标的逆单应矩阵 H_inv
        H_inv = np.linalg.inv(H)
        
        # 仅在第一次盲猜时打印这些底层的变换矩阵，供人工 Debug
        if mprint:
            mprint(f"      [矩阵] T_CS (相机到局部系): \n{np.round(T_CS, 4)}")
            mprint(f"      [矩阵] H_inv (逆单应映射): \n{np.round(H_inv, 6)}")
        return H_inv
        
    except np.linalg.LinAlgError:
        # 【极度安全防御】如果优化器随机游走产生了一个奇异矩阵(行列式为0)无法求逆，
        # 直接返回 3x3 单位阵，防止抛出 LinAlgError 导致整个进程暴毙
        return np.eye(3) 

def pixel_to_sign_plane(u, v, H_inv):
    """
    核心工具函数：单应透视除法 (Homogeneous Perspective Division)。
    给它一个 2D 像素点的坐标 (u, v) 以及逆映射矩阵 H_inv，它就能算出这个像素点在这个物理铁皮面上的 X, Y 坐标(单位是米)。
    """
    # 将普通的 2D 坐标扩展为齐次坐标 [u, v, 1.0]
    p_img = np.array([u, v, 1.0]) 
    # 矩阵乘法：得到 3D 齐次向量 p_sign
    p_sign = np.dot(H_inv, p_img) 
    
    # 🛡️ 极小常量 1e-8 防御：在齐次坐标转回笛卡尔坐标时，必须除以第三个元素(齐次缩放因子)。
    # 防止非线性优化器在乱转时，恰好把平面转到了通过相机光心的地方(导致分母为 0)，从而引发除以零的硬件崩溃！
    epsilon = 1e-8
    
    # 透视除法，返回铁皮面上的物理坐标 X 和 Y
    return p_sign[0] / (p_sign[2] + epsilon), p_sign[1] / (p_sign[2] + epsilon)

# =========================================================
# 【绝无造假·绝对物理防线】真实的 PLY 点云读取 + 视锥截取 + 3D PCA 拟合
# =========================================================
def load_and_fit_background_plane(ply_file, T_CW, bbox, mprint):
    """
    功能：从包含几千个杂乱点的 SLAM 全局点云里，精准“手术刀式”切除出贴在标志牌背后的那堵墙的平面方程。
    没有任何封装黑盒，纯靠物理几何推导！
    ply_file: 导出的全局点云文件
    T_CW: 目标帧相机的绝对位姿
    bbox: 目标帧中 YOLO 给出的标志牌 2D 边界框
    """
    mprint(f"\n⚙️ [真实点云解析] 正在执行语义视锥截取与 PCA 主成分平面拟合...")
    
    # 文件检查防线
    if not os.path.exists(ply_file): 
        mprint(f"   ❌ 找不到点云文件: {ply_file}")
        return None

    try:
        # 1. 智能剥离 PLY 文件的 ASCII 头 (Header)
        with open(ply_file, 'r') as f:
            lines = f.readlines()
        # 找到 "end_header" 这一行，下面的全是坐标数据
        data_start_idx = next(i for i, line in enumerate(lines) if "end_header" in line) + 1
        
        # 将纯数字读取进 numpy 数组，形状为 (N, 3)，此时的点云处于世界坐标系下
        global_points_3d = np.loadtxt(lines[data_start_idx:])
        
        # 2. 🌟 视锥投影截取 (Frustum Culling) 🌟
        # 这是本系统极为神妙的一步：用 2D 的语义框，去空间里套取对应的 3D 点云！
        
        # [步骤 A] 外参刚体变换：把世界系的点云，全部挪到当前相机的坐标系下 (P_C)
        P_C = np.dot(global_points_3d, T_CW[:3, :3].T) + T_CW[:3, 3]
        
        # [步骤 B] Z 轴深度裁剪防线：剔除所有跑到相机背后、或者极其贴近相机的点 (Z < 0.1 米)
        valid_z_mask = P_C[:, 2] > 0.1
        P_C_valid = P_C[valid_z_mask]
        P_W_valid = global_points_3d[valid_z_mask] # 同步保留这些点在世界系下的坐标，后面拟合要用
        
        # [步骤 C] 内参透视投影：将相机前方所有的 3D 点，按针孔模型拍扁成当前照片上的 2D 像素 (u, v)
        u = FX * (P_C_valid[:, 0] / P_C_valid[:, 2]) + CX
        v = FY * (P_C_valid[:, 1] / P_C_valid[:, 2]) + CY
        
        # [步骤 D] 语义掩码膨胀：读取 YOLO 检测框
        u_min, v_min, u_max, v_max = bbox
        w, h = u_max - u_min, v_max - v_min
        # 因为标志牌本身是黑的提不出特征，背景墙通常在标志牌周围一圈，所以把框向外四周膨胀 30%
        roi_u_min = u_min - 0.3 * w
        roi_u_max = u_max + 0.3 * w
        roi_v_min = v_min - 0.3 * h
        roi_v_max = v_max + 0.3 * h
        
        # [步骤 E] 视锥剔除：利用布尔索引数组，只保留投影后刚好落在这个膨胀框里面的那些 3D 点
        roi_mask = (u >= roi_u_min) & (u <= roi_u_max) & (v >= roi_v_min) & (v <= roi_v_max)
        
        # 提取出最终被框住的物理点云 (提取出的是世界坐标)
        roi_points_3d = P_W_valid[roi_mask]

        mprint(f"   ✅ 视锥截取成功！全局 {len(global_points_3d)} 个点 -> 局部周边结构 {len(roi_points_3d)} 个点！")

        # 若截取后发现点太少，说明背后可能是蓝天白云(提不出特征)，直接返回 None，不强求拟合
        if len(roi_points_3d) < 10: 
            mprint("   ❌ 视锥内特征点过少(可能是蓝天背景)，放弃拟合！启用降级托底模式。")
            return None

        # 3. 纯 3D RANSAC 内点搜索：抵抗飞鸟、路灯等严重离群点(Outliers)的干扰
        best_inliers_mask = None
        max_inliers = 0
        for _ in range(300): # 掷骰子 300 次
            # 从这几十个点里，随机盲抽 3 个不同的点
            idx = np.random.choice(len(roi_points_3d), 3, replace=False)
            p1, p2, p3 = roi_points_3d[idx]
            
            # 【纯物理计算】利用空间向量的叉乘 (Cross Product)，算出垂直于这 3 个点的法向量 n
            # 这种做法比 2D 回归好一万倍，因为它不受任何视角的坐标轴限制，具有绝对的 3D 旋转不变性！
            n = np.cross(p2 - p1, p3 - p1)
            
            # 奇异性拦截：如果抽到的三个点恰好在一条直线上，叉乘结果为零向量，拦截以防后续除以 0
            if np.linalg.norm(n) < 1e-6: continue
            
            # 将法向量的长度归一化为 1
            n = n / np.linalg.norm(n)
            # 通过带入 p1 点，反推出这个平面方程 (nx*X + ny*Y + nz*Z + d = 0) 中的截距 d
            d = -np.dot(n, p1)
            
            # 向量点乘运算：批量计算视锥里所有点到这个平面的正交垂直距离
            distances = np.abs(np.dot(roi_points_3d, n) + d)
            
            # 距离小于 0.15 米 (15厘米物理厚度) 的点，我们认为是“同一堵墙”的内点 (Inliers)
            mask = distances < 0.15 
            inliers = np.sum(mask) # 统计内点的数量
            
            # 像打擂台一样，保留拥有最多支持者的那个平面所对应的掩码
            if inliers > max_inliers:
                max_inliers = inliers
                best_inliers_mask = mask

        # 🛡️ 降维崩溃防御：如果 300 次抽签全军覆没，连 3 个刚好在一面墙上的点都凑不出来，直接放弃！
        if best_inliers_mask is None or np.sum(best_inliers_mask) < 3:
            mprint("   ❌ RANSAC 失败：未能找到超过 3 个有效共面点！系统降级。")
            return None
                
        # 4. SVD / PCA 主成分分析精修 (消除单次抽样带来的随机噪声)
        # 提取出刚刚赢得擂台赛的所有“内点” (可能是几十上百个点)
        inlier_points = roi_points_3d[best_inliers_mask]
        
        # 计算这群内点的空间几何质心 (Center of Mass)
        center = np.mean(inlier_points, axis=0) 
        # 计算这群点在 X、Y、Z 三个方向上的分布协方差矩阵 (Covariance Matrix)
        cov_matrix = np.cov(inlier_points.T)    
        
        # 对 3x3 的协方差矩阵进行特征值分解 (SVD / Eigen Decomposition)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 【最高级别数学操作】找到散布程度最弱 (即特征值最小) 的那个维度方向，
        # 这个特征向量 (eigenvector) 就是综合了所有上百个内点算出来的、最平滑、最准确的物理面法向量 n！
        refined_n = eigenvectors[:, np.argmin(eigenvalues)] 
        # 根据质心，反推出最终严谨的距离 d
        refined_d = -np.dot(refined_n, center)
                
        # 物理法向二义性纠正：PCA 算出的法向量可能朝里也可能朝外。
        # 我们用相机光心减去质心构造一个视线向量，用法向点乘视线，如果是负数，说明法向背对着相机，立刻取反纠正。
        cam_pos = np.linalg.inv(T_CW)[:3, 3]
        if np.dot(refined_n, cam_pos - center) < 0:
            refined_n, refined_d = -refined_n, -refined_d

        mprint(f"   ✅ PCA/SVD 精拟合成功！内点数: {len(inlier_points)}/{len(roi_points_3d)}")
        mprint(f"   -> 物理法向量 n: {np.round(refined_n, 4)} | 绝对深度 d: {refined_d:.4f} m")
        
        # 将代表这面物理墙体的 n 和 d 封装返回，这就是优化器防止深度发散的终极武器！
        return {'n': refined_n, 'd': refined_d}
        
    except Exception as e:
        mprint(f"   ❌ 点云处理发生意料之外的系统异常: {e}")
        return None

# =========================================================
# 4. 构造四重联合优化非线性误差函数 (Cost Function)
# =========================================================
def joint_residuals(x, clear_frames, blur_frames, plane_prior, mprint_fn):
    """
    功能：这是 Levenberg-Marquardt (LM) 最小二乘法优化器的核心反馈源。
    优化器在每一轮都会瞎猜一个状态向量 x，本函数负责用四重物理规则去“拷问”这个 x。
    如果 x 猜得不对，各项规则就会产生巨大的残差 (Residuals)，并返回给优化器指明修改梯度。
    """
    global GLOBAL_EVAL_COUNT
    is_first_eval = (GLOBAL_EVAL_COUNT == 0)
    res = [] # 存放所有约束产生的误差值的长池子
    
    if is_first_eval:
        mprint_fn("\n" + "▼"*60)
        mprint_fn("【深度数学审计】优化器第 0 次计算 (Initial State Verification)")
        mprint_fn(f"当前猜测的状态向量 x = [tx, ty, tz, rx, ry, rz]: \n{np.round(x, 4)}")
        mprint_fn("▼"*60)

    # ================= 遍历序列中所有被近距离拍摄的清晰帧 =================
    for idx, f in enumerate(clear_frames): 
        # 解析优化器当前试错的标志牌位姿矩阵
        t_WS = x[0:3] 
        r_WS = R.from_rotvec(x[3:6]).as_matrix() 
        
        # 【约束 1：多视图宏观基线角点重投影】
        # (1) 将局部虚拟铁皮 4 个角点 P_S，乘以 r_WS 和加 t_WS 变到世界系
        P_W = np.dot(P_S, r_WS.T) + t_WS
        # (2) 将世界系的 4 个角点，乘以外参 T_CW 变到当前相机的相对坐标系
        P_C = np.dot(P_W, f['T_CW'][:3, :3].T) + f['T_CW'][:3, 3]
        # (3) 防御极小深度，防止透视除法除以零
        Z_C = np.maximum(P_C[:, 2], 1e-6) 
        # (4) 乘以内参 FX, FY，投影得到虚拟角点在图像上的 2D 像素坐标 pts_hat
        pts_hat = np.column_stack((FX * (P_C[:, 0] / Z_C) + CX, FY * (P_C[:, 1] / Z_C) + CY))
        
        # 将虚拟像素点与照片上实际观测到的角点 (corners) 相减，差值乘以设定的权重，推入残差池
        res.append((f['corners'] - pts_hat).flatten() * W_REPROJ)
        
        if is_first_eval and idx == 0:
            mprint_fn(f"\n[A] 几何角点重投影 (Reprojection Error) - 目标帧 1:")
            mprint_fn(f"  公式: P_C = T_CW * (R_WS * P_S + t_WS)  |  u = fx*(X_C/Z_C)+cx, v = fy*(Y_C/Z_C)+cy")
            mprint_fn(f"  -> 计算得 3D 世界坐标 P_W: \n{np.round(P_W, 3)}")
            mprint_fn(f"  -> 计算得相机坐标系 P_C: \n{np.round(P_C, 3)}")
            mprint_fn(f"  -> 虚拟投影 2D 像素 pts_hat: \n{np.round(pts_hat, 2)}")
            mprint_fn(f"  -> 真实观测 2D 像素 corners: \n{np.round(f['corners'], 2)}")
            mprint_fn(f"  -> [误差结果] X/Y 像素偏差: {np.round((f['corners'] - pts_hat).flatten(), 2)}")

        # 【约束 2：微观文本正交刚性死锁 (The Masterpiece)】
        # 如果当前图片成功提取了汉字的横竖线段...
        if 'h_lines' in f and 'v_lines' in f: 
            # 根据当前试错的位姿，求得反向消除透视畸变的映射算子 H_inv
            H_inv = get_inverse_homography(x, f['T_CW'], mprint=mprint_fn if (is_first_eval and idx == 0) else None) 
            
            # 将照片里所有的横线 (一横，横折钩的横段等) 逆向投影回物理 3D 面
            for u1, v1, u2, v2, weight in f['h_lines']: 
                _, Y1 = pixel_to_sign_plane(u1, v1, H_inv) 
                _, Y2 = pixel_to_sign_plane(u2, v2, H_inv) 
                # 计算其左右两个端点在此物理面上的高度差 Y1 - Y2。
                # 理想状态下横线绝对水平，Y1 - Y2 应该绝对等于 0！不等于0说明标志牌发生了翻滚或偏航，重罚！
                res.append(np.array([(Y1 - Y2) * weight * W_ORTHO]))
                
            # 将照片里所有的竖线 (一竖，竖弯钩的竖段等) 逆向投影回物理 3D 面
            for u1, v1, u2, v2, weight in f['v_lines']: 
                X1, _ = pixel_to_sign_plane(u1, v1, H_inv) 
                X2, _ = pixel_to_sign_plane(u2, v2, H_inv) 
                # 计算上下端点的宽度差 X1 - X2。不等于 0 说明标志牌产生了俯仰角错误，重罚！
                res.append(np.array([(X1 - X2) * weight * W_ORTHO]))
                
            if is_first_eval and idx == 0:
                mprint_fn(f"\n[B] 正交线段结构死锁 (Orthogonal Structural Lock):")
                mprint_fn(f"  公式: P_sign = H_inv * [u, v, 1]^T  |  e_horiz = (Y1 - Y2)*w  |  e_vert = (X1 - X2)*w")
                if len(f['h_lines']) > 0:
                    u1, v1, u2, v2, w = f['h_lines'][0]
                    _, Y1 = pixel_to_sign_plane(u1, v1, H_inv)
                    _, Y2 = pixel_to_sign_plane(u2, v2, H_inv)
                    mprint_fn(f"  -> [抽样验证] 第一条横线端点: ({u1:.1f}, {v1:.1f}) -> ({u2:.1f}, {v2:.1f})")
                    mprint_fn(f"  -> 映射到物理平面高度: Y1 = {Y1:.4f} m, Y2 = {Y2:.4f} m")
                    mprint_fn(f"  -> [误差结果] 高度差 (倾斜畸变): {(Y1-Y2):.6f} m (目标为0)")
        
    # ================= 遍历所有远距离拍摄的、字看不清的模糊帧 =================
    for f in blur_frames: 
        t_WS = x[0:3] 
        r_WS = R.from_rotvec(x[3:6]).as_matrix() 
        # 将虚拟标志牌投影到当前相机的像素坐标系中
        P_W = np.dot(P_S, r_WS.T) + t_WS
        P_C = np.dot(P_W, f['T_CW'][:3, :3].T) + f['T_CW'][:3, 3]
        Z_C = np.maximum(P_C[:, 2], 1e-6) 
        pts_hat = np.column_stack((FX * (P_C[:, 0] / Z_C) + CX, FY * (P_C[:, 1] / Z_C) + CY))
        
        # 利用投影得到的四个像素点，强行框出一个最小外接矩形 [u_min, v_min, u_max, v_max]
        b_hat = np.array([np.min(pts_hat[:,0]), np.min(pts_hat[:,1]), np.max(pts_hat[:,0]), np.max(pts_hat[:,1])])
        
        # 【约束 3：YOLO 视锥宏观兜底 (Coarse Attraction)】
        # 强制用猜出的矩形框去减真实图片上的 YOLO 检测框。
        # W_BBOX 权重极低(0.02)，优化器平时不会在意它。但在深度暴走到几十上百米时，由于投影面积的近大远小，
        # 这个面积偏差项会指数级爆炸，从而把优化器生生拉回正确的深度量级附近。
        res.append((f['bbox'] - b_hat) * W_BBOX)
        
    # 【约束 4：背景物理墙体强制深空托底 (Depth Wall Soft-Prior)】
    if plane_prior:
        # 提取上面 PCA 手推算出的墙体方程的法向 n 和截距 d
        n, d = plane_prior['n'], plane_prior['d'] 
        
        # 数学定理：计算当前盲猜的标志牌绝对中心坐标 (x[0:3]) 到这堵物理墙的垂直距离
        # 公式 D = n_x * X + n_y * Y + n_z * Z + d
        dist = np.dot(n, x[0:3]) + d 
        
        # 标志牌是贴在墙上的，如果它距离这堵墙过远(或者悬空在墙的前后方)，直接施加巨大的深度方向残差惩罚
        res.append(np.array([dist]) * W_STRUCT) 
        
        if is_first_eval:
            mprint_fn(f"\n[D] 背景平面约束 (Background Plane Prior):")
            mprint_fn(f"  公式: dist = n^T * t_WS + d")
            mprint_fn(f"  -> n = {np.round(n, 4)}, d = {d:.2f}, t_WS = {np.round(x[0:3],2)}")
            mprint_fn(f"  -> [误差结果] 距离背景平面偏差: {dist:.4f} m")

    if is_first_eval:
        mprint_fn("▲"*60 + "\n")
        
    # 快照次数递增，避免疯狂刷屏
    GLOBAL_EVAL_COUNT += 1
    # 把长长的四项总残差池拼接成一维向量，退还给 SciPy 让它计算雅可比矩阵和下降梯度
    return np.concatenate(res) 

# =========================================================
# 5. 主程序管线集成
# =========================================================
def main():
    # 🛡️ 防御机制 1：【防全局变量污染】保证在连续调用(比如流水线批处理多条视频)时状态清零
    global GLOBAL_EVAL_COUNT
    GLOBAL_EVAL_COUNT = 0 

    # 预创建所需的调试图纸和日志存储目录
    os.makedirs(DEBUG_OUT_DIR, exist_ok=True)
    log_file_path = os.path.join(DEBUG_OUT_DIR, "optimization_detailed_log.txt")
    
    # 物理清空上次运行遗留的残余日志
    open(log_file_path, 'w').close()
    
    # 自定义一个双向打印机：一方面把日志打印到黑框终端给你看，一方面物理落盘保存以便导师验收
    def mprint(text):
        print(text) 
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
            
    mprint("=========================================================")
    mprint(f"🚀 初始化纯物理自适应单目感知管线 | 调试记录存放至: {DEBUG_OUT_DIR}")
    mprint("=========================================================")
    
    # 打印给导师看的各个核心组件的能量比配方案
    mprint("\n【核心数学流形约束权重配置 (Information Matrix Diagonal)】")
    mprint(f" W_REPROJ (角点定平移): {W_REPROJ} | W_ORTHO (微观绝对死锁旋转): {W_ORTHO}")
    mprint(f" W_BBOX (视锥引力限位): {W_BBOX} | W_STRUCT (物理实体深度托底): {W_STRUCT}")
    
    # 【加载四大核心数据表】 
    # 1. 读入 SLAM 时序轨迹 
    traj_data = np.loadtxt(TRAJ_FILE) 
    traj_dict = {row[0]: row[1:] for row in traj_data} 
    # 2. 读入图文和绝对机器时间的映射表
    mapping = pd.read_csv(MAP_FILE)   
    # 3. 读入包含 YOLO 视锥框的数据表
    bboxes_df = pd.read_csv(BBOX_FILE, comment='#', header=None, names=['filename', 'u_min', 'v_min', 'u_max', 'v_max']) 
    
    # 初始化双列车：blur 装远处驶来的模糊序列，clear 装抵近的高分辨率高特征帧
    blur_frames, clear_frames = [], [] 
    all_ts = np.array(list(traj_dict.keys())) 

    mprint("\n【多传感器时序同步与解构】")
    for i, row in bboxes_df.iterrows(): 
        # [步骤 A] 基于文件名查找时间戳，实现图像与机器时钟的软对齐
        match = mapping[mapping['filename'] == row['filename']] 
        if match.empty: continue 
        ts = match.iloc[0]['timestamp_s'] 
        
        # [步骤 B] 由于相机帧率和 SLAM 的更新频率可能不同，因此暴力搜寻最近的一个里程计时间戳做物理绑定
        closest_ts = all_ts[np.argmin(np.abs(all_ts - ts))] 
        # 解析相机在这一刻的绝对外参
        T_CW = get_T_CW(traj_dict[closest_ts]) 
        # 剥离 YOLO 框像素数据
        bbox = np.array([row['u_min'], row['v_min'], row['u_max'], row['v_max']]) 
        img_name = row['filename']
        
        # 🐱 【自适应距离防线】：由于远处的框就是个糊黑点，去提边缘纯属胡闹。
        # 利用几何近大远小的天然属性：只要标志牌框的横向像素宽度大于 120 个网格点，就说明车离得够近了，可以开切。
        bbox_width = bbox[2] - bbox[0]
        is_clear_frame = bbox_width > 120 
        
        if not is_clear_frame:
            # 针对模糊阵列，只抽取宏观的轮廓和相机的外参，只为提供引力盆和弱基线参考。
            # 🛡️ 防时空错位补丁：强制绑定图片名字 'img_name'，否则后续退化时索引将引发张冠李戴的灾难！
            blur_frames.append({'T_CW': T_CW, 'bbox': bbox, 'img_name': img_name}) 
        else:
            img_path = os.path.join(IMAGE_DIR, img_name) 
            img = cv2.imread(img_path) 
            
            # 🛡️ 图像I/O防线：机械硬盘 I/O 不稳定可能导致读取为空阵。必须加上防空补丁阻止后续的 `AttributeError`
            if img is None:
                mprint(f"  ⚠️ 系统警告: 读取载体 {img_path} 异常失效，操作略过。")
                continue

            mprint(f"\n⚙️ 解构高维特征系: {img_name} (尺度 {bbox_width:.1f} px)")
            
            # 开启 LSD 对画面提取局部的亚像素正交约束特征
            h_lines, v_lines = extract_subpixel_orthogonal_lines(img, roi_bbox=bbox, mprint=mprint)
            
            # [暂定 Placeholder] 提取外轮廓顶点。未来将替换为 Canny 求交提取法解决尺度漂移。
            obs_corners = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
            # 将多模态的全套特征挂载至这一帧的字典对象上，打包装车
            clear_frames.append({'T_CW': T_CW, 'corners': obs_corners, 'h_lines': h_lines, 'v_lines': v_lines, 'bbox': bbox, 'img_name': img_name})

    # 🛡️ 系统级熔断协议：如果送进来的视频是一团空切废料流，什么都没有触发，直接切断进程拒绝运算。
    if not clear_frames and not blur_frames:
        mprint("❌ 灾难级异常：时序中全空，YOLO 检测流失联，解耦管线全线终止引爆！")
        return

    # 🛡️ 优雅的软降级协议：如果在高速公路上车速飙到 120 码，导致根本没有超过 120 宽度的“完美帧”。
    # 不要在此时崩溃，必须从模糊列车的最后一节车厢里，强行提取一帧充当锚点，只为保留矩阵运算的一口生气！
    if not clear_frames and blur_frames:
        last_f = blur_frames.pop()
        img_name = last_f['img_name'] 
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = cv2.imread(img_path)
        
        if img is not None:
            mprint(f"⚠️ 触发安全底线：被迫提取模糊锚点帧 {img_name} 以进行劣质态势估算。")
            h_lines, v_lines = extract_subpixel_orthogonal_lines(img, roi_bbox=last_f['bbox'], mprint=mprint)
            obs_corners = np.array([[last_f['bbox'][0], last_f['bbox'][1]], [last_f['bbox'][2], last_f['bbox'][1]], 
                                    [last_f['bbox'][2], last_f['bbox'][3]], [last_f['bbox'][0], last_f['bbox'][3]]])
            last_f.update({'corners': obs_corners, 'h_lines': h_lines, 'v_lines': v_lines})
            clear_frames.append(last_f)
        else:
            mprint("❌ 彻底报废：连降级保底图载体也遭遇 I/O 损毁，管道全线切断。")
            return

    # 将离得最近的这一张高清主视觉帧锁定，作为系统解算核心
    target_frame = clear_frames[-1]
    
    # 【唤醒 3D 点云处理中枢】利用这一帧相机的透视投影，把上万个 SLAM 全局点云投到镜头前，
    # 根据 YOLO 的框精准裁切下对应范围的废墟点，启动 3D PCA 主成分分析，求出那面唯一的物理背景墙面方程。
    plane_prior = load_and_fit_background_plane(POINT_CLOUD_FILE, target_frame['T_CW'], target_frame['bbox'], mprint)

    # 🐱 【自适应初值投喂器】
    # 原理：单目无基线时，如果乱给深度初值，海森矩阵的雅可比梯度的下降方向会疯狂散漫直到迭代死机。
    # 我们利用经典的古老针孔几何法：真实的物理高度 H 映射到传感器上，其像素高度 h 与其在空间中的距离 Z 成严格的反比！
    # 推导 Z = fy * H_real / h_pixel。算出这个粗糙的 Z，给到优化器，它就能在一两步内稳定找到正道。
    last_T_WC = np.linalg.inv(target_frame['T_CW']) 
    target_bbox_height = target_frame['bbox'][3] - target_frame['bbox'][1]
    # max(..., 10.0) 极小值保护，免遭遮挡引发小尺寸检测，导致把深度猜到几万米外！
    dynamic_guess_z = FY * HEIGHT / max(target_bbox_height, 10.0) 
    
    # 构造 x0 (仅向车前 Z 方向推出 dynamic_guess_z，不带任何 X Y 和旋转，一切交给图优化网络求解)
    init_t = last_T_WC[:3, 3] + last_T_WC[:3, 2] * dynamic_guess_z 
    x0 = np.concatenate([init_t, [0, 0, 0]]) 

    mprint(f"\n>>> 物理先验注入: 利用 2D 边框张力逆推标定绝对锚点初值 Z={dynamic_guess_z:.2f} 米")
    mprint(f">>> 引擎全速运转：点火释放 Levenberg-Marquardt (LM) 高维联合图优化器...")
    
    # 核心黑盒弹射室。把刚刚定制好的 joint_residuals 误差计算函数及一系列包裹了时空属性的数据流投入黑盒！
    res = least_squares(lambda x: joint_residuals(x, clear_frames, blur_frames, plane_prior, mprint), 
                        x0, method='lm', xtol=1e-8, ftol=1e-8, verbose=2) 

    # 从优化结束的一堆数学池水里，将精妙解构出来的 6-DoF 位姿抽离
    x_opt = res.x
    # scipy 四元旋转转换至经典 3x3 空间表征
    R_opt_matrix = R.from_rotvec(x_opt[3:6]).as_matrix()
    # 李代数流形，化为工程师肉眼可测的直观航向欧拉角表达体系
    euler_angles = R.from_rotvec(x_opt[3:6]).as_euler('zyx', degrees=True) 

    mprint("\n" + "="*60)
    mprint("🎉 [里程碑] 单目 6-DoF 三维强关联约束解算网全流程触底圆满收敛！")
    mprint("="*60)
    mprint(f"📍 世界绝对空间质心定标点 (Translation [X, Y, Z]): \n   [{x_opt[0]:.4f}, {x_opt[1]:.4f}, {x_opt[2]:.4f}] 米")
    mprint(f"📐 目标物理刚性姿态解旋 (Rotation Matrix R_WS): \n{np.round(R_opt_matrix, 5)}") 
    mprint(f"📐 欧拉域视角反演解析 (Roll, Pitch, Yaw): \n   Roll(相对翻滚): {euler_angles[2]:.2f}° | Pitch(俯仰倾角): {euler_angles[1]:.2f}° | Yaw(水平偏航): {euler_angles[0]:.2f}°")
    mprint(f"📉 数学信赖域下降指征: Success: {res.success} | Hessian矩阵震荡估算次数: {res.nfev} | 最终泛函残差压降 (Cost): {res.cost:.6f}")
    mprint("="*60)

    # =========================================================
    # 7. 3D 可视化图形交互渲染及文件固化 (UI Event Hook)
    # =========================================================
    mprint(">>> 三维全息场景态势编织中，请进入弹射界面任意滑动查看系统构建解...")
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
    
    # 拉起一个可拖拽的 3D 画布框架
    fig = plt.figure(figsize=(12, 10)) 
    ax = fig.add_subplot(111, projection='3d') 

    # 解析所有历史轨迹记录，并描绘相机的时空路径
    cam_xs, cam_ys, cam_zs = [], [], [] 
    for f in blur_frames + clear_frames: 
        T_WC = np.linalg.inv(f['T_CW']) 
        cam_xs.append(T_WC[0, 3]); cam_ys.append(T_WC[1, 3]); cam_zs.append(T_WC[2, 3]) 

    # 将自动驾驶车载相机的运行序列画出蓝色的虚线尾迹，并标注好红绿端点
    ax.plot(cam_xs, cam_ys, cam_zs, label='Autonomous Vehicle Ego-Trajectory', color='blue', marker='.', linestyle='dashed', alpha=0.7)
    ax.scatter(cam_xs[0], cam_ys[0], cam_zs[0], color='green', s=100, marker='s', label='Mission Start Anchor')
    ax.scatter(cam_xs[-1], cam_ys[-1], cam_zs[-1], color='orange', s=100, marker='^', label='Mission Final Observation')

    # 【灵魂展现】将那个平平无奇的 1.26*0.6 长方形框框，赋予算好的那 6 自由度空间矩阵系。
    # 随后把它狠狠地“插在”那个 3D 全局坐标空间系上的对应方位里。
    sign_corners_world = np.dot(P_S, R_opt_matrix.T) + x_opt[0:3] 
    verts = [list(zip(sign_corners_world[:, 0], sign_corners_world[:, 1], sign_corners_world[:, 2]))]
    # 利用红色不透明实体多边形，渲染那块交通铁皮物理占据的空间范围边界
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.8, facecolor='red', edgecolor='black', linewidths=2))

    # 为了不让图片在画出 3D 后产生极度扭曲视觉，我们抓取所有轨迹的跨度范围，算出个最居中的缩放正方体约束边框
    max_range = np.array([np.max(cam_xs)-np.min(cam_xs), np.max(cam_ys)-np.min(cam_ys), np.max(cam_zs)-np.min(cam_zs), 5.0]).max() / 2.0
    mid_x, mid_y, mid_z = (np.max(cam_xs)+np.min(cam_xs))*0.5, (np.max(cam_ys)+np.min(cam_ys))*0.5, (np.max(cam_zs)+np.min(cam_zs))*0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 增加三轴标定及物理刻度说明
    ax.set_xlabel('Global Physical X (m)'); ax.set_ylabel('Global Physical Y (m)'); ax.set_zlabel('Global Physical Z (m)')
    ax.legend(loc='upper left') 
    
    # 赋予一个舒适的带些微俯视倾角的上帝初见视角
    ax.view_init(elev=20, azim=-45) 
    
    # 指定图片自动固化路径位置
    plot_save_path = os.path.join(DEBUG_OUT_DIR, "step4_final_3d_trajectory.png")

    # 【OS 界面系统交互挂钩】
    def on_close(event):
        """
        闭包函数。专门拦截系统的“窗口被关闭”回调事件。
        在用户点击红叉的那 0.1 秒前，抢下内存截图固化存盘！
        """
        fig.savefig(plot_save_path, dpi=300, bbox_inches='tight')
        mprint(f"✅ 系统渲染中端触发：三维态势场已在您撤离视线时锁定，最后一刻视野光刻落盘至: {plot_save_path}")

    # 给底层 Matplotlib 的 Canvas GUI 窗口，打上这个红叉监听劫持挂钩
    fig.canvas.mpl_connect('close_event', on_close)
    
    mprint(">>> 💡 提示指引：左键拉拽把弄观察视网，中键调整景深尺度。当且仅当点击退出关闭页面时，视角镜像将即刻完成固化。")
    # 阻塞并把计算线程挂起，渲染系统进入高优监听无限循环...直到用户销毁窗体。
    plt.show()

# 模块防御与标准 Python 入口标尺线
if __name__ == "__main__":
    main()