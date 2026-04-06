import numpy as np
import pandas as pd
import cv2
import os
import math
from scipy.spatial.transform import Rotation as R # 用于处理四元数到旋转矩阵的转换

# =========================================================
# 1. 核心配置与文件路径
# =========================================================
# SLAM 轨迹文件 (TUM格式: timestamp tx ty tz qx qy qz qw)
TRAJ_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/output/FrameTrajectory_TUM_20260331_154838.txt"
# 图像文件名与时间戳的映射关系表
MAP_FILE  = "/home/zah/ORB_SLAM3-master/landmarkslam/output/Filename_Mapping_20260331_154838.csv"
# YOLO 检测输出的结果 (格式: fname, u1, v1, u2, v2)
BBOX_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/yolo_simulated_bboxes.txt"
# 原始图像存放目录
IMAGE_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/" 

# 调试图像的输出路径，用于存放画了绿线、紫线的可视化结果
DEBUG_OUT_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/output/looming_debug_vis/"

# =========================================================
# 2. 核心数学：SLAM位姿解析
# =========================================================
def get_T_WC(pose_tum):
    """
    将 TUM 格式的一维数组位姿，转换为 4x4 的齐次变换矩阵 T_WC。
    T_WC 代表从 Camera 坐标系到 World 坐标系的变换。
    - pose_tum[0:3] 是平移向量 t (tx, ty, tz)
    - pose_tum[3:7] 是四元数 q (qx, qy, qz, qw)
    """
    T_WC = np.eye(4) # 初始化 4x4 单位矩阵
    # 利用 scipy 将四元数转为 3x3 旋转矩阵，填入左上角
    T_WC[:3, :3] = R.from_quat(pose_tum[3:7]).as_matrix()
    # 将平移向量填入右上角
    T_WC[:3, 3] = pose_tum[0:3]
    return T_WC

# =========================================================
# 3. 核心创新：基于长度海选与空间极值的边缘重构 (V4.1 - 纯净版)
# =========================================================
def extract_physical_hw(img_path, u1, v1, u2, v2, fname):
    """
    该函数的核心使命：剥离 YOLO 的粗糙边框，通过图像底层的亮暗梯度，
    提取出标志牌真实的“亚像素级”物理宽度和高度。
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    
    # ---------------------------------------------------------
    # 3.1 ROI 结界外扩 (Padding)
    # ---------------------------------------------------------
    # 为什么要外扩？因为 YOLO 的框可能偏小，刚好把标志牌真实的物理边缘切掉在了框外。
    # 我们向外扩展 2 个像素，给底层特征提取留出搜寻空间。
    pad = 2
    crop_u1, crop_v1 = max(0, int(u1) - pad), max(0, int(v1) - pad)
    crop_u2, crop_v2 = min(img_w, int(u2) + pad), min(img_h, int(v2) + pad)
    
    # 抠出这块感兴趣区域 (Region of Interest)
    roi = gray[crop_v1:crop_v2, crop_u1:crop_u2]
    roi_h, roi_w = roi.shape
    
    # ---------------------------------------------------------
    # 3.2 图像增强与亚像素线段探测 (LSD)
    # ---------------------------------------------------------
    # 使用 CLAHE (限制对比度自适应直方图均衡化) 来对抗逆光和强曝光，凸显边缘
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    # 初始化 LSD (Line Segment Detector)，这是一种极其敏锐的亚像素线段提取算法
    lsd = cv2.createLineSegmentDetector(0)
    # 在增强后的 ROI 上提取所有线段
    lines, _, _, _ = lsd.detect(clahe.apply(roi))
    
    # 记下 YOLO 给出的宏观尺寸，作为最后的保底(Fallback)手段
    fallback_w, fallback_h = float(u2 - u1), float(v2 - v1)
    vis_img = img.copy() # 用于画图的画布

    # 如果极度模糊导致 LSD 什么都没提取出来，直接返回 YOLO 的尺寸
    if lines is None:
        cv2.imwrite(os.path.join(DEBUG_OUT_DIR, fname), vis_img)
        return fallback_w, fallback_h
        
    all_x, all_y = [], [] # 用于存放所有合格线段的端点坐标
    
    # ---------------------------------------------------------
    # 3.3 长度海选逻辑 (废弃角度过滤的精髓)
    # ---------------------------------------------------------
    # 只要线段的长度超过 ROI 短边的 40%，我们就认为它是有效的结构边缘。
    # 这样做可以完美免疫近距离造成的“严重透视畸变”（边缘变成大斜线）。
    min_len = min(roi_w, roi_h) * 0.4 
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.hypot(x2-x1, y2-y1) # 计算线段长度
        
        if length > min_len:
            # LSD 提取的坐标是相对 ROI 左上角的，这里需要加上 crop 偏移量，映射回原图的绝对坐标
            gx1, gy1 = x1 + crop_u1, y1 + crop_v1
            gx2, gy2 = x2 + crop_u1, y2 + crop_v1
            
            # 将这条长线的两个端点坐标全部扔进池子里
            all_x.extend([gx1, gx2])
            all_y.extend([gy1, gy2])
            
            # 可视化：把参与极值竞争的长线画成细绿色
            cv2.line(vis_img, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 255, 0), 1)
                
    # ---------------------------------------------------------
    # 3.4 空间极值挤压与长宽比保底 (The Moat 护城河机制)
    # ---------------------------------------------------------
    final_w, final_h = fallback_w, fallback_h # 默认先使用保底值
    w_triggered, h_triggered = True, True     # 标记是否触发了保底
    
    # 计算 YOLO 框给出的基准常识长宽比
    fallback_ratio = fallback_w / fallback_h

    # [X轴/宽度 计算]：利用所有 X 坐标的最大值减去最小值，硬生生把宽度“撑”出来
    if len(all_x) >= 2:
        ext_w = max(all_x) - min(all_x)
        # 护城河校验：用提取的宽度和 YOLO 高度算一个临时长宽比。
        # 如果这个比值与 YOLO 基准长宽比的偏差在 15% 以内，说明没抓到离谱的电线，提取有效！
        if 0.85 < (ext_w / fallback_h) / fallback_ratio < 1.15:
            final_w = float(ext_w)
            w_triggered = False # 提取成功，解除保底
            # 画紫色的 X 轴边界锁定线
            cv2.line(vis_img, (int(min(all_x)), int(v1)), (int(min(all_x)), int(v2)), (255, 0, 255), 2)
            cv2.line(vis_img, (int(max(all_x)), int(v1)), (int(max(all_x)), int(v2)), (255, 0, 255), 2)

    # [Y轴/高度 计算]：利用所有 Y 坐标的最大值减去最小值
    if len(all_y) >= 2:
        ext_h = max(all_y) - min(all_y)
        # 护城河校验：同理，防止抓到内部水平的汉字导致高度严重缩水
        if 0.85 < (fallback_w / ext_h) / fallback_ratio < 1.15:
            final_h = float(ext_h)
            h_triggered = False # 提取成功，解除保底
            # 画紫色的 Y 轴边界锁定线
            cv2.line(vis_img, (int(u1), int(min(all_y))), (int(u2), int(min(all_y))), (255, 0, 255), 2)
            cv2.line(vis_img, (int(u1), int(max(all_y))), (int(u2), int(max(all_y))), (255, 0, 255), 2)

    # 只有当提取彻底失败（抓到电线/文字被护城河拦截），才会在图片上打上黄色警告
    if w_triggered or h_triggered:
        cv2.putText(vis_img, "Fallback Mode", (int(u1), int(v2)+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
    cv2.imwrite(os.path.join(DEBUG_OUT_DIR, fname), vis_img)
    return final_w, final_h

# =========================================================
# 4. 主程序：数据对齐与深度解算管线
# =========================================================
def main():
    print(">>> [系统启动] 正在初始化【长度优先策略-纯净版】Looming 探测器...")
    os.makedirs(DEBUG_OUT_DIR, exist_ok=True)

    # 1. 加载 SLAM 轨迹，建立 时间戳 -> 位姿 的字典
    traj_data = np.loadtxt(TRAJ_FILE)
    traj_dict = {row[0]: row[1:] for row in traj_data}
    all_ts = np.array(list(traj_dict.keys()))
    
    # 2. 加载 图片->时间戳 的映射表，以及 YOLO 的检测框结果
    mapping_df = pd.read_csv(MAP_FILE)
    bbox_df = pd.read_csv(BBOX_FILE, header=None, names=['fname', 'u1', 'v1', 'u2', 'v2'], sep=',', comment='#')
    
    frames = [] # 用于存储通过了检验的有效帧数据
    print(">>> 正在执行亚像素边缘重构...")
    
    # 遍历每一个 YOLO 框数据
    for _, row in bbox_df.iterrows():
        fname = str(row['fname']).strip()
        
        # 在映射表中找到这张图片对应的时间戳
        match_rows = mapping_df[mapping_df['filename'] == fname]
        if match_rows.empty: continue
            
        ts = match_rows.iloc[0]['timestamp_s']
        
        # 核心：时间戳同步。在 SLAM 轨迹中找离图片曝光时间最近的那个位姿
        closest_ts = all_ts[np.argmin(np.abs(all_ts - ts))]
        T_WC = get_T_WC(traj_dict[closest_ts]) # 获取该帧相机的 3D 空间坐标
        
        u1, v1, u2, v2 = float(row['u1']), float(row['v1']), float(row['u2']), float(row['v2'])
        img_path = os.path.join(IMAGE_DIR, fname)
        if not os.path.exists(img_path): continue
            
        # 呼叫重构器，获取亚像素长宽
        w, h = extract_physical_hw(img_path, u1, v1, u2, v2, fname)
        if w is None or h is None: continue
        
        # 将成功重构的数据压入列表。pos_w 是相机在世界坐标系下的绝对三维位置 (tx, ty, tz)
        frames.append({'fname': fname, 'width': w, 'height': h, 'pos_w': T_WC[:3, 3]})
        
    if len(frames) < 2:
        print("❌ 错误：有效帧数不足。")
        return

    # ---------------------------------------------------------
    # 5. Looming 视觉迫近深度解算核心数学
    # ---------------------------------------------------------
    f_start, f_end = frames[0], frames[-1] # 取远端(t1)和近端(t2)两帧
    w1, h1, w2, h2 = f_start['width'], f_start['height'], f_end['width'], f_end['height']
    
    # 计算相机在这两帧之间行驶的绝对物理位移 (欧氏距离)
    delta_d = np.linalg.norm(f_end['pos_w'] - f_start['pos_w'])

    print("\n" + "="*50)
    print(" 🛠️ 亚像素双维膨胀数据面板")
    print("="*50)
    print(f" 远端 (t1): {f_start['fname']} | W1={w1:.2f}, H1={h1:.2f}")
    print(f" 近端 (t2): {f_end['fname']} | W2={w2:.2f}, H2={h2:.2f}")
    print(f" 物理位移 (Δd): {delta_d:.4f} units")
    print("-" * 50)

    # 计算图像上的像素膨胀量
    delta_w, delta_h = w2 - w1, h2 - h1
    
    # 利用公式 Z = (w2 * Δd) / Δw 独立计算两个维度的目标深度
    # 设定最小膨胀阈值 8.0 像素，防止除以极小数导致系统发散爆炸
    Z_w = (w2 * delta_d) / delta_w if delta_w > 8.0 else None
    Z_h = (h2 * delta_d) / delta_h if delta_h > 8.0 else None
        
    # ---------------------------------------------------------
    # 6. 双维正交交叉验证 (The Cross-Validation Arbitrator)
    # ---------------------------------------------------------
    print("\n 🚀 [联合解算结果]")
    if Z_w and Z_h:
        # 计算两个引擎给出的深度偏差率
        diff = abs(Z_w - Z_h) / max(Z_w, Z_h)
        print(f"   ├─ X轴深度 (Z_w): {Z_w:.4f}\n   ├─ Y轴深度 (Z_h): {Z_h:.4f}")
        
        # 偏差小于 15%，认为特征提取完美且自洽，直接取均值降低统计方差
        if diff < 0.15:
            print(f"   🟢 交叉验证通过! 最终锚点: 【 {(Z_w + Z_h) / 2.0:.4f} 】")
        else:
            # 偏差过大，说明其中一个维度被严重污染（抓了电线或文字）。
            # 此时采取“信噪比优先”策略：谁膨胀的像素多（信号更强、抗噪更好），就用谁的深度。
            final_z = Z_h if delta_h > delta_w else Z_w
            print(f"   🔴 偏差过大({diff*100:.1f}%)，已选择高置信度维度。")
            print(f"   🎯 建议锚点: 【 {final_z:.4f} 】")
            
    # 如果有一个维度膨胀量不足 8 像素，直接退化为单维度计算保底
    elif Z_w or Z_h:
        print(f"   🟠 单维保底锚点: 【 {Z_w if Z_w else Z_h:.4f} 】")

    print("="*50)

if __name__ == "__main__":
    main()