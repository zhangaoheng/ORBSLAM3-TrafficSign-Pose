import numpy as np
import pandas as pd
import cv2
import os
import math
import logging
import csv
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# =========================================================
# 1. 核心配置与实验输出路径
# =========================================================
TRAJ_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/output/FrameTrajectory_TUM_20260331_154838.txt"
MAP_FILE  = "/home/zah/ORB_SLAM3-master/landmarkslam/output/Filename_Mapping_20260331_154838.csv"
BBOX_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/yolo_simulated_bboxes.txt"
IMAGE_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/" 

# --- 新增：实验数据输出目录规划 ---
EXPERIMENT_OUT_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/output/looming_results/"
LOG_FILE = os.path.join(EXPERIMENT_OUT_DIR, "looming_experiment.log")
CSV_FILE = os.path.join(EXPERIMENT_OUT_DIR, "looming_results.csv")
VIS_STEPS_DIR = os.path.join(EXPERIMENT_OUT_DIR, "vis_steps/") # 存放中间过程图片

os.makedirs(EXPERIMENT_OUT_DIR, exist_ok=True)
os.makedirs(VIS_STEPS_DIR, exist_ok=True)

# 初始化日志记录器 (同时输出到终端和 Log 文件)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================================================
# 2. 核心数学：SLAM位姿解析
# =========================================================
def get_T_WC(pose_tum):
    T_WC = np.eye(4)
    T_WC[:3, :3] = R.from_quat(pose_tum[3:7]).as_matrix()
    T_WC[:3, 3] = pose_tum[0:3]
    return T_WC

# =========================================================
# 3. 核心创新：提取光学几何投影 (带全中间过程输出)
# =========================================================
def extract_optical_projection_hw(img_path, u1, v1, u2, v2, fname):
    """
    剥离语义边框，提取标志牌真实物理边界在图像上的“亚像素级”投影尺寸。
    【实验版】：保存提取过程的所有中间图像，用于消融实验展示。
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    fname_base = os.path.splitext(fname)[0] # 去掉后缀的文件名，用于拼接
    
    # 3.1 ROI 结界外扩
    pad = 2
    crop_u1, crop_v1 = max(0, int(u1) - pad), max(0, int(v1) - pad)
    crop_u2, crop_v2 = min(img_w, int(u2) + pad), min(img_h, int(v2) + pad)
    
    roi = gray[crop_v1:crop_v2, crop_u1:crop_u2]
    roi_h, roi_w = roi.shape
    # [📸 输出中间图片 1]：保存原始裁剪的 ROI
    cv2.imwrite(os.path.join(VIS_STEPS_DIR, f"{fname_base}_step1_ROI.jpg"), roi)
    
    # 3.2 图像增强与探测
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(roi)
    # [📸 输出中间图片 2]：保存 CLAHE 增强后的 ROI (展示抗逆光效果)
    cv2.imwrite(os.path.join(VIS_STEPS_DIR, f"{fname_base}_step2_CLAHE.jpg"), roi_enhanced)

    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(roi_enhanced)
    
    fallback_w, fallback_h = float(u2 - u1), float(v2 - v1)
    vis_img_final = img.copy() 
    vis_img_all_lines = img.copy()
    vis_img_filtered = img.copy()

    if lines is None:
        logger.warning(f"Frame {fname}: LSD failed to detect any lines. Triggered Fallback.")
        return fallback_w, fallback_h
        
    all_x, all_y = [], []
    min_len = min(roi_w, roi_h) * 0.4 
    
    # 3.3 长度海选
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.hypot(x2-x1, y2-y1)
        
        gx1, gy1 = x1 + crop_u1, y1 + crop_v1
        gx2, gy2 = x2 + crop_u1, y2 + crop_v1
        
        # [📸 画中间图片 3]：绘制所有 LSD 探测到的原始线段 (红色)
        cv2.line(vis_img_all_lines, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 0, 255), 1)
        
        if length > min_len:
            all_x.extend([gx1, gx2])
            all_y.extend([gy1, gy2])
            # [📸 画中间图片 4]：绘制通过长度过滤的长线段 (绿色)
            cv2.line(vis_img_filtered, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 255, 0), 1)
                
    # 输出图片 3 和 4
    cv2.imwrite(os.path.join(VIS_STEPS_DIR, f"{fname_base}_step3_LSD_All.jpg"), vis_img_all_lines[crop_v1:crop_v2, crop_u1:crop_u2])
    cv2.imwrite(os.path.join(VIS_STEPS_DIR, f"{fname_base}_step4_LSD_Filtered.jpg"), vis_img_filtered[crop_v1:crop_v2, crop_u1:crop_u2])

    # 3.4 空间极值挤压
    final_w, final_h = fallback_w, fallback_h
    w_triggered, h_triggered = True, True
    fallback_ratio = fallback_w / fallback_h

    if len(all_x) >= 2:
        ext_w = max(all_x) - min(all_x)
        if 0.85 < (ext_w / fallback_h) / fallback_ratio < 1.15:
            final_w = float(ext_w)
            w_triggered = False
            cv2.line(vis_img_final, (int(min(all_x)), int(v1)), (int(min(all_x)), int(v2)), (255, 0, 255), 2)
            cv2.line(vis_img_final, (int(max(all_x)), int(v1)), (int(max(all_x)), int(v2)), (255, 0, 255), 2)

    if len(all_y) >= 2:
        ext_h = max(all_y) - min(all_y)
        if 0.85 < (fallback_w / ext_h) / fallback_ratio < 1.15:
            final_h = float(ext_h)
            h_triggered = False
            cv2.line(vis_img_final, (int(u1), int(min(all_y))), (int(u2), int(min(all_y))), (255, 0, 255), 2)
            cv2.line(vis_img_final, (int(u1), int(max(all_y))), (int(u2), int(max(all_y))), (255, 0, 255), 2)

    if w_triggered or h_triggered:
        cv2.putText(vis_img_final, "Fallback Mode", (int(u1), int(v2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        logger.info(f"Frame {fname}: Fallback Triggered. W_fallback={w_triggered}, H_fallback={h_triggered}")
        
    # [📸 输出中间图片 5]：保存最终用于计算深度的紫线包围盒
    cv2.imwrite(os.path.join(VIS_STEPS_DIR, f"{fname_base}_step5_FinalBox.jpg"), vis_img_final)
    
    return final_w, final_h

# =========================================================
# 4. 主程序：数据对齐与深度解算管线
# =========================================================
def main():
    logger.info(">>> [系统启动] 实验版 Looming 探测器初始化...")
    logger.info(f">>> 实验日志、数据表和中间图片将保存在: {EXPERIMENT_OUT_DIR}")

    traj_data = np.loadtxt(TRAJ_FILE)
    traj_dict = {row[0]: row[1:] for row in traj_data}
    all_ts = np.array(list(traj_dict.keys()))
    
    mapping_df = pd.read_csv(MAP_FILE)
    bbox_df = pd.read_csv(BBOX_FILE, header=None, names=['fname', 'u1', 'v1', 'u2', 'v2'], sep=',', comment='#')
    
    frames = [] 
    logger.info(">>> 正在执行亚像素边缘重构并输出中间过程图片...")
    
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
            
        w, h = extract_optical_projection_hw(img_path, u1, v1, u2, v2, fname)
        if w is None or h is None: continue
        
        frames.append({'fname': fname, 'width': w, 'height': h, 'pos_w': T_WC[:3, 3], 'ts': ts})
        
    if len(frames) < 2:
        logger.error("❌ 错误：有效帧数不足，无法计算膨胀。")
        return

    # ---------------------------------------------------------
    # 5. 联合解算与结果收集
    # ---------------------------------------------------------
    # 准备 CSV 写入列表
    csv_results = []
    
    f_start = frames[0]
    w1, h1 = f_start['width'], f_start['height']
    
    # 我们用第一帧作为固定的 t1，让后续每一帧都跟 t1 算一次深度，这样就能描绘出车辆迫近时的深度收敛曲线！
    for i in range(1, len(frames)):
        f_end = frames[i]
        w2, h2 = f_end['width'], f_end['height']
        delta_d = np.linalg.norm(f_end['pos_w'] - f_start['pos_w'])

        delta_w, delta_h = w2 - w1, h2 - h1
        
        Z_w = (w2 * delta_d) / delta_w if delta_w > 8.0 else None
        Z_h = (h2 * delta_d) / delta_h if delta_h > 8.0 else None
        
        Z_final = None
        status = ""
        
        if Z_w and Z_h:
            diff = abs(Z_w - Z_h) / max(Z_w, Z_h)
            if diff < 0.15:
                Z_final = (Z_w + Z_h) / 2.0
                status = "Cross-Validated (Avg)"
            else:
                Z_final = Z_h if delta_h > delta_w else Z_w
                status = "Fallback Arbitrated (Max SNR)"
        elif Z_w:
            Z_final = Z_w
            status = "Single-Dim (W)"
        elif Z_h:
            Z_final = Z_h
            status = "Single-Dim (H)"
        else:
            status = "Failed (Insufficient Growth)"
            
        # 记录到 CSV 缓存
        csv_results.append({
            't1_fname': f_start['fname'],
            't2_fname': f_end['fname'],
            't2_timestamp': f_end['ts'],
            'delta_d': delta_d,
            'w1': w1, 'h1': h1,
            'w2': w2, 'h2': h2,
            'delta_w': delta_w, 'delta_h': delta_h,
            'Z_w': Z_w if Z_w else -1,
            'Z_h': Z_h if Z_h else -1,
            'Z_final': Z_final if Z_final else -1,
            'status': status
        })
        
        # 仅在日志中打印最后一帧的详细信息，防止刷屏
        if i == len(frames) - 1:
            logger.info("="*50)
            logger.info(" 🛠️ 最终解算面板 (t1 vs t_last)")
            logger.info(f" 远端 (t1): {f_start['fname']} | W1={w1:.2f}, H1={h1:.2f}")
            logger.info(f" 近端 (t2): {f_end['fname']} | W2={w2:.2f}, H2={h2:.2f}")
            logger.info(f" 物理位移 (Δd): {delta_d:.4f} units")
            if Z_w and Z_h:
                logger.info(f" ├─ X轴深度 (Z_w): {Z_w:.4f} | Y轴深度 (Z_h): {Z_h:.4f}")
            logger.info(f" 🎯 最终输出深度: 【 {Z_final:.4f} 】 ({status})")
            logger.info("="*50)

    # ---------------------------------------------------------
    # 6. 将实验数据落盘为 CSV 文件
    # ---------------------------------------------------------
    logger.info(f">>> 正在将连续解算结果导出至 CSV: {CSV_FILE}")
    keys = csv_results[0].keys()
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(csv_results)
        
    logger.info(">>> 实验数据收集完毕！请查看 output/experiment_results/ 目录。")

if __name__ == "__main__":
    main()