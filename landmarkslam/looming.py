import numpy as np
import pandas as pd
import cv2
import os
import math
import logging
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# =========================================================
# 1. 核心配置与实验输出路径
# =========================================================
TRAJ_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/output/FrameTrajectory_TUM_20260331_154838.txt"
MAP_FILE  = "/home/zah/ORB_SLAM3-master/landmarkslam/output/Filename_Mapping_20260331_154838.csv"
BBOX_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/yolo_simulated_bboxes.txt"
IMAGE_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/" 

EXPERIMENT_OUT_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/output/looming_results/"
LOG_FILE = os.path.join(EXPERIMENT_OUT_DIR, "looming_experiment.log")
CSV_FILE = os.path.join(EXPERIMENT_OUT_DIR, "looming_results.csv")
VIS_STEPS_DIR = os.path.join(EXPERIMENT_OUT_DIR, "vis_steps/") 

os.makedirs(EXPERIMENT_OUT_DIR, exist_ok=True)
os.makedirs(VIS_STEPS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
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
# 3. 核心创新：提取光学几何投影
# =========================================================
def extract_optical_projection_hw(img_path, u1, v1, u2, v2, fname):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    fname_base = os.path.splitext(fname)[0] 
    
    pad = 2
    crop_u1, crop_v1 = max(0, int(u1) - pad), max(0, int(v1) - pad)
    crop_u2, crop_v2 = min(img_w, int(u2) + pad), min(img_h, int(v2) + pad)
    
    roi = gray[crop_v1:crop_v2, crop_u1:crop_u2]
    roi_h, roi_w = roi.shape 
    
    cv2.imwrite(os.path.join(VIS_STEPS_DIR, f"{fname_base}_step1_ROI.jpg"), roi)
    
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(roi)
    cv2.imwrite(os.path.join(VIS_STEPS_DIR, f"{fname_base}_step2_CLAHE.jpg"), roi_enhanced)

    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(roi_enhanced)
    
    fallback_w, fallback_h = float(u2 - u1), float(v2 - v1)
    vis_img_final = img.copy() 
    vis_img_all_lines = img.copy()
    vis_img_filtered = img.copy()

    if lines is None:
        logger.warning(f"[特征提取] {fname}: LSD 提取失败 (图像极度模糊)，已强制回退至 YOLO 原始宽/高。")
        return fallback_w, fallback_h
        
    all_x, all_y = [], []
    min_len = min(roi_w, roi_h) * 0.4 
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.hypot(x2-x1, y2-y1)
        gx1, gy1 = x1 + crop_u1, y1 + crop_v1
        gx2, gy2 = x2 + crop_u1, y2 + crop_v1
        cv2.line(vis_img_all_lines, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 0, 255), 1)
        
        if length > min_len:
            all_x.extend([gx1, gx2])
            all_y.extend([gy1, gy2])
            cv2.line(vis_img_filtered, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 255, 0), 1)
                
    cv2.imwrite(os.path.join(VIS_STEPS_DIR, f"{fname_base}_step3_LSD_All.jpg"), vis_img_all_lines[crop_v1:crop_v2, crop_u1:crop_u2])
    cv2.imwrite(os.path.join(VIS_STEPS_DIR, f"{fname_base}_step4_LSD_Filtered.jpg"), vis_img_filtered[crop_v1:crop_v2, crop_u1:crop_u2])

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
        logger.info(f"[物理护城河] {fname}: 侦测到形变异常 (疑似干扰), 触发回退机制 -> W使用保底: {w_triggered}, H使用保底: {h_triggered}")
        
    cv2.imwrite(os.path.join(VIS_STEPS_DIR, f"{fname_base}_step5_FinalBox.jpg"), vis_img_final)
    
    return final_w, final_h

# =========================================================
# 4. 空间 3D 绘图引擎 (含收敛轨迹)
# =========================================================
def plot_spatial_trajectory_3d(frames, csv_results):
    """提取相机的 XYZ 轨迹及地标推算轨迹，保存三视图，并开启 3D 交互窗口"""
    logger.info(">>> [阶段4] 正在渲染全量空间 3D 轨迹与推断收敛点阵...")
    
    cam_x = [f['pos_w'][0] for f in frames]
    cam_y = [f['pos_w'][1] for f in frames]
    cam_z = [f['pos_w'][2] for f in frames]
    
    lm_x_list, lm_y_list, lm_z_list = [], [], []
    for i, res in enumerate(csv_results):
        z = res['Z_final']
        if z != -1:
            T_WC = frames[i+1]['T_WC']
            P_camera = np.array([0, 0, z, 1.0]) 
            P_world = T_WC @ P_camera
            lm_x_list.append(P_world[0])
            lm_y_list.append(P_world[1])
            lm_z_list.append(P_world[2])

    views = [
        ('XY Plane (Top View)', cam_x, cam_y, lm_x_list, lm_y_list, 'X', 'Y', 'traj_view_XY_Top.png'),
        ('XZ Plane (Front View)', cam_x, cam_z, lm_x_list, lm_z_list, 'X', 'Z', 'traj_view_XZ_Front.png'),
        ('YZ Plane (Side View)', cam_y, cam_z, lm_y_list, lm_z_list, 'Y', 'Z', 'traj_view_YZ_Side.png')
    ]
    
    for title, cx, cy, lx, ly, x_label, y_label, filename in views:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(cx, cy, linestyle='-', color='b', alpha=0.5, label='Camera Path')
        if len(cx) > 2: ax.scatter(cx[1:-1], cy[1:-1], color='cyan', s=15, edgecolor='k', label='Cam Inter-poses')
        ax.scatter(cx[0], cy[0], color='green', s=100, edgecolor='k', label='Cam Start')
        ax.scatter(cx[-1], cy[-1], color='red', s=100, edgecolor='k', label='Cam End')
        
        if lx:
            if len(lx) > 1: ax.scatter(lx[:-1], ly[:-1], color='orange', marker='x', s=30, alpha=0.6, label='LM Inter-Est')
            ax.scatter(lx[-1], ly[-1], color='gold', marker='*', s=300, edgecolor='k', label='Final LM ⭐')
            
        ax.set_title(title)
        ax.set_xlabel(f'World {x_label}')
        ax.set_ylabel(f'World {y_label}')
        ax.grid(True, linestyle='--')
        ax.legend(loc='best')
        ax.axis('equal') 
        fig.savefig(os.path.join(EXPERIMENT_OUT_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    logger.info(">>> 🖼️ 已成功输出含有收敛过程的 XY, XZ, YZ 空间三视图至本地。")

    fig3d = plt.figure(figsize=(12, 9))
    ax3d = fig3d.add_subplot(111, projection='3d')
    
    ax3d.plot(cam_x, cam_y, cam_z, color='blue', linewidth=2, alpha=0.6, label='Camera Trajectory')
    if len(cam_x) > 2:
        ax3d.scatter(cam_x[1:-1], cam_y[1:-1], cam_z[1:-1], color='cyan', s=30, edgecolor='k', label='Intermediate Poses')
    ax3d.scatter(cam_x[0], cam_y[0], cam_z[0], color='green', s=100, edgecolor='k', label='Start Point')
    ax3d.scatter(cam_x[-1], cam_y[-1], cam_z[-1], color='red', s=100, edgecolor='k', label='Current Point (t2)')
    
    if lm_x_list:
        if len(lm_x_list) > 1:
            ax3d.scatter(lm_x_list[:-1], lm_y_list[:-1], lm_z_list[:-1], color='orange', marker='x', s=40, label='Convergence Path')
        
        final_lm_x, final_lm_y, final_lm_z = lm_x_list[-1], lm_y_list[-1], lm_z_list[-1]
        ax3d.scatter(final_lm_x, final_lm_y, final_lm_z, color='gold', marker='*', s=500, edgecolor='black', label='Final Signboard (⭐)')
        ax3d.plot([cam_x[-1], final_lm_x], [cam_y[-1], final_lm_y], [cam_z[-1], final_lm_z], color='gray', linestyle='--', label=f'Final LOS (Z={csv_results[-1]["Z_final"]:.2f})')

    ax3d.set_title('3D SLAM Trajectory & Landmark Convergence', fontsize=14, fontweight='bold')
    ax3d.set_xlabel('World X')
    ax3d.set_ylabel('World Y')
    ax3d.set_zlabel('World Z')
    ax3d.legend()
    ax3d.view_init(elev=20, azim=-45) 
    
    logger.info(">>> 🚀 正在弹出 3D 轨迹交互窗口，您可以用鼠标拖拽旋转视角！(关掉窗口程序结束)")
    plt.show()

# =========================================================
# 5. 主程序：数据对齐与深度解算管线
# =========================================================
def main():
    logger.info("="*60)
    logger.info(" 🚀 [系统启动] Looming 深度探测器 (实验版)")
    logger.info(f" 📂 实验输出根目录: {EXPERIMENT_OUT_DIR}")
    logger.info("="*60)

    traj_data = np.loadtxt(TRAJ_FILE)
    traj_dict = {row[0]: row[1:] for row in traj_data}
    all_ts = np.array(list(traj_dict.keys()))
    
    mapping_df = pd.read_csv(MAP_FILE)
    bbox_df = pd.read_csv(BBOX_FILE, header=None, names=['fname', 'u1', 'v1', 'u2', 'v2'], sep=',', comment='#')
    
    frames = [] 
    logger.info(">>> [阶段1] 开始逐帧执行亚像素边缘重构 (LSD + 极值法)...")
    
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
        
        frames.append({'fname': fname, 'width': w, 'height': h, 'pos_w': T_WC[:3, 3], 'ts': ts, 'T_WC': T_WC})
        
    if len(frames) < 2:
        logger.error("❌ 有效帧数不足，需要至少2帧才能进行膨胀计算。")
        return

    logger.info(f">>> [阶段1完成] 成功重构 {len(frames)} 帧数据的亚像素边界。")
    logger.info("")
    logger.info(">>> [阶段2] 开始时序 Looming 深度解算与交叉验证...")

    csv_results = []
    f_start = frames[0]
    w1, h1 = f_start['width'], f_start['height']

    # 打印收敛过程表头
    logger.info(" --------------------------------------------------------------------------------------------------")
    logger.info(" | 当前帧(t2)           | 物理位移(Δd) | 宽膨胀(ΔW) | 高膨胀(ΔH) | Z_w(宽算) | Z_h(高算) | 融合锚点 |")
    logger.info(" --------------------------------------------------------------------------------------------------")

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
                status = f"Cross-Validated (Avg, 偏差 {diff*100:.1f}%)"
            else:
                Z_final = Z_h if delta_h > delta_w else Z_w
                status = f"Arbitrated (Max SNR, 偏差 {diff*100:.1f}%)"
        elif Z_w:
            Z_final = Z_w
            status = "Single-Dim (W_only)"
        elif Z_h:
            Z_final = Z_h
            status = "Single-Dim (H_only)"
        else:
            status = "Failed (Δ < 8px)"
            
        csv_results.append({
            't1_fname': f_start['fname'], 't2_fname': f_end['fname'], 't2_timestamp': f_end['ts'],
            'delta_d': delta_d, 'w1': w1, 'h1': h1, 'w2': w2, 'h2': h2,
            'delta_w': delta_w, 'delta_h': delta_h,
            'Z_w': Z_w if Z_w else -1, 'Z_h': Z_h if Z_h else -1,
            'Z_final': Z_final if Z_final else -1, 'status': status
        })
        
        # 将每一帧的计算过程打印在日志里，恢复终端表格输出
        log_zw = f"{Z_w:.3f}" if Z_w else "N/A"
        log_zh = f"{Z_h:.3f}" if Z_h else "N/A"
        log_zfinal = f"{Z_final:.3f}" if Z_final else "N/A"
        logger.info(f" | {f_end['fname'][-20:]:<20} | {delta_d:>11.4f} | {delta_w:>9.2f} | {delta_h:>9.2f} | {log_zw:>9} | {log_zh:>9} | {log_zfinal:>8} |")
        
        # 最后一帧输出终极备忘录总结
        if i == len(frames) - 1:
            logger.info(" --------------------------------------------------------------------------------------------------")
            logger.info("")
            logger.info("="*60)
            logger.info(" 📚 [原理备忘录] 本地解算逻辑回顾")
            logger.info("  1. Looming公式 : Z_2 = (W_2 * Δd) / ΔW  (无需真实物理大小)")
            logger.info("  2. 双向验证原理: 理想刚体宽高解算的 Z 必须相等。")
            logger.info("  3. 测谎仪与融合: 若 |Z_w - Z_h| 偏差 < 15%，视为纯白噪声，取均值滤波；")
            logger.info("                   若偏差 >= 15%，说明某维提取异常，强制取膨胀量大的维度(高信噪比)。")
            logger.info("="*60)
            logger.info(" 🎯 [终极结论] 最终状态与物理意义解读")
            logger.info(f"  - 远端参照帧(t1) : {f_start['fname']}")
            logger.info(f"  - 当前最新帧(t2) : {f_end['fname']}")
            logger.info(f"  - 期间车辆总位移 : {delta_d:.4f} units")
            logger.info(f"  - 双维解算报告   : 宽引擎得出 {log_zw}，高引擎得出 {log_zh} ({status})")
            logger.info(f"  - 📍 物理结论    : 在拍下 t2 最新帧的瞬间，相机光心距离标志牌的绝对真实深度为 【 {Z_final:.4f} 】 units。")
            logger.info("="*60)

    # ---------------------------------------------------------
    # 6. 将实验数据落盘为 CSV 文件
    # ---------------------------------------------------------
    logger.info(f">>> [阶段3] 正在将连续收敛数据落盘至 CSV: {CSV_FILE}")
    keys = csv_results[0].keys()
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(csv_results)
        
    # ---------------------------------------------------------
    # 7. 调用 3D 绘图引擎
    # ---------------------------------------------------------
    plot_spatial_trajectory_3d(frames, csv_results)
    logger.info(">>> 🎉 全部实验流程执行完毕！")

if __name__ == "__main__":
    main()