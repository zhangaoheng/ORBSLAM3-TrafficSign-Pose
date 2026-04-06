import numpy as np
import pandas as pd
import cv2
import os
import math
from scipy.spatial.transform import Rotation as R

# =========================================================
# 1. 核心配置与文件路径
# =========================================================
TRAJ_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/output/FrameTrajectory_TUM_20260331_154838.txt"
MAP_FILE  = "/home/zah/ORB_SLAM3-master/landmarkslam/output/Filename_Mapping_20260331_154838.csv"
BBOX_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/yolo_simulated_bboxes.txt"
IMAGE_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/data/capture_1774599243/black_data/" 

# 调试图像输出路径
DEBUG_OUT_DIR = "/home/zah/ORB_SLAM3-master/landmarkslam/output/looming_debug_vis/"

# =========================================================
# 2. 核心数学：SLAM位姿解析
# =========================================================
def get_T_WC(pose_tum):
    T_WC = np.eye(4)
    T_WC[:3, :3] = R.from_quat(pose_tum[3:7]).as_matrix()
    T_WC[:3, 3] = pose_tum[0:3]
    return T_WC

# =========================================================
# 3. 核心创新：基于长度海选与空间极值的边缘重构 (V4.1 - 纯净版)
# =========================================================
def extract_physical_hw(img_path, u1, v1, u2, v2, fname):
    """
    通过线段长度锁定物理边界，不再显示 YOLO 红色参考框
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    
    # 1. ROI 结界外扩
    pad = 2
    crop_u1, crop_v1 = max(0, int(u1) - pad), max(0, int(v1) - pad)
    crop_u2, crop_v2 = min(img_w, int(u2) + pad), min(img_h, int(v2) + pad)
    
    roi = gray[crop_v1:crop_v2, crop_u1:crop_u2]
    roi_h, roi_w = roi.shape
    
    # 2. 图像增强与亚像素线段探测
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(clahe.apply(roi))
    
    fallback_w, fallback_h = float(u2 - u1), float(v2 - v1)
    vis_img = img.copy()

    if lines is None:
        cv2.imwrite(os.path.join(DEBUG_OUT_DIR, fname), vis_img)
        return fallback_w, fallback_h
        
    all_x, all_y = [], []
    
    # 3. 长度海选逻辑
    min_len = min(roi_w, roi_h) * 0.4 
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.hypot(x2-x1, y2-y1)
        
        if length > min_len:
            gx1, gy1 = x1 + crop_u1, y1 + crop_v1
            gx2, gy2 = x2 + crop_u1, y2 + crop_v1
            all_x.extend([gx1, gx2])
            all_y.extend([gy1, gy2])
            # 参与计算的长线画绿色（变薄一点，减少视觉干扰）
            cv2.line(vis_img, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 255, 0), 1)
                
    # 4. 空间极值挤压与长宽比保底
    final_w, final_h = fallback_w, fallback_h
    w_triggered, h_triggered = True, True
    fallback_ratio = fallback_w / fallback_h

    # 宽度计算
    if len(all_x) >= 2:
        ext_w = max(all_x) - min(all_x)
        if 0.85 < (ext_w / fallback_h) / fallback_ratio < 1.15:
            final_w = float(ext_w)
            w_triggered = False
            # 紫色宽度锁定线
            cv2.line(vis_img, (int(min(all_x)), int(v1)), (int(min(all_x)), int(v2)), (255, 0, 255), 2)
            cv2.line(vis_img, (int(max(all_x)), int(v1)), (int(max(all_x)), int(v2)), (255, 0, 255), 2)

    # 高度计算
    if len(all_y) >= 2:
        ext_h = max(all_y) - min(all_y)
        if 0.85 < (fallback_w / ext_h) / fallback_ratio < 1.15:
            final_h = float(ext_h)
            h_triggered = False
            # 紫色高度锁定线
            cv2.line(vis_img, (int(u1), int(min(all_y))), (int(u2), int(min(all_y))), (255, 0, 255), 2)
            cv2.line(vis_img, (int(u1), int(max(all_y))), (int(u2), int(max(all_y))), (255, 0, 255), 2)

    # 仅在保底触发时显示黄色文字提示，平时不显示
    if w_triggered or h_triggered:
        cv2.putText(vis_img, "Fallback Mode", (int(u1), int(v2)+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
    cv2.imwrite(os.path.join(DEBUG_OUT_DIR, fname), vis_img)
    return final_w, final_h

# =========================================================
# 4. 主程序
# =========================================================
def main():
    print(">>> [系统启动] 正在初始化【长度优先策略-纯净版】Looming 探测器...")
    os.makedirs(DEBUG_OUT_DIR, exist_ok=True)

    traj_data = np.loadtxt(TRAJ_FILE)
    traj_dict = {row[0]: row[1:] for row in traj_data}
    all_ts = np.array(list(traj_dict.keys()))
    mapping_df = pd.read_csv(MAP_FILE)
    bbox_df = pd.read_csv(BBOX_FILE, header=None, names=['fname', 'u1', 'v1', 'u2', 'v2'], sep=',', comment='#')
    
    frames = []
    print(">>> 正在执行亚像素边缘重构...")
    
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
            
        w, h = extract_physical_hw(img_path, u1, v1, u2, v2, fname)
        if w is None or h is None: continue
        
        frames.append({'fname': fname, 'width': w, 'height': h, 'pos_w': T_WC[:3, 3]})
        
    if len(frames) < 2:
        print("❌ 错误：有效帧数不足。")
        return

    f_start, f_end = frames[0], frames[-1]
    w1, h1, w2, h2 = f_start['width'], f_start['height'], f_end['width'], f_end['height']
    delta_d = np.linalg.norm(f_end['pos_w'] - f_start['pos_w'])

    print("\n" + "="*50)
    print(" 🛠️ 亚像素双维膨胀数据面板")
    print("="*50)
    print(f" 远端 (t1): {f_start['fname']} | W1={w1:.2f}, H1={h1:.2f}")
    print(f" 近端 (t2): {f_end['fname']} | W2={w2:.2f}, H2={h2:.2f}")
    print(f" 物理位移 (Δd): {delta_d:.4f} units")
    print("-" * 50)

    delta_w, delta_h = w2 - w1, h2 - h1
    Z_w = (w2 * delta_d) / delta_w if delta_w > 8.0 else None
    Z_h = (h2 * delta_d) / delta_h if delta_h > 8.0 else None
        
    print("\n 🚀 [联合解算结果]")
    if Z_w and Z_h:
        diff = abs(Z_w - Z_h) / max(Z_w, Z_h)
        print(f"   ├─ X轴深度 (Z_w): {Z_w:.4f}\n   ├─ Y轴深度 (Z_h): {Z_h:.4f}")
        if diff < 0.15:
            print(f"   🟢 交叉验证通过! 最终锚点: 【 {(Z_w + Z_h) / 2.0:.4f} 】")
        else:
            final_z = Z_h if delta_h > delta_w else Z_w
            print(f"   🔴 偏差过大({diff*100:.1f}%)，已选择高置信度维度。")
            print(f"   🎯 建议锚点: 【 {final_z:.4f} 】")
    elif Z_w or Z_h:
        print(f"   🟠 单维保底锚点: 【 {Z_w if Z_w else Z_h:.4f} 】")

    print("="*50)

if __name__ == "__main__":
    main()