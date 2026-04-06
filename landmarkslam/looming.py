import numpy as np
import pandas as pd
import cv2
import os
import math
from scipy.spatial.transform import Rotation as R

# =========================================================
# 1. 核心配置与文件路径 (请替换为你本地的真实路径)
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
# 3. 核心创新：由粗到精的亚像素物理边框提取 (带可视化与长宽比护城河)
# =========================================================
def extract_physical_hw(img_path, u1, v1, u2, v2, fname):
    """
    YOLO 提供 ROI 结界，LSD 提取真实的亚像素物理宽高
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    
    # 1. 结界外扩 (Padding)
    pad = 10
    crop_u1 = max(0, int(u1) - pad)
    crop_v1 = max(0, int(v1) - pad)
    crop_u2 = min(img_w, int(u2) + pad)
    crop_v2 = min(img_h, int(v2) + pad)
    
    roi = gray[crop_v1:crop_v2, crop_u1:crop_u2]
    roi_h, roi_w = roi.shape
    
    # 2. 逆光增强 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(roi)
    
    # 3. 提取亚像素线段
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(roi_enhanced)
    
    fallback_w = float(u2 - u1)
    fallback_h = float(v2 - v1)
    
    # ====== 可视化画板准备 ======
    vis_img = img.copy()
    cv2.rectangle(vis_img, (int(u1), int(v1)), (int(u2), int(v2)), (0, 0, 255), 2)
    cv2.putText(vis_img, "YOLO ROI", (int(u1), int(v1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if lines is None:
        cv2.imwrite(os.path.join(DEBUG_OUT_DIR, fname), vis_img)
        return fallback_w, fallback_h
        
    horiz_y = []
    vert_x = []
    
    # 4. 剔除碎线与角度过滤
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2-y1, x2-x1))) % 180
        length = math.hypot(x2-x1, y2-y1)
        
        gx1, gy1 = int(x1 + crop_u1), int(y1 + crop_v1)
        gx2, gy2 = int(x2 + crop_u1), int(y2 + crop_v1)
        
        if angle < 15 or angle > 165: # 横线
            if length > roi_w * 0.3:
                horiz_y.extend([gy1, gy2])
                cv2.line(vis_img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 1)
        elif abs(angle - 90) < 15:    # 竖线
            if length > roi_h * 0.3:
                vert_x.extend([gx1, gx2])
                cv2.line(vis_img, (gx1, gy1), (gx2, gy2), (255, 0, 0), 1)
                
    # 5. 👉 终极护城河：基于物理常识的长宽比保底校验
    fallback_ratio = fallback_w / fallback_h
    
    final_h = fallback_h
    h_triggered = True
    if len(horiz_y) >= 2:
        extracted_h = float(max(horiz_y) - min(horiz_y))
        extracted_ratio = fallback_w / extracted_h
        # 容忍度设为 10%：偏离过大说明抓到了标志牌内部文字
        if abs(extracted_ratio - fallback_ratio) / fallback_ratio < 0.10:
            final_h = extracted_h
            h_triggered = False
            # 画高度锁定紫线
            y_min, y_max = min(horiz_y), max(horiz_y)
            cv2.line(vis_img, (int(u1), y_min), (int(u2), y_min), (255, 0, 255), 2)
            cv2.line(vis_img, (int(u1), y_max), (int(u2), y_max), (255, 0, 255), 2)

    final_w = fallback_w
    w_triggered = True
    if len(vert_x) >= 2:
        extracted_w = float(max(vert_x) - min(vert_x))
        extracted_ratio = extracted_w / fallback_h
        if abs(extracted_ratio - fallback_ratio) / fallback_ratio < 0.10:
            final_w = extracted_w
            w_triggered = False
            # 画宽度锁定紫线
            x_min, x_max = min(vert_x), max(vert_x)
            cv2.line(vis_img, (x_min, int(v1)), (x_min, int(v2)), (255, 0, 255), 2)
            cv2.line(vis_img, (x_max, int(v1)), (x_max, int(v2)), (255, 0, 255), 2)

    # 标注状态
    if w_triggered or h_triggered:
        cv2.putText(vis_img, f"Fallback! W:{w_triggered} H:{h_triggered}", 
                    (int(u1), int(v2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
    cv2.imwrite(os.path.join(DEBUG_OUT_DIR, fname), vis_img)
    
    return float(final_w), float(final_h)

# =========================================================
# 4. 主程序：双维交叉验证 Looming 管线
# =========================================================
def main():
    print(">>> [系统启动] 正在初始化【双维交叉验证】Looming 深度探测器...")

    os.makedirs(DEBUG_OUT_DIR, exist_ok=True)
    print(f">>> 调试图像将输出至: {DEBUG_OUT_DIR}")

    traj_data = np.loadtxt(TRAJ_FILE)
    traj_dict = {row[0]: row[1:] for row in traj_data}
    all_ts = np.array(list(traj_dict.keys()))
    mapping_df = pd.read_csv(MAP_FILE)
    
    bbox_df = pd.read_csv(BBOX_FILE, header=None, names=['fname', 'u1', 'v1', 'u2', 'v2'], sep=',', comment='#')
    
    frames = []
    print(">>> 正在切片图像，执行亚像素级边框重构并生成可视化...")
    
    for _, row in bbox_df.iterrows():
        fname = str(row['fname']).strip()
        match_rows = mapping_df[mapping_df['filename'] == fname]
        if match_rows.empty: continue
            
        ts = match_rows.iloc[0]['timestamp_s']
        closest_ts = all_ts[np.argmin(np.abs(all_ts - ts))]
        T_WC = get_T_WC(traj_dict[closest_ts])
        
        u1, v1 = float(row['u1']), float(row['v1'])
        u2, v2 = float(row['u2']), float(row['v2'])
        
        img_path = os.path.join(IMAGE_DIR, fname)
        if not os.path.exists(img_path): continue
            
        w, h = extract_physical_hw(img_path, u1, v1, u2, v2, fname)
        
        if w is None or h is None: continue
        
        frames.append({
            'fname': fname,
            'width': w,
            'height': h,
            'pos_w': T_WC[:3, 3] 
        })
        
    print(f"\n>>> 雷达扫描与画图完毕！成功收集到有效帧数: {len(frames)} 帧")
        
    if len(frames) < 2:
        print("❌ 错误：有效帧数不足。")
        return

    f_start = frames[0]
    f_end = frames[-1]
    
    w1, h1 = f_start['width'], f_start['height']
    w2, h2 = f_end['width'], f_end['height']
    delta_d = np.linalg.norm(f_end['pos_w'] - f_start['pos_w'])

    print("\n" + "="*50)
    print(" 🛠️ 亚像素双维膨胀数据面板")
    print("="*50)
    print(f" 远端 (t1): {f_start['fname']} | W1={w1:.2f}, H1={h1:.2f}")
    print(f" 近端 (t2): {f_end['fname']} | W2={w2:.2f}, H2={h2:.2f}")
    print(f" 物理位移 (Δd): {delta_d:.4f} units")
    
    delta_w = w2 - w1
    delta_h = h2 - h1
    print(f" 膨胀梯度: ΔW = {delta_w:.2f}px, ΔH = {delta_h:.2f}px")
    print("-" * 50)

    MIN_GROWTH = 8.0 
    Z_w, Z_h = None, None
    
    if delta_w >= MIN_GROWTH:
        Z_w = (w2 * delta_d) / delta_w
    else:
        print(" ⚠️ 宽度膨胀不足，X轴维度暂不可观。")
        
    if delta_h >= MIN_GROWTH:
        Z_h = (h2 * delta_d) / delta_h
    else:
        print(" ⚠️ 高度膨胀不足，Y轴维度暂不可观。")
        
    print("\n 🚀 [联合解算结果]")
    if Z_w and Z_h:
        diff_ratio = abs(Z_w - Z_h) / max(Z_w, Z_h)
        print(f"   ├─ X轴独立深度 (Z_w): {Z_w:.4f}")
        print(f"   ├─ Y轴独立深度 (Z_h): {Z_h:.4f}")
        
        if diff_ratio < 0.15:
            Z_final = (Z_w + Z_h) / 2.0
            print(f"   🟢 交叉验证通过 (偏差 {diff_ratio*100:.1f}%)。统计学方差减半触发！")
            print(f"   🎯 最终高置信度锚点深度: 【 {Z_final:.4f} 】 units")
        else:
            print(f"   🔴 交叉验证警告 (偏差 {diff_ratio*100:.1f}%)。疑似一侧物理边框被杂纹污染。")
            Z_final = Z_h if delta_h > delta_w else Z_w
            print(f"   🎯 系统已自适应采用高信噪比维度锚点: 【 {Z_final:.4f} 】 units")
            
    elif Z_w:
        print(f"   🟠 单维保底(仅宽可观)锚点: 【 {Z_w:.4f} 】 units")
    elif Z_h:
        print(f"   🟠 单维保底(仅高可观)锚点: 【 {Z_h:.4f} 】 units")

    print("="*50)

if __name__ == "__main__":
    main()