import sys
import os
import cv2
import numpy as np
import glob
import math
import matplotlib.pyplot as plt

# 配置环境变量以导入子文件夹中的模块
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# --- 1. 从 lines 提取位姿/法向单应性约束 ---
from lines.lines_tool import detect_and_filter_lines_plslam
from lines.Homography_lines_n import evaluate_orthogonality

# --- 2. 从 mid_FOE_Z_d 提取膨胀深度测距与 SLAM 轨迹关联 ---
from mid_FOE_Z_d.pure_looming_depth import (
    load_tum_trajectory, get_closest_pose, 
    calculate_relative_motion, calculate_pure_looming_Z, load_saved_rois, FRAME_STEP, fx, fy, cx, cy
)
from tools.mid import extract_four_lines_from_real_image, calculate_rectangle_center
from mid_FOE_Z_d.imgs_mid import process_sequence_with_cached_rois

# ======= 相机内参 (以 D456 为例) =======
K = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float64)
K_inv = np.linalg.inv(K)

# ==================================================================
# 🌟 1. GUI 选帧模块
# ==================================================================
def select_frame_gui(images, default_idx, sequence_name):
    """弹出窗口展示图像序列以供选择"""
    idx = default_idx if default_idx != -1 else 0
    window_name = f"Select Frame for {sequence_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        img = cv2.imread(images[idx])
        if img is None:
            break
            
        display_img = img.copy()
        cv2.putText(display_img, f"Seq: {sequence_name} | Frame: {idx}/{len(images)-1}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display_img, "[SPACE] or [D]: Next | [A]: Prev | [ENTER]: Select", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(0) & 0xFF
        
        if key == 32 or key == ord('d'):
            idx = (idx + 1) % len(images)
        elif key == ord('a'):
            idx = (idx - 1) % len(images)
        elif key == 13: # 回车选定
            break
        elif key == 27: # ESC退出
            break
            
    cv2.destroyWindow(window_name)
    return idx

# ==================================================================
# 🌟 2. 鼠标点击 4 个点模块
# ==================================================================
def get_four_points_manually(img, window_name="Select 4 Points"):
    """
    弹出一个窗口，让用户手动点击4个点。
    返回一个 Nx2 的 numpy 数组 (float32)。
    """
    points = []
    img_copy = img.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append([x, y])
                cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(img_copy, str(len(points)), (x + 10, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow(window_name, img_copy)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img_copy)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\n👉 请在弹出的 '{window_name}' 窗口中点击 4 个点。")
    print("   (⚠️ 请确保两张图的点击顺序完全一致！建议：左上 -> 右上 -> 右下 -> 左下)")
    
    while True:
        cv2.waitKey(10)
        if len(points) == 4:
            cv2.waitKey(500) # 停顿半秒让用户看到第四个点
            break
            
    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.float32)

# ==================================================================
# 🌟 3. 3D姿态与图像联合可视化模块
# ==================================================================
def draw_camera(ax, R, t, label, scale=0.4, color='black'):
    t = t.flatten()
    ax.scatter(*t, color=color, s=50)
    ax.text(t[0], t[1], t[2], f'  {label}', fontsize=10, weight='bold', color=color)
    
    x_ax = t + R @ np.array([scale, 0, 0])
    y_ax = t + R @ np.array([0, scale, 0])
    z_ax = t + R @ np.array([0, 0, scale])
    
    ax.plot([t[0], x_ax[0]], [t[1], x_ax[1]], [t[2], x_ax[2]], color='r', linewidth=2)
    ax.plot([t[0], y_ax[0]], [t[1], y_ax[1]], [t[2], y_ax[2]], color='g', linewidth=2)
    ax.plot([t[0], z_ax[0]], [t[1], z_ax[1]], [t[2], z_ax[2]], color='b', linewidth=2)

def visualize_dashboard(img1, img2, R_rel, t_real, X_3d, n_cv):
    fig = plt.figure(figsize=(18, 6))
    fig.canvas.manager.set_window_title("Metric Pose & 3D Spatial Layout")
    
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax1.set_title("Sequence 1 (Base - With Z_Looming)")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax2.set_title("Sequence 2 (Target)")
    ax2.axis('off')
    
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.set_title("3D Relative Pose & Sign Position")
    
    origin1 = np.array([0, 0, 0])
    R1 = np.eye(3)
    draw_camera(ax3, R1, origin1, "Cam1 (Ref)", color='black')
    
    origin2 = (-R_rel.T @ t_real).flatten()
    R2 = R_rel.T
    draw_camera(ax3, R2, origin2, "Cam2", color='dodgerblue')
    
    target = X_3d.flatten()
    ax3.scatter(*target, color='orange', s=100, marker='*')
    ax3.text(target[0], target[1], target[2], '  Sign Center', color='darkorange')
    
    normal_end = target + n_cv.flatten() * 0.5 
    ax3.plot([target[0], normal_end[0]], [target[1], normal_end[1]], [target[2], normal_end[2]], 
             color='purple', linestyle='-', linewidth=2, label='Surface Normal')
    
    ax3.plot([origin1[0], target[0]], [origin1[1], target[1]], [origin1[2], target[2]], 'k:', alpha=0.4)
    ax3.plot([origin2[0], target[0]], [origin2[1], target[1]], [origin2[2], target[2]], 'b:', alpha=0.4)
    
    all_pts = np.vstack([origin1, origin2, target, normal_end])
    max_range = np.array([all_pts[:,0].max()-all_pts[:,0].min(), 
                          all_pts[:,1].max()-all_pts[:,1].min(), 
                          all_pts[:,2].max()-all_pts[:,2].min()]).max() / 2.0
    mid_x = (all_pts[:,0].max()+all_pts[:,0].min()) * 0.5
    mid_y = (all_pts[:,1].max()+all_pts[:,1].min()) * 0.5
    mid_z = (all_pts[:,2].max()+all_pts[:,2].min()) * 0.5
    
    ax3.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3.set_zlim(mid_z - max_range, mid_z + max_range)
    ax3.set_xlabel('X (Right)')
    ax3.set_ylabel('Y (Down)')
    ax3.set_zlabel('Z (Forward)')
    
    ax3.invert_yaxis()
    ax3.view_init(elev=-25, azim=-45)
    
    plt.tight_layout()
    plt.show()

# ==================================================================
# 核心主流程
# ==================================================================
def integrate_and_solve_metric_pose():
    # ==== 路径配置 ====
    FOLDER_PATH_1 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/lines2/cam0/data"
    TRAJ_PATH_1 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/output/all_frames_lines2_slam_traj.txt"
    ROI_PATH_1 = os.path.join(FOLDER_PATH_1, "saved_rois.txt")

    FOLDER_PATH_2 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/lines1/cam0/data"
    TRAJ_PATH_2 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/output/all_frames_lines1_slam_traj.txt"

    images1 = sorted(glob.glob(os.path.join(FOLDER_PATH_1, "*.png")))
    images2 = sorted(glob.glob(os.path.join(FOLDER_PATH_2, "*.png")))
    
    slam_poses1 = load_tum_trajectory(TRAJ_PATH_1)

    print(f">>> 🚀 启动 [物理度量恢复] 跨序列双剑合璧！(手动点4点 + Looming测Z + H矩阵正交检验)...")
    
    # ---------------------------------------------------------
    # 1. GUI 手动挑选计算帧 
    # ---------------------------------------------------------
    idx1_base, idx2_base = 0, 0 
    cache_file = os.path.join(os.path.dirname(__file__), "selected_frames_cache.txt")
    
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            lines = f.read().strip().split()
            if len(lines) >= 2:
                idx1_base = int(lines[0])
                idx2_base = int(lines[1])
        print(f"\n📁 发现缓存文件，已自动加载选帧：Lines1[{idx1_base}] <---> Lines2[{idx2_base}]")
        print(f"   (如需重新使用窗口选帧，请删除 {cache_file} 文件)\n")
    else:
        print("\n🔔 [GUI 选帧模式] 正在打开窗口，请在弹出的图像窗口中操作...")
        idx1_base = select_frame_gui(images1, idx1_base, "Sequence 1 (Lines1)")
        idx2_base = select_frame_gui(images2, idx2_base, "Sequence 2 (Lines2)")
        
        print(f"👉 最终选定计算帧：Lines1_img[{idx1_base}] <---> Lines2_img[{idx2_base}]")
        with open(cache_file, "w") as f:
            f.write(f"{idx1_base}\n{idx2_base}")
        print(f"💾 选择结果已保存至 {cache_file}，下次运行将自动加载。\n")

    # ---------------------------------------------------------
    # 2. Looming 测距逻辑 (完全依赖序列 1 已有的框)
    # ---------------------------------------------------------
    print("👉 检验序列 1 (基准) 的连续测距掩码框...")
    process_sequence_with_cached_rois(FOLDER_PATH_1)
    saved_rois1 = load_saved_rois(ROI_PATH_1)

    idx1_next = idx1_base + FRAME_STEP
    if idx1_next >= len(images1) or idx1_next >= len(saved_rois1) or idx1_base >= len(saved_rois1):
        print("❌ Lines1 序列后续帧不足以计算 Looming，请调整 FRAME_STEP 或选取靠前的帧。")
        return

    time1_A = float(os.path.basename(images1[idx1_base]).replace(".png", "")) / 1e9
    time1_B = float(os.path.basename(images1[idx1_next]).replace(".png", "")) / 1e9

    pose1_A = get_closest_pose(time1_A, slam_poses1)
    pose1_B = get_closest_pose(time1_B, slam_poses1)

    R_12, t_12 = calculate_relative_motion(pose1_A, pose1_B)
    tx, ty, tz = t_12
    delta_d = tz
    if delta_d < 0.02:
        print(f"⚠️ 序列 1 的运动 delta_d ({delta_d:.4f}) 太小，无法计算 Looming Z！")
        return

    FOE = (fx * (tx / tz) + cx, fy * (ty / tz) + cy)
    
    img1_A = cv2.imread(images1[idx1_base])
    img1_B = cv2.imread(images1[idx1_next])
    
    # 序列 1 利用历史保存的框算交点 (仅用于获得中心点计算 Looming Z)
    lines1_A = extract_four_lines_from_real_image(img1_A, saved_rois1[idx1_base])
    lines1_B = extract_four_lines_from_real_image(img1_B, saved_rois1[idx1_next])
    
    if not lines1_A or not lines1_B:
        print("❌ 序列 1 提取 ROI 框失败")
        return

    center1_A_looming, corners1_A_looming = calculate_rectangle_center(*lines1_A)
    center1_B_looming, _ = calculate_rectangle_center(*lines1_B)

    # 获得关键深度 Z!
    Z_looming, _, _, _ = calculate_pure_looming_Z(center1_A_looming, center1_B_looming, FOE, delta_d, R_12)
    if Z_looming is None:
        print("❌ 序列 1 Looming 计算失败")
        return
    print(f"\n[Lines1 内部] ✅ (被动测距模块) FOE 物理深度计算成功: Z = {Z_looming:.3f} m")

    # ---------------------------------------------------------
    # 3. 手动点击 4 个角点进行 H 矩阵计算 (Seq 1 & Seq 2)
    # ---------------------------------------------------------
    print("\n🔄 启动跨序列对齐，请依次在两张图上点击 4 个相同的角点...")
    
    pts1 = get_four_points_manually(img1_A, "Seq 1 - Click 4 Points")
    if len(pts1) < 4:
        print("❌ 序列 1 选点不足，退出！")
        return

    img2_base = cv2.imread(images2[idx2_base])
    pts2 = get_four_points_manually(img2_base, "Seq 2 - Click 4 Points")
    if len(pts2) < 4:
        print("❌ 序列 2 选点不足，退出！")
        return
        
    print(f"\n👉 正在使用你点击的角点计算单应性矩阵 H ...")
    H, _ = cv2.findHomography(pts1, pts2, 0)
    
    if H is None:
        print("❌ 单应性矩阵 H 计算失败！")
        return

    # ---------------------------------------------------------
    # 4. 提取汉字骨架并进行正交验证筛选
    # ---------------------------------------------------------
    print("\n🛠️ [提取汉字骨架] 正在提取 Seq 1 中的汉字横竖线用于正交校验...")
    # 💡 提示：如果此函数内部还在弹出 cv2.selectROI，建议你去修改 lines_tool.py，
    # 💡 让它直接接收我们从 Looming 里读取的 saved_rois1[idx1_base] 作为截取范围，就可以真正告别画框了！
    h_lines, v_lines, _ = detect_and_filter_lines_plslam(images1[idx1_base], "Extract Char Lines (Seq 1)")
    if len(h_lines) < 2 or len(v_lines) < 2:
        print("❌ 有效线段太少，无法进行正交验证，程序终止。")
        return

    num_solutions, Rs, Ts, normals = cv2.decomposeHomographyMat(H, K)
    
    print("\n🕵️‍♂️ 启动真理验算 (利用汉字 3D 正交性误差筛选最佳姿态)...")
    best_idx = -1
    min_error = float('inf')

    for i in range(num_solutions):
        n_cv = normals[i].flatten()
        if n_cv[2] > 0: # 平面法向必须面向相机
            # 利用汉字横竖在3D空间的正交性验证物理法向
            error = evaluate_orthogonality(h_lines, v_lines, n_cv, K)
            print(f"🟢 [候选解 {i+1}] 正交误差: {error:.4f}  |  法向 n: {np.round(n_cv, 4)}")
            
            if error < min_error:
                min_error = error
                best_idx = i

    if best_idx == -1:
        print("❌ 所有的解都不符合物理规律！")
        return

    R_rel = Rs[best_idx]
    t_norm = Ts[best_idx]
    n_cv = normals[best_idx].flatten()
    print(f"🎉 验算完毕！【解 {best_idx+1}】完美通过汉字 3D 正交检验！")

    # ---------------------------------------------------------
    # 5. 利用 Looming 的 Z 恢复绝对平移尺度
    # ---------------------------------------------------------
    # 使用你手动点击的四个点的几何中心，这样计算 d 最符合你手点的平面位置
    center_pts1 = np.mean(pts1, axis=0)
    p_2d = np.array([center_pts1[0], center_pts1[1], 1.0]).reshape(3, 1)
    ray = K_inv @ p_2d
    
    # 利用序列 1 的深度 Z_looming 恢复 3D 绝对坐标
    X_3d = ray * (Z_looming / ray[2][0])
    # 反算摄像机到平面的绝对垂直距离 d
    d = float((n_cv.T @ X_3d)[0])
    
    # 把真正的尺度投射到无尺度的 t_norm 之上
    t_real = t_norm * abs(d)

    print("\n🏆 (两极反转) 双序列跨界绝对姿态提取结果！")
    print("="*40)
    print(f" -> Looming 计算出的深度 Z: {Z_looming:.3f} m")
    print(f" -> 摄像机到平面绝对垂直距离: {abs(d):.3f} m (代数 d={d:.3f})")
    print(f" -> 跨序列真实的绝对平移 t_real (m):\n{t_real}")
    print(f" -> 跨序列的旋转矩阵 R 变换:\n{R_rel}")
    print("="*40)

    # 🚀 压轴大戏：启动 3D 可视化面板！
    print("\n🎨 正在启动 3D 姿态与图像联合可视化窗口 (可拖拽旋转)...")
    visualize_dashboard(img1_A, img2_base, R_rel, t_real, X_3d, n_cv)

if __name__ == "__main__":
    integrate_and_solve_metric_pose()