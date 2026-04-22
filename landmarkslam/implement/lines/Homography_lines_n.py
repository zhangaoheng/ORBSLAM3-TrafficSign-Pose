import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==========================================
# 导包：调用你的两个底层模块
# ==========================================
import sys
import os
# 【自动适配系统路径】为了支持其他目录 (如 main/) 跨文件夹导入，自动将当前文件目录加入环境
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lines_tool import detect_and_filter_lines_plslam
from Homography_loftr_tool import match_images_with_loftr_roi, visualize_matches_one_by_one

# ==========================================
# 🌟 全新数学引擎：汉字 3D 正交性校验 (Orthogonality Check)
# ==========================================
def backproject_line_to_plane(line_2d, n, K_inv):
    """将图像上的 2D 线段逆投影到指定的 3D 平面上，返回 3D 空间中的方向向量"""
    # 获取 2D 端点 (齐次坐标)
    p1 = np.array([line_2d[0], line_2d[1], 1.0])
    p2 = np.array([line_2d[2], line_2d[3], 1.0])

    # 转换为 3D 射线方向
    ray1 = K_inv @ p1
    ray2 = K_inv @ p2

    # 射线与平面 (n^T * X = 1) 求精确交点
    # s = 1 / (n^T * ray)
    dot1 = np.dot(n, ray1)
    dot2 = np.dot(n, ray2)
    
    # 防止分母为0 (射线平行于平面)
    if abs(dot1) < 1e-6 or abs(dot2) < 1e-6: return None

    P1_3d = (1.0 / dot1) * ray1
    P2_3d = (1.0 / dot2) * ray2

    # 计算 3D 空间中的线段方向向量并归一化
    vec_3d = P2_3d - P1_3d
    norm = np.linalg.norm(vec_3d)
    if norm < 1e-6: return None
    return vec_3d / norm

def evaluate_orthogonality(h_lines, v_lines, n, K):
    """
    计算横竖笔画在这个 3D 候选平面上的垂直程度。
    返回值 (error) 越接近 0，说明十字架越完美，越接近真实物理平面。
    """
    K_inv = np.linalg.inv(K)
    
    # 1. 挑选最长的高质量线段以抵抗噪点
    h_lines_sorted = sorted(h_lines, key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2, reverse=True)
    v_lines_sorted = sorted(v_lines, key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2, reverse=True)
    
    # 取前 15 条最长的横线和竖线进行交叉验证
    top_h = h_lines_sorted[:15]
    top_v = v_lines_sorted[:15]

    # 2. 将它们逆投影到当前的候选 3D 平面上
    h_vecs_3d = [backproject_line_to_plane(l, n, K_inv) for l in top_h]
    v_vecs_3d = [backproject_line_to_plane(l, n, K_inv) for l in top_v]
    
    h_vecs_3d = [v for v in h_vecs_3d if v is not None]
    v_vecs_3d = [v for v in v_vecs_3d if v is not None]

    if not h_vecs_3d or not v_vecs_3d: 
        return float('inf') # 投影失败，说明是个无效平面

    # 3. 交叉计算所有横竖线的点乘绝对值平均误差
    ortho_error = 0.0
    count = 0
    for vh in h_vecs_3d:
        for vv in v_vecs_3d:
            ortho_error += abs(np.dot(vh, vv)) # 完美的 90 度，点乘结果为 0
            count += 1
            
    return ortho_error / count if count > 0 else float('inf')

# ==========================================
# 📊 3D 交互式可视化窗口
# ==========================================
def visualize_3d_scene(R, t, n, K, roi):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("SLAM 3D Viewer: Relative Pose & True Planar Target", fontsize=14)

    # 1. 绘制相机 1 (作为世界原点)
    C1 = np.array([0.0, 0.0, 0.0])
    ax.scatter(*C1, color='black', s=50, marker='s')
    ax.text(*C1, " Cam 1 (Ref)", color='black', fontsize=12)
    ax.quiver(*C1, 1, 0, 0, color='r', length=0.2, arrow_length_ratio=0.2)
    ax.quiver(*C1, 0, 1, 0, color='g', length=0.2, arrow_length_ratio=0.2)
    ax.quiver(*C1, 0, 0, 1, color='b', length=0.2, arrow_length_ratio=0.2)

    # 2. 计算并绘制相机 2
    R_inv = R.T
    C2 = (-R_inv @ t).flatten()
    
    ax.scatter(*C2, color='black', s=50, marker='^')
    ax.text(*C2, " Cam 2", color='black', fontsize=12)
    ax.quiver(*C2, *R_inv[:, 0], color='r', length=0.2, arrow_length_ratio=0.2)
    ax.quiver(*C2, *R_inv[:, 1], color='g', length=0.2, arrow_length_ratio=0.2)
    ax.quiver(*C2, *R_inv[:, 2], color='b', length=0.2, arrow_length_ratio=0.2)

    ax.plot([C1[0], C2[0]], [C1[1], C2[1]], [C1[2], C2[2]], color='gray', linestyle='--')

    # 3. 基于你点击的4个物理形变角点，计算在三维空间中这块平面的真实轮廓
    pts = np.array(roi, dtype="float32")
    # 对4个角点进行几何排序: 左上(TL), 右上(TR), 右下(BR), 左下(BL)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]     # 左上: x+y 最小
    rect[2] = pts[np.argmax(s)]     # 右下: x+y 最大
    diff = np.diff(pts, axis=1)     # y - x
    rect[1] = pts[np.argmin(diff)]  # 右上: y-x 最小 (即x很大,y很小)
    rect[3] = pts[np.argmax(diff)]  # 左下: y-x 最大 (即x很小,y很大)

    K_inv = np.linalg.inv(K)

    def get_3d_point(p_2d):
        ray = K_inv @ np.array([p_2d[0], p_2d[1], 1.0])
        s = 1.0 / np.dot(n, ray)
        return s * ray

    # 逆投影获取三维空间中物理纸张真实的 4 个顶点
    P_TL = get_3d_point(rect[0])
    P_TR = get_3d_point(rect[1])
    P_BR = get_3d_point(rect[2])
    P_BL = get_3d_point(rect[3])

    plane_center = P_TL # 【核心】将左上角作为平面的正交原点
    vec_x = P_TR - P_TL # X轴（物体的长）
    vec_y = P_BL - P_TL # Y轴（物体的宽）
    
    # 确保 Z 轴正向总是朝向相机的！(抵消opencv归一化的翻转)
    Z_axis = -n if n[2] > 0 else n 
    
    # 绘制目标的局部坐标系
    ax_len = np.linalg.norm(vec_x) * 0.5 
    ax.quiver(*plane_center, *vec_x, color='red', length=1.0, linewidth=2.5) # 长
    ax.quiver(*plane_center, *vec_y, color='green', length=1.0, linewidth=2.5) # 宽
    ax.quiver(*plane_center, *Z_axis, color='blue', length=ax_len, linewidth=2.5, arrow_length_ratio=0.3) # 垂向Z
    
    ax.text(*plane_center, " Origin (TL)", color='black')

    # 将 4 个点连结成平面绘制
    poly = [[P_TL, P_TR, P_BR, P_BL]]
    ax.add_collection3d(Poly3DCollection(poly, alpha=0.4, facecolor='cyan', edgecolor='k'))

    # 4. 强制坐标轴比例 1:1:1
    all_pts = np.array([C1, C2, P_TL, P_TR, P_BL, P_BR])
    max_range = np.array([all_pts[:,0].max()-all_pts[:,0].min(), 
                          all_pts[:,1].max()-all_pts[:,1].min(), 
                          all_pts[:,2].max()-all_pts[:,2].min()]).max() / 2.0
    mid_x = (all_pts[:,0].max()+all_pts[:,0].min()) * 0.5
    mid_y = (all_pts[:,1].max()+all_pts[:,1].min()) * 0.5
    mid_z = (all_pts[:,2].max()+all_pts[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # （重要）翻转 Y/Z 轴应对相机投影，使它视觉自然！
    ax.invert_yaxis()
    ax.invert_zaxis()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    print("📺 3D 窗口已弹出！您可以使用鼠标左键拖拽旋转，右键缩放。")
    plt.show()

# ==========================================
# 终极主流程 (The Caller)
# ==========================================
if __name__ == "__main__":
    IMG1_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/Homography/1.png"
    IMG2_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/Homography/2.png"
    
    K = np.array([[429.78,  0,     429.94],
                  [ 0,      429.78, 241.57],
                  [ 0,       0,      1    ]], dtype=np.float64)

    img1 = cv2.imread(IMG1_PATH)
    img2 = cv2.imread(IMG2_PATH)

    print("\n" + "="*40)
    print(" 🛠️ 阶段 1：提取汉字骨架 (提供正交先验数据)")
    print("="*40)
    h_lines, v_lines, roi_target = detect_and_filter_lines_plslam(IMG1_PATH, "Image 1: Select Lines ROI")
    
    if len(h_lines) < 2 or len(v_lines) < 2:
        exit("❌ 有效线段太少，无法进行正交验证，程序终止。")
    print(f"✅ 成功提取汉字骨架：横向 {len(h_lines)} 条，竖向 {len(v_lines)} 条。准备进行 3D 逆投影大逃杀！")

    print("\n" + "="*40)
    print(" 🛠️ 阶段 2：启动 LoFTR 引擎解算数学位姿")
    print("="*40)
    pts1, pts2 = match_images_with_loftr_roi(img1, img2)
    
    if pts1 is None or len(pts1) < 4:
        exit("❌ 有效匹配点不足，程序终止。")

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    
    visualize_matches_one_by_one(img1, img2, pts1, pts2, mask)

    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)

    print("\n" + "="*40)
    print(" 🕵️‍♂️ 阶段 3：真理验算 (3D 正交性误差比对)")
    print("="*40)

    best_idx = -1
    min_error = float('inf') # 误差越小越好

    for i in range(num_solutions):
        n_cv = normals[i].flatten()
        
        # 关卡 1：物理常识，必须面向相机
        if n_cv[2] > 0:
            # 关卡 2：你的专属创新点！测量在这个 3D 平面上，汉字的横竖到底有多垂直！
            error = evaluate_orthogonality(h_lines, v_lines, n_cv, K)
            print(f"🟢 [候选解 {i+1}] 3D正交误差: {error:.4f}  |  数学解 n_cv: {np.round(n_cv, 4)}")
            
            if error < min_error:
                min_error = error
                best_idx = i

    if best_idx != -1:
        print("\n" + "🏆"*20)
        print(f"🎉 验算完毕！【解 {best_idx+1}】完美通过了汉字 3D 正交检验，是唯一真实姿态！")
        
        best_R = rotations[best_idx]
        best_t = translations[best_idx]
        best_n = normals[best_idx].flatten()
        
        print(f"📍 归一化平移 t :\n{np.round(best_t.flatten(), 4)}")
        print(f"🌀 绝对旋转矩阵 R :\n{np.round(best_R, 4)}")
        print("🏆"*20 + "\n")

        # 🌟 弹出 3D 窗口 (传入框选信息和内参用于还原长宽)
        visualize_3d_scene(best_R, best_t, best_n, K, roi_target)
        
    else:
        print("⚠️ 所有的解都不符合物理规律！")