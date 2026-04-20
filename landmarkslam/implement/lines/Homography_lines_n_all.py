import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==========================================
# 导包：调用你的两个底层模块
# ==========================================
from lines_tool import detect_and_filter_lines_plslam
from Homography_loftr_tool import match_images_with_loftr_roi, visualize_matches_one_by_one

# ==========================================
# 🌟 全新数学引擎：汉字 3D 正交性校验
# ==========================================
def backproject_line_to_plane(line_2d, n, K_inv):
    p1 = np.array([line_2d[0], line_2d[1], 1.0])
    p2 = np.array([line_2d[2], line_2d[3], 1.0])

    ray1 = K_inv @ p1
    ray2 = K_inv @ p2

    dot1 = np.dot(n, ray1)
    dot2 = np.dot(n, ray2)
    
    if abs(dot1) < 1e-6 or abs(dot2) < 1e-6: return None

    P1_3d = (1.0 / dot1) * ray1
    P2_3d = (1.0 / dot2) * ray2

    vec_3d = P2_3d - P1_3d
    norm = np.linalg.norm(vec_3d)
    if norm < 1e-6: return None
    return vec_3d / norm

def evaluate_orthogonality(h_lines, v_lines, n, K):
    K_inv = np.linalg.inv(K)
    
    h_lines_sorted = sorted(h_lines, key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2, reverse=True)
    v_lines_sorted = sorted(v_lines, key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2, reverse=True)
    
    top_h = h_lines_sorted[:15]
    top_v = v_lines_sorted[:15]

    h_vecs_3d = [backproject_line_to_plane(l, n, K_inv) for l in top_h]
    v_vecs_3d = [backproject_line_to_plane(l, n, K_inv) for l in top_v]
    
    h_vecs_3d = [v for v in h_vecs_3d if v is not None]
    v_vecs_3d = [v for v in v_vecs_3d if v is not None]

    if not h_vecs_3d or not v_vecs_3d: 
        return float('inf')

    ortho_error = 0.0
    count = 0
    for vh in h_vecs_3d:
        for vv in v_vecs_3d:
            ortho_error += abs(np.dot(vh, vv)) 
            count += 1
            
    return ortho_error / count if count > 0 else float('inf')

# ==========================================
# 📊 3D 交互式可视化窗口 (修改为支持多窗口)
# ==========================================
def visualize_3d_scene(R, t, n, sol_idx, error, is_best):
    """
    修改点：去掉了内部的 plt.show()，允许外部循环创建多个 Figure。
    添加了标题动态显示，标明是真实解还是假解。
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # 动态标题
    status = "TRUE POSE (Winner)" if is_best else "GHOST POSE (Fake)"
    title_color = "green" if is_best else "red"
    ax.set_title(f"Solution {sol_idx} | Error: {error:.4f} | {status}", fontsize=14, color=title_color, fontweight='bold')

    C1 = np.array([0.0, 0.0, 0.0])
    ax.scatter(*C1, color='black', s=50, marker='s')
    ax.text(*C1, " Cam 1", color='black', fontsize=12)
    ax.quiver(*C1, 1, 0, 0, color='r', length=0.2)
    ax.quiver(*C1, 0, 1, 0, color='g', length=0.2)
    ax.quiver(*C1, 0, 0, 1, color='b', length=0.2)

    R_inv = R.T
    C2 = (-R_inv @ t).flatten()
    
    ax.scatter(*C2, color='black', s=50, marker='^')
    ax.text(*C2, " Cam 2", color='black', fontsize=12)
    ax.quiver(*C2, *R_inv[:, 0], color='r', length=0.2)
    ax.quiver(*C2, *R_inv[:, 1], color='g', length=0.2)
    ax.quiver(*C2, *R_inv[:, 2], color='b', length=0.2)

    ax.plot([C1[0], C2[0]], [C1[1], C2[1]], [C1[2], C2[2]], color='gray', linestyle='--')

    plane_center = np.array([0, 0, 1.0 / n[2]])
    ax.quiver(*plane_center, *n, color='magenta', length=0.3, linewidth=2)
    ax.text(*plane_center, " Normal", color='magenta', fontsize=12)

    u = np.array([1, 0, 0])
    if np.abs(np.dot(u, n)) > 0.9: u = np.array([0, 1, 0])
    u = u - np.dot(u, n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    # 用不同的颜色区分真假平面
    plane_color = 'cyan' if is_best else 'salmon'
    
    s = 0.4 
    p1 = plane_center + s * u + s * v
    p2 = plane_center - s * u + s * v
    p3 = plane_center - s * u - s * v
    p4 = plane_center + s * u - s * v
    poly = [[p1, p2, p3, p4]]
    
    ax.add_collection3d(Poly3DCollection(poly, alpha=0.4, facecolor=plane_color, edgecolor='k'))

    all_pts = np.array([C1, C2, plane_center, p1, p2, p3, p4])
    max_range = np.array([all_pts[:,0].max()-all_pts[:,0].min(), 
                          all_pts[:,1].max()-all_pts[:,1].min(), 
                          all_pts[:,2].max()-all_pts[:,2].min()]).max() / 2.0
    mid_x = (all_pts[:,0].max()+all_pts[:,0].min()) * 0.5
    mid_y = (all_pts[:,1].max()+all_pts[:,1].min()) * 0.5
    mid_z = (all_pts[:,2].max()+all_pts[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

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
    h_lines, v_lines = detect_and_filter_lines_plslam(IMG1_PATH, "Image 1: Select Lines ROI")
    
    if len(h_lines) < 2 or len(v_lines) < 2:
        exit("❌ 有效线段太少，无法进行正交验证，程序终止。")
    print(f"✅ 成功提取汉字骨架。")

    print("\n" + "="*40)
    print(" 🛠️ 阶段 2：启动 LoFTR 引擎解算数学位姿")
    print("="*40)
    pts1, pts2 = match_images_with_loftr_roi(img1, img2)
    if pts1 is None or len(pts1) < 4: exit("❌ 有效匹配点不足。")

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    visualize_matches_one_by_one(img1, img2, pts1, pts2, mask)
    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)

    print("\n" + "="*40)
    print(" 🕵️‍♂️ 阶段 3：多解同时展示与验算对比")
    print("="*40)

    # 🌟 核心修改：保存所有合法的候选解
    valid_solutions = []

    for i in range(num_solutions):
        n_cv = normals[i].flatten()
        
        # 只要面朝相机，就将其收入“决赛圈”
        if n_cv[2] > 0:
            error = evaluate_orthogonality(h_lines, v_lines, n_cv, K)
            valid_solutions.append({
                'idx': i + 1,
                'R': rotations[i],
                't': translations[i],
                'n': n_cv,
                'error': error
            })
            print(f"🟢 [候选解 {i+1}] 3D正交误差: {error:.4f}  |  数学解 n_cv: {np.round(n_cv, 4)}")

    if not valid_solutions:
        exit("⚠️ 所有的解都在相机后方，计算彻底失败！")

    # 找出误差最小的那个作为“真理”
    best_sol = min(valid_solutions, key=lambda x: x['error'])

    print("\n" + "🏆"*20)
    print(f"🎉 验算对比完毕！")
    print(f"✅ 【解 {best_sol['idx']}】误差极小 ({best_sol['error']:.4f})，完美垂直！是真实姿态！")
    for sol in valid_solutions:
        if sol['idx'] != best_sol['idx']:
            print(f"❌ 【解 {sol['idx']}】误差极大 ({sol['error']:.4f})，严重畸变！是幽灵姿态！")
    print("🏆"*20 + "\n")

    # 🌟 为每一个有效解创建一个 3D 窗口
    print("📺 正在生成多视角 3D 对比窗口... 请注意任务栏，可能会有多个窗口弹出！")
    for sol in valid_solutions:
        is_best = (sol['idx'] == best_sol['idx'])
        visualize_3d_scene(sol['R'], sol['t'], sol['n'], sol['idx'], sol['error'], is_best)

    # 最后统一 show，让两个窗口同时停留在屏幕上供你对比
    plt.show()