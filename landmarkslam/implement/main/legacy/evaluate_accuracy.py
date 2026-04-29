#!/usr/bin/env python3
"""
精度评估脚本 —— 对比 Looming 深度、正交筛选、绝对位姿与真值
使用方法：
    1. 修改下方“用户配置区”中的路径和参数
    2. 运行 python evaluate_accuracy.py
"""

import os
import sys
import cv2
import numpy as np
import math
import re
import json
from scipy.spatial.transform import Rotation as R_scipy

# ==================== 用户配置区 ====================
# 实验日志文件 (主程序输出的 experiment_results.txt)
LOG_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/main/experiment_results.txt"

# 序列1深度图文件夹
DEPTH_DIR_SEQ1 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/deepseek/lines1/depth"

# 序列1和2的轨迹文件 (TUM格式，已分割好的)
TRAJ_SEQ1 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/deepseek/lines1/trajectory.txt"
TRAJ_SEQ2 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/deepseek/lines2/trajectory.txt"

# 相机内参 (848x480 那次录制的)
FX = 426.372
FY = 425.671
CX = 435.525
CY = 244.974
K = np.array([[FX, 0, CX],
              [0, FY, CY],
              [0, 0, 1]], dtype=np.float64)

# 选择的帧索引 (对应图像列表中的序号)
IDX_SEQ1_BASE = 221          # 序列1 当前帧 B
FRAME_STEP = 15              # 与 config.yaml 一致
IDX_SEQ1_PREV = IDX_SEQ1_BASE - FRAME_STEP   # 序列1 前一帧 A
IDX_SEQ2_BASE = 232          # 序列2 匹配帧 C

# ROI 中心点像素坐标 (备用，若 USE_SAVED_ROIS=True 则自动从文件读取)
CENTER_X = 300   # 示例，请替换为实际值
CENTER_Y = 240

# 是否自动从 saved_rois.txt 中提取中心点 (若为True，则忽略上面的手动坐标)
USE_SAVED_ROIS = True
SAVED_ROIS_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/deepseek/lines1/rgb/saved_rois.txt"
# ==================================================

K_INV = np.linalg.inv(K)

# ============== 工具函数 ==============
def load_tum_trajectory(filename):
    """读取 TUM 轨迹文件，返回列表 [(timestamp, [tx, ty, tz, qx, qy, qz, qw])]"""
    traj = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            ts = float(parts[0])
            data = [float(x) for x in parts[1:8]]
            traj.append((ts, data))
    return traj

def get_pose_at_index(traj, idx):
    """通过索引获取位姿 (假设轨迹与图像一一对应)"""
    if idx < len(traj):
        return np.array(traj[idx][1][0:3]), np.array(traj[idx][1][3:7])
    else:
        return None, None

def calculate_motion_rt(pose1, pose2):
    """计算 pose1 到 pose2 的相对变换 (R_12, t_12)，两者均为 [tx,ty,tz,qx,qy,qz,qw]"""
    t1, q1 = pose1[0:3], pose1[3:7]
    t2, q2 = pose2[0:3], pose2[3:7]
    R1 = R_scipy.from_quat(q1).as_matrix()
    R2 = R_scipy.from_quat(q2).as_matrix()
    R_12 = R1.T @ R2          # 世界→A 乘以 B→世界
    t_12 = R1.T @ (t2 - t1)
    return R_12, t_12

def get_gt_depth(depth_dir, img_idx, center_x, center_y):
    """通过索引读取深度图，返回 (深度_米, 深度图文件名)"""
    depth_files = sorted(os.listdir(depth_dir))
    if img_idx >= len(depth_files):
        print(f"❌ 深度图索引 {img_idx} 超出范围 ({len(depth_files)})")
        return None, None
    fname = depth_files[img_idx]
    depth_path = os.path.join(depth_dir, fname)
    depth_mm = cv2.imread(depth_path, -1)
    if depth_mm is None:
        print(f"❌ 无法读取深度图: {depth_path}")
        return None, None

    # 3x3 邻域中值 (忽略0)
    x, y = int(center_x), int(center_y)
    patch = depth_mm[max(0, y-1):y+2, max(0, x-1):x+2]
    patch = patch[patch > 0]
    if len(patch) == 0:
        print(f"⚠️  像素 ({x},{y}) 周围无有效深度值")
        return None, None
    Z_gt = np.median(patch) / 1000.0   # 毫米转米
    return Z_gt, fname

def fit_plane_from_depth(depth_dir, img_idx, roi_polygon):
    """利用深度图和ROI多边形内的点拟合平面方程 n·X = d (n单位化)"""
    depth_files = sorted(os.listdir(depth_dir))
    if img_idx >= len(depth_files):
        return None, None, None
    fname = depth_files[img_idx]
    depth_path = os.path.join(depth_dir, fname)
    depth_mm = cv2.imread(depth_path, -1)
    if depth_mm is None:
        return None, None, None

    # 在ROI多边形内均匀采样点 (这里简单取ROI内所有像素点)
    mask = np.zeros(depth_mm.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi_polygon, dtype=np.int32)], 255)
    ys, xs = np.where(mask > 0)
    points_3d = []
    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        Z_mm = depth_mm[y, x]
        if Z_mm <= 0:
            continue
        Z_m = Z_mm / 1000.0
        # 反投影
        pt = np.array([x, y, 1.0])
        ray = K_INV @ pt
        X_3d = ray * (Z_m / ray[2])
        points_3d.append(X_3d)
    if len(points_3d) < 3:
        return None, None, None
    points = np.array(points_3d)
    # 最小二乘拟合平面：ax+by+cz = d
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    b = -points[:, 2]
    coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_coeff, d = coeff
    n_raw = np.array([a, b_coeff, 1.0])
    n = n_raw / np.linalg.norm(n_raw)
    # 调整方向使 n_z > 0
    if n[2] < 0:
        n = -n
        d = -d
    return n, d, fname

def parse_experiment_log(log_path):
    """从实验日志中提取关键结果"""
    results = {}
    with open(log_path, 'r') as f:
        content = f.read()
    # Looming 深度
    m = re.search(r'物理深度计算成功:\s*Z\s*=\s*([\d.]+)\s*m', content)
    if m:
        results['Z_looming'] = float(m.group(1))
    m = re.search(r'dr\s*=\s*([\d.]+)\s*px', content)
    if m:
        results['dr'] = float(m.group(1))
    # delta_d
    m = re.search(r'tz\(delta_d\):\s*([\d.]+)', content)
    if m:
        results['delta_d'] = float(m.group(1))

    # 正交误差与法向量 (提取所有候选解)
    ortho_errors = []
    normals = []
    for line in content.split('\n'):
        if '候选解' in line and '3D正交误差' in line:
            err_m = re.search(r'3D正交误差:\s*([\d.]+)', line)
            n_m = re.search(r'n_cv:\s*\[([^\]]+)\]', line)
            if err_m and n_m:
                ortho_errors.append(float(err_m.group(1)))
                n_vals = [float(x.strip()) for x in n_m.group(1).split()]
                normals.append(n_vals)
    results['ortho_errors'] = ortho_errors
    results['normals'] = normals

    # 选中的最佳解索引 (从日志“验算完毕！【解 N】”)
    best_m = re.search(r'验算完毕！【解\s*(\d+)】', content)
    if best_m:
        results['best_solution_idx'] = int(best_m.group(1)) - 1  # 转为0基

    # 绝对平移 t_real
    t_matches = re.findall(r'\[\[\s*([\-\d.]+)\s*\]\s*\[\s*([\-\d.]+)\s*\]\s*\[\s*([\-\d.]+)\s*\]\]', content)
    if t_matches:
        t = [float(t_matches[-1][i]) for i in range(3)]
        results['t_real'] = np.array(t).reshape(3, 1)

    # 旋转矩阵 R (粗略提取)
    r_lines = []
    in_rot = False
    for line in content.split('\n'):
        if '跨序列的绝对旋转矩阵 R' in line:
            in_rot = True
            continue
        if in_rot:
            if line.startswith('[[') or line.startswith(' ['):
                r_lines.append(line)
            else:
                break
    if len(r_lines) >= 3:
        rot_vals = []
        for l in r_lines[:3]:
            vals = re.findall(r'[\-\d.]+', l)
            rot_vals.extend([float(v) for v in vals])
        if len(rot_vals) == 9:
            results['R'] = np.array(rot_vals).reshape(3, 3)

    return results

# ============== 主评估流程 ==============
def main():
    print("=" * 60)
    print("🎯 精度评估开始")
    print("=" * 60)

    # 1. 加载轨迹真值
    traj1 = load_tum_trajectory(TRAJ_SEQ1)
    traj2 = load_tum_trajectory(TRAJ_SEQ2)
    if len(traj1) <= IDX_SEQ1_PREV or len(traj2) <= IDX_SEQ2_BASE:
        print("❌ 轨迹索引超出范围")
        return

    # 2. 获取真实相对位姿
    pose_A = np.array(traj1[IDX_SEQ1_PREV][1])  # [tx,ty,tz,qx,qy,qz,qw]
    pose_C = np.array(traj2[IDX_SEQ2_BASE][1])
    R_gt, t_gt = calculate_motion_rt(pose_A, pose_C)

    # 3. 获取深度真值
    # 尝试从 saved_rois.txt 中提取 ROI 中心点
    roi_polygon = None
    if USE_SAVED_ROIS and os.path.exists(SAVED_ROIS_PATH):
        with open(SAVED_ROIS_PATH, 'r') as f:
            lines = f.readlines()
        if IDX_SEQ1_BASE < len(lines):
            roi_line = lines[IDX_SEQ1_BASE].strip()
            # 处理可能的逗号分隔，提取所有数字
            coords = [int(float(x)) for x in re.split(r'[ ,\t]+', roi_line) if x != '']
            if len(coords) >= 4:
                # 如果是8个数，视为四个角点；如果只有4个数，视为左上和右下
                if len(coords) == 8:
                    pts = np.array([[coords[i], coords[i+1]] for i in range(0, 8, 2)])
                elif len(coords) == 4:
                    # 矩形框：x1,y1,x2,y2，取中心
                    center = np.array([(coords[0]+coords[2])/2, (coords[1]+coords[3])/2])
                    pts = np.array([[coords[0], coords[1]], [coords[2], coords[1]],
                                    [coords[2], coords[3]], [coords[0], coords[3]]])
                else:
                    print(f"⚠️  saved_rois 坐标个数为 {len(coords)}，无法确定ROI格式")
                    roi_polygon = None
                    pts = None
                if pts is not None:
                    center = np.mean(pts, axis=0)
                    global CENTER_X, CENTER_Y
                    CENTER_X, CENTER_Y = center[0], center[1]
                    print(f"✅ 从 saved_rois 提取 ROI 中心: ({CENTER_X:.1f}, {CENTER_Y:.1f})")
                    roi_polygon = pts.astype(int).tolist()
                else:
                    print("❌ saved_rois 格式无法识别，使用手动坐标")
                    roi_polygon = None
            else:
                print("❌ saved_rois 中没有有效坐标")
                roi_polygon = None
    else:
        print("使用手动指定的中心点坐标")
        roi_polygon = None

    # 获取深度真值
    Z_gt, depth_fname = get_gt_depth(DEPTH_DIR_SEQ1, IDX_SEQ1_BASE, CENTER_X, CENTER_Y)
    if Z_gt is None:
        print("❌ 无法获取深度真值")
    else:
        print(f"📏 深度真值 (帧{IDX_SEQ1_BASE}): {Z_gt:.4f} m   (深度图: {depth_fname})")

    # 4. 拟合平面获取真实法向量
    n_gt = None
    if roi_polygon is not None:
        n_gt, _, _ = fit_plane_from_depth(DEPTH_DIR_SEQ1, IDX_SEQ1_BASE, roi_polygon)
    else:
        print("⚠️  未提供 ROI 多边形，无法计算法向量真值，跳过正交筛选命中率检测")

    # 5. 解析日志结果
    results = parse_experiment_log(LOG_PATH)
    if not results:
        print("❌ 无法解析实验日志")
        return

    print("\n--- 解析结果 ---")
    Z_looming = results.get('Z_looming')
    if Z_looming:
        print(f"🌊 Looming 深度: {Z_looming:.4f} m")
    if 'dr' in results:
        print(f"   膨胀量 dr = {results['dr']:.2f} px")
    if 'delta_d' in results:
        print(f"   前向位移 delta_d = {results['delta_d']:.4f} m")

    # 6. 计算深度误差
    if Z_gt is not None and Z_looming is not None:
        err_abs = abs(Z_looming - Z_gt)
        err_rel = err_abs / Z_gt * 100
        print(f"\n✅ Looming 深度误差: {err_abs:.4f} m, 相对误差: {err_rel:.2f}%")

    # 7. 正交筛选命中率
    if n_gt is not None and 'normals' in results and 'best_solution_idx' in results:
        normals = results['normals']
        best_idx = results['best_solution_idx']
        print(f"\n--- 正交筛选验证 ---")
        print(f"真实法向量 n_gt: {np.round(n_gt, 4)}")
        # 计算每组解与真值的夹角
        angles = []
        for i, n_i in enumerate(normals):
            angle = math.degrees(math.acos(min(np.abs(np.dot(n_i, n_gt)), 1.0)))
            angles.append(angle)
            mark = " 👈 选中" if i == best_idx else ""
            print(f"  解{i+1}: 正交误差={results['ortho_errors'][i]:.4f}, 与真值夹角={angle:.2f}°{mark}")
        # 命中检查
        min_angle_idx = np.argmin(angles)
        print(f"最小夹角对应解 {min_angle_idx+1}, 正交误差最小对应解 {best_idx+1}")
        if min_angle_idx == best_idx:
            print("🎯 正交筛选成功命中真实解！")
        else:
            print("⚠️  正交筛选未命中真实解，请检查正交线条质量或参数")

    # 8. 绝对位姿误差
    if 't_real' in results and 'R' in results:
        t_est = results['t_real'].flatten()
        R_est = results['R']
        print(f"\n--- 跨序列绝对位姿精度 ---")
        # 平移误差
        t_err = np.linalg.norm(t_est - t_gt)
        gt_len = np.linalg.norm(t_gt)
        t_err_rel = np.linalg.norm(t_est - t_gt) / gt_len * 100 if gt_len > 0 else 0
        # 旋转误差
        R_diff = R_est.T @ R_gt
        rot_err_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2.0, -1, 1))
        rot_err_deg = math.degrees(rot_err_rad)

        print(f"真实平移 t_gt: {t_gt.flatten()} m")
        print(f"估计平移 t_est: {t_est}")
        print(f"平移误差 (欧式距离): {t_err:.4f} m, 相对错误: {t_err_rel:.2f}%")
        print(f"真实旋转矩阵 R_gt:\n{np.round(R_gt, 4)}")
        print(f"估计旋转矩阵 R_est:\n{np.round(R_est, 4)}")
        print(f"旋转误差: {rot_err_deg:.4f}°")

    print("\n" + "=" * 60)
    print("评估结束")
    print("=" * 60)

if __name__ == "__main__":
    main()