import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def read_ply_with_colors(filename):
    """解析带有RGB颜色的 ASCII 格式 PLY 文件"""
    points = []
    colors = []
    try:
        with open(filename, 'r') as f:
            in_data = False
            for line in f:
                if line.strip() == "end_header":
                    in_data = True
                    continue
                if in_data:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        # xyz r g b
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        colors.append([float(parts[3])/255.0, float(parts[4])/255.0, float(parts[5])/255.0])
                    elif len(parts) >= 3:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        colors.append([1.0, 0.0, 0.0]) # 默认红色
    except Exception as e:
        print(f"Error reading PLY: {e}")
    return np.array(points), np.array(colors)

def read_ply(filename):
    """手动解析 ASCII 格式的 PLY 文件 (无颜色版保留兼容性)"""
    points = []
    try:
        with open(filename, 'r') as f:
            in_data = False
            for line in f:
                if line.strip() == "end_header":
                    in_data = True
                    continue
                if in_data:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    except Exception as e:
        pass
    return np.array(points)

def read_tum(filename):
    try:
        data = np.loadtxt(filename)
        if len(data) > 0 and len(data.shape) > 1 and data.shape[1] >= 4:
            return data[:, 1:4]
    except Exception as e:
        pass
    return np.array([])


def main():
    parser = argparse.ArgumentParser(description="Visualize SLAM and YOLO 3D Point Clouds.")
    parser.add_argument('--mode', type=str, choices=['yolo', 'fused', 'sign'], default='fused',
                        help="'yolo' shows YOLO points, 'fused' shows SLAM+YOLO, 'sign' shows the newly computed text feature sign point cloud.")
    args = parser.parse_args()

    output_dir = "output2"
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    points_for_bounds = []

    # ==========================
    # 如果用户想看最新生成的彩色字体特征路牌点云
    # ==========================
    if args.mode == 'sign':
        target_ply = os.path.join(output_dir, "sign_cloud.ply")
        if not os.path.exists(target_ply):
            print(f"[报错] 找不到 {target_ply}，请先运行: ./run_scripts/run_compute_sign_pose.sh")
            return
            
        print(f"正在加载高密度文字/箭头特征路牌点云: {target_ply}")
        sign_pts, sign_cols = read_ply_with_colors(target_ply)
        
        if len(sign_pts) > 0:
            ax.set_title(f"Traffic Sign High-Density PointCloud ({len(sign_pts)} Points)")
            ax.scatter(sign_pts[:, 0], sign_pts[:, 1], sign_pts[:, 2], 
                       c=sign_cols, s=15, alpha=1.0) # 使用原图提出来的颜色绘制
            points_for_bounds = list(sign_pts)
        else:
            print("[警告] 点云文件没有数据！")
    
    # ==========================
    # 其他模式 (原有的 SLAM+YOLO 模式)
    # ==========================
    else:
        traj_files = sorted(glob.glob(os.path.join(output_dir, "*KeyFrameTrajectory_*.txt")))
        if not traj_files:
            print(f"未找到轨迹文件 (在 {output_dir}/*KeyFrameTrajectory_*.txt)！")
            return
        
        latest_traj = traj_files[-1]
        file_basename = os.path.basename(latest_traj).replace('.txt', '')
        parts = file_basename.split('_')
        timestamp = f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else parts[-1]
            
        latest_ply = os.path.join(output_dir, f"SLAM_PointCloud_{timestamp}.ply")
        latest_yolo_ply = os.path.join(output_dir, f"YOLO_Landmarks_{timestamp}.ply")
        
        print(f"正在加载轨迹: {latest_traj}")
        traj_points = read_tum(latest_traj)
        if len(traj_points) > 0:
            points_for_bounds.extend(traj_points)
        
        mode_title = "YOLO Only" if args.mode == 'yolo' else "Fused SLAM & YOLO"
        ax.set_title(f"{mode_title} 3D Trajectory ({timestamp})")

        if args.mode == 'fused' and os.path.exists(latest_ply):
            pc_points = read_ply(latest_ply)
            if len(pc_points) > 0:
                idx = np.random.choice(len(pc_points), min(20000, len(pc_points)), replace=False)
                dp = pc_points[idx]
                ax.scatter(dp[:, 0], dp[:, 1], dp[:, 2], c='gray', s=2, alpha=0.3, label='SLAM Maps')
                points_for_bounds.extend(dp[::10])

        if os.path.exists(latest_yolo_ply):
            yolo_points = read_ply(latest_yolo_ply)
            if len(yolo_points) > 0:
                ax.scatter(yolo_points[:, 0], yolo_points[:, 1], yolo_points[:, 2], c='red', s=20, marker='^', label='YOLO')
                points_for_bounds.extend(yolo_points)

        if len(traj_points) > 0:
            ax.plot(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2], c='blue', linewidth=1.5, label='Trajectory')
            ax.scatter(traj_points[0,0], traj_points[0,1], traj_points[0,2], c='green', s=100, label='Start')
            ax.scatter(traj_points[-1,0], traj_points[-1,1], traj_points[-1,2], c='purple', s=100, marker='X', label='End')
            
        ax.legend(loc='upper right')

    # ==========================
    # 通用设置缩放
    # ==========================
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if len(points_for_bounds) > 0:
        points_for_bounds = np.array(points_for_bounds)
        max_range = np.array([points_for_bounds[:,0].max()-points_for_bounds[:,0].min(), 
                              points_for_bounds[:,1].max()-points_for_bounds[:,1].min(), 
                              points_for_bounds[:,2].max()-points_for_bounds[:,2].min()]).max() / 2.0
        mid_x = (points_for_bounds[:,0].max()+points_for_bounds[:,0].min()) * 0.5
        mid_y = (points_for_bounds[:,1].max()+points_for_bounds[:,1].min()) * 0.5
        mid_z = (points_for_bounds[:,2].max()+points_for_bounds[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

if __name__ == "__main__":
    main()
