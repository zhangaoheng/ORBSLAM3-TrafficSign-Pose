import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def visualize_txt_point_cloud(txt_file):
    if not os.path.exists(txt_file):
        print(f"Error: File not found: {txt_file}")
        return

    points = []
    colors = []
    
    print(f"Loading {txt_file}...")
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    # 格式: X Y Z R G B [Frame]
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    colors.append([float(parts[3])/255.0, float(parts[4])/255.0, float(parts[5])/255.0])
                except ValueError:
                    pass
            elif len(parts) == 4:
                try:
                    # 兼容纯轨迹文件: Frame_Name X Y Z
                    points.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    colors.append([0.0, 1.0, 0.0]) # 默认给轨迹染成绿色
                except ValueError:
                    pass
                
    points = np.array(points)
    colors = np.array(colors)
    
    if len(points) == 0:
        print("No valid points found in the file.")
        return
        
    print(f"Loaded {len(points)} points. Filtering outliers and plotting...")
    
    # === 移除异常值 (Outliers) ===
    # 修改：识别特殊关键点（红色角点 [1,0,0] 和 绿色相机点 [0,1,0]），给予“免死金牌”防过滤
    is_special = ((colors[:, 0] == 1.0) & (colors[:, 1] == 0.0) & (colors[:, 2] == 0.0)) | \
                 ((colors[:, 0] == 0.0) & (colors[:, 1] == 1.0) & (colors[:, 2] == 0.0))
                 
    normal_points = points[~is_special]
    if len(normal_points) > 0:
        # 将原先极严的 5%-95% 放宽到 1%-99%，避免把周边有效点给删了
        p1 = np.percentile(normal_points, 1, axis=0)
        p99 = np.percentile(normal_points, 99, axis=0)
        
        normal_mask = (normal_points[:, 0] >= p1[0]) & (normal_points[:, 0] <= p99[0]) & \
                      (normal_points[:, 1] >= p1[1]) & (normal_points[:, 1] <= p99[1]) & \
                      (normal_points[:, 2] >= p1[2]) & (normal_points[:, 2] <= p99[2])
                      
        filtered_normal_pts = normal_points[normal_mask]
        filtered_normal_colors = colors[~is_special][normal_mask]
        
        # 拼接回被保护的特殊点 (相机、角点)
        if np.any(is_special):
            filtered_points = np.vstack([filtered_normal_pts, points[is_special]])
            filtered_colors = np.vstack([filtered_normal_colors, colors[is_special]])
        else:
            filtered_points = filtered_normal_pts
            filtered_colors = filtered_normal_colors
    else:
        filtered_points = points
        filtered_colors = colors
    
    print(f"Remaining {len(filtered_points)} points after thresholding bounds.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 调大点的大小，从 s=1.0 调大为 s=5.0 或者更大
    ax.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], 
               c=filtered_colors, s=15.0, marker='o', alpha=0.8)
    
    # === 仅根据过滤后的数据中心点来设置坐标轴范围 ===
    if len(filtered_points) > 0:
        max_range = np.array([filtered_points[:, 0].max()-filtered_points[:, 0].min(), 
                              filtered_points[:, 1].max()-filtered_points[:, 1].min(), 
                              filtered_points[:, 2].max()-filtered_points[:, 2].min()]).max() / 2.0
        
        mid_x = (filtered_points[:, 0].max()+filtered_points[:, 0].min()) * 0.5
        mid_y = (filtered_points[:, 1].max()+filtered_points[:, 1].min()) * 0.5
        mid_z = (filtered_points[:, 2].max()+filtered_points[:, 2].min()) * 0.5
        
        # 增加点冗余空间 1.1倍
        ax.set_xlim(mid_x - max_range*1.1, mid_x + max_range*1.1)
        ax.set_ylim(mid_y - max_range*1.1, mid_y + max_range*1.1)
        ax.set_zlim(mid_z - max_range*1.1, mid_z + max_range*1.1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Sign Point Cloud (Filtered)")
    
    # 反转Y轴和Z轴使其符合常见图像坐标习惯 (可选)
    ax.invert_yaxis()
    ax.invert_zaxis()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize XYZRGB text point cloud")
    parser.add_argument("--file", type=str, default="/home/zah/ORB_SLAM3-master/landmarkslam/output2/sign_cloud_py_trajectory.txt", help="Path to the .txt file")
    args = parser.parse_args()
    
    visualize_txt_point_cloud(args.file)
