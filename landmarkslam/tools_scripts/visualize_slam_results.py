import open3d as o3d
import pandas as pd
import numpy as np
import os

def visualize_results(results_dir="analysis_results"):
    # 1. 设置文件路径
    scene_ply_path = os.path.join(results_dir, "/home/zah/ORB_SLAM3-master/landmarkslam/analysis_results/Global_Scene.ply")
    sign_ply_path = os.path.join(results_dir, "/home/zah/ORB_SLAM3-master/landmarkslam/analysis_results/RoadSign_Points.ply")
    csv_path = os.path.join(results_dir, "/home/zah/ORB_SLAM3-master/landmarkslam/analysis_results/Sign_Poses_Analysis.csv")

    geometries = []

    # 2. 加载全局场景点云 (灰色)
    if os.path.exists(scene_ply_path):
        print(f"-> 正在加载全局场景: {scene_ply_path}")
        pcd_scene = o3d.io.read_point_cloud(scene_ply_path)
        # 如果颜色没读出来，强制设为浅灰色
        if not pcd_scene.has_colors():
            pcd_scene.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(pcd_scene)
    else:
        print("警告: 找不到 Global_Scene.ply")

    # 3. 加载标识牌点云簇 (红色)
    if os.path.exists(sign_ply_path):
        print(f"-> 正在加载标识牌点云: {sign_ply_path}")
        pcd_sign = o3d.io.read_point_cloud(sign_ply_path)
        # 强制设为鲜艳的红色以便区分
        pcd_sign.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(pcd_sign)
    else:
        print("警告: 找不到 RoadSign_Points.ply")

    # 4. 加载 CSV 轨迹并绘制中心点与法线
    if os.path.exists(csv_path):
        print(f"-> 正在处理位姿分析数据: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 提取点坐标
        points = df[['x', 'y', 'z']].values
        normals = df[['nx', 'ny', 'nz']].values

        # A. 绘制标识牌中心点轨迹 (蓝色点)
        traj_pcd = o3d.geometry.PointCloud()
        traj_pcd.points = o3d.utility.Vector3dVector(points)
        traj_pcd.paint_uniform_color([0.0, 0.0, 1.0]) 
        geometries.append(traj_pcd)

        # B. 绘制轨迹连接线
        if len(points) > 1:
            lines = [[i, i+1] for i in range(len(points)-1)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(lines))]) # 绿色线
            geometries.append(line_set)

        # C. 绘制法向量 (从中心点射出的黄色短线)
        line_points = []
        line_indices = []
        for i in range(len(points)):
            p_start = points[i]
            p_end = p_start + normals[i] * 0.5 # 线段长度设为 0.5 米
            line_points.extend([p_start, p_end])
            line_indices.append([2*i, 2*i + 1])
        
        normal_lines = o3d.geometry.LineSet()
        normal_lines.points = o3d.utility.Vector3dVector(np.array(line_points))
        normal_lines.lines = o3d.utility.Vector2iVector(np.array(line_indices))
        normal_lines.colors = o3d.utility.Vector3dVector([[1, 0.8, 0] for _ in range(len(line_indices))]) # 黄色法线
        geometries.append(normal_lines)
    else:
        print("警告: 找不到 Sign_Poses_Analysis.csv")

    # 5. 添加世界坐标系参考 (RGB -> XYZ)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    geometries.append(coord_frame)

    # 6. 启动可视化窗口
    print("\n[操作指南]")
    print(" - 鼠标左键: 旋转")
    print(" - 鼠标右键: 平移")
    print(" - 滚轮: 缩放")
    print(" - 关闭窗口退出程序")
    
    o3d.visualization.draw_geometries(geometries, 
                                      window_name="Road Sign SLAM Analysis",
                                      width=1280, height=720)

if __name__ == "__main__":
    visualize_results()