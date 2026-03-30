import open3d as o3d
import numpy as np
import os

# 替换为你实际生成的文件名
FILE_MAP_POINTS = "/home/zah/ORB_SLAM3-master/landmarkslam/output/PointCloud_20260330_181452.ply"
FILE_KF_POSES = "/home/zah/ORB_SLAM3-master/landmarkslam/output/KeyFrames_Poses_20260330_181452.ply"
FILE_TRAJECTORY = "/home/zah/ORB_SLAM3-master/landmarkslam/output/KeyFrameTrajectory_20260330_181452.txt"

def main():
    geometries = []

    # 1. 加载并处理地图点云 (Map Points)
    if os.path.exists(FILE_MAP_POINTS):
        pcd_map = o3d.io.read_point_cloud(FILE_MAP_POINTS)
        # 将地图点涂成暗灰色，作为背景，避免视觉干扰
        pcd_map.paint_uniform_color([0.6, 0.6, 0.6])
        geometries.append(pcd_map)
        print(f"已加载地图点云: {len(pcd_map.points)} 个点")

    # 2. 加载并处理关键帧位置点云 (KeyFrame Poses)
    if os.path.exists(FILE_KF_POSES):
        pcd_kfs = o3d.io.read_point_cloud(FILE_KF_POSES)
        # 将关键帧位置涂成红色，以示突出
        pcd_kfs.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(pcd_kfs)
        print(f"已加载关键帧: {len(pcd_kfs.points)} 个帧")

    # 3. 解析 TUM 轨迹文件，生成连续的轨迹连线
    if os.path.exists(FILE_TRAJECTORY):
        # TUM格式: timestamp tx ty tz qx qy qz qw
        traj_data = np.loadtxt(FILE_TRAJECTORY)
        # 提取 tx, ty, tz 坐标 (第2, 3, 4列，即索引 1, 2, 3)
        points = traj_data[:, 1:4]

        # 创建线段连接相邻的位姿点
        lines = [[i, i+1] for i in range(len(points)-1)]
        # 轨迹线涂成鲜绿色
        colors = [[0.0, 0.8, 0.0] for _ in range(len(lines))] 

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        geometries.append(line_set)
        print(f"已加载连续轨迹线: {len(points)} 个位姿节点")

    # 4. 启动渲染窗口同步显示
    if geometries:
        print("\n正在显示 3D 结果...")
        print("💡 提示: 鼠标左键拖动旋转，右键平移，滚轮缩放")
        o3d.visualization.draw_geometries(
            geometries, 
            window_name="ORB-SLAM3 Algorithm Verification Visualization",
            width=1024, 
            height=768,
            point_show_normal=False
        )
    else:
        print("未找到任何文件，请检查当前路径或文件名设置。")

if __name__ == "__main__":
    main()