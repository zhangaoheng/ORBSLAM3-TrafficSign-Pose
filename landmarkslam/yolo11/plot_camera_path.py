import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ====== 配置：最多五个csv文件路径 ======
csv_files = [
	'D:/zah/Latex/yolo11/pose_inference_results/base/camera_trajectory_pnp.csv',
	'D:/zah/Latex/yolo11/pose_inference_results/new/camera_trajectory_pnp.csv',
    'D:/zah/Latex/unityDataset/CameraLogs/base/camera_log.csv',
	'D:/zah/Latex/unityDataset/CameraLogs/new/camera_log.csv',
	'D:/zah/Latex/unityDataset/CameraLogs/20251230_200224/camera_log.csv'
	# 可添加更多文件，最多5个
	# 'D:/zah/Latex/yolo11/camera_trajectory_pnp.csv',
	# 'your/other/path1.csv',
	# 'your/other/path2.csv',
	# 'your/other/path3.csv',
]
labels = [
	'Base_estimate',
	'New_estimate',
	'Base_real',
	'New_real',
	'20251230_200224_real',
	# 'PnP Trajectory',
	# 'Other1',
	# 'Other2',
	# 'Other3',
]


for idx, csv_path in enumerate(csv_files[:5]):
	try:
		df = pd.read_csv(csv_path)
	except Exception as e:
		print(f"读取失败: {csv_path}, 错误: {e}")
		continue
	# 自动判断字段名
	if {'position_x', 'position_y', 'position_z'}.issubset(df.columns):
		x = df['position_x']
		y = df['position_y']
		z = df['position_z']
	elif {'tx', 'ty', 'tz'}.issubset(df.columns):
		x = df['tx']
		y = df['ty']
		z = df['tz']
	else:
		print(f"文件 {csv_path} 不包含可识别的轨迹字段")
		continue
	label = labels[idx] if idx < len(labels) else f'Trajectory {idx+1}'
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(x, y, z, marker='o', label=label)
	ax.set_xlabel('X')
	ax.set_ylabel('z')
	ax.set_zlabel('y')
	ax.set_title(f'Camera 3D Path: {label}')
	ax.legend()
	ax.set_box_aspect([1, 1, 1])  # 保持xyz比例一致
plt.show()