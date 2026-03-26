with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "r") as f:
    code = f.read()

# 1. 扩充保存函数，支持写入一列 frame_name
code = code.replace(
    "def save_point_cloud_txt(points, colors, filename):", 
    "def save_point_cloud_txt(points, colors, frames, filename):"
)
code = code.replace(
    "for p, c in zip(points, colors):", 
    "for p, c, fn in zip(points, colors, frames):"
)
code = code.replace(
    "f.write(f\"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[2])} {int(c[1])} {int(c[0])}\\n\")", 
    "f.write(f\"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[2])} {int(c[1])} {int(c[0])} {fn}\\n\")"
)

# 2. 增加 frame list 的收集
code = code.replace(
    "all_sign_colors = []", 
    "all_sign_colors = []\n    all_sign_frames = []"
)
code = code.replace(
    "all_sign_colors.append(color) # [B, G, R]\n                    valid_pts_count += 1", 
    "all_sign_colors.append(color) # [B, G, R]\n                    all_sign_frames.append(base_name)\n                    valid_pts_count += 1"
)
code = code.replace(
    "save_point_cloud_txt(all_sign_points, all_sign_colors, args.output)", 
    "save_point_cloud_txt(all_sign_points, all_sign_colors, all_sign_frames, args.output)"
)

with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "w") as f:
    f.write(code)
