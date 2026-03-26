import re

with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "r") as f:
    code = f.read()

# Change the save function
old_func = """def save_point_cloud_txt(points, colors, filename):
    \"\"\"保存带颜色的TXT文件: X Y Z R G B\"\"\"
    if not points: return
    with open(filename, 'w') as f:
        for p, c in zip(points, colors):
            # c 是 BGR，所以保存 RGB 是 c[2], c[1], c[0]
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[2])} {int(c[1])} {int(c[0])}\\n")"""

new_func = """def save_point_cloud_txt(points, colors, frame_names, filename):
    \"\"\"保存带颜色的TXT文件: X Y Z R G B FrameName\"\"\"
    if not points: return
    with open(filename, 'w') as f:
        for p, c, fn in zip(points, colors, frame_names):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[2])} {int(c[1])} {int(c[0])} {fn}\\n")"""

code = code.replace(old_func, new_func)

# Append logic
code = code.replace("all_sign_colors = []", "all_sign_colors = []\\n    all_sign_frames = []")
code = code.replace("all_sign_colors.append(color) # [B, G, R]", "all_sign_colors.append(color)\\n                    all_sign_frames.append(base_name)")

# Function call
code = code.replace("save_point_cloud_txt(all_sign_points, all_sign_colors, args.output)", "save_point_cloud_txt(all_sign_points, all_sign_colors, all_sign_frames, args.output)")

with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "w") as f:
    f.write(code)
