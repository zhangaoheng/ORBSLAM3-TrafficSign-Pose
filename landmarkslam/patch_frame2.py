with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "r") as f:
    code = f.read()

import re

# We need to correctly patch the function to save txt WITH frame name
new_func = """def save_point_cloud_txt(points, colors, frames, filename):
    \"\"\"保存带颜色的TXT文件 (含Frame名)\"\"\"
    if not points: return
    with open(filename, 'w') as f:
        for p, c, fn in zip(points, colors, frames):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[2])} {int(c[1])} {int(c[0])} {fn}\\n")"""

# Pattern to replace everything from def save_point_cloud_txt down to the main function
code = re.sub(r'def save_point_cloud_txt\(.*?\):.*?(?=def main\(\):)', new_func + '\n\n', code, flags=re.DOTALL)

# Add frames collection
if "all_sign_frames = []" not in code:
    code = code.replace("all_sign_colors = []", "all_sign_colors = []\n    all_sign_frames = []")
    code = code.replace("all_sign_colors.append(color) # [B, G, R]", "all_sign_colors.append(color) # [B, G, R]\n                    all_sign_frames.append(base_name)")
    code = code.replace("save_point_cloud_txt(all_sign_points, all_sign_colors, args.output)", "save_point_cloud_txt(all_sign_points, all_sign_colors, all_sign_frames, args.output)")

with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "w") as f:
    f.write(code)
