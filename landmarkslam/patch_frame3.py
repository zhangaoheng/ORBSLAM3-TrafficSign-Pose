with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "r") as f:
    code = f.read()

import re
old_func = """def save_point_cloud_txt(points, colors, frames, filename):
    \"\"\"保存带颜色的TXT文件 (含Frame名)\"\"\"
    if not points: return
    with open(filename, 'w') as f:
        for p, c, fn in zip(points, colors, frames):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[2])} {int(c[1])} {int(c[0])} {fn}\\n")"""

# It seems the ply headers are still there? Let's just blindly force replace
def_pattern = r'def save_point_cloud_txt\(.*?def main\(\):'
clean_func = """def save_point_cloud_txt(points, colors, frames, filename):
    \"\"\"保存带颜色的TXT文件 (含Frame名)\"\"\"
    if not points: return
    with open(filename, 'w') as f:
        for p, c, fn in zip(points, colors, frames):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[2])} {int(c[1])} {int(c[0])} {fn}\\n")

def main():"""

code = re.sub(def_pattern, clean_func, code, flags=re.DOTALL)

with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "w") as f:
    f.write(code)
