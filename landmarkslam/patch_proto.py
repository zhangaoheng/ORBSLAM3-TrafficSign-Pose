with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "r") as f:
    code = f.read()

import re
old_func = """def save_point_cloud_txt(points, colors, filename):
    \"\"\"保存带颜色的PLY文件\"\"\"
    if not points: return
    with open(filename, 'w') as f:
        f.write("ply\\n")
        f.write("format ascii 1.0\\n")
        f.write(f"element vertex {len(points)}\\n")
        f.write("property float x\\n")
        f.write("property float y\\n")
        f.write("property float z\\n")
        f.write("property uchar red\\n")
        f.write("property uchar green\\n")
        f.write("property uchar blue\\n")
        f.write("end_header\\n")
        for p, c in zip(points, colors):
            # c 是 BGR，所以保存 RGB 是 c[2], c[1], c[0]
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[2])} {int(c[1])} {int(c[0])}\\n")"""

new_func = """def save_point_cloud_txt(points, colors, filename):
    \"\"\"保存带颜色的TXT文件: X Y Z R G B\"\"\"
    if not points: return
    with open(filename, 'w') as f:
        for p, c in zip(points, colors):
            # c 是 BGR，所以保存 RGB 是 c[2], c[1], c[0]
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[2])} {int(c[1])} {int(c[0])}\\n")"""

code = code.replace(old_func, new_func)

# 还要把 default 名字改一下
code = code.replace("sign_cloud_py.ply", "sign_cloud_py.txt")
code = code.replace('parser.add_argument("--output", default="../output2/sign_cloud_py.txt", help="Output ply file")', 'parser.add_argument("--output", default="../output2/sign_cloud_py.txt", help="Output txt file")')

with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "w") as f:
    f.write(code)
