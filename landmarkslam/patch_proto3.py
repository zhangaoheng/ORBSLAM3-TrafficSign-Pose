import sys

with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "all_sign_colors = []" in line:
        lines.insert(i+1, "    all_sign_frames = []\n")
        break

for i, line in enumerate(lines):
    if "all_sign_colors.append(color)" in line:
        lines.insert(i+1, "                    all_sign_frames.append(base_name)\n")
        break

for i, line in enumerate(lines):
    if "save_point_cloud_txt(all_sign_points, all_sign_colors, args.output)" in line:
        lines[i] = line.replace("save_point_cloud_txt(all_sign_points, all_sign_colors, args.output)", "save_point_cloud_txt(all_sign_points, all_sign_colors, all_sign_frames, args.output)")

with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "w") as f:
    f.writelines(lines)
