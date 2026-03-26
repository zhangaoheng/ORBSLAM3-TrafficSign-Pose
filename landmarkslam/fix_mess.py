lines = []
with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if line.startswith("def save_point_cloud_txt"):
        skip = True
        new_lines.append("def save_point_cloud_txt(points, colors, frames, filename):\n")
        new_lines.append("    if not points: return\n")
        new_lines.append("    with open(filename, 'w') as f:\n")
        new_lines.append("        for p, c, fn in zip(points, colors, frames):\n")
        new_lines.append("            f.write(f\"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[2])} {int(c[1])} {int(c[0])} {fn}\\n\")\n\n")
        continue
    
    if skip and line.startswith("def main():"):
        skip = False
        
    if not skip:
        # Also fix the call inside main()
        line = line.replace("save_point_cloud_txt(all_sign_points, all_sign_colors, args.output)", "save_point_cloud_txt(all_sign_points, all_sign_colors, all_sign_frames, args.output)")
        if "all_sign_colors = []" in line:
            new_lines.append(line)
            if "all_sign_frames" not in line:
                new_lines.append("    all_sign_frames = []\n")
            continue
        if "all_sign_colors.append(color) # [B, G, R]" in line:
            new_lines.append(line)
            if "all_sign_frames" not in line:
                new_lines.append("                    all_sign_frames.append(base_name)\n")
            continue
            
        new_lines.append(line)

with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "w") as f:
    f.writelines(new_lines)
