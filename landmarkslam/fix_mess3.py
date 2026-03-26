with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "r") as f:
    text = f.read()

# I am completely losing my mind. This file is not getting saved correctly. I will overwrite it fully.
with open("/home/zah/ORB_SLAM3-master/landmarkslam/python_prototypes/compute_sign_pose_proto.py", "w") as f:
    f.write("""import cv2
import numpy as np
import os
import glob
import argparse

def load_yolo_detections(filename):
    \"\"\"加载YOLO检测结果\"\"\"
    detections = {}
    if not os.path.exists(filename):
        return detections
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 9:
                img_name = parts[0]
                pts = []
                for i in range(4):
                    pts.append([float(parts[1 + i*2]), float(parts[2 + i*2])])
                detections[img_name] = np.array(pts, dtype=np.float32)
    return detections

def load_camera_intrinsics(yaml_file):
    fx = 458.654
    fy = 457.296
    cx = 367.215
    cy = 248.375
    try:
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r') as f:
                for line in f:
                    if "Camera1.fx:" in line: fx = float(line.split(":")[1].strip())
                    if "Camera1.fy:" in line: fy = float(line.split(":")[1].strip())
                    if "Camera1.cx:" in line: cx = float(line.split(":")[1].strip())
                    if "Camera1.cy:" in line: cy = float(line.split(":")[1].strip())
    except: pass
        
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0,  1]], dtype=np.float64)
    return K

def is_point_inside_polygon(pt, polygon):
    return cv2.pointPolygonTest(polygon, (float(pt[0]), float(pt[1])), False) >= 0

def backproject_to_plane(pt, K_inv, R, t):
    pt_c = np.array([pt[0], pt[1], 1.0], dtype=np.float64).reshape(3, 1)
    ray_c = K_inv @ pt_c
    R_T = R.T
    r_row2 = R_T[2, :]
    num = np.dot(r_row2, t)
    den = np.dot(r_row2, ray_c)
    if abs(float(den)) < 1e-6: return None
    s = num / den
    P_c = s * ray_c
    P_w = R_T @ (P_c - t)
    return [float(P_w[0]), float(P_w[1]), float(P_w[2])]

def save_point_cloud_txt(points, colors, frames, filename):
    if not points: return
    with open(filename, 'w') as f:
        for p, c, fn in zip(points, colors, frames):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[2])} {int(c[1])} {int(c[0])} {fn}\\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="../landmarkslam.yaml")
    parser.add_argument("--images", default="../data/20260321_111801/")
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--output", default="/home/zah/ORB_SLAM3-master/landmarkslam/output2/sign_cloud_py.txt")
    args = parser.parse_args()

    K = load_camera_intrinsics(args.settings)
    K_inv = np.linalg.inv(K)

    yolo_data = load_yolo_detections(args.yolo)
    if not yolo_data: return

    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        image_paths.extend(glob.glob(os.path.join(args.images, ext)))
    image_paths.sort()

    sign_w, sign_h = 1.0, 1.0
    obj_pts = np.array([[-sign_w/2, -sign_h/2], [sign_w/2, -sign_h/2], [sign_w/2, sign_h/2], [-sign_w/2, sign_h/2]], dtype=np.float32)

    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)

    all_sign_points = []
    all_sign_colors = []
    all_sign_frames = []

    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        if base_name not in yolo_data: continue
        corners = yolo_data[base_name]
        im = cv2.imread(img_path)
        if im is None: continue

        H, status = cv2.findHomography(obj_pts, corners)
        if H is None: continue

        num, Rs, ts, normals = cv2.decomposeHomographyMat(H, K)
        if num == 0: continue

        best_idx = 0
        for j in range(num):
            if normals[j][2][0] < 0 and ts[j][2][0] > 0:
                best_idx = j
                break
                
        R = Rs[best_idx]
        t = ts[best_idx]

        keypoints, des = orb.detectAndCompute(im, None)
        if keypoints is None: continue
        
        valid_pts_count = 0
        for kp in keypoints:
            if is_point_inside_polygon(kp.pt, corners):
                pt3d = backproject_to_plane(kp.pt, K_inv, R, t)
                if pt3d is not None:
                    all_sign_points.append(pt3d)
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    color = im[y, x] 
                    all_sign_colors.append(color)
                    all_sign_frames.append(base_name)
                    valid_pts_count += 1

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_point_cloud_txt(all_sign_points, all_sign_colors, all_sign_frames, args.output)

if __name__ == "__main__":
    main()
""")
