import sys, os, cv2, numpy as np, math, glob
from scipy.spatial.transform import Rotation as R_scipy

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
from tools.mid import extract_four_lines_from_real_image, calculate_rectangle_center

fx, fy, cx, cy = 426.372, 425.671, 435.525, 244.974
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
K_inv = np.linalg.inv(K)
FRAME_STEP = 15
IDX_B = 221
IDX_A = IDX_B - FRAME_STEP

IMG_DIR = '/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/deepseek/lines1/rgb'
ROI_PATH = '/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/deepseek/lines1/rgb/saved_rois.txt'
TRAJ_PATH = '/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/deepseek/lines1/trajectory.txt'

images = sorted(glob.glob(os.path.join(IMG_DIR, '*.png')))

rois = []
with open(ROI_PATH) as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 4:
            rois.append(tuple(map(int, parts)))

def load_tum_trajectory(filename):
    traj = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            traj.append((float(parts[0]), [float(x) for x in parts[1:8]]))
    return traj

def get_closest_pose(timestamp, traj):
    best = min(traj, key=lambda x: abs(x[0] - timestamp))
    return np.array(best[1])

def calculate_motion_rt(pose1, pose2):
    t1, q1 = pose1[0:3], pose1[3:7]
    t2, q2 = pose2[0:3], pose2[3:7]
    R1 = R_scipy.from_quat(q1).as_matrix()
    R2 = R_scipy.from_quat(q2).as_matrix()
    R_12 = R1.T @ R2
    t_12 = R1.T @ (t2 - t1)
    return R_12, t_12

def derotate_point(P_raw, R_mat):
    H = K @ R_mat @ K_inv
    P_homo = np.array([P_raw[0], P_raw[1], 1.0])
    P_pure_homo = H @ P_homo
    return (P_pure_homo[0] / P_pure_homo[2], P_pure_homo[1] / P_pure_homo[2])

traj = load_tum_trajectory(TRAJ_PATH)

print('='*60)
print('Looming Calculation Diagnosis')
print('='*60)

time_A = float(os.path.basename(images[IDX_A]).replace('.png', '')) / 1e9
time_B = float(os.path.basename(images[IDX_B]).replace('.png', '')) / 1e9
print(f'Frame A: idx={IDX_A}, time={time_A}')
print(f'Frame B: idx={IDX_B}, time={time_B}')

pose_A = get_closest_pose(time_A, traj)
pose_B = get_closest_pose(time_B, traj)
print(f'pose_A (t): {pose_A[0:3]}')
print(f'pose_B (t): {pose_B[0:3]}')

R_12, t_12 = calculate_motion_rt(pose_A, pose_B)
tx, ty, tz = t_12
print(f'R_12:\n{np.round(R_12, 4)}')
print(f't_12: tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}')
print(f'delta_d = tz = {tz:.4f}')

FOE = (fx * (tx / tz) + cx, fy * (ty / tz) + cy)
print(f'FOE = ({FOE[0]:.2f}, {FOE[1]:.2f})')

print(f'\nROI[frameA={IDX_A}]: {rois[IDX_A]}')
print(f'ROI[frameB={IDX_B}]: {rois[IDX_B]}')

roiA = rois[IDX_A]
roiB = rois[IDX_B]
raw_center_A = (roiA[0] + roiA[2]/2, roiA[1] + roiA[3]/2)
raw_center_B = (roiB[0] + roiB[2]/2, roiB[1] + roiB[3]/2)
print(f'\nRaw ROI center A: {raw_center_A}')
print(f'Raw ROI center B: {raw_center_B}')
dist_raw_A = math.sqrt((raw_center_A[0]-FOE[0])**2 + (raw_center_A[1]-FOE[1])**2)
dist_raw_B = math.sqrt((raw_center_B[0]-FOE[0])**2 + (raw_center_B[1]-FOE[1])**2)
print(f'Raw dist to FOE: A={dist_raw_A:.2f}px, B={dist_raw_B:.2f}px')

img_A = cv2.imread(images[IDX_A])
img_B = cv2.imread(images[IDX_B])

lines_A = extract_four_lines_from_real_image(img_A, rois[IDX_A])
lines_B = extract_four_lines_from_real_image(img_B, rois[IDX_B])

if lines_A:
    print(f'\nLSD lines A: found')
else:
    print(f'\nLSD lines A: FAILED')
if lines_B:
    print(f'LSD lines B: found')
else:
    print(f'LSD lines B: FAILED')

if lines_A and lines_B:
    center_A_raw, corners_A = calculate_rectangle_center(*lines_A)
    center_B, corners_B = calculate_rectangle_center(*lines_B)
    print(f'\nLSD center A (raw): {center_A_raw}')
    print(f'LSD center B: {center_B}')
    
    center_A_pure_v1 = derotate_point(center_A_raw, R_12)
    center_A_pure_v2 = derotate_point(center_A_raw, R_12.T)
    print(f'\nDerotated center A (R_12):   {center_A_pure_v1}  (A->B)')
    print(f'Derotated center A (R_12.T): {center_A_pure_v2}  (B->A, current code)')
    
    dist_B = math.sqrt((center_B[0]-FOE[0])**2 + (center_B[1]-FOE[1])**2)
    dist_A_v1 = math.sqrt((center_A_pure_v1[0]-FOE[0])**2 + (center_A_pure_v1[1]-FOE[1])**2)
    dist_A_v2 = math.sqrt((center_A_pure_v2[0]-FOE[0])**2 + (center_A_pure_v2[1]-FOE[1])**2)
    
    print(f'\nDistances from FOE:')
    print(f'  center B (closer):   r_B = {dist_B:.3f} px')
    print(f'  center A derot(R_12): r_A = {dist_A_v1:.3f} px')
    print(f'  center A derot(R^T):  r_A = {dist_A_v2:.3f} px (current)')
    
    dr_v1 = dist_B - dist_A_v1
    dr_v2 = dist_B - dist_A_v2
    print(f'\nExpansion dr:')
    print(f'  dr (R_12):   r_B - r_A = {dr_v1:.3f} px')
    print(f'  dr (R_12.T): r_B - r_A = {dr_v2:.3f} px (current code)')
    
    if dr_v1 > 0.2:
        Z_v1 = dist_B * tz / dr_v1
        Z_v1_correct = dist_B * tz / dr_v1 - tz
        print(f'\nZ Looming (using R_12 derotation):')
        print(f'  Z (simple)   = r_B*delta_d/dr = {Z_v1:.3f} m')
        print(f'  Z (corrected) = r_B*delta_d/dr - delta_d = {Z_v1_correct:.3f} m')
    
    if dr_v2 > 0.2:
        Z_v2 = dist_B * tz / dr_v2
        Z_v2_correct = dist_B * tz / dr_v2 - tz
        print(f'\nZ Looming (using R_12.T derotation, current code):')
        print(f'  Z (simple)   = r_B*delta_d/dr = {Z_v2:.3f} m')
        print(f'  Z (corrected) = r_B*delta_d/dr - delta_d = {Z_v2_correct:.3f} m')
    else:
        print(f'\nFAIL: dr_v2={dr_v2:.3f} <= 0.2, Looming fails with R_12.T!')

print(f'\nGround truth: Z ~ 2.01 m (from depth map at frame {IDX_B})')

print(f'\n--- Alternative: Raw ROI centers ---')
dr_raw = dist_raw_B - dist_raw_A
print(f'Raw ROI dr = {dr_raw:.3f} px')
if dr_raw > 0.2:
    Z_raw = dist_raw_B * tz / dr_raw
    Z_raw_correct = dist_raw_B * tz / dr_raw - tz
    print(f'Z (raw ROI, simple) = {Z_raw:.3f} m')
    print(f'Z (raw ROI, corrected) = {Z_raw_correct:.3f} m')