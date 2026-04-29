"""
分析 Looming 测距 37.83% 误差的来源：
是公式推导错误，还是测量噪声？
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2, math, numpy as np, glob
from scipy.spatial.transform import Rotation as R_scipy
import yaml

# 加载配置
with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r') as f:
    cfg = yaml.safe_load(f)

fx, fy = cfg['Camera']['fx'], cfg['Camera']['fy']
cx, cy = cfg['Camera']['cx'], cfg['Camera']['cy']
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
K_inv = np.linalg.inv(K)
FRAME_STEP = cfg['Algorithm']['frame_step']

FOLDER_PATH_1 = cfg['Sequence1']['image_dir']
TRAJ_PATH_1   = cfg['Sequence1']['trajectory_path']
ROI_PATH_1    = cfg['Sequence1']['roi_path']
DEPTH_DIR_1   = cfg['Sequence1'].get('depth_dir', '')

idx1_base = 240
idx1_prev = idx1_base - FRAME_STEP

# ---------- 1. 加载 SLAM 轨迹，计算真实 delta_d ----------
poses = {}
with open(TRAJ_PATH_1, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        poses[float(parts[0])] = np.array([float(x) for x in parts[1:8]])

images1 = sorted(glob.glob(os.path.join(FOLDER_PATH_1, "*.png")))
ts_A = float(os.path.basename(images1[idx1_prev]).replace('.png', '')) / 1e9
ts_B = float(os.path.basename(images1[idx1_base]).replace('.png', '')) / 1e9

times = np.array(list(poses.keys()))
iA = np.argmin(np.abs(times - ts_A))
iB = np.argmin(np.abs(times - ts_B))

pose_A = poses[times[iA]]
pose_B = poses[times[iB]]

t1, q1 = pose_A[0:3], pose_A[3:7]
t2, q2 = pose_B[0:3], pose_B[3:7]
R1 = R_scipy.from_quat(q1).as_matrix()
R2 = R_scipy.from_quat(q2).as_matrix()
R_12 = R1.T @ R2
t_12 = R1.T @ (t2 - t1)

tx, ty, tz = t_12
delta_d_actual = tz

print("=" * 70)
print("1. SLAM 轨迹信息")
print("=" * 70)
print(f"  帧 A (far,  idx={idx1_prev}): ts={ts_A:.6f}, matched SLAM idx={iA}")
print(f"  帧 B (near, idx={idx1_base}): ts={ts_B:.6f}, matched SLAM idx={iB}")
print(f"  相对平移 t_12 (在帧A系下): tx={tx:.4f}, ty={ty:.4f}, tz(delta_d)={tz:.4f} m")
print(f"  FOE = ({fx * tx/tz + cx:.1f}, {fy * ty/tz + cy:.1f})")

# ---------- 2. 加载 ROI 中心 ----------
if os.path.exists(ROI_PATH_1):
    with open(ROI_PATH_1, 'r') as f:
        roi_lines = [line.strip() for line in f if line.strip()]
    xA, yA, wA, hA = map(int, roi_lines[idx1_prev].split(','))
    xB, yB, wB, hB = map(int, roi_lines[idx1_base].split(','))
    cA_raw = np.array([xA + wA/2, yA + hA/2])
    cB_raw = np.array([xB + wB/2, yB + hB/2])
    print(f"\n  ROI 中心 (raw):")
    print(f"    帧A: ({cA_raw[0]:.1f}, {cA_raw[1]:.1f})")
    print(f"    帧B: ({cB_raw[0]:.1f}, {cB_raw[1]:.1f})")

# ---------- 3. 去旋 ----------
H = K @ R_12.T @ K_inv
p_homo = np.array([cA_raw[0], cA_raw[1], 1.0])
p_pure = H @ p_homo
cA_pure = np.array([p_pure[0]/p_pure[2], p_pure[1]/p_pure[2]])
print(f"\n2. 去旋:")
print(f"  R_12 角度: roll={math.degrees(math.atan2(R_12[2,1],R_12[2,2])):.2f}°, "
      f"pitch={math.degrees(-math.asin(R_12[2,0])):.2f}°, "
      f"yaw={math.degrees(math.atan2(R_12[1,0],R_12[0,0])):.2f}°")
print(f"  帧A去旋后中心: ({cA_pure[0]:.1f}, {cA_pure[1]:.1f})")
print(f"  去旋偏移: ({cA_pure[0]-cA_raw[0]:.2f}, {cA_pure[1]-cA_raw[1]:.2f}) px")

# ---------- 4. 计算 FOE 和 Looming 膨胀量 ----------
FOE = np.array([fx * tx/tz + cx, fy * ty/tz + cy])
r_far  = np.linalg.norm(cA_pure - FOE)
r_near = np.linalg.norm(cB_raw - FOE)
dr = r_near - r_far

Z_gt = 0.542  # 从深度图验证的深度真值

print(f"\n3. Looming 膨胀量:")
print(f"  r_far (去旋后)  = {r_far:.2f} px")
print(f"  r_near           = {r_near:.2f} px")
print(f"  dr = r_near - r_far = {dr:.2f} px")

# ---------- 5. 公式推导 ----------
print("\n" + "=" * 70)
print("4. Looming 公式推导 (关键!)")
print("=" * 70)
print("""
假设: 相机纯向前平移 delta_d，标志物在帧B系下深度 = Z。

相似三角形 (针孔模型):
  r_near / f = X / Z        → r_near = f * X / Z
  r_far  / f = X / (Z+Δd)   → r_far  = f * X / (Z + Δd)

所以:
  r_far / r_near = Z / (Z + Δd)                         (1)

膨胀量 dr:
  dr = r_near - r_far = r_near * (1 - Z/(Z+Δd))
     = r_near * Δd / (Z + Δd)                           (2)

从 (2) 解出 Z:
  Z + Δd = r_near * Δd / dr
  Z = r_near * Δd / dr - Δd                             ← 正确公式!

用 r_far 的等价形式 (从 (1)):
  Z = r_far * Δd / dr                                    ← 等价正确公式!
""")

# ---------- 6. 对比 ----------
Z_code     = r_near * delta_d_actual / dr     # 代码当前公式 (缺 -Δd)
Z_correct  = Z_code - delta_d_actual           # 修正后
Z_correct2 = r_far * delta_d_actual / dr       # 用 r_far 的等价形式

print("5. 计算结果对比:")
print("-" * 70)
print(f"  代码当前公式 Z = r_near * Δd / dr     = {Z_code:.4f} m")
print(f"  正确公式     Z = r_near * Δd / dr - Δd = {Z_correct:.4f} m  ←")
print(f"  等价正确公式 Z = r_far  * Δd / dr     = {Z_correct2:.4f} m")
print(f"  深度真值     Z_gt                    = {Z_gt:.4f} m")
print()
print(f"  代码 Z 与真值差: {Z_code - Z_gt:.4f} m")
print(f"  Δd = {delta_d_actual:.4f} m")
print(f"  Z_code - Z_gt ≈ Δd ? {abs((Z_code - Z_gt) - delta_d_actual) < 1e-6}")
print()
print(f"  修正后 Z 与真值差: {Z_correct - Z_gt:.4f} m = {(Z_correct - Z_gt)*1000:.1f} mm")
print(f"  修正后相对误差: {abs(Z_correct - Z_gt) / Z_gt * 100:.2f}%")

# ---------- 7. 误差来源分解 ----------
print("\n" + "=" * 70)
print("6. 误差来源分解")
print("=" * 70)

# 代码误差 = delta_d (公式错误) + ε (测量噪声)
# 实测: Z_code - Z_gt = 0.747 - 0.542 = 0.205
# delta_d_actual = ?
# ε = (Z_code - Z_gt) - delta_d_actual

total_err = Z_code - Z_gt
formula_err = delta_d_actual
noise = total_err - formula_err

print(f"  总误差 Z_code - Z_gt = {total_err*1000:.1f} mm (100%)")
print(f"    公式偏置误差 Δd    = {formula_err*1000:.1f} mm ({abs(formula_err/total_err)*100:.1f}%)")
print(f"    残留测量噪声 ε    = {noise*1000:.1f} mm ({abs(noise/total_err)*100:.1f}%)")
print()
if abs(noise) < 0.01:
    print("  🎯 结论: 误差 100% 来自公式推导错误 (缺少 -Δd 项)，非测量噪声!")
elif abs(noise) < 0.05:
    print("  🔶 结论: 误差主要来自公式推导错误 ({:.0f}%)，测量噪声很小 ({:.0f}%)".format(
        abs(formula_err/total_err)*100, abs(noise/total_err)*100))
else:
    print("  ⚠️ 结论: 公式错误是主要原因，但也有一定的测量噪声")

print()
print("=" * 70)
print("🔧 修复方案: 将 test.py 第 ~146/103 行 Z = (r_near * delta_d) / dr")
print("   改为 Z = (r_near * delta_d) / dr - delta_d")
print("   或 Z = (r_far * delta_d) / dr")
print("=" * 70)