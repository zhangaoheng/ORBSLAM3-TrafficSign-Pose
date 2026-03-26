import os
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

# ========== 配置区域 ==========
MODEL_PATH = r"D:/zah/Latex/road_sign_pose_project/yolo11n_pose_run2/weights/best.pt"
TEST_SOURCE = r"D:/zah/Latex/unityDataset/VideoFrames/base"
# 每次运行结果保存到带时间戳的独立文件夹
import datetime
run_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
# 获取source文件夹名
source_name = os.path.basename(os.path.normpath(TEST_SOURCE))
OUTPUT_DIR = os.path.join('D:/zah/Latex/yolo11/pose_inference_results', f'{source_name}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'camera_trajectory_pnp.csv')
# 路牌实际尺寸（单位：米）
BOARD_WIDTH = 2.0
BOARD_HEIGHT = 1.0
# 假设内参（需根据实际相机调整）
CAMERA_MATRIX = np.array([[1200, 0, 640], [0, 1200, 360], [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros((4, 1))
# ============================

def main():
    model = YOLO(MODEL_PATH)
    results = model.predict(source=TEST_SOURCE, save=False, conf=0.1)

    # 世界坐标（左上、右上、右下、左下）
    obj_points = np.array([
        [0, 0, 0],
        [BOARD_WIDTH, 0, 0],
        [BOARD_WIDTH, BOARD_HEIGHT, 0],
        [0, BOARD_HEIGHT, 0]
    ], dtype=np.float32)

    trajectory = []
    save_img_dir = os.path.join(OUTPUT_DIR, 'yolo_vis')
    os.makedirs(save_img_dir, exist_ok=True)
    for result in results:
        file_name = os.path.basename(result.path)
        save_path = os.path.join(save_img_dir, file_name)
        # 默认保存原图
        img_to_save = None
        try:
            orig_img = cv2.imread(result.path)
        except Exception:
            orig_img = None
        # 检查是否有检测结果
        if result.keypoints is not None and result.boxes is not None:
            kpts = result.keypoints.xy.cpu().numpy()  # (N, 4, 2)  N为目标数
            confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else None
            if kpts.shape[0] > 0 and confs is not None and len(confs) > 0:
                # 只选置信度最高的目标
                best_idx = np.argmax(confs)
                kpt = kpts[best_idx]
                if kpt.shape[0] >= 4:
                    img_points = kpt[:4].astype(np.float32)  # 取前4个点
                    # PnP
                    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE)
                    if success:
                        trajectory.append({
                            'image': file_name,
                            'target': int(best_idx),
                            'tx': tvec[0][0], 'ty': tvec[1][0], 'tz': tvec[2][0],
                            'rx': rvec[0][0], 'ry': rvec[1][0], 'rz': rvec[2][0]
                        })
                # 有检测结果时保存可视化
                img_to_save = result.plot(line_width=1, kpt_radius=2)
        # 没有检测结果时保存原图
        if img_to_save is None and orig_img is not None:
            img_to_save = orig_img
        if img_to_save is not None:
            cv2.imwrite(save_path, img_to_save)


    # 保存为csv
    df = pd.DataFrame(trajectory)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"轨迹已保存到: {OUTPUT_CSV}")

    # ===== 轨迹三维可视化 =====
    if len(df) > 0:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(df['tx'], df['ty'], df['tz'], marker='o', label='Camera Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Estimated Camera Trajectory (PnP)')
        ax.legend()
        ax.set_box_aspect([1, 1, 1])  # 保持比例尺一致
        plt.show()

if __name__ == "__main__":
    main()
