import numpy as np
import cv2
import glob

# ================= 1. 核心参数配置 =================
# 内部角点数量 (长边角点数, 短边角点数) - 我已经帮你数好填上了
CHECKERBOARD = (11, 8) 

# 请将这里的 0.030 替换为你实际测量的一个黑方块的边长（单位：米）
SQUARE_SIZE = 0.030  
IMAGE_FOLDER = 'calib_images/*.jpg'
# ===================================================

# 亚像素角点提取的迭代终止条件 (最大迭代30次或精度达到0.001)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备 3D 真实世界坐标点 (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# 存储所有图片的 3D 点和 2D 点
objpoints = [] # 真实世界中的 3D 点
imgpoints = [] # 图像平面中的 2D 点

images = glob.glob(IMAGE_FOLDER)
if not images:
    print("没有找到图片，请检查路径！")
    exit()

print(f"找到 {len(images)} 张图片，开始高精度角点提取...")
image_size = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1]

    # 1. 寻找棋盘格的粗略角点
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 2. 如果找齐了所有角点，则进入高精度模式
    if ret == True:
        objpoints.append(objp)
        
        # 核心：亚像素级精确化（提升精度的秘密武器）
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # 可视化：把找到的角点画出来看看 (可选)
        # cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)

cv2.destroyAllWindows()
print("特征点提取完毕，开始计算内参...")

# 3. 执行相机标定
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None)

print("\n✅ 标定成功！")
print("====================================")
print("重投影误差 (越接近0.1越好):", ret)
print("\n内参矩阵 (Camera Matrix):\n", cameraMatrix)
print("\n畸变系数 (DistCoeffs):\n", distCoeffs.ravel())
print("====================================")