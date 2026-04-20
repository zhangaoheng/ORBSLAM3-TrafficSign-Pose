import cv2
import numpy as np

# ==========================================
# 1. 鼠标交互选点工具 (功能不变，仅作为框选 ROI 用)
# ==========================================
def select_four_points(img, window_name):
    """
    弹出窗口，允许用户鼠标左键点击选择 4 个点，划定特征提取的 ROI 区域。
    """
    points = []
    display_img = img.copy()
    
    print(f"\n👉 请在 '{window_name}' 中，顺时针或逆时针点击靶标外围的 4 个点。")
    print("这 4 个点将构成一个多边形区域，算法只会在此区域内寻找特征点！")

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
                
                # 如果超过 1 个点，画连线，方便看清包围框
                if len(points) > 1:
                    cv2.line(display_img, points[-2], points[-1], (0, 255, 0), 2)
                # 连闭合线
                if len(points) == 4:
                    cv2.line(display_img, points[3], points[0], (0, 255, 0), 2)
                    
                cv2.imshow(window_name, display_img)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display_img)
    cv2.setMouseCallback(window_name, mouse_callback)

    while len(points) < 4:
        cv2.waitKey(50) 
        
    cv2.destroyWindow(window_name)
    print(f"✅ {window_name} 框选完成。")
    return np.array(points, dtype=np.int32)

# ==========================================
# 2. 核心计算主流程
# ==========================================
if __name__ == "__main__":
    # 【请修改这里】替换为你两张图片的实际路径
    IMG1_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/Homography/1.png"
    IMG2_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/Homography/2.png"

    # 相机内参 K
    K = np.array([[429.78,  0,     429.94],
                  [ 0,      429.78, 241.57],
                  [ 0,       0,      1    ]], dtype=np.float64)

    # 1. 读取图像
    img1 = cv2.imread(IMG1_PATH)
    img2 = cv2.imread(IMG2_PATH)
    if img1 is None or img2 is None:
        print("❌ 图像读取失败，请检查路径。")
        exit()
        
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 2. 用户手动框选 4 个角点 (仅用于生成 Mask)
    pts1_roi = select_four_points(img1, "Image 1: Select ROI")
    pts2_roi = select_four_points(img2, "Image 2: Select ROI")

    # 3. 创建纯黑的掩码 (Mask)，并将用户框选的多边形内部涂白 (255)
    mask1 = np.zeros_like(img1_gray)
    cv2.fillPoly(mask1, [pts1_roi], 255)
    
    mask2 = np.zeros_like(img2_gray)
    cv2.fillPoly(mask2, [pts2_roi], 255)

    # 4. SIFT 提取与匹配 (神级操作：把 mask 传进去，它就只在框里提取！)
    print("\n🔍 正在框选区域内提取 SIFT 特征并进行匹配...")
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, mask=mask1)
    kp2, des2 = sift.detectAndCompute(img2_gray, mask=mask2)
    
    if des1 is None or des2 is None:
        print("❌ 在框选区域内没有找到足够的特征点！请重新框选包含丰富纹理（棋盘格）的区域。")
        exit()

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Lowe's Ratio Test 剔除误匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    print(f"🎯 提取到 {len(good_matches)} 对高质量内部匹配点。")

    if len(good_matches) < 4:
        print("❌ 优质匹配点不足 4 个，无法计算单应性矩阵。")
        exit()

    # 5. 可视化匹配点对 (窗口将一直悬停，直到按下任意键)
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.namedWindow("SIFT Matches inside ROI", cv2.WINDOW_NORMAL)
    cv2.imshow("SIFT Matches inside ROI", img_matches)
    print("👉 请在弹出的匹配图像窗口中按下【任意键盘按键】继续计算...")
    cv2.waitKey(0) # 👈 修改了这里：0 代表无限期等待键盘输入
    cv2.destroyWindow("SIFT Matches inside ROI")

    # 6. 获取匹配点的像素坐标
    pts1_matched = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2_matched = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 7. 使用 RANSAC 计算单应性矩阵 H (这比人手点 4 个点准无数倍)
    H, mask_ransac = cv2.findHomography(pts1_matched, pts2_matched, cv2.RANSAC, 3.0)
    
    print("\n" + "="*40)
    print("--- 基于内部特征点算出的高精度 H ---")
    print(np.round(H, 4))
    print("="*40)

    # 8. 分解单应性矩阵，提取位姿
    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
    print(f"\n✅ 成功分解出 {num_solutions} 组数学解。正在进行物理过滤...\n")

    for i in range(num_solutions):
        R = rotations[i]
        t = translations[i]
        n = normals[i].flatten()
        
        is_physically_possible = n[2] > 0
        
        if is_physically_possible:
            print(f"🟢 【解 {i+1}】: 物理可能 (在相机前方)")
        else:
            print(f"🔴 【解 {i+1}】: 物理不可能 (在相机背后，直接排除！)")
            
        print(f"   法向量 n : {np.round(n, 4)}")
        print(f"   旋转 R   :\n{np.round(R, 4)}")
        print(f"   归一化 t :\n{np.round(t.flatten(), 4)}")
        print("-" * 40)