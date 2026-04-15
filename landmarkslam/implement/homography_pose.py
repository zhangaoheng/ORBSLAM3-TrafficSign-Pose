import cv2
import numpy as np


def compute_homography_and_pose(img_path1, img_path2, K):
    """
    读取两张图像，进行特征匹配，计算单应性矩阵 H，并分解出 R 和 t
    """
    # 1. 读取图像 (转为灰度图)
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print("❌ 无法读取图像，请检查路径。")
        return None

    # 2. 初始化 SIFT 探测器
    sift = cv2.SIFT_create()
    
    # 寻找关键点和描述子
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # 3. 暴力匹配器 (BFMatcher) 配合 L2 距离
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # 4. 应用 Lowe's ratio test 过滤误匹配 (极大地提高 H 矩阵精度)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    print(f">>> 提取到优质匹配点对: {len(good_matches)} 个")
    
    if len(good_matches) < 4:
        print("❌ 匹配点不足 4 个，无法计算单应性矩阵！")
        return None
        
    # 5. 提取匹配点的像素坐标
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 6. 计算单应性矩阵 H (使用 RANSAC 剔除离群点)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    print("\n--- 算出的单应性矩阵 H ---")
    print(H)
    
    # [可视化] 画出 RANSAC 筛选后的正确匹配连线
    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,
                       matchesMask=matchesMask, flags=2)
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
    cv2.imshow("Good Matches", img_matches)
    cv2.waitKey(1) # 闪现一下窗口，0表示阻塞等待
    
    # ==========================================
    # 7. 核心：分解单应性矩阵，获取 R, t 和法向量 n
    # ==========================================
    # OpenCV 直接提供了神级函数 decomposeHomographyMat
    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
    
    print(f"\n--- 单应性分解完成，获得 {num_solutions} 组数学解 ---")
    
    return rotations, translations, normals

if __name__ == "__main__":
    # 使用你之前的 D456 内参
    K = np.array([[429.78045654,  0,          429.94277954],
                  [ 0,          429.78045654, 241.57313537],
                  [ 0,           0,           1         ]], dtype=np.float64)
                  
    # 找两张你数据集里的图片来测试 (替换为你的真实路径)
    # 建议选一张近的(轨迹2)，一张远的(轨迹1)，但保证牌子都在画面里
    IMG1 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/mid/1776083697.100000000.png" 
    IMG2 = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/mid/1776083715.100000000.png"
    
    results = compute_homography_and_pose(IMG1, IMG2, K)
    
    if results:
        rotations, translations, normals = results
        # 打印出这 4 组解
        for i in range(4):
            print(f"\n[解 {i+1}]")
            print(f"旋转 R:\n{rotations[i]}")
            print(f"平移 t (归一化尺度):\n{translations[i].T}")
            print(f"法向量 n:\n{normals[i].T}")