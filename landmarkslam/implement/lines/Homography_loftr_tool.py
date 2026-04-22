import cv2
import torch
import numpy as np
from kornia.feature import LoFTR

# ==========================================
# 1. 鼠标交互选点工具：生成多边形 ROI (加入空间防抖)
# ==========================================
def select_four_points(img, window_name):
    """
    弹出窗口，允许用户鼠标左键点击选择 4 个点，划定 ROI 区域。
    加入空间防抖，防止鼠标硬件连击导致重复录入。
    """
    points = []
    display_img = img.copy()
    
    print(f"\n👉 请在 '{window_name}' 中，顺时针或逆时针点击靶标外围的 4 个点。")

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                # 🚀 核心优化：空间防抖逻辑
                if len(points) > 0:
                    last_x, last_y = points[-1]
                    # 计算当前点击位置与上一个点的欧氏距离
                    distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                    # 如果距离小于 10 个像素，判定为鼠标连击误触，直接忽略
                    if distance < 10:
                        print("⚠️ 检测到重叠点击/鼠标连击，已自动忽略！")
                        return
                        
                points.append((x, y))
                cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
                
                if len(points) > 1:
                    cv2.line(display_img, points[-2], points[-1], (0, 255, 0), 2)
                if len(points) == 4:
                    cv2.line(display_img, points[3], points[0], (0, 255, 0), 2)
                    
                cv2.imshow(window_name, display_img)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display_img)
    cv2.setMouseCallback(window_name, mouse_callback)

    while len(points) < 4:
        cv2.waitKey(50) 
        
    cv2.destroyWindow(window_name)
    print(f"✅ {window_name} 框选完成: {points}")
    return np.array(points, dtype=np.int32)

# ==========================================
# 2. 核心匹配模块 (加入 ROI 掩码逻辑)
# ==========================================
def match_images_with_loftr_roi(img1_color, img2_color):
    """
    结合手动框选的 ROI 进行 LoFTR 匹配
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"👉 正在使用设备: {device}")

    pts1_roi = select_four_points(img1_color, "Image 1: Select ROI")
    pts2_roi = select_four_points(img2_color, "Image 2: Select ROI")

    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    mask1 = np.zeros_like(img1_gray)
    cv2.fillPoly(mask1, [pts1_roi], 255)
    mask2 = np.zeros_like(img2_gray)
    cv2.fillPoly(mask2, [pts2_roi], 255)

    img1_masked = cv2.bitwise_and(img1_gray, img1_gray, mask=mask1)
    img2_masked = cv2.bitwise_and(img2_gray, img2_gray, mask=mask2)

    img1_tensor = torch.from_numpy(img1_masked)[None, None].float() / 255.0
    img2_tensor = torch.from_numpy(img2_masked)[None, None].float() / 255.0
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)

    print("⏳ 正在加载 LoFTR 模型进行局部精准匹配...")
    matcher = LoFTR(pretrained='outdoor').to(device)
    matcher.eval()

    with torch.no_grad():
        correspondences = matcher({"image0": img1_tensor, "image1": img2_tensor})

    mkpts1 = correspondences['keypoints0'].cpu().numpy()
    mkpts2 = correspondences['keypoints1'].cpu().numpy()
    confidence = correspondences['confidence'].cpu().numpy()
    
    print(f"🎯 LoFTR 在掩码区域提取到 {len(mkpts1)} 对匹配点。")

    conf_thresh = 0.5
    good_pts1, good_pts2 = [], []
    
    for p1, p2, conf in zip(mkpts1, mkpts2, confidence):
        if conf > conf_thresh:
            in_poly1 = cv2.pointPolygonTest(pts1_roi, (float(p1[0]), float(p1[1])), False) >= 0
            in_poly2 = cv2.pointPolygonTest(pts2_roi, (float(p2[0]), float(p2[1])), False) >= 0
            
            if in_poly1 and in_poly2:
                good_pts1.append(p1)
                good_pts2.append(p2)

    print(f"🔪 经过置信度与严格 ROI 边界过滤，剩下 {len(good_pts1)} 对纯净匹配点！")
    return np.array(good_pts1), np.array(good_pts2)

# ==========================================
# 3. 可视化模块 (交互式逐个查看)
# ==========================================
def visualize_matches_one_by_one(img1_color, img2_color, pts1, pts2, mask):
    """
    交互式可视化：逐个查看匹配点连线
    操作：按 'd' 看下一个，按 'a' 看上一个，按 'q' 退出进入下一步计算
    """
    keypoints1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts1]
    keypoints2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts2]
    
    valid_indices = [i for i, is_inlier in enumerate(mask.ravel()) if is_inlier]
    
    if not valid_indices:
        print("❌ 没有有效的匹配点可供查看！")
        return

    print(f"\n🔬 开启显微镜模式：准备逐个审查 {len(valid_indices)} 对匹配点！")
    print("="*40)
    print("👉 操作指南:")
    print("   [ D ] 键 : 下一个匹配点 (Next)")
    print("   [ A ] 键 : 上一个匹配点 (Previous)")
    print("   [ Q ] 键 : 退出查看，继续输出位姿结果 (Quit)")
    print("="*40)

    current_idx = 0
    window_name = "Microscope Mode: Inspect Matches (Press Q to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        match_idx = valid_indices[current_idx]
        single_match = [cv2.DMatch(match_idx, match_idx, 0)]

        matched_img = cv2.drawMatches(
            img1_color, keypoints1, 
            img2_color, keypoints2, 
            single_match, None, 
            matchColor=(0, 255, 0),       
            singlePointColor=(255, 0, 0), 
            flags=cv2.DrawMatchesFlags_DEFAULT 
        )

        text = f"Match: {current_idx + 1} / {len(valid_indices)} | ID: {match_idx}"
        cv2.putText(matched_img, text, (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow(window_name, matched_img)
        
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27:  
            break
        elif key == ord('d'):             
            current_idx = (current_idx + 1) % len(valid_indices)
        elif key == ord('a'):             
            current_idx = (current_idx - 1) % len(valid_indices)

    cv2.destroyAllWindows()
    print("✅ 审查结束，继续往下执行...")

# ==========================================
# 4. 主流程
# ==========================================
if __name__ == "__main__":
    IMG1_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/Homography/1.png"
    IMG2_PATH = "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/Homography/2.png"
    
    K = np.array([[429.78,  0,     429.94],
                  [ 0,      429.78, 241.57],
                  [ 0,       0,      1    ]], dtype=np.float64)

    img1_color = cv2.imread(IMG1_PATH)
    img2_color = cv2.imread(IMG2_PATH)
    
    if img1_color is None or img2_color is None:
        print("❌ 图像读取失败！")
        exit()

    pts1, pts2 = match_images_with_loftr_roi(img1_color, img2_color)
    
    if pts1 is not None and len(pts1) >= 4:
        print("\n🚀 正在计算 Homography 矩阵...")
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
        
        print("--- 高精度单应性矩阵 H ---")
        print(np.round(H, 4))
        
        # 🌟 修复点：调用交互式的显微镜查看函数 🌟
        visualize_matches_one_by_one(img1_color, img2_color, pts1, pts2, mask)
        
        num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
        print(f"\n✅ 成功分解出 {num_solutions} 组位姿解。")
        
        for i in range(num_solutions):
            R = rotations[i]
            t = translations[i]
            n = normals[i].flatten()
            
            if n[2] > 0:
                print(f"🟢 【解 {i+1}】: 物理可能 (法向量指向相机前方)")
                print(f"   法向量 n : {np.round(n, 4)}")
                print(f"   旋转 R   :\n{np.round(R, 4)}")
                print(f"   归一化 t :\n{np.round(t.flatten(), 4)}")
                print("-" * 40)
def match_images_with_auto_roi(img1_color, img2_color, pts1_roi, pts2_roi):
    """
    自动通过外部传入的物理边界（四角点）划定 ROI 进行 LoFTR 匹配，无需人工点击
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"👉 正在使用设备: {device} 进行自动化共面 ROI 匹配")

    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    mask1 = np.zeros_like(img1_gray)
    cv2.fillPoly(mask1, [pts1_roi], 255)
    mask2 = np.zeros_like(img2_gray)
    cv2.fillPoly(mask2, [pts2_roi], 255)

    img1_masked = cv2.bitwise_and(img1_gray, img1_gray, mask=mask1)
    img2_masked = cv2.bitwise_and(img2_gray, img2_gray, mask=mask2)

    img1_tensor = torch.from_numpy(img1_masked)[None, None].float() / 255.0
    img2_tensor = torch.from_numpy(img2_masked)[None, None].float() / 255.0
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)

    print("⏳ 正在加载 LoFTR 模型进行自动局部精准匹配...")
    matcher = LoFTR(pretrained='outdoor').to(device)
    matcher.eval()

    with torch.no_grad():
        correspondences = matcher({"image0": img1_tensor, "image1": img2_tensor})
        
    mkpts1 = correspondences['keypoints0'].cpu().numpy()
    mkpts2 = correspondences['keypoints1'].cpu().numpy()
    confidence = correspondences['confidence'].cpu().numpy()
    
    print(f"🎯 LoFTR 在自动掩码区域提取到 {len(mkpts1)} 对共面匹配点。")

    conf_thresh = 0.5
    good_pts1, good_pts2 = [], []
    
    for p1, p2, conf in zip(mkpts1, mkpts2, confidence):
        if conf > conf_thresh:
            in_poly1 = cv2.pointPolygonTest(pts1_roi, (float(p1[0]), float(p1[1])), False) >= 0
            in_poly2 = cv2.pointPolygonTest(pts2_roi, (float(p2[0]), float(p2[1])), False) >= 0
            if in_poly1 and in_poly2:
                good_pts1.append(p1)
                good_pts2.append(p2)

    print(f"🔪 经过置信度与严格自动 ROI 边界过滤，重构出 {len(good_pts1)} 对有效共面点！")
    return np.array(good_pts1), np.array(good_pts2)
