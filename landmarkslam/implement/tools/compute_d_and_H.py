import cv2
import numpy as np

class HomographyPoseEstimator:
    def __init__(self, K):
        self.K = K
        self.K_inv = np.linalg.inv(K)

    def _filter_valid_solution(self, Rs, Ts, Ns, pts1):
        """
        [架构师的避坑点] 
        单应分解必然产生 4 组数学解，但物理世界只有 1 组是真的！
        过滤规则：
        1. 所有的特征点在相机前方 (Z > 0)
        2. 路牌平面的法向量必须朝向相机 (n_z < 0，假设相机朝向正Z轴)
        """
        valid_idx = -1
        max_positive_depths = 0

        for i in range(len(Rs)):
            R = Rs[i]
            T = Ts[i]
            n = Ns[i]

            # 物理规则 1: 法向量检查 (路牌面朝相机，法向量 z 分量应为负)
            if n[2][0] > 0:
                continue

            # 物理规则 2: 深度正向性检查
            positive_count = 0
            for pt in pts1:
                # 逆投影到归一化平面
                p_homo = np.array([pt[0], pt[1], 1.0])
                X_norm = self.K_inv.dot(p_homo)
                # 计算在平面上的投影深度
                # 根据平面方程 n^T * X = d_norm (这里的 d_norm 在无尺度下设为1)
                Z_test = 1.0 / np.dot(n.T, X_norm)[0]
                
                # 检查在当前帧和下一帧的深度是否都为正
                X_3d = X_norm * Z_test
                X_3d_next = R.dot(X_3d.reshape(3, 1)) + T
                
                if Z_test > 0 and X_3d_next[2][0] > 0:
                    positive_count += 1

            if positive_count > max_positive_depths:
                max_positive_depths = positive_count
                valid_idx = i

        if valid_idx == -1:
            raise ValueError("未能找到符合物理规律的位姿解！请检查特征匹配或 H 矩阵质量。")

        return Rs[valid_idx], Ts[valid_idx], Ns[valid_idx]

    def recover_scaled_pose(self, img1, img2, bbox1, bbox2, Z_center, center_pt1):
        """
        完整流程：特征提取 -> RANSAC 算 H -> 分解 -> 物理过滤 -> 尺度注入
        
        参数:
        Z_center: Step 3 (Looming) 算出的中心点绝对物理深度
        center_pt1: 第一帧的标志牌中心像素坐标 (u_c, v_c)
        """
        # 1. 在 ROI 区域内提取特征点并匹配 (这里以 ORB 为例，实战可换 SuperPoint)
        # 注意：一定要把 bbox 传进去，只在路牌区域内找点，背景点会直接毁掉 H 矩阵！
        pts1, pts2 = self._extract_and_match_features_in_roi(img1, img2, bbox1, bbox2)
        
        if len(pts1) < 4:
            raise ValueError("匹配到的特征点少于 4 对，无法计算单应矩阵！")

        # 2. RANSAC 鲁棒解算 H 矩阵
        # 这里的 3.0 是重投影误差阈值，像素单位。路牌如果是平的，阈值可以设小一点(如 2.0)
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 2.0)
        
        # 保留 Inliers 用于后续解的校验
        inlier_pts1 = pts1[mask.ravel() == 1]

        # 3. 单应矩阵分解 (得到无尺度的 t 和 法向量 n)
        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, self.K)

        # 4. 提取唯一的物理真实解
        R_true, t_normalized, n_true = self._filter_valid_solution(Rs, Ts, Ns, inlier_pts1)

        # ==========================================
        # 5. 灵魂操作：打破尺度魔咒，注入绝对深度！
        # ==========================================
        # a. 构造第一帧中心点的齐次坐标
        p_c_homo = np.array([center_pt1[0], center_pt1[1], 1.0])
        
        # b. 结合已知的光轴绝对深度 Z_center，恢复中心点的 3D 绝对坐标
        X_center_3d = Z_center * self.K_inv.dot(p_c_homo)
        
        # c. 利用点法式求取相机光心到平面的正交垂距 d
        # d = |n^T * X_center_3d|
        d_real = np.abs(np.dot(n_true.reshape(-1), X_center_3d))
        
        # d. 致命缝合：恢复真实的物理平移向量
        t_real = t_normalized * d_real

        return R_true, t_real, n_true, inlier_pts1

    def _extract_and_match_features_in_roi(self, img1, img2, bbox1, bbox2):
        """
        (工程桩代码) 在 BBox 内进行特征提取和匹配
        建议：路牌内部纹理较弱，尽量提取汉字的角点或使用 SuperPoint
        """
        # ... (OpenCV ORB/SIFT + FlannBasedMatcher 逻辑) ...
        pass