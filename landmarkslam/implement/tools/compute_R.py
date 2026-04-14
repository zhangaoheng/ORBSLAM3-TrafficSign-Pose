import cv2
import numpy as np
from scipy.optimize import least_squares

class SemanticOrthogonalOptimizer:
    """Step 6: 基于汉字拓扑的位姿终极优化器"""
    def __init__(self, K):
        self.K = K
        self.K_T = K.T  # K的转置，用于计算射线平面法向量

    def _line_from_pts(self, p1, p2):
        """已知两点，求2D直线的一般式齐次坐标 l = [a, b, c]^T"""
        x1, y1 = p1
        x2, y2 = p2
        # 叉乘求直线：l = p1_homo x p2_homo
        l = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
        # 归一化以保证数值稳定性
        return l / np.linalg.norm(l[:2])

    def _cost_function(self, rvec, lines_h, lines_v, rvec_init):
        """
        误差函数 (Cost Function)
        rvec: 当前正在优化的旋转向量 (3x1)
        lines_h: 提取到的所有近似横向笔画的 2D 直线方程
        lines_v: 提取到的所有近似纵向笔画的 2D 直线方程
        """
        # 1. 恢复旋转矩阵 R
        R, _ = cv2.Rodrigues(rvec)
        
        # 2. 提取标志牌的 3D 法向量 n 
        # 假设路牌的局部坐标系 Z 轴垂直于路牌向外，则法向量就是 R 的第三列
        n = R[:, 2] 
        
        residuals = []
        
        # 3. 语义正交残差 (Semantic Orthogonal Residuals)
        # 遍历所有的横竖线段对，它们在 3D 世界中必须正交
        for lh in lines_h:
            # 射线平面法向量: pi_h = K^T * lh
            pi_h = self.K_T.dot(lh)
            # 3D 直线方向向量: v_h = pi_h x n
            v_h = np.cross(pi_h, n)
            v_h = v_h / (np.linalg.norm(v_h) + 1e-8)
            
            for lv in lines_v:
                pi_v = self.K_T.dot(lv)
                v_v = np.cross(pi_v, n)
                v_v = v_v / (np.linalg.norm(v_v) + 1e-8)
                
                # 核心惩罚项：横竖笔画的 3D 点积应该等于 0 (正交)
                dot_product = np.dot(v_h, v_v)
                residuals.append(dot_product)
                
        # 4. 先验正则化惩罚 (Prior Regularization)
        # 防止优化器跑飞，把路牌完全翻转过去。优化后的 R 不能偏离初始 R 太远
        # 权重 lambda 可以根据初始单应矩阵的置信度来调节
        lambda_reg = 0.5 
        reg_error = lambda_reg * (rvec - rvec_init).flatten()
        residuals.extend(reg_error)
        
        return np.array(residuals)

    def optimize_pose(self, R_init, strokes_horizontal, strokes_vertical):
        """
        执行优化
        strokes_horizontal: list of tuples ((x1,y1), (x2,y2)) 横向笔画端点
        strokes_vertical: list of tuples ((x1,y1), (x2,y2)) 纵向笔画端点
        """
        # 将端点转换为 2D 直线方程
        lines_h = [self._line_from_pts(p1, p2) for p1, p2 in strokes_horizontal]
        lines_v = [self._line_from_pts(p1, p2) for p1, p2 in strokes_vertical]
        
        # 初始姿态转换为旋转向量
        rvec_init, _ = cv2.Rodrigues(R_init)
        rvec_init = rvec_init.flatten()
        
        # 启动 Levenberg-Marquardt 非线性优化
        # loss='huber' 极其关键！因为提取的笔画中可能混有 '撇' 和 '捺'，
        # Huber Loss 可以自动抑制这些非垂直线条对全局优化的干扰！
        result = least_squares(
            self._cost_function, 
            rvec_init, 
            args=(lines_h, lines_v, rvec_init),
            loss='huber',  
            f_scale=0.1,   # Huber loss 的截断阈值
            method='lm' if len(lines_h)*len(lines_v) > 0 else 'trf'
        )
        
        # 恢复出被打磨到极致的最优旋转矩阵
        R_opt, _ = cv2.Rodrigues(result.x)
        
        # 由于 n_opt 改变了，我们还需要把法向量更新回去
        n_opt = R_opt[:, 2]
        
        return R_opt, n_opt