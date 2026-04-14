import numpy as np

class MonocularVIOSolver:
    """使用纯单目 SLAM 配合底盘数据进行绝对深度与位姿解算"""
    def __init__(self, K):
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

    def _get_relative_motion(self, T_cw1, T_cw2):
        """从世界位姿计算两帧之间的相对运动"""
        T_wc2 = np.linalg.inv(T_cw2)
        T_12 = T_cw1 @ T_wc2
        R_12 = T_12[:3, :3]
        t_12_unscaled = T_12[:3, 3] # 注意：这是无尺度的 t
        return R_12, t_12_unscaled

    def calculate_foe_from_mono_slam(self, t_12_unscaled):
        """
        [奇迹一]：利用无尺度的平移向量计算绝对准确的 FOE
        """
        tx, ty, tz = t_12_unscaled
        
        # 即使 t_12 是无尺度的，只要 tz 不为 0，比例关系是不变的
        if abs(tz) < 1e-6:
            return None # 纯横向移动或静止
            
        u_foe = (self.fx * tx / tz) + self.cx
        v_foe = (self.fy * ty / tz) + self.cy
        return np.array([u_foe, v_foe])

    def derotate_point(self, p2_pixel, R_12):
        """利用单目 SLAM 提供的精确旋转，进行去旋补偿"""
        H_rot = self.K @ R_12 @ self.K_inv
        p2_homo = np.array([p2_pixel[0], p2_pixel[1], 1.0])
        p2_warped = H_rot @ p2_homo
        return p2_warped[:2] / p2_warped[2]

    def solve_absolute_depth(self, p1, p2, T_cw1, T_cw2, chassis_delta_d):
        """
        [终极缝合]：单目 SLAM 提供姿态与方向 + 底盘提供尺度 = 绝对深度 Z
        
        T_cw1, T_cw2: 单目 SLAM 输出的无尺度位姿
        chassis_delta_d: 底盘轮速计/CAN总线读取的真实前向位移 (米)
        """
        # 1. 提取相对旋转和无尺度平移
        R_12, t_12_unscaled = self._get_relative_motion(T_cw1, T_cw2)
        
        # 2. 计算 FOE (免疫尺度模糊)
        foe = self.calculate_foe_from_mono_slam(t_12_unscaled)
        if foe is None: return None
        
        # 3. 对 P2 进行纯平移去旋
        p2_pure = self.derotate_point(p2, R_12)
        
        # 4. 计算纯净像素半径
        r1 = np.linalg.norm(p1 - foe)
        r2 = np.linalg.norm(p2_pure - foe)
        
        delta_r = r2 - r1
        if delta_r <= 1.0: # 像素量化噪声保护
            return None 
            
        # 5. [奇迹二] 引入底盘绝对尺度，计算光轴深度 Z
        Z2 = (r1 * chassis_delta_d) / delta_r
        
        return Z2, R_12