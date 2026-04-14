import numpy as np

class FOEEstimatorFromSLAM:
    """使用 SLAM 位姿计算膨胀焦点 (FOE)"""
    def __init__(self, K):
        self.K = K  # 相机内参 3x3 矩阵
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

    def _inverse_pose(self, T):
        """求 SE(3) 变换矩阵的逆 T^{-1}"""
        R = T[:3, :3]
        t = T[:3, 3:]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3:] = -R.T @ t
        return T_inv

    def get_foe(self, T_cw1, T_cw2):
        """
        计算 Frame 1 图像平面上的 FOE
        T_cw1: Frame 1 的 World to Camera 变换矩阵 (4x4)
        T_cw2: Frame 2 的 World to Camera 变换矩阵 (4x4)
        """
        # 1. 计算相对位姿 T_12 (将 Frame 2 的点变换到 Frame 1)
        # T_12 = T_cw1 * T_wc2 = T_cw1 * T_cw2^{-1}
        T_wc2 = self._inverse_pose(T_cw2)
        T_12 = T_cw1 @ T_wc2

        # 2. 提取平移向量 t_12
        # 这是 Frame 2 的相机光心 (0,0,0) 在 Frame 1 坐标系下的 3D 物理坐标
        t_12 = T_12[:3, 3]
        tx, ty, tz = t_12[0], t_12[1], t_12[2]

        # 3. 物理极性检查 (必须有纵向位移)
        if abs(tz) < 1e-5:
            # 纯横向移动或静止，FOE 在无穷远处
            return None 

        # 4. 相机针孔模型投影，获得 FOE 的像素坐标
        u_foe = (self.fx * tx / tz) + self.cx
        v_foe = (self.fy * ty / tz) + self.cy

        return np.array([u_foe, v_foe])