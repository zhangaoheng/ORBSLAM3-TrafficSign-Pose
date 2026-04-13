import pyrealsense2 as rs
import numpy as np

# ⚠️ 换成你的 bag 包绝对路径
BAG_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/data/sterero_imu.bag"

pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, BAG_FILE, repeat_playback=False)

print("\n🚀 正在检测 Bag 包，尝试提取【双目红外 + IMU】全套硬核参数...\n")
try:
    profile = pipeline.start(config)
    
    # ================= 1. 尝试获取左右红外流 (生死劫) =================
    try:
        ir1_stream = profile.get_stream(rs.stream.infrared, 1) # 左红外
        ir2_stream = profile.get_stream(rs.stream.infrared, 2) # 右红外
    except RuntimeError:
        print("❌ 灾难性失败：你的 Bag 包里根本没有录制 Infrared (红外) 数据流！")
        print("👉 解决方案：你只能用当前的包跑 RGB-D，或者重新录制一个包（录制时必须在 Viewer 里打开 Infrared 1 和 2）。")
        exit()

    ir1_intrinsics = ir1_stream.as_video_stream_profile().get_intrinsics()
    ir2_intrinsics = ir2_stream.as_video_stream_profile().get_intrinsics()

    print("✅ 恭喜！包里有纯正的双目数据！分辨率: {}x{}".format(ir1_intrinsics.width, ir1_intrinsics.height))
    print("\n请将以下内容组装成你的 Stereo-Inertial YAML:\n")
    print("=" * 60)

    # ================= 2. 左红外内参 (Camera1) =================
    print("Camera.type: \"PinHole\"\n")
    print("# LEFT Infrared Camera")
    print(f"Camera1.fx: {ir1_intrinsics.fx:.5f}")
    print(f"Camera1.fy: {ir1_intrinsics.fy:.5f}")
    print(f"Camera1.cx: {ir1_intrinsics.ppx:.5f}")
    print(f"Camera1.cy: {ir1_intrinsics.ppy:.5f}")
    print(f"Camera1.k1: {ir1_intrinsics.coeffs[0]:.5f}")
    print(f"Camera1.k2: {ir1_intrinsics.coeffs[1]:.5f}")
    print(f"Camera1.p1: {ir1_intrinsics.coeffs[2]:.5f}")
    print(f"Camera1.p2: {ir1_intrinsics.coeffs[3]:.5f}\n")

    # ================= 3. 右红外内参 (Camera2) =================
    print("# RIGHT Infrared Camera")
    print(f"Camera2.fx: {ir2_intrinsics.fx:.5f}")
    print(f"Camera2.fy: {ir2_intrinsics.fy:.5f}")
    print(f"Camera2.cx: {ir2_intrinsics.ppx:.5f}")
    print(f"Camera2.cy: {ir2_intrinsics.ppy:.5f}")
    print(f"Camera2.k1: {ir2_intrinsics.coeffs[0]:.5f}")
    print(f"Camera2.k2: {ir2_intrinsics.coeffs[1]:.5f}")
    print(f"Camera2.p1: {ir2_intrinsics.coeffs[2]:.5f}")
    print(f"Camera2.p2: {ir2_intrinsics.coeffs[3]:.5f}\n")

    # 全局分辨率
    print(f"Camera.width: {ir1_intrinsics.width}")
    print(f"Camera.height: {ir1_intrinsics.height}")
    print("Camera.fps: 30\n")

    # ================= 4. 双目外参 (T_c1_c2: 从右眼到左眼的变换) =================
    ext_ir2_to_ir1 = ir2_stream.get_extrinsics_to(ir1_stream)
    R_lr = np.array(ext_ir2_to_ir1.rotation).reshape(3,3)
    t_lr = np.array(ext_ir2_to_ir1.translation)
    
    print("# Transformation from Camera 2 (Right) to Camera 1 (Left)")
    print("Stereo.T_c1_c2: !!opencv-matrix")
    print("   rows: 4\n   cols: 4\n   dt: f")
    print(f"   data: [{R_lr[0,0]:.6f}, {R_lr[0,1]:.6f}, {R_lr[0,2]:.6f}, {t_lr[0]:.6f},")
    print(f"          {R_lr[1,0]:.6f}, {R_lr[1,1]:.6f}, {R_lr[1,2]:.6f}, {t_lr[1]:.6f},")
    print(f"          {R_lr[2,0]:.6f}, {R_lr[2,1]:.6f}, {R_lr[2,2]:.6f}, {t_lr[2]:.6f},")
    print("          0.000000, 0.000000, 0.000000, 1.000000]\n")

    # ================= 5. IMU 外参 (Tbc: 从 IMU 到 左眼) =================
    gyro_stream = profile.get_stream(rs.stream.gyro)
    ext_imu_to_ir1 = gyro_stream.get_extrinsics_to(ir1_stream)
    R_bc = np.array(ext_imu_to_ir1.rotation).reshape(3,3)
    t_bc = np.array(ext_imu_to_ir1.translation)

    print("# Transformation from body-frame (IMU) to Camera 1 (Left)")
    print("Tbc: !!opencv-matrix")
    print("   rows: 4\n   cols: 4\n   dt: f")
    print(f"   data: [{R_bc[0,0]:.6f}, {R_bc[0,1]:.6f}, {R_bc[0,2]:.6f}, {t_bc[0]:.6f},")
    print(f"          {R_bc[1,0]:.6f}, {R_bc[1,1]:.6f}, {R_bc[1,2]:.6f}, {t_bc[1]:.6f},")
    print(f"          {R_bc[2,0]:.6f}, {R_bc[2,1]:.6f}, {R_bc[2,2]:.6f}, {t_bc[2]:.6f},")
    print("          0.000000, 0.000000, 0.000000, 1.000000]\n")

    # ================= 6. IMU 内参 (保持不变) =================
    print("# IMU Parameters")
    print("IMU.NoiseGyro: 0.002\nIMU.NoiseAccel: 0.02")
    print("IMU.GyroWalk: 0.00002\nIMU.AccelWalk: 0.0002")
    print("IMU.Frequency: 200.0")

    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 程序异常: {e}")

finally:
    pipeline.stop()