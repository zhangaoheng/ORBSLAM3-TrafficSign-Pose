import pyrealsense2 as rs
import numpy as np

# ⚠️ 换成你的 bag 包绝对路径
BAG_FILE = "/home/zah/ORB_SLAM3-master/landmarkslam/data/orbslam.bag"

pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, BAG_FILE, repeat_playback=False)
profile = pipeline.start(config)

print("\n🚀 正在从 Bag 包中暴力破解全套 ORB-SLAM3 标定参数...\n")
print("=" * 60)

try:
    # ================= 1. 提取 RGB 相机内参 =================
    color_stream = profile.get_stream(rs.stream.color)
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    print("#--------------------------------------------------------------------------------------------")
    print("# 1. Camera Parameters (RGB 相机内参)")
    print("#--------------------------------------------------------------------------------------------")
    print(f"Camera.fx: {color_intrinsics.fx:.5f}")
    print(f"Camera.fy: {color_intrinsics.fy:.5f}")
    print(f"Camera.cx: {color_intrinsics.ppx:.5f}")
    print(f"Camera.cy: {color_intrinsics.ppy:.5f}")
    print("")
    print(f"Camera.k1: {color_intrinsics.coeffs[0]:.5f}")
    print(f"Camera.k2: {color_intrinsics.coeffs[1]:.5f}")
    print(f"Camera.p1: {color_intrinsics.coeffs[2]:.5f}")
    print(f"Camera.p2: {color_intrinsics.coeffs[3]:.5f}")
    print("")

    # ================= 2. 提取 IMU 外参 (Tbc) =================
    gyro_stream = profile.get_stream(rs.stream.gyro)
    accel_stream = profile.get_stream(rs.stream.accel)

    extrinsics = gyro_stream.get_extrinsics_to(color_stream)
    R = np.array(extrinsics.rotation).reshape(3,3)
    t = np.array(extrinsics.translation)

    print("#--------------------------------------------------------------------------------------------")
    print("# 2. Transformation from body-frame (imu) to camera (IMU 到 RGB 相机的绝对位姿)")
    print("#--------------------------------------------------------------------------------------------")
    print("Tbc: !!opencv-matrix")
    print("   rows: 4")
    print("   cols: 4")
    print("   dt: f")
    print(f"   data: [{R[0,0]:.6f}, {R[0,1]:.6f}, {R[0,2]:.6f}, {t[0]:.6f},")
    print(f"          {R[1,0]:.6f}, {R[1,1]:.6f}, {R[1,2]:.6f}, {t[1]:.6f},")
    print(f"          {R[2,0]:.6f}, {R[2,1]:.6f}, {R[2,2]:.6f}, {t[2]:.6f},")
    print("          0.000000, 0.000000, 0.000000, 1.000000]")
    print("")

    # ================= 3. 提取 IMU 内参 (Noise & Walk) =================
    gyro_intrinsics = gyro_stream.as_motion_stream_profile().get_motion_intrinsics()
    accel_intrinsics = accel_stream.as_motion_stream_profile().get_motion_intrinsics()

    # RealSense SDK 提供的是 variance (方差)，ORB-SLAM3 需要的是 standard deviation (标准差)
    # 所以必须做开根号处理
    gyro_noise = np.sqrt(gyro_intrinsics.noise_variances[0])
    gyro_walk = np.sqrt(gyro_intrinsics.bias_variances[0])
    accel_noise = np.sqrt(accel_intrinsics.noise_variances[0])
    accel_walk = np.sqrt(accel_intrinsics.bias_variances[0])

    print("#--------------------------------------------------------------------------------------------")
    print("# 3. IMU Parameters (IMU 白噪声与随机游走内参)")
    print("#--------------------------------------------------------------------------------------------")
    
    # 物理防抖：如果 SDK 未读出校准方差（值为0），则自动 Fallback 到博世 BMI085 的标准标称值
    if gyro_noise == 0 or accel_noise == 0:
        print("# ⚠️ 注意: 你的 Bag 包底层未挂载硬件方差，已自动切换为 BMI085 出厂推荐物理真值！")
        print("IMU.NoiseGyro: 0.002")
        print("IMU.NoiseAccel: 0.02")
        print("IMU.GyroWalk: 0.00002")
        print("IMU.AccelWalk: 0.0002")
    else:
        print(f"IMU.NoiseGyro: {gyro_noise:.6f}")
        print(f"IMU.NoiseAccel: {accel_noise:.6f}")
        print(f"IMU.GyroWalk: {gyro_walk:.6f}")
        print(f"IMU.AccelWalk: {accel_walk:.6f}")

    print("IMU.Frequency: 200.0  # RealSense IMU (Gyro) 数据流频率通常设为 200.0")
    print("=" * 60)
    print("\n✅ 大功告成！直接将上面等号框内的所有文本，覆盖/粘贴进你的 D456.yaml 中！")

except Exception as e:
    print(f"\n❌ 提取失败：{e}")
    print("请确认你的 Bag 包确实同时包含 Color、Gyro 和 Accel 流。")

pipeline.stop()