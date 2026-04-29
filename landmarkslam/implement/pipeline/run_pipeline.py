#!/usr/bin/env python3
"""
🌀 ORB-SLAM3 + Looming 一键流水线
=====================================
从 .bag 原始数据到最终评估结果，全自动运行，自动跳过已完成阶段。

所有路径和参数集中配置在 pipeline_config.yaml。

阶段:
  1. Bag 提取         — 从 RealSense .bag 提取 RGB/Depth → lines_all/
  2. ORB-SLAM3 建图    — 运行 rgbd_tum → AllFrames_trajectory.txt
  3. 序列分割           — 按帧范围割出 lines1/ + lines2/（含 trajectory.txt）
  4. ROI 标注           — 运行 process_sequence_with_cached_rois() → saved_rois.txt
  5. 主流水线           — test.py (batch/interactive/full)

用法:
  python run_pipeline.py
  python run_pipeline.py --config my_config.yaml
  python run_pipeline.py --skip 1 2 3
"""

import os
import sys
import json
import yaml
import shutil
import subprocess
import glob
import datetime
import argparse
import importlib.util

# ==============================================================================
# 🌟 路径设置
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # landmarkslam/implement 上层
ORB_ROOT = os.path.join(PROJECT_ROOT, "..")  # ORB_SLAM3-master

def resolve_path(path, allow_relative_to_project=True):
    """将路径解释为：已存在路径/绝对路径/相对于 project_root 的路径"""
    if path is None:
        return None
    path = str(path)
    if os.path.exists(path):
        return os.path.abspath(path)
    if path.startswith("/"):
        return path
    if allow_relative_to_project:
        return os.path.abspath(os.path.join(ORB_ROOT, path))
    return os.path.abspath(path)


# ==============================================================================
# 🚀 流水线引擎
# ==============================================================================
class PipelineRunner:
    STAGE_NAMES = {
        1: "🎬  Stage 1: Bag 提取",
        2: "🏗️  Stage 2: ORB-SLAM3 建图",
        3: "✂️   Stage 3: 序列分割",
        4: "🎯  Stage 4: ROI 标注",
        5: "📊  Stage 5: 主流水线",
    }

    def __init__(self, config_path=None):
        # 加载配置
        if config_path is None:
            config_path = os.path.join(SCRIPT_DIR, "pipeline_config.yaml")
        self.config_path = str(config_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # 解析根路径
        self.project_root = resolve_path(self.cfg.get("project_root", ORB_ROOT))
        if not os.path.exists(self.project_root):
            print(f"❌ project_root 不存在: {self.project_root}")
            sys.exit(1)

        # 解析所有路径
        self._resolve_paths()
        # test.py 所在目录
        self.main_dir = os.path.join(self.project_root, "landmarkslam", "implement", "main")

        # 跳过阶段
        self.skip_stages = set(self.cfg.get("pipeline", {}).get("skip_stages", []))
        # 模式
        self.mode = self.cfg.get("pipeline", {}).get("mode", "batch")

        print(f"\n{'=' * 60}")
        print(f"  🌀 Looming 一键流水线")
        print(f"  配置: {self.config_path}")
        print(f"  模式: {self.mode}")
        print(f"  跳过: {list(self.skip_stages) if self.skip_stages else '无'}")
        print(f"{'=' * 60}\n")

    def _resolve_paths(self):
        """将配置中的相对路径转为绝对路径"""
        o = self.cfg.get("orbslam", {})
        self.orbslam_bin = resolve_path(o.get("binary"))
        self.orbslam_vocab = resolve_path(o.get("vocabulary"))
        self.orbslam_template = resolve_path(o.get("config_template"))

        d = self.cfg.get("data", {})
        self.bag_file = resolve_path(d.get("bag_file"))
        self.data_root = resolve_path(d.get("output_root"))

        # 自动推断的子目录
        self.lines_all_dir = os.path.join(self.data_root, "lines_all")
        self.lines1_dir = os.path.join(self.data_root, "lines1")
        self.lines2_dir = os.path.join(self.data_root, "lines2")

    def _p(self, path):
        """返回相对 project_root 的路径（用于打印）"""
        try:
            return os.path.relpath(path, self.project_root)
        except ValueError:
            return path

    # ---- 检测函数 ----

    def stage_done(self, stage_id):
        """检测某阶段是否已完成"""
        if stage_id == 1:
            rgb_dir = os.path.join(self.lines_all_dir, "rgb")
            depth_dir = os.path.join(self.lines_all_dir, "depth")
            has_rgb = os.path.isdir(rgb_dir) and len(os.listdir(rgb_dir)) > 10
            has_depth = os.path.isdir(depth_dir) and len(os.listdir(depth_dir)) > 10
            return has_rgb and has_depth

        if stage_id == 2:
            return os.path.exists(os.path.join(self.lines_all_dir, "AllFrames_trajectory.txt"))

        if stage_id == 3:
            has_1 = os.path.isdir(os.path.join(self.lines1_dir, "rgb")) and len(os.listdir(os.path.join(self.lines1_dir, "rgb"))) > 0
            has_2 = os.path.isdir(os.path.join(self.lines2_dir, "rgb")) and len(os.listdir(os.path.join(self.lines2_dir, "rgb"))) > 0
            return has_1 and has_2

        if stage_id == 4:
            return os.path.exists(os.path.join(self.lines1_dir, "rgb", "saved_rois.txt"))

        if stage_id == 5:
            return False  # 始终运行（增量生成）

        return False

    def run(self):
        """顺序运行所有阶段"""
        for sid in sorted(self.STAGE_NAMES.keys()):
            if sid in self.skip_stages:
                print(f"⏭️  {self.STAGE_NAMES[sid]} — 已跳过（skip_stages 配置）\n")
                continue

            if self.stage_done(sid):
                print(f"✅ {self.STAGE_NAMES[sid]} — 已有输出，跳过\n")
                continue

            print(f"\n{'─' * 60}")
            print(f"  {self.STAGE_NAMES[sid]}")
            print(f"{'─' * 60}")
            self._run_stage(sid)

        self._print_summary()

    def _run_stage(self, sid):
        if sid == 1:
            self._stage1_extract_bag()
        elif sid == 2:
            self._stage2_orbslam()
        elif sid == 3:
            self._stage3_split()
        elif sid == 4:
            self._stage4_roi()
        elif sid == 5:
            self._stage5_main()
        else:
            print(f"❌ 未知阶段: {sid}")

    # ======================================================================
    # Stage 1: Bag 提取
    # ======================================================================
    def _stage1_extract_bag(self):
        """从 .bag 文件提取 RGB/Depth/IMU 到 lines_all/"""
        if not os.path.exists(self.bag_file):
            print(f"❌ Bag 文件不存在: {self.bag_file}")
            return

        print(f"  📦 Bag 文件: {self.bag_file}")
        print(f"  📁 输出目录: {self.lines_all_dir}")

        # 动态导入 pyrealsense2（仅在此阶段需要）
        try:
            import pyrealsense2 as rs
        except ImportError:
            print("❌ 需要 pyrealsense2 库。请安装: pip install pyrealsense2")
            return

        rgb_dir = os.path.join(self.lines_all_dir, "rgb")
        depth_dir = os.path.join(self.lines_all_dir, "depth")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        intrinsics_path = os.path.join(self.lines_all_dir, "camera_intrinsics.json")
        imu_path = os.path.join(self.lines_all_dir, "imu.txt")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(self.bag_file, repeat_playback=False)
        profile = pipeline.start(config)
        device = profile.get_device()
        playback = device.as_playback()
        playback.set_real_time(False)

        # 获取内参
        try:
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_stream.get_intrinsics()
            camera_params = {
                "fx": intr.fx, "fy": intr.fy,
                "cx": intr.ppx, "cy": intr.ppy,
                "width": intr.width, "height": intr.height,
            }
            print(f"  ✅ 内参: fx={intr.fx:.3f}, fy={intr.fy:.3f}, cx={intr.ppx:.3f}, cy={intr.ppy:.3f}")
        except Exception:
            camera_params = {"fx": 426.372, "fy": 425.671, "cx": 435.525, "cy": 244.974, "width": 848, "height": 480}
            print(f"  ⚠️  使用默认内参: {camera_params}")

        with open(intrinsics_path, "w") as f:
            json.dump(camera_params, f, indent=4)

        # 对齐深度到彩色
        align = rs.align(rs.stream.color)
        depth_scale = 0.001

        imu_f = open(imu_path, "w")
        imu_f.write("# timestamp_ns w_x w_y w_z a_x a_y a_z\n")

        last_gyro = [0.0, 0.0, 0.0]
        last_accel = [0.0, 0.0, 0.0]
        frame_idx = 0
        imu_count = 0
        start_t = datetime.datetime.now()

        print("  🚀 开始提取...")
        try:
            while True:
                success, frames = pipeline.try_wait_for_frames(2000)
                if not success:
                    break

                # IMU
                gyro_frame = frames.first_or_default(rs.stream.gyro)
                accel_frame = frames.first_or_default(rs.stream.accel)
                if gyro_frame or accel_frame:
                    if gyro_frame:
                        gd = gyro_frame.as_motion_frame().get_motion_data()
                        last_gyro = [gd.x, gd.y, gd.z]
                        imu_ts = int(gyro_frame.get_timestamp() * 1e6)
                    if accel_frame:
                        ad = accel_frame.as_motion_frame().get_motion_data()
                        last_accel = [ad.x, ad.y, ad.z]
                        if not gyro_frame:
                            imu_ts = int(accel_frame.get_timestamp() * 1e6)
                    imu_f.write(f"{imu_ts} {last_gyro[0]:.6f} {last_gyro[1]:.6f} {last_gyro[2]:.6f} "
                                f"{last_accel[0]:.6f} {last_accel[1]:.6f} {last_accel[2]:.6f}\n")
                    imu_count += 1

                # 图像
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame:
                    continue

                ts_ns = int(color_frame.get_timestamp() * 1e6)
                color_img = np.asanyarray(color_frame.get_data())
                cv2.imwrite(os.path.join(rgb_dir, f"{ts_ns}.png"), color_img)

                if depth_frame:
                    depth_img = np.asanyarray(depth_frame.get_data())
                    depth_mm = (depth_img * depth_scale * 1000).astype(np.uint16)
                    cv2.imwrite(os.path.join(depth_dir, f"{ts_ns}.png"), depth_mm)

                frame_idx += 1
                if frame_idx % 50 == 0:
                    elapsed = (datetime.datetime.now() - start_t).total_seconds()
                    print(f"    📦 已提取 {frame_idx} 帧 | {elapsed:.0f}s")
        except Exception as e:
            print(f"  ⚠️  提取结束: {e}")
        finally:
            pipeline.stop()
            imu_f.close()

        # 生成 associations.txt
        self._generate_associations(self.lines_all_dir)
        elapsed = (datetime.datetime.now() - start_t).total_seconds()
        print(f"  ✅ 提取完成: {frame_idx} 帧, {imu_count} IMU 行, 耗时 {elapsed:.1f}s")

    @staticmethod
    def _generate_associations(data_dir):
        """生成 TUM 格式 associations.txt"""
        files = sorted(glob.glob(os.path.join(data_dir, "rgb", "*.png")))
        assoc_path = os.path.join(data_dir, "associations.txt")
        with open(assoc_path, "w") as f:
            for fp in files:
                fname = os.path.basename(fp)
                ns_str = fname.split(".")[0]
                sec = float(ns_str) / 1e9
                f.write(f"{sec:.6f} rgb/{fname} {sec:.6f} depth/{fname}\n")
        print(f"  ✅ associations.txt: {len(files)} 帧")

    # ======================================================================
    # Stage 2: ORB-SLAM3 建图
    # ======================================================================
    def _stage2_orbslam(self):
        """运行 rgbd_tum 生成 AllFrames_trajectory.txt"""
        if not os.path.exists(self.orbslam_bin):
            print(f"❌ ORB-SLAM3 二进制不存在: {self.orbslam_bin}")
            print(f"   请先编译 ORB-SLAM3 或在配置中指定正确路径")
            return
        if not os.path.exists(self.orbslam_vocab):
            print(f"❌ ORB 词典不存在: {self.orbslam_vocab}")
            return

        assoc_path = os.path.join(self.lines_all_dir, "associations.txt")
        if not os.path.exists(assoc_path):
            self._generate_associations(self.lines_all_dir)

        # 生成 ORB-SLAM3 相机 YAML 配置
        orbslam_config = self._generate_orbslam_yaml()
        print(f"  📝 ORB-SLAM3 配置: {orbslam_config}")

        # 运行 rgbd_tum
        cmd = [
            self.orbslam_bin,
            self.orbslam_vocab,
            orbslam_config,
            self.lines_all_dir,
            assoc_path,
            os.path.join(self.lines_all_dir, "out"),
        ]
        print(f"  🏗️  运行 ORB-SLAM3: {' '.join(os.path.basename(c) if c.startswith('/') else c for c in cmd[:3])} ...")
        sys.stdout.flush()

        # ORB-SLAM3 输出文件默认写到 CWD，改为写目标目录
        saved_cwd = os.getcwd()
        os.chdir(self.lines_all_dir)
        try:
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                print(f"  ⚠️  ORB-SLAM3 返回非零状态码: {result.returncode}")
        except Exception as e:
            print(f"  ❌ ORB-SLAM3 执行失败: {e}")
        finally:
            os.chdir(saved_cwd)

        # 移动生成的轨迹文件到 lines_all/
        for suffix in ["out", "CameraTrajectory"]:
            src = os.path.join(self.lines_all_dir, f"AllFrames_{suffix}.txt")
            if os.path.exists(src):
                dst = os.path.join(self.lines_all_dir, "AllFrames_trajectory.txt")
                shutil.move(src, dst)
                print(f"  ✅ 轨迹文件已保存: {self._p(dst)}")
                return

        # 尝试 AlternativeFrames 输出
        for f in sorted(glob.glob(os.path.join(self.lines_all_dir, "AllFrames_*.txt"))):
            dst = os.path.join(self.lines_all_dir, "AllFrames_trajectory.txt")
            shutil.move(f, dst)
            print(f"  ✅ 轨迹文件已保存: {self._p(dst)}")
            return

        print("  ⚠️  ORB-SLAM3 未生成轨迹文件，请检查输出。")

    def _generate_orbslam_yaml(self):
        """基于配置生成 ORB-SLAM3 相机 YAML 配置文件"""
        cam = self.cfg.get("camera", {})
        orb = self.cfg.get("orb", {})

        # 加载模板
        template_path = self.orbslam_template
        if not os.path.exists(template_path):
            print(f"  ⚠️  模板文件不存在: {template_path}，使用默认参数")
            yaml_lines = [
                "%YAML:1.0",
                "File.version: '1.0'",
                "Camera.type: 'PinHole'",
            ]
        else:
            with open(template_path, "r") as f:
                yaml_lines = f.read().splitlines()

        # 替换相机内参
        replacements = {
            "Camera1.fx": str(cam.get("fx", 426.372)),
            "Camera1.fy": str(cam.get("fy", 425.671)),
            "Camera1.cx": str(cam.get("cx", 435.525)),
            "Camera1.cy": str(cam.get("cy", 244.974)),
            "Camera1.k1": str(cam.get("k1", 0.0)),
            "Camera1.k2": str(cam.get("k2", 0.0)),
            "Camera1.p1": str(cam.get("p1", 0.0)),
            "Camera1.p2": str(cam.get("p2", 0.0)),
            "Camera1.k3": str(cam.get("k3", 0.0)),
            "Camera.width": str(cam.get("width", 848)),
            "Camera.height": str(cam.get("height", 480)),
            "Camera.fps": str(cam.get("fps", 30)),
            "Camera.RGB": str(cam.get("rgb_order", 1)),
            "RGBD.DepthMapFactor": str(cam.get("depth_map_factor", 1000.0)),
            "Stereo.ThDepth": str(cam.get("depth_threshold", 40.0)),
            "ORBextractor.nFeatures": str(orb.get("n_features", 1000)),
            "ORBextractor.scaleFactor": str(orb.get("scale_factor", 1.2)),
            "ORBextractor.nLevels": str(orb.get("n_levels", 8)),
            "ORBextractor.iniThFAST": str(orb.get("ini_th_fast", 20)),
            "ORBextractor.minThFAST": str(orb.get("min_th_fast", 7)),
        }

        out_lines = []
        for line in yaml_lines:
            replaced = False
            for key, val in replacements.items():
                # 匹配 "key: value" 模式（带冒号）
                if line.strip().startswith(key + ":"):
                    indent = line[: len(line) - len(line.lstrip())]
                    out_lines.append(f"{indent}{key}: {val}")
                    replaced = True
                    break
            if not replaced:
                out_lines.append(line)

        output_path = os.path.join(SCRIPT_DIR, "orbslam_config.yaml")
        with open(output_path, "w") as f:
            f.write("\n".join(out_lines) + "\n")

        return output_path

    # ======================================================================
    # Stage 3: 序列分割
    # ======================================================================
    def _stage3_split(self):
        """按配置帧范围自动分割 lines_all → lines1/ + lines2/"""
        split = self.cfg.get("split", {})
        s1s = split.get("seq1_start")
        s1e = split.get("seq1_end")
        s2s = split.get("seq2_start")
        s2e = split.get("seq2_end")

        has_gui = (s1s is None or s1e is None or s2s is None or s2e is None
                   or s1s == -1 or s1e == -1 or s2s == -1 or s2e == -1)

        if has_gui:
            print("  🖱️  启动 GUI 序列分割界面...")
            self._split_with_gui()
        else:
            print(f"  自动分割: seq1=[{s1s}:{s1e}], seq2=[{s2s}:{s2e}]")
            self._auto_split("lines1", s1s, s1e)
            self._auto_split("lines2", s2s, s2e)
            print("  ✅ 序列分割完成")

    def _auto_split(self, seq_name, start_idx, end_idx):
        """自动按帧索引分割（无 GUI）"""
        rgb_dir = os.path.join(self.lines_all_dir, "rgb")
        all_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        if start_idx >= len(all_images) or end_idx >= len(all_images):
            print(f"  ⚠️  索引超出范围（最大 {len(all_images)-1}），跳过 {seq_name}")
            return

        seq_dir = os.path.join(self.data_root, seq_name)
        rgb_out = os.path.join(seq_dir, "rgb")
        depth_out = os.path.join(seq_dir, "depth")
        os.makedirs(rgb_out, exist_ok=True)
        os.makedirs(depth_out, exist_ok=True)

        selected = all_images[start_idx:end_idx + 1]
        timestamps = []
        for img_path in selected:
            fname = os.path.basename(img_path)
            ts = int(fname.split(".")[0]) / 1e9
            timestamps.append(ts)

            # 复制 RGB
            shutil.copy(img_path, os.path.join(rgb_out, fname))
            # 复制 depth
            depth_src = os.path.join(self.lines_all_dir, "depth", fname)
            if os.path.exists(depth_src):
                shutil.copy(depth_src, os.path.join(depth_out, fname))

        # 内参
        intrinsics_src = os.path.join(self.lines_all_dir, "camera_intrinsics.json")
        if os.path.exists(intrinsics_src):
            shutil.copy(intrinsics_src, os.path.join(seq_dir, "camera_intrinsics.json"))

        # associations.txt
        assoc_path = os.path.join(seq_dir, "associations.txt")
        with open(assoc_path, "w") as f:
            for ts, img_path in zip(timestamps, selected):
                fname = os.path.basename(img_path)
                f.write(f"{ts:.6f} rgb/{fname} {ts:.6f} depth/{fname}\n")

        # trajectory.txt
        traj_src = os.path.join(self.lines_all_dir, "AllFrames_trajectory.txt")
        if os.path.exists(traj_src):
            self._save_trajectory(traj_src, timestamps, os.path.join(seq_dir, "trajectory.txt"))

        print(f"    ✅ {seq_name}: {len(selected)} 帧 → {self._p(seq_dir)}")

    def _split_with_gui(self):
        """交互式 GUI 序列分割"""
        # 导入并运行交互式分割
        self._add_implement_path()
        from deepseek_cut_all import main as gui_split_main
        # 注入路径
        import deepseek_cut_all as dca
        dca.LINES_ALL_DIR = self.lines_all_dir
        dca.OUT_DIR = self.data_root
        dca.TRAJECTORY_FILE = "AllFrames_trajectory.txt"
        gui_split_main()

    @staticmethod
    def _save_trajectory(traj_path, timestamps, output_path):
        """根据时间戳列表从完整轨迹中提取子序列轨迹"""
        traj = {}
        with open(traj_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                ts = float(parts[0])
                traj[ts] = [float(x) for x in parts[1:8]]

        with open(output_path, "w") as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            for ts in timestamps:
                if not traj:
                    continue
                closest_ts = min(traj.keys(), key=lambda k: abs(k - ts))
                if abs(closest_ts - ts) < 0.01:
                    p = traj[closest_ts]
                    f.write(f"{ts:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                            f"{p[3]:.6f} {p[4]:.6f} {p[5]:.6f} {p[6]:.6f}\n")

    # ======================================================================
    # Stage 4: ROI 标注
    # ======================================================================
    def _stage4_roi(self):
        """对 lines1/rgb/ 运行 ROI 标注"""
        seq1_rgb = os.path.join(self.lines1_dir, "rgb")
        if not os.path.isdir(seq1_rgb):
            print(f"  ❌ lines1/rgb/ 不存在: {seq1_rgb}")
            return

        self._add_implement_path()
        print(f"  🖱️  启动 ROI 标注界面（序列: {self._p(seq1_rgb)}）")
        print("  ⏺️   提示: 拖动鼠标框选目标，按 SPACE/Enter 确认，按 ESC 跳过")

        from mid_FOE_Z_d.imgs_mid import process_sequence_with_cached_rois
        process_sequence_with_cached_rois(seq1_rgb)

        saved_rois = os.path.join(seq1_rgb, "saved_rois.txt")
        if os.path.exists(saved_rois):
            print(f"  ✅ ROI 已保存: {self._p(saved_rois)}")
        else:
            print(f"  ⚠️  saved_rois.txt 未生成")

    # ======================================================================
    # Stage 5: 主流水线
    # ======================================================================
    def _stage5_main(self):
        """运行 test.py（主实验脚本）"""
        test_py = os.path.join(self.main_dir, "test.py")
        if not os.path.exists(test_py):
            print(f"  ❌ test.py 不存在: {test_py}")
            return

        # 准备 config.yaml 供 test.py 使用
        self._write_test_config()

        if self.mode in ("batch", "full"):
            print(f"  📊 批量评估模式...")
            self._run_test_py(quiet=True)

        if self.mode in ("interactive", "full"):
            print(f"  🖥️  交互模式（GUI）...")
            self._run_test_py(quiet=False)

    def _write_test_config(self):
        """将 pipeline_config 中的序列路径写入 main/config.yaml"""
        cam = self.cfg.get("camera", {})
        algo = self.cfg.get("algorithm", {})

        test_config = {
            "Camera": {
                "fx": cam.get("fx", 426.372),
                "fy": cam.get("fy", 425.671),
                "cx": cam.get("cx", 435.525),
                "cy": cam.get("cy", 244.974),
            },
            "Algorithm": {
                "frame_step": algo.get("frame_step", 15),
                "cache_file": "selected_frames_cache.json",
            },
            "Sequence1": {
                "image_dir": os.path.join(self.lines1_dir, "rgb"),
                "depth_dir": os.path.join(self.lines1_dir, "depth"),
                "trajectory_path": os.path.join(self.lines1_dir, "trajectory.txt"),
                "roi_path": os.path.join(self.lines1_dir, "rgb", "saved_rois.txt"),
            },
            "Sequence2": {
                "image_dir": os.path.join(self.lines2_dir, "rgb"),
                "trajectory_path": os.path.join(self.lines2_dir, "trajectory.txt"),
            },
        }

        config_dst = os.path.join(self.main_dir, "config.yaml")
        with open(config_dst, "w", encoding="utf-8") as f:
            yaml.dump(test_config, f, default_flow_style=False, allow_unicode=True)

        print(f"  ✅ test.py 配置已同步: {self._p(config_dst)}")

    def _run_test_py(self, quiet=True):
        """子进程运行 test.py"""
        test_py = os.path.join(self.main_dir, "test.py")
        python = sys.executable

        cmd = [python, test_py]
        if quiet:
            cmd.append("-q")

        print(f"  🚀 运行: {' '.join(cmd)}")
        sys.stdout.flush()

        # 需要先添加 implement/ 到 PYTHONPATH
        env = os.environ.copy()
        impl_dir = os.path.join(self.project_root, "landmarkslam", "implement")
        pythonpath = env.get("PYTHONPATH", "")
        if impl_dir not in pythonpath:
            env["PYTHONPATH"] = f"{impl_dir}:{pythonpath}" if pythonpath else impl_dir

        result = subprocess.run(cmd, env=env, cwd=self.main_dir)
        if result.returncode != 0:
            print(f"  ⚠️  test.py 返回状态码: {result.returncode}")
        else:
            print(f"  ✅ test.py 完成")

    # ---- 辅助 ----

    def _add_implement_path(self):
        """将 landmarkslam/implement/ 加入 sys.path（用于导入工具模块）"""
        impl_dir = os.path.join(self.project_root, "landmarkslam", "implement")
        if impl_dir not in sys.path:
            sys.path.insert(0, impl_dir)

    # ---- 汇总 ----
    def _print_summary(self):
        """输出最终文件清单"""
        files = []
        # Stage 1 产物
        rgb_dir = os.path.join(self.lines_all_dir, "rgb")
        if os.path.isdir(rgb_dir):
            files.append((f"{self._p(rgb_dir)}/ (N 帧 RGB)", "🎬"))
        depth_dir = os.path.join(self.lines_all_dir, "depth")
        if os.path.isdir(depth_dir):
            files.append((f"{self._p(depth_dir)}/ (N 帧 Depth)", "🎬"))
        # Stage 2 产物
        traj_all = os.path.join(self.lines_all_dir, "AllFrames_trajectory.txt")
        if os.path.exists(traj_all):
            files.append((self._p(traj_all), "🏗️"))
        # Stage 3 产物
        for seq_name in ["lines1", "lines2"]:
            seq_dir = os.path.join(self.data_root, seq_name)
            traj_seq = os.path.join(seq_dir, "trajectory.txt")
            if os.path.exists(traj_seq):
                files.append((self._p(traj_seq), "✂️"))
        # Stage 4 产物
        rois = os.path.join(self.lines1_dir, "rgb", "saved_rois.txt")
        if os.path.exists(rois):
            files.append((self._p(rois), "🎯"))
        # Stage 5 产物
        results_json = os.path.join(self.main_dir, "batch_results.json")
        if os.path.exists(results_json):
            files.append((self._p(results_json), "📊"))
        report_md = os.path.join(self.main_dir, "batch_plots", "batch_report.md")
        if os.path.exists(report_md):
            files.append((self._p(report_md), "📊"))

        print(f"\n{'=' * 60}")
        print(f"  📦 流水线输出汇总")
        print(f"{'=' * 60}")
        if not files:
            print(f"  （流水线未产生新文件，或所有阶段均已跳过）")
        else:
            for path, icon in files:
                print(f"  {icon} {path}")
        print(f"{'=' * 60}\n")


# ==============================================================================
# 🏁 入口
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORB-SLAM3 Looming 一键流水线")
    parser.add_argument("--config", "-c", default=None,
                        help="配置文件路径（默认: pipeline_config.yaml）")
    parser.add_argument("--skip", "-s", type=int, nargs="*", default=None,
                        help="跳过指定阶段（如: --skip 1 2 3）")
    parser.add_argument("--mode", "-m", default=None,
                        choices=["batch", "interactive", "full"],
                        help="覆盖 pipeline.mode 配置")
    args = parser.parse_args()

    runner = PipelineRunner(config_path=args.config)

    # 命令行跳过覆盖
    if args.skip is not None:
        runner.skip_stages = set(args.skip)
    # 命令行模式覆盖
    if args.mode is not None:
        runner.mode = args.mode

    # 检查关键外部依赖
    if not os.path.exists(runner.orbslam_bin):
        print(f"  ⚠️  ORB-SLAM3 二进制不存在: {runner.orbslam_bin}")
        print(f"     请先编译, 或修改配置中的 orbslam.binary 路径")

    runner.run()
