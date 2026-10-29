#!/usr/bin/env python3
"""Smart D456/GPS recorder with live ORB-SLAM3 or recording-only fallback."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

from record_d456_gps_synced import (
    GPSReader,
    fit_device_clock,
    resolve_gps_port,
    write_gps_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="智能选择实时建图/纯录制模式，同步保存 D456、IMU 和 GPS"
    )
    parser.add_argument(
        "--mode", choices=("auto", "live", "record"), default="auto",
        help="auto 优先实时 ORB-SLAM3，不可用时才回退到纯录制",
    )
    parser.add_argument("--name", help="实验名；默认 live_YYYYmmdd_HHMMSS")
    parser.add_argument("--out", type=Path, default=Path("recordings"), help="输出根目录")
    parser.add_argument("--duration", type=float, default=0.0, help="录制秒数；0 表示 Ctrl+C 停止")
    parser.add_argument(
        "--viewer", choices=("auto", "on", "off"), default="auto",
        help="auto 有图形会话时显示 Pangolin（默认）",
    )
    parser.add_argument(
        "--preview", choices=("auto", "on", "off"), default="auto",
        help="Viewer 关闭时的独立 D456 图像窗口",
    )
    parser.add_argument(
        "--gps-viewer", choices=("auto", "on", "off"), default="auto",
        help="独立 GPS 状态和轨迹窗口；auto 有图形会话和 GPS 时启用",
    )
    parser.add_argument("--no-viewer", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--gps-port", default="auto",
        help="GPS NMEA 串口；auto 自动发现，none 禁用（默认: auto）",
    )
    parser.add_argument("--gps-baud", type=int, default=115200)
    parser.add_argument("--gps-max-dt-ms", type=float, default=600.0)
    parser.add_argument(
        "--allow-no-gps", action="store_true",
        help="GPS 不存在或无定位时仍允许实验返回成功",
    )
    parser.add_argument(
        "--build", choices=("auto", "always", "never"), default="auto",
        help="实时程序的构建策略（默认: auto）",
    )
    return parser.parse_args()


def find_executable(project_root: Path, source: Path) -> Path | None:
    candidates = [
        project_root / "Examples/Monocular-Inertial/mono_inertial_realsense_D456_live",
        project_root / "Examples/Monocular-Inertial/mono_inertial_realsense_D456",
    ]
    return next(
        (
            path for path in candidates
            if path.is_file() and os.access(path, os.X_OK)
            and path.stat().st_mtime_ns >= source.stat().st_mtime_ns
        ),
        None,
    )


def run_record_only(args: argparse.Namespace, project_root: Path, gps_port: str | None) -> int:
    recorder = Path(__file__).with_name("record_d456_gps_synced.py")
    command = [
        sys.executable, str(recorder), "--out", str(args.out),
        "--gps-port", gps_port or "none", "--gps-baud", str(args.gps_baud),
        "--gps-max-dt-ms", str(args.gps_max_dt_ms),
    ]
    if args.name:
        command += ["--name", args.name]
    if args.duration:
        command += ["--duration", str(args.duration)]
    if args.allow_no_gps or gps_port is None:
        command.append("--allow-no-gps")
    print("模式：record（同步录制，不启动实时地图）", flush=True)
    return subprocess.call(command, cwd=project_root)


def finish_process(
    process: subprocess.Popen, *, send_interrupt: bool, reason: str
) -> int:
    """Escalate shutdown without abandoning buffered experiment data."""
    if process.poll() is not None:
        return process.returncode
    if send_interrupt:
        print(f"正在安全结束（{reason}）：停止传感器并刷新数据…", flush=True)
        process.send_signal(signal.SIGINT)
    try:
        return process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        print("警告：安全结束等待超过 45 秒，发送 SIGTERM。", file=sys.stderr)
        process.terminate()
    try:
        return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("警告：进程仍未退出，最后发送 SIGKILL；输出将保留 .incomplete。", file=sys.stderr)
        process.kill()
        return process.wait(timeout=5)


def wait_with_gps_viewer(process: subprocess.Popen, gps: GPSReader) -> int:
    """Show GPS status/track only; ORB-SLAM3 owns all SLAM visualization."""
    try:
        import tkinter as tk
    except ImportError:
        print("警告：缺少 tkinter，GPS 可视化已关闭", file=sys.stderr)
        return process.wait()

    try:
        root = tk.Tk()
    except tk.TclError as error:
        print(f"警告：无法打开 GPS 窗口：{error}", file=sys.stderr)
        return process.wait()
    root.title("GPS Live")
    root.geometry("700x560")
    root.configure(bg="#10151c")
    status = tk.StringVar(value="Waiting for NMEA...")
    tk.Label(
        root, textvariable=status, anchor="w", justify="left",
        bg="#10151c", fg="#e6edf3", font=("Sans", 12), padx=12, pady=8,
    ).pack(fill="x")
    canvas = tk.Canvas(root, width=680, height=480, bg="#071018", highlightthickness=0)
    canvas.pack(padx=10, pady=(0, 10), fill="both", expand=True)
    close_requested = False

    def close() -> None:
        nonlocal close_requested
        close_requested = True

    root.protocol("WM_DELETE_WINDOW", close)
    root.bind("<Escape>", lambda _event: close())
    root.bind("q", lambda _event: close())

    try:
        while process.poll() is None and not close_requested:
            all_fixes = list(gps.fixes)
            fixes = all_fixes[-5000:]
            width = max(canvas.winfo_width(), 100)
            height = max(canvas.winfo_height(), 100)
            canvas.delete("all")
            for fraction in (0.25, 0.5, 0.75):
                canvas.create_line(width * fraction, 0, width * fraction, height,
                                   fill="#1d2a36")
                canvas.create_line(0, height * fraction, width, height * fraction,
                                   fill="#1d2a36")
            if fixes:
                lat0 = fixes[0].latitude
                lon0 = fixes[0].longitude
                cos_lat = max(0.01, math.cos(math.radians(lat0)))
                points_m = [
                    ((fix.longitude - lon0) * 111320.0 * cos_lat,
                     (fix.latitude - lat0) * 110540.0)
                    for fix in fixes
                ]
                xs = [point[0] for point in points_m]
                ys = [point[1] for point in points_m]
                span = max(max(xs) - min(xs), max(ys) - min(ys), 5.0)
                scale = 0.8 * min(width, height) / span
                cx = 0.5 * (min(xs) + max(xs))
                cy = 0.5 * (min(ys) + max(ys))
                pixels = [
                    (width * 0.5 + (x - cx) * scale,
                     height * 0.5 - (y - cy) * scale)
                    for x, y in points_m
                ]
                if len(pixels) >= 2:
                    canvas.create_line(*[coordinate for point in pixels for coordinate in point],
                                       fill="#26c6da", width=3, smooth=True)
                x, y = pixels[-1]
                canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#ff5252", outline="")
                latest = fixes[-1]
                status.set(
                    f"GPS FIX  fixes={len(all_fixes)}  satellites={latest.satellites}  "
                    f"quality={latest.fix_quality}  HDOP={latest.hdop:g}\n"
                    f"lat={latest.latitude:.8f}  lon={latest.longitude:.8f}  "
                    f"alt={latest.altitude:.2f} m"
                )
            else:
                canvas.create_text(
                    width / 2, height / 2, text="WAITING FOR GPS FIX",
                    fill="#ffb74d", font=("Sans", 20, "bold"),
                )
                status.set(
                    f"NMEA={gps.nmea_sentences}  GGA={gps.gga_sentences}  "
                    f"no-fix={gps.no_fix_sentences}  bad={gps.bad_sentences}\n"
                    "Receiver is online; move antenna to open sky if no-fix keeps increasing."
                )
            root.update_idletasks()
            root.update()
            time.sleep(0.08)
    except tk.TclError:
        close()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
    return finish_process(
        process, send_interrupt=close_requested, reason="GPS 窗口关闭或快捷键"
    )


def main() -> int:
    args = parse_args()
    if args.duration < 0:
        raise SystemExit("--duration 不能小于 0")

    project_root = Path(__file__).resolve().parents[3]
    source = project_root / "Examples/Monocular-Inertial/mono_inertial_realsense_D456.cc"
    gps_port = resolve_gps_port(args.gps_port)
    if gps_port is None and not args.allow_no_gps:
        print("错误：未发现 GPS 串口。请指定 --gps-port，或显式加 --allow-no-gps。", file=sys.stderr)
        return 2

    if args.mode == "record":
        return run_record_only(args, project_root, gps_port)

    executable = find_executable(project_root, source)
    should_build = args.build == "always" or (args.build == "auto" and executable is None)
    if should_build:
        builder = Path(__file__).with_name("build_d456_orbslam_live.sh")
        print("实时程序缺失或已过期，正在自动构建…", flush=True)
        result = subprocess.run([str(builder)], cwd=project_root, check=False)
        executable = find_executable(project_root, source) if result.returncode == 0 else None
    if executable is None:
        if args.mode == "auto":
            print("警告：实时 ORB-SLAM3 不可用，auto 回退到可完整保存数据的 record 模式。", file=sys.stderr)
            return run_record_only(args, project_root, gps_port)
        print("错误：实时程序不存在或构建失败。", file=sys.stderr)
        return 2

    name = args.name or dt.datetime.now().strftime("live_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        print("错误：--name 只能包含字母、数字、点、下划线和连字符。", file=sys.stderr)
        return 2
    session = (args.out / name).resolve()
    vocabulary = project_root / "Vocabulary/ORBvoc.txt"

    viewer_requested = (
        False if args.no_viewer or args.viewer == "off" else
        True if args.viewer == "on" else
        bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    )
    display_available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    preview_requested = (
        False if args.preview == "off" else
        True if args.preview == "on" else
        display_available
    )
    has_render_node = Path("/dev/dri/renderD128").exists()
    viewer_enabled = viewer_requested
    # ORB-SLAM3's own Viewer already contains its Current Frame view. Avoid a
    # second OpenCV window in the C++ process whenever Pangolin is enabled.
    preview_enabled = preview_requested and not viewer_enabled
    gps_viewer_enabled = bool(gps_port) and (
        False if args.gps_viewer == "off" else
        True if args.gps_viewer == "on" else
        display_available
    )
    command = [
        str(executable),
        str(vocabulary),
        str(session),
        str(args.duration),
        "1" if viewer_enabled else "0",
        "1" if preview_enabled else "0",
    ]
    environment = os.environ.copy()
    ros_lib = "/opt/ros/lyrical/lib/x86_64-linux-gnu"
    environment["LD_LIBRARY_PATH"] = ros_lib + ":" + environment.get("LD_LIBRARY_PATH", "")
    if viewer_enabled and not has_render_node:
        # Keep the native Pangolin Viewer, but force Mesa software rendering on
        # machines without a DRM render node.
        environment.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        environment.setdefault("GALLIUM_DRIVER", "llvmpipe")
        environment.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
        print("OpenGL：未发现 /dev/dri/renderD128，ORB-SLAM3 Viewer 使用 llvmpipe 软件渲染", flush=True)

    print("模式：live（ORB-SLAM3 + D456 + GPS 同源录制）", flush=True)
    print(f"Viewer：{'ORB-SLAM3 Pangolin' if viewer_enabled else 'off'} ({args.viewer})", flush=True)
    preview_strategy = "native-camera" if preview_enabled else "ORB-SLAM3 Current Frame" if viewer_enabled else "off"
    print(f"图像预览：{preview_strategy} ({args.preview})", flush=True)
    print(f"GPS 可视化：{'on' if gps_viewer_enabled else 'off'} ({args.gps_viewer})", flush=True)
    print(f"输出目录：{session}", flush=True)
    print("启动后先静置约 2 秒，再进行平移和多轴旋转；Ctrl+C 安全结束。", flush=True)

    gps_stop = threading.Event()
    gps = None
    if gps_port:
        gps = GPSReader(gps_port, args.gps_baud, gps_stop)
        try:
            gps.connect()
        except RuntimeError as error:
            print(f"错误：{error}", file=sys.stderr)
            return 2
        gps.start()
        print(f"GPS：{gps_port} @ {args.gps_baud}")

    # Put the recorder in its own process group. Ctrl+C reaches this supervisor
    # once, then is forwarded exactly once through the staged shutdown below.
    process = subprocess.Popen(
        command, cwd=project_root, env=environment, start_new_session=True
    )
    try:
        if gps_viewer_enabled and gps is not None:
            return_code = wait_with_gps_viewer(process, gps)
        else:
            return_code = process.wait()
    except KeyboardInterrupt:
        return_code = finish_process(
            process, send_interrupt=True, reason="收到 Ctrl+C"
        )
    finally:
        if gps is not None:
            gps.request_stop()
            gps.join(timeout=3.0)
        else:
            gps_stop.set()

    gps_ready = False
    gps_metadata = {
        "enabled": gps is not None,
        "required": not args.allow_no_gps,
        "port": gps_port,
        "fix_count": 0,
        "aligned_image_count": 0,
        "alignment_coverage": 0.0,
        "sync_ready": False,
    }
    if gps is not None:
        gps_metadata.update({
            "received_lines": gps.received_lines,
            "nmea_sentences": gps.nmea_sentences,
            "gga_sentences": gps.gga_sentences,
            "no_fix_sentences": gps.no_fix_sentences,
            "bad_sentences": gps.bad_sentences,
            "reader_error": gps.error,
        })
    if gps is not None and session.is_dir():
        clock_path = session / "device_host_clock.csv"
        camera_path = session / "mav0/cam0/data.csv"
        if clock_path.is_file() and camera_path.is_file():
            with clock_path.open(newline="", encoding="utf-8") as handle:
                clock_rows = [
                    (int(row["device_ns"]), int(row["host_utc_ns"]))
                    for row in csv.DictReader(handle)
                ]
            with camera_path.open(newline="", encoding="utf-8") as handle:
                images = [
                    (int(row[0]), row[1]) for row in csv.reader(handle)
                    if row and not row[0].startswith("#")
                ]
            if clock_rows:
                clock_fit = fit_device_clock(clock_rows)
                valid, errors = write_gps_outputs(
                    session, images, gps.fixes, clock_fit, args.gps_max_dt_ms
                )
                coverage = valid / len(images) if images else 0.0
                gps_ready = bool(gps.fixes and coverage >= 0.9 and not gps.error)
                gps_metadata.update({
                    "fix_count": len(gps.fixes),
                    "aligned_image_count": valid,
                    "image_count": len(images),
                    "alignment_coverage": coverage,
                    "sync_ready": gps_ready,
                    "clock_fit": clock_fit.__dict__,
                    "nearest_dt_max_ms": max(errors) if errors else None,
                    "reader_error": gps.error,
                    "bad_sentences": gps.bad_sentences,
                })
                print(f"GPS 对齐：{valid}/{len(images)} 帧 ({coverage:.1%})")
        if not gps.fixes and gps.no_fix_sentences:
            print(
                f"GPS 状态：串口/NMEA 正常，但 {gps.no_fix_sentences} 条 GGA 均无定位；"
                "请到开阔室外等待卫星锁定。",
                file=sys.stderr,
            )

    if session.is_dir():
        live_metadata = {}
        live_path = session / "live_metadata.json"
        if live_path.is_file():
            live_metadata = json.loads(live_path.read_text(encoding="utf-8"))
        experiment_ready = bool(
            return_code == 0 and live_metadata.get("slam_initialized")
            and (gps_ready or args.allow_no_gps)
        )
        metadata = {
            "format_version": 2,
            "mode": "live",
            "viewer_enabled": viewer_enabled,
            "viewer_requested": viewer_requested,
            "preview_enabled": preview_requested,
            "preview_strategy": preview_strategy,
            "gps_viewer_enabled": gps_viewer_enabled,
            "process_return_code": return_code,
            "orbslam": live_metadata,
            "gps": gps_metadata,
            "experiment_ready": experiment_ready,
            "synchronization_note": (
                "Camera and IMU share the D456 device clock. GPS uses NMEA UTC; "
                "device_host_clock.csv provides an affine software clock fit. "
                "Without PPS wiring this is measured software synchronization, not hardware sync."
            ),
        }
        (session / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"实验数据检查：{'通过' if experiment_ready else '未通过'}")
        if not experiment_ready and return_code == 0:
            return_code = 1

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
