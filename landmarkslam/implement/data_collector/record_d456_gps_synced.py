#!/usr/bin/env python3
"""Record RealSense D456 monocular+IMU data and NMEA GPS on auditable clocks.

The recorder deliberately keeps two time domains:

* Camera and IMU use the D456 device timestamp.  This is the time domain used
  by ORB-SLAM3 and avoids adding USB/OS scheduling jitter between the sensors.
* GPS fixes use NMEA UTC.  Periodic (device timestamp, host UTC) observations
  estimate an affine device->UTC clock mapping after recording.  The generated
  frame_gps.csv contains one time-aligned/interpolated GPS row per image.

Without a GPS PPS signal wired to the camera, software cannot provide true
hardware synchronization.  The output therefore includes clock-fit residuals
and GPS match errors instead of claiming zero-error synchronization.

Output is directly consumable by ORB-SLAM3's mono_inertial_euroc example:

  <session>/
    mav0/cam0/data/<device_timestamp_ns>.png
    mav0/cam0/data.csv
    mav0/imu0/data.csv
    gps/data.csv
    frame_gps.csv
    camera_calibration.json
    orbslam_d456_mono_imu.yaml
    times.txt
    metadata.json
    run_orbslam.sh

Examples:
  python3 record_d456_gps_synced.py --name road_01 --gps-port /dev/ttyACM0
  python3 record_d456_gps_synced.py --name road_01 --duration 120
  python3 record_d456_gps_synced.py --self-test
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import glob
import json
import math
import os
import queue
import signal
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


NS_PER_MS = 1_000_000
NS_PER_SECOND = 1_000_000_000


def discover_gps_port() -> Optional[str]:
    """Return the most likely GNSS serial port, preferring stable by-id paths."""
    candidates: dict[str, tuple[int, str]] = {}
    for path in glob.glob("/dev/serial/by-id/*"):
        lower = path.lower()
        score = 100 + (50 if any(word in lower for word in ("gps", "gnss", "ublox")) else 0)
        candidates[path] = (score, path)

    try:
        from serial.tools import list_ports

        for port in list_ports.comports():
            text = " ".join(
                str(value or "") for value in
                (port.device, port.description, port.manufacturer, port.product, port.hwid)
            ).lower()
            score = 80 if any(word in text for word in ("gps", "gnss", "u-blox", "ublox")) else 0
            if "ttyacm" in text:
                score += 20
            elif "ttyusb" in text:
                score += 10
            candidates[port.device] = max(candidates.get(port.device, (-1, "")), (score, port.device))
    except ImportError:
        pass

    for pattern, score in (("/dev/ttyACM*", 20), ("/dev/ttyUSB*", 10)):
        for path in glob.glob(pattern):
            candidates.setdefault(path, (score, path))
    return max(candidates.values())[1] if candidates else None


def resolve_gps_port(value: Optional[str]) -> Optional[str]:
    if value is None or value.lower() in ("none", "off", "disabled"):
        return None
    if value.lower() == "auto":
        return discover_gps_port()
    return value


@dataclass(frozen=True)
class GPSFix:
    utc_ns: int
    host_rx_ns: int
    latitude: float
    longitude: float
    altitude: float
    fix_quality: int
    satellites: int
    hdop: float
    sentence: str


@dataclass(frozen=True)
class MotionSample:
    device_ns: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class ClockFit:
    slope: float
    intercept_ns: float
    residual_median_ms: float
    residual_p95_ms: float
    sample_count: int

    def to_utc_ns(self, device_ns: int) -> int:
        return int(round(self.slope * device_ns + self.intercept_ns))


def _parse_hhmmss(value: str) -> tuple[int, int, int, int]:
    if len(value) < 6:
        raise ValueError("invalid NMEA time")
    hour = int(value[0:2])
    minute = int(value[2:4])
    seconds = float(value[4:])
    second = int(seconds)
    microsecond = int(round((seconds - second) * 1_000_000))
    if microsecond == 1_000_000:
        second += 1
        microsecond = 0
    return hour, minute, second, microsecond


def _nmea_coordinate(value: str, hemisphere: str) -> float:
    if not value:
        raise ValueError("empty NMEA coordinate")
    dot = value.find(".")
    if dot < 3:
        raise ValueError("invalid NMEA coordinate")
    degrees = float(value[: dot - 2])
    minutes = float(value[dot - 2 :])
    result = degrees + minutes / 60.0
    if hemisphere in ("S", "W"):
        result = -result
    return result


def _checksum_ok(line: str) -> bool:
    if not line.startswith("$") or "*" not in line:
        return False
    body, claimed = line[1:].split("*", 1)
    checksum = 0
    for char in body:
        checksum ^= ord(char)
    try:
        return checksum == int(claimed[:2], 16)
    except ValueError:
        return False


class NMEAParser:
    """Stateful GGA/RMC parser; RMC supplies the date for GGA fixes."""

    def __init__(self) -> None:
        self.last_date: Optional[dt.date] = None

    @staticmethod
    def _closest_date(candidate: dt.datetime, host_rx_ns: int) -> dt.datetime:
        host = dt.datetime.fromtimestamp(host_rx_ns / NS_PER_SECOND, tz=dt.timezone.utc)
        choices = (candidate - dt.timedelta(days=1), candidate, candidate + dt.timedelta(days=1))
        return min(choices, key=lambda item: abs((item - host).total_seconds()))

    def parse(self, line: str, host_rx_ns: int) -> Optional[GPSFix]:
        line = line.strip()
        if not _checksum_ok(line):
            return None
        fields = line.split("*", 1)[0].split(",")
        kind = fields[0]

        if kind in ("$GPRMC", "$GNRMC"):
            if len(fields) < 10 or fields[2] != "A" or not fields[9]:
                return None
            day = int(fields[9][0:2])
            month = int(fields[9][2:4])
            short_year = int(fields[9][4:6])
            year = 1900 + short_year if short_year >= 80 else 2000 + short_year
            self.last_date = dt.date(year, month, day)
            # GGA is preferred because it carries fix quality and altitude.
            return None

        if kind not in ("$GPGGA", "$GNGGA") or len(fields) < 10:
            return None
        fix_quality = int(fields[6] or "0")
        if fix_quality <= 0 or not fields[2] or not fields[4]:
            return None

        hour, minute, second, microsecond = _parse_hhmmss(fields[1])
        host_date = dt.datetime.fromtimestamp(
            host_rx_ns / NS_PER_SECOND, tz=dt.timezone.utc
        ).date()
        date = self.last_date or host_date
        candidate = dt.datetime(
            date.year, date.month, date.day, hour, minute, second, microsecond,
            tzinfo=dt.timezone.utc,
        )
        candidate = self._closest_date(candidate, host_rx_ns)
        return GPSFix(
            utc_ns=int(candidate.timestamp() * NS_PER_SECOND),
            host_rx_ns=host_rx_ns,
            latitude=_nmea_coordinate(fields[2], fields[3]),
            longitude=_nmea_coordinate(fields[4], fields[5]),
            altitude=float(fields[9] or "nan"),
            fix_quality=fix_quality,
            satellites=int(fields[7] or "0"),
            hdop=float(fields[8] or "nan"),
            sentence=kind[1:],
        )


class GPSReader(threading.Thread):
    def __init__(self, port: str, baud: int, stop_event: threading.Event) -> None:
        super().__init__(name="gps-reader", daemon=True)
        self.port = port
        self.baud = baud
        self.stop_event = stop_event
        self.fixes: list[GPSFix] = []
        self.bad_sentences = 0
        self.received_lines = 0
        self.nmea_sentences = 0
        self.gga_sentences = 0
        self.no_fix_sentences = 0
        self.error: Optional[str] = None
        self.ready = threading.Event()
        self._serial = None

    def connect(self) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("缺少 pyserial，请执行: pip install pyserial") from exc
        try:
            self._serial = serial.Serial(self.port, self.baud, timeout=0.2)
        except Exception as exc:
            raise RuntimeError(f"无法打开 GPS 串口 {self.port}: {exc}") from exc

    def request_stop(self) -> None:
        """Wake a blocking serial read so shutdown does not wait on the driver."""
        self.stop_event.set()
        serial_port = self._serial
        if serial_port is not None:
            try:
                serial_port.cancel_read()
            except (AttributeError, OSError):
                pass

    def run(self) -> None:
        parser = NMEAParser()
        self.ready.set()
        try:
            while not self.stop_event.is_set():
                raw = self._serial.readline()
                if not raw:
                    continue
                host_rx_ns = time.time_ns()
                line = raw.decode("ascii", errors="replace").strip()
                self.received_lines += 1
                if line.startswith("$"):
                    self.nmea_sentences += 1
                is_gga = line.startswith(("$GPGGA", "$GNGGA"))
                if is_gga:
                    self.gga_sentences += 1
                try:
                    fix = parser.parse(line, host_rx_ns)
                except (ValueError, IndexError, OverflowError):
                    self.bad_sentences += 1
                    continue
                if fix is not None:
                    self.fixes.append(fix)
                elif is_gga:
                    fields = line.split("*", 1)[0].split(",")
                    if _checksum_ok(line) and len(fields) >= 7 and (fields[6] or "0") == "0":
                        self.no_fix_sentences += 1
                    else:
                        self.bad_sentences += 1
        except Exception as exc:
            self.error = str(exc)
            self.stop_event.set()
        finally:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass


def fit_device_clock(samples: Sequence[tuple[int, int]]) -> ClockFit:
    """Robust affine fit: host_utc_ns = slope * device_ns + intercept."""
    if not samples:
        raise ValueError("no device/host clock samples")
    if len(samples) == 1:
        device_ns, host_ns = samples[0]
        return ClockFit(1.0, float(host_ns - device_ns), 0.0, 0.0, 1)

    def fit(points: Sequence[tuple[int, int]]) -> tuple[float, float]:
        x0, y0 = points[0]
        xs = [float(x - x0) for x, _ in points]
        ys = [float(y - y0) for _, y in points]
        # Keep compatibility with the project's older system Python, where
        # statistics.fmean is unavailable.
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denominator = sum((x - x_mean) ** 2 for x in xs)
        slope = 1.0 if denominator == 0 else sum(
            (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
        ) / denominator
        intercept = float(y0) - slope * float(x0) + y_mean - slope * x_mean
        return slope, intercept

    points = sorted(samples)
    slope, intercept = fit(points)
    residuals = [host - (slope * device + intercept) for device, host in points]
    median = statistics.median(residuals)
    deviations = [abs(value - median) for value in residuals]
    mad = statistics.median(deviations) or 1.0
    filtered = [
        point for point, residual in zip(points, residuals)
        if abs(residual - median) <= max(6.0 * mad, 5_000_000.0)
    ]
    if len(filtered) >= 3:
        slope, intercept = fit(filtered)
        points = filtered
    abs_residual_ms = sorted(
        abs(host - (slope * device + intercept)) / NS_PER_MS
        for device, host in points
    )
    p95_index = min(len(abs_residual_ms) - 1, math.ceil(0.95 * len(abs_residual_ms)) - 1)
    return ClockFit(
        slope=slope,
        intercept_ns=intercept,
        residual_median_ms=statistics.median(abs_residual_ms),
        residual_p95_ms=abs_residual_ms[p95_index],
        sample_count=len(points),
    )


def interpolate_accel(
    gyros: Sequence[MotionSample], accels: Sequence[MotionSample]
) -> list[tuple[int, float, float, float, float, float, float]]:
    """Interpolate accelerometer readings onto every gyro timestamp."""
    gyros = sorted({sample.device_ns: sample for sample in gyros}.values(), key=lambda s: s.device_ns)
    accels = sorted({sample.device_ns: sample for sample in accels}.values(), key=lambda s: s.device_ns)
    if not gyros or not accels:
        return []
    accel_times = [sample.device_ns for sample in accels]
    rows = []
    for gyro in gyros:
        # Only retain the interval in which acceleration is genuinely
        # available; do not fabricate boundary IMU data by extrapolation.
        if gyro.device_ns < accel_times[0] or gyro.device_ns > accel_times[-1]:
            continue
        right = bisect.bisect_left(accel_times, gyro.device_ns)
        if right == 0:
            accel = accels[0]
            ax, ay, az = accel.x, accel.y, accel.z
        elif right == len(accels):
            accel = accels[-1]
            ax, ay, az = accel.x, accel.y, accel.z
        else:
            before, after = accels[right - 1], accels[right]
            span = after.device_ns - before.device_ns
            alpha = 0.0 if span == 0 else (gyro.device_ns - before.device_ns) / span
            ax = before.x + alpha * (after.x - before.x)
            ay = before.y + alpha * (after.y - before.y)
            az = before.z + alpha * (after.z - before.z)
        rows.append((gyro.device_ns, gyro.x, gyro.y, gyro.z, ax, ay, az))
    return rows


def _median_rate_hz(timestamps: Sequence[int]) -> float:
    if len(timestamps) < 2:
        return 0.0
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    return 0.0 if not gaps else NS_PER_SECOND / statistics.median(gaps)


def _max_gap_ms(timestamps: Sequence[int]) -> float:
    if len(timestamps) < 2:
        return float("inf")
    return max(b - a for a, b in zip(timestamps, timestamps[1:])) / NS_PER_MS


def _write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_gps_outputs(
    session_dir: Path,
    images: Sequence[tuple[int, str]],
    fixes: Sequence[GPSFix],
    clock_fit: ClockFit,
    max_match_ms: float,
) -> tuple[int, list[float]]:
    gps_dir = session_dir / "gps"
    gps_dir.mkdir(exist_ok=True)
    fixes = sorted({fix.utc_ns: fix for fix in fixes}.values(), key=lambda fix: fix.utc_ns)
    _write_csv(
        gps_dir / "data.csv",
        ("utc_ns", "host_rx_ns", "serial_latency_ms", "latitude", "longitude",
         "altitude_m", "fix_quality", "satellites", "hdop", "sentence"),
        ((fix.utc_ns, fix.host_rx_ns, (fix.host_rx_ns - fix.utc_ns) / NS_PER_MS,
          f"{fix.latitude:.10f}", f"{fix.longitude:.10f}", f"{fix.altitude:.4f}",
          fix.fix_quality, fix.satellites, fix.hdop, fix.sentence) for fix in fixes),
    )

    gps_times = [fix.utc_ns for fix in fixes]
    aligned_rows = []
    valid_count = 0
    nearest_errors = []
    for device_ns, filename in images:
        camera_utc_ns = clock_fit.to_utc_ns(device_ns)
        insert = bisect.bisect_left(gps_times, camera_utc_ns)
        before = fixes[insert - 1] if insert > 0 else None
        after = fixes[insert] if insert < len(fixes) else None
        candidates = [fix for fix in (before, after) if fix is not None]
        nearest = min(candidates, key=lambda fix: abs(fix.utc_ns - camera_utc_ns)) if candidates else None
        nearest_ms = (
            abs(nearest.utc_ns - camera_utc_ns) / NS_PER_MS if nearest is not None else float("inf")
        )
        valid = nearest is not None and nearest_ms <= max_match_ms
        mode = "nearest"
        alpha = 0.0
        chosen = nearest
        latitude = longitude = altitude = float("nan")
        fix_quality = satellites = 0
        hdop = float("nan")
        before_ns = before.utc_ns if before else ""
        after_ns = after.utc_ns if after else ""
        if valid and before is not None and after is not None and after.utc_ns > before.utc_ns:
            alpha = (camera_utc_ns - before.utc_ns) / (after.utc_ns - before.utc_ns)
            if 0.0 <= alpha <= 1.0:
                mode = "interpolated"
                latitude = before.latitude + alpha * (after.latitude - before.latitude)
                longitude = before.longitude + alpha * (after.longitude - before.longitude)
                altitude = before.altitude + alpha * (after.altitude - before.altitude)
                chosen = before if alpha < 0.5 else after
        if valid and mode == "nearest" and chosen is not None:
            latitude, longitude, altitude = chosen.latitude, chosen.longitude, chosen.altitude
        if valid and chosen is not None:
            fix_quality = chosen.fix_quality
            satellites = chosen.satellites
            hdop = chosen.hdop
            valid_count += 1
            nearest_errors.append(nearest_ms)
        aligned_rows.append((
            device_ns, camera_utc_ns, filename, before_ns, after_ns, f"{alpha:.9f}", mode,
            f"{nearest_ms:.3f}" if math.isfinite(nearest_ms) else "", int(valid),
            f"{latitude:.10f}" if valid else "", f"{longitude:.10f}" if valid else "",
            f"{altitude:.4f}" if valid else "", fix_quality, satellites,
            f"{hdop:.3f}" if valid and math.isfinite(hdop) else "",
        ))
    _write_csv(
        session_dir / "frame_gps.csv",
        ("camera_device_ns", "camera_utc_ns", "image", "gps_before_utc_ns",
         "gps_after_utc_ns", "interpolation_alpha", "mode", "nearest_dt_ms", "valid",
         "latitude", "longitude", "altitude_m", "fix_quality", "satellites", "hdop"),
        aligned_rows,
    )
    return valid_count, nearest_errors


def write_orb_config(path: Path, calibration: dict, gyro_rate: float) -> None:
    intr = calibration["color_intrinsics"]
    coeffs = list(intr["coefficients"]) + [0.0] * 5
    matrix = calibration["T_imu_camera"]
    rows = [", ".join(f"{value:.9g}" for value in row) for row in matrix]
    text = f'''%YAML:1.0

File.version: "1.0"
Camera.type: "PinHole"

# Read from the connected D456 at recording time.
Camera1.fx: {intr['fx']:.9g}
Camera1.fy: {intr['fy']:.9g}
Camera1.cx: {intr['cx']:.9g}
Camera1.cy: {intr['cy']:.9g}
Camera1.k1: {coeffs[0]:.9g}
Camera1.k2: {coeffs[1]:.9g}
Camera1.p1: {coeffs[2]:.9g}
Camera1.p2: {coeffs[3]:.9g}
Camera1.k3: {coeffs[4]:.9g}
Camera.width: {intr['width']}
Camera.height: {intr['height']}
Camera.fps: {intr['fps']:.9g}
Camera.RGB: 0

# Camera-to-IMU transform, queried from the same physical device.
IMU.T_b_c1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: f
   data: [{rows[0]},
          {rows[1]},
          {rows[2]},
          {rows[3]}]
IMU.Frequency: {gyro_rate:.1f}
IMU.NoiseGyro: 1.5e-1
IMU.NoiseAcc: 1.5e0
IMU.GyroWalk: 1e-4
IMU.AccWalk: 1e-3
IMU.InsertKFsWhenLost: 1

ORBextractor.nFeatures: 2000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 10
ORBextractor.minThFAST: 5

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
'''
    path.write_text(text, encoding="utf-8")


def matrix_from_extrinsics(extrinsics) -> list[list[float]]:
    """Convert librealsense's column-major rotation into a row-major 4x4 matrix."""
    rotation = list(extrinsics.rotation)
    translation = list(extrinsics.translation)
    return [
        [rotation[0], rotation[3], rotation[6], translation[0]],
        [rotation[1], rotation[4], rotation[7], translation[1]],
        [rotation[2], rotation[5], rotation[8], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def run_self_test() -> None:
    parser = NMEAParser()
    host_ns = int(dt.datetime(1994, 3, 23, 12, 35, 20, tzinfo=dt.timezone.utc).timestamp() * 1e9)
    parser.parse("$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A", host_ns)
    fix = parser.parse("$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47", host_ns)
    assert fix is not None
    assert dt.datetime.fromtimestamp(fix.utc_ns / NS_PER_SECOND, tz=dt.timezone.utc).year == 1994
    assert abs(fix.latitude - 48.1173) < 1e-7
    assert abs(fix.longitude - 11.5166666667) < 1e-7
    assert fix.fix_quality == 1
    samples = [(i * 1_000_000_000, 1_700_000_000_000_000_000 + i * 1_000_010_000) for i in range(10)]
    fit = fit_device_clock(samples)
    assert abs(fit.slope - 1.00001) < 1e-8
    gyro = [MotionSample(5, 1, 2, 3)]
    accel = [MotionSample(0, 0, 0, 0), MotionSample(10, 10, 20, 30)]
    row = interpolate_accel(gyro, accel)[0]
    assert row[4:] == (5.0, 10.0, 15.0)
    fake_extrinsics = type("Extrinsics", (), {
        "rotation": [1, 4, 7, 2, 5, 8, 3, 6, 9],
        "translation": [10, 11, 12],
    })()
    assert matrix_from_extrinsics(fake_extrinsics)[:3] == [
        [1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]
    ]
    print("✅ self-test passed: NMEA, clock fit, IMU interpolation, extrinsics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D456 mono+IMU and GPS synchronized recorder")
    parser.add_argument("--name", help="session name; default is run_YYYYmmdd_HHMMSS")
    parser.add_argument("--out", type=Path, default=Path("recordings"), help="output root")
    parser.add_argument(
        "--gps-port", default="auto",
        help="GPS serial port; auto discovers GNSS/ttyACM/ttyUSB (default: auto)",
    )
    parser.add_argument("--gps-baud", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; 0 means until Ctrl+C")
    parser.add_argument("--warmup", type=float, default=3.0, help="camera warm-up seconds")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--gps-max-dt-ms", type=float, default=600.0,
        help="maximum distance to a real GPS fix (600 ms also supports 1 Hz GPS)",
    )
    parser.add_argument("--allow-no-gps", action="store_true", help="continue if GPS cannot open")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs
    except ImportError as exc:
        print(f"❌ 缺少依赖: {exc}", file=sys.stderr)
        print("安装: pip install pyrealsense2 pyserial opencv-python numpy", file=sys.stderr)
        return 2

    args.gps_port = resolve_gps_port(args.gps_port)
    if args.gps_port is None and not args.allow_no_gps:
        print("❌ 未发现 GPS 串口；请用 --gps-port /dev/tty... 指定，或用 --allow-no-gps 继续", file=sys.stderr)
        return 2

    name = args.name or dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if Path(name).name != name or name in (".", ".."):
        print("❌ --name 只能是目录名，不能包含路径", file=sys.stderr)
        return 2
    session_dir = args.out.expanduser().resolve() / name
    if session_dir.exists():
        print(f"❌ 输出目录已存在，拒绝覆盖: {session_dir}", file=sys.stderr)
        return 2
    image_dir = session_dir / "mav0" / "cam0" / "data"
    imu_dir = session_dir / "mav0" / "imu0"
    image_dir.mkdir(parents=True)
    imu_dir.mkdir(parents=True)
    (session_dir / "rgb").symlink_to(Path("mav0/cam0/data"), target_is_directory=True)
    incomplete_marker = session_dir / ".incomplete"
    incomplete_marker.write_text("recording\n", encoding="utf-8")

    stop_event = threading.Event()
    recording_event = threading.Event()
    fatal_errors: list[str] = []
    image_queue: queue.Queue = queue.Queue(maxsize=max(300, args.fps * 20))
    images: list[tuple[int, str]] = []
    gyros: list[MotionSample] = []
    accels: list[MotionSample] = []
    clock_samples: list[tuple[int, int]] = []
    timestamp_domains: set[str] = set()
    stream_domains: dict[str, set[str]] = {
        "color": set(), "gyro": set(), "accel": set()
    }
    image_quality_samples: list[tuple[float, float, float]] = []
    callback_lock = threading.Lock()
    image_count_seen = 0
    last_color_timestamp_ns = 0
    discarded_color_frames = 0

    def request_stop(_signum=None, _frame=None) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def image_writer() -> None:
        while not stop_event.is_set() or not image_queue.empty():
            try:
                device_ns, filename, image = image_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if not cv2.imwrite(str(image_dir / filename), image):
                    raise RuntimeError("cv2.imwrite returned false")
                images.append((device_ns, filename))
                if len(images) % 30 == 0:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image_quality_samples.append((
                        float(gray.mean()), float(gray.std()),
                        float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    ))
            except Exception as exc:
                fatal_errors.append(f"写图失败: {exc}")
                stop_event.set()
            finally:
                image_queue.task_done()

    writer = threading.Thread(target=image_writer, name="image-writer", daemon=True)
    writer.start()

    gps = GPSReader(args.gps_port or "", args.gps_baud, stop_event)
    if args.gps_port is not None:
        try:
            gps.connect()
            gps.start()
            print(f"✅ GPS: {args.gps_port} @ {args.gps_baud}")
        except RuntimeError as exc:
            if not args.allow_no_gps:
                print(f"❌ {exc}", file=sys.stderr)
                stop_event.set()
                writer.join(timeout=2)
                return 2
            print(f"⚠️ {exc}; 按 --allow-no-gps 继续")
    else:
        print("⚠️ 未连接 GPS，按 --allow-no-gps 仅录制 D456")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    # Let librealsense select the device-supported motion profiles. Their
    # actual rates are read from the resulting pipeline profile below.
    config.enable_stream(rs.stream.gyro)
    config.enable_stream(rs.stream.accel)
    def handle_images(color) -> None:
        nonlocal image_count_seen, last_color_timestamp_ns, discarded_color_frames
        if not color:
            return
        device_ns = int(round(color.get_timestamp() * NS_PER_MS))
        if device_ns <= last_color_timestamp_ns:
            discarded_color_frames += 1
            return
        last_color_timestamp_ns = device_ns
        domain = str(color.get_frame_timestamp_domain())
        timestamp_domains.add(domain)
        stream_domains["color"].add(domain)
        host_ns = time.time_ns()
        image_count_seen += 1
        if image_count_seen % max(1, args.fps) == 0:
            clock_samples.append((device_ns, host_ns))
        if not recording_event.is_set():
            return
        image = np.asanyarray(color.as_video_frame().get_data()).copy()
        filename = f"{device_ns}.png"
        try:
            image_queue.put_nowait((device_ns, filename, image))
        except queue.Full:
            fatal_errors.append("图像写入队列溢出；磁盘速度不足，数据集判定无效")
            stop_event.set()

    def handle_frame(frame) -> None:
        if frame.is_frameset():
            handle_images(frame.as_frameset().get_color_frame())
            return
        stream_type = frame.profile.stream_type()
        device_ns = int(round(frame.get_timestamp() * NS_PER_MS))
        domain = str(frame.get_frame_timestamp_domain())
        timestamp_domains.add(domain)
        if recording_event.is_set() and stream_type in (rs.stream.gyro, rs.stream.accel):
            stream_domains["gyro" if stream_type == rs.stream.gyro else "accel"].add(domain)
            motion = frame.as_motion_frame().get_motion_data()
            sample = MotionSample(device_ns, float(motion.x), float(motion.y), float(motion.z))
            with callback_lock:
                (gyros if stream_type == rs.stream.gyro else accels).append(sample)

    profile = None
    start_epoch_ns = 0
    end_epoch_ns = 0
    try:
        profile = pipeline.start(config, handle_frame)
        device = profile.get_device()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        gyro_profile = profile.get_stream(rs.stream.gyro)
        accel_profile = profile.get_stream(rs.stream.accel)
        intr = color_profile.get_intrinsics()
        extr = color_profile.get_extrinsics_to(gyro_profile)
        gyro_rate = float(gyro_profile.fps())
        accel_rate = float(accel_profile.fps())
        calibration = {
            "device_name": device.get_info(rs.camera_info.name),
            "device_serial": device.get_info(rs.camera_info.serial_number),
            "color_intrinsics": {
                "width": intr.width, "height": intr.height, "fps": float(color_profile.fps()),
                "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
                "distortion_model": str(intr.model), "coefficients": list(intr.coeffs),
            },
            "gyro_fps": gyro_rate,
            "accel_fps": accel_rate,
            "recording_mode": "monocular_imu",
            "depth_recorded": False,
            "T_imu_camera": matrix_from_extrinsics(extr),
            "extrinsics_definition": "camera coordinates to gyro/IMU coordinates (RealSense get_extrinsics_to)",
        }
        (session_dir / "camera_calibration.json").write_text(
            json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_orb_config(session_dir / "orbslam_d456_mono_imu.yaml", calibration, gyro_rate)

        print(f"✅ D456: {calibration['device_name']} S/N {calibration['device_serial']}")
        print(f"   Color: {intr.width}x{intr.height}@{color_profile.fps()}  IMU: gyro={gyro_rate:g}Hz accel={accel_rate:g}Hz")
        print(f"⏳ 预热 {args.warmup:.1f}s（本阶段不落盘）...")
        warmup_until = time.monotonic() + max(0.0, args.warmup)
        while time.monotonic() < warmup_until and not stop_event.is_set():
            time.sleep(0.05)

        start_epoch_ns = time.time_ns()
        recording_event.set()
        print(f"🔴 开始录制 → {session_dir}")
        print("   建议：开始静置约2秒，随后进行平移+转动以初始化惯性；Ctrl+C停止。")
        deadline = time.monotonic() + args.duration if args.duration > 0 else None
        last_report = time.monotonic()
        while not stop_event.is_set() and (deadline is None or time.monotonic() < deadline):
            time.sleep(0.1)
            if time.monotonic() - last_report >= 5.0:
                with callback_lock:
                    gyro_count, accel_count = len(gyros), len(accels)
                print(f"   RGB={len(images)} queued={image_queue.qsize()} gyro={gyro_count} accel={accel_count} GPS={len(gps.fixes)}")
                last_report = time.monotonic()
    except Exception as exc:
        fatal_errors.append(f"RealSense错误: {exc}")
        stop_event.set()
    finally:
        recording_event.clear()
        end_epoch_ns = time.time_ns()
        if profile is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        stop_event.set()
        if gps.is_alive():
            gps.join(timeout=2)
        writer.join(timeout=30)

    if writer.is_alive():
        fatal_errors.append("图像写入线程未能在30秒内结束")
    if gps.error:
        fatal_errors.append(f"GPS读取错误: {gps.error}")
    if not images or not gyros or not accels:
        fatal_errors.append("RGB、gyro或accel数据为空")
    if not clock_samples:
        fatal_errors.append("没有相机时钟映射样本")
    nonempty_domains = [values for values in stream_domains.values() if values]
    if nonempty_domains and len(set().union(*nonempty_domains)) != 1:
        fatal_errors.append(f"相机与IMU不在同一RealSense时间域: {stream_domains}")
    if fatal_errors:
        print("❌ 录制未通过完整性检查:", file=sys.stderr)
        for error in fatal_errors:
            print(f"   - {error}", file=sys.stderr)
        return 1

    images.sort()
    with callback_lock:
        imu_rows = interpolate_accel(gyros, accels)
    if not imu_rows:
        print("❌ 无法生成同步 IMU", file=sys.stderr)
        return 1
    imu_start, imu_end = imu_rows[0][0], imu_rows[-1][0]
    valid_images = [(stamp, name) for stamp, name in images if imu_start < stamp <= imu_end]
    dropped_boundary = len(images) - len(valid_images)

    _write_csv(
        session_dir / "mav0" / "cam0" / "data.csv",
        ("#timestamp [ns]", "filename"),
        ((stamp, name) for stamp, name in valid_images),
    )
    with (session_dir / "times.txt").open("w", encoding="utf-8") as handle:
        for stamp, _ in valid_images:
            handle.write(f"{stamp}\n")
    _write_csv(
        session_dir / "mav0" / "imu0" / "data.csv",
        ("#timestamp [ns]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]",
         "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]",
         "a_RS_S_z [m s^-2]"),
        ((row[0], *(f"{value:.9f}" for value in row[1:])) for row in imu_rows),
    )

    clock_fit = fit_device_clock(clock_samples)
    session_fixes = [
        fix for fix in gps.fixes
        if start_epoch_ns - NS_PER_SECOND <= fix.host_rx_ns <= end_epoch_ns + NS_PER_SECOND
    ]
    valid_gps, gps_errors = write_gps_outputs(
        session_dir, valid_images, session_fixes, clock_fit, args.gps_max_dt_ms
    )

    image_times = [row[0] for row in valid_images]
    imu_times = [row[0] for row in imu_rows]
    brightness = [row[0] for row in image_quality_samples]
    contrast = [row[1] for row in image_quality_samples]
    sharpness = [row[2] for row in image_quality_samples]
    gyro_axis_std = [
        statistics.pstdev([row[index] for row in imu_rows])
        for index in (1, 2, 3)
    ]
    accel_norms = [math.sqrt(row[4] ** 2 + row[5] ** 2 + row[6] ** 2) for row in imu_rows]
    quality = {
        "image_count": len(valid_images),
        "imu_count": len(imu_rows),
        "gps_fix_count": len(session_fixes),
        "gps_aligned_image_count": valid_gps,
        "gps_alignment_coverage": valid_gps / len(valid_images) if valid_images else 0.0,
        "image_rate_hz": _median_rate_hz(image_times),
        "imu_rate_hz": _median_rate_hz(imu_times),
        "image_max_gap_ms": _max_gap_ms(image_times),
        "imu_max_gap_ms": _max_gap_ms(imu_times),
        "boundary_images_excluded": dropped_boundary,
        "gps_nearest_dt_median_ms": statistics.median(gps_errors) if gps_errors else None,
        "gps_nearest_dt_max_ms": max(gps_errors) if gps_errors else None,
        "clock_slope": clock_fit.slope,
        "clock_drift_ppm": (clock_fit.slope - 1.0) * 1e6,
        "clock_fit_residual_median_ms": clock_fit.residual_median_ms,
        "clock_fit_residual_p95_ms": clock_fit.residual_p95_ms,
        "brightness_median": statistics.median(brightness) if brightness else None,
        "contrast_median": statistics.median(contrast) if contrast else None,
        "laplacian_variance_median": statistics.median(sharpness) if sharpness else None,
        "gyro_axis_std_rad_s": gyro_axis_std,
        "accel_norm_std_m_s2": statistics.pstdev(accel_norms) if len(accel_norms) > 1 else 0.0,
    }
    quality["image_quality_ok"] = bool(
        contrast and sharpness
        and statistics.median(contrast) >= 15.0
        and statistics.median(sharpness) >= 20.0
    )
    quality["imu_excitation_ok"] = bool(
        max(gyro_axis_std) >= 0.01 or quality["accel_norm_std_m_s2"] >= 0.05
    )
    quality["orbslam_ready"] = bool(
        len(valid_images) >= 60
        and len(imu_rows) >= 300
        and quality["image_max_gap_ms"] < 150.0
        and quality["imu_max_gap_ms"] < 50.0
        and quality["image_quality_ok"]
        and quality["imu_excitation_ok"]
    )
    quality["gps_sync_ready"] = bool(
        session_fixes and quality["gps_alignment_coverage"] >= 0.9
    )
    metadata = {
        "format_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recording_start_utc_ns": start_epoch_ns,
        "recording_end_utc_ns": end_epoch_ns,
        "timestamp_domains": sorted(timestamp_domains),
        "stream_timestamp_domains": {key: sorted(value) for key, value in stream_domains.items()},
        "clock_fit": asdict(clock_fit),
        "quality": quality,
        "recording_mode": "monocular_imu",
        "depth_recorded": False,
        "discarded_color_frames": discarded_color_frames,
        "gps_bad_sentences": gps.bad_sentences,
        "synchronization_note": (
            "Camera/IMU share the D456 device clock. GPS uses NMEA UTC. "
            "frame_gps.csv uses an affine device-to-host-UTC fit; without PPS this is software synchronization."
        ),
    }
    (session_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    project_root = Path(__file__).resolve().parents[3]
    runner = session_dir / "run_orbslam.sh"
    runner.write_text(
        "#!/usr/bin/env bash\nset -e\n"
        f'PROJECT_ROOT="{project_root}"\n'
        f'DATASET="{session_dir}"\n'
        'export LD_LIBRARY_PATH=/opt/ros/lyrical/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}\n'
        'cd "$DATASET"\n'
        '"$PROJECT_ROOT/Examples/Monocular-Inertial/mono_inertial_euroc" \\\n'
        '  "$PROJECT_ROOT/Vocabulary/ORBvoc.txt" \\\n'
        '  "$DATASET/orbslam_d456_mono_imu.yaml" \\\n'
        '  "$DATASET" "$DATASET/times.txt" synced_run\n',
        encoding="utf-8",
    )
    runner.chmod(0o755)
    incomplete_marker.unlink(missing_ok=True)

    print("\n✅ 录制与整理完成")
    print(f"   RGB: {len(valid_images)} ({quality['image_rate_hz']:.2f} Hz)")
    print(f"   IMU: {len(imu_rows)} ({quality['imu_rate_hz']:.2f} Hz)")
    print(f"   GPS: {len(session_fixes)}, 覆盖图像 {quality['gps_alignment_coverage']:.1%}")
    print(f"   时钟拟合P95残差: {clock_fit.residual_p95_ms:.3f} ms")
    print(f"   图像质量: {'通过' if quality['image_quality_ok'] else '未通过'}")
    print(f"   IMU运动激励: {'通过' if quality['imu_excitation_ok'] else '未通过'}")
    print(f"   ORB-SLAM3数据检查: {'通过' if quality['orbslam_ready'] else '未通过'}")
    print(f"   GPS同步检查: {'通过' if quality['gps_sync_ready'] else '未通过'}")
    print(f"   输出: {session_dir}")
    print(f"   建图: {runner}")
    return 0 if quality["orbslam_ready"] and (quality["gps_sync_ready"] or args.allow_no_gps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
