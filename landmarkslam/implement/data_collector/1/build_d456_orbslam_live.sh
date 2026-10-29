#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SOURCE="$PROJECT_ROOT/Examples/Monocular-Inertial/mono_inertial_realsense_D456.cc"
OUTPUT="$PROJECT_ROOT/Examples/Monocular-Inertial/mono_inertial_realsense_D456_live"

g++ -std=c++14 -O3 "$SOURCE" -o "$OUTPUT" \
  -I"$PROJECT_ROOT" \
  -I"$PROJECT_ROOT/include" \
  -I"$PROJECT_ROOT/include/CameraModels" \
  -I"$PROJECT_ROOT/Thirdparty/Sophus" \
  -I/usr/include/eigen3 \
  -I/usr/include/opencv4 \
  -I/opt/ros/lyrical/include \
  -L"$PROJECT_ROOT/lib" \
  -Wl,-rpath,"$PROJECT_ROOT/lib:/opt/ros/lyrical/lib/x86_64-linux-gnu:/usr/local/lib" \
  -lORB_SLAM3 \
  -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core \
  -L/opt/ros/lyrical/lib/x86_64-linux-gnu -lrealsense2 \
  /usr/local/lib/libpangolin.so -lGL -lGLEW -lpthread

echo "Built: $OUTPUT"
