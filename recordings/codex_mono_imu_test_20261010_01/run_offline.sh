#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/home/ehuy/ORBSLAM3-TrafficSign-Pose"
DATASET="recordings/codex_mono_imu_test_20261010_01"
export LD_LIBRARY_PATH=/opt/ros/lyrical/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
cd "$DATASET"
"$PROJECT_ROOT/Examples/Monocular-Inertial/mono_inertial_euroc" \
  "$PROJECT_ROOT/Vocabulary/ORBvoc.txt" \
  "$DATASET/orbslam_d456_mono_imu.yaml" \
  "$DATASET" "$DATASET/times.txt" offline_replay
