#!/bin/bash
# 带可视化界面的 ORB-SLAM3 运行脚本
# 在 WSL 中直接运行: bash run_viz.sh

export MESA_GL_VERSION_OVERRIDE=4.5
export MESA_GLSL_VERSION_OVERRIDE=450
export LIBGL_ALWAYS_SOFTWARE=1

cd /home/zah/ORB_SLAM3-master/landmarkslam/implement/output

echo ">>> 开始运行 ORB-SLAM3（等待加载字典约20秒）..."
../build/run_mono_fast \
  ../../Vocabulary/ORBvoc.txt \
  ../data/D456i_140500.yaml \
  ../data/euroc_20260613_140500 \
  ../data/euroc_20260613_140500/times.txt \
  140500_viz

echo "完成！"
