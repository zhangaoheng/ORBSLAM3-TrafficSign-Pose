#!/bin/bash
cd /home/zah/ORB_SLAM3-master
exec python3 -u /home/zah/ORB_SLAM3-master/landmarkslam/implement/slam_outside_real/extract_bag.py \
  "/mnt/d/zah/20260529_114122.bag" \
  "/home/zah/ORB_SLAM3-master/landmarkslam/implement/data/extracted_data"
