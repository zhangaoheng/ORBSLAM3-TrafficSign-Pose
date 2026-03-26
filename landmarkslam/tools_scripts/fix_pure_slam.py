import sys

with open('/home/zah/ORB_SLAM3-master/landmarkslam/run_landmarkslam_yolo.cc', 'r', encoding='utf-8') as f:
    lines = f.readlines()

out_lines = []
in_block = False
brace_count = 0

for line in lines:
    if 'if (yoloData.find(baseFilename) != yoloData.end()) {' in line:
        in_block = True
        brace_count = 1
        out_lines.append(line)
        # Append our new simplified logic
        out_lines.append('''                vector<cv::Point2f> corners = yoloData[baseFilename];
                printAndLog("\\n[Landmark Extractor] Frame " + baseFilename + " has YOLO keypoints!");
                
                // 获取 SLAM 当前帧跟踪到的所有特征点及其对应的真实 3D 地图点
                vector<cv::KeyPoint> vKeys = SLAM.GetTrackedKeyPointsUn();
                vector<ORB_SLAM3::MapPoint*> vMPs = SLAM.GetTrackedMapPoints();
                
                // 【完全遵循您的要求】不进行任何反投影、深度估计等额外计算。
                // 仅记录SLAM已经完成的内部计算过程，将4个角点对应的SLAM 3D MapPoint剥离出来
                for (int i = 0; i < 4; ++i) {
                    float min_dist = 100000.0f;
                    ORB_SLAM3::MapPoint* best_mp = nullptr;
                    cv::Point2f best_kp(0,0);
                    
                    // 遍历目前整个Frame中由SLAM系统“内部计算并产生”的三维MapPoint
                    for (size_t j = 0; j < vKeys.size(); ++j) {
                        if (vMPs[j] != nullptr && !vMPs[j]->isBad()) {
                            float dx = vKeys[j].pt.x - corners[i].x;
                            float dy = vKeys[j].pt.y - corners[i].y;
                            float dist = dx * dx + dy * dy; 
                            
                            // 找到与角点信息最贴合的一个追踪点
                            if (dist < min_dist) {
                                min_dist = dist;
                                best_mp = vMPs[j];
                                best_kp = vKeys[j].pt;
                            }
                        }
                    }
                    
                    // 如果系统为这个角点算出了三维空间信息，直接提取并输出
                    if (best_mp != nullptr && min_dist < 625.0f) { // 允许25个像素的追踪偏差容差
                        Eigen::Vector3f pos3d = best_mp->GetWorldPos();
                        printAndLog("   Corner " + to_string(i) + " [" + to_string(corners[i].x) + ", " + to_string(corners[i].y) + "] -> SLAM extracted POS: [" + to_string(pos3d(0)) + ", " + to_string(pos3d(1)) + ", " + to_string(pos3d(2)) + "]");
                        
                        yolo_cloud_file << pos3d(0) << " " << pos3d(1) << " " << pos3d(2) << "\\n";
                    } else {
                        printAndLog("   Corner " + to_string(i) + " [" + to_string(corners[i].x) + ", " + to_string(corners[i].y) + "] -> SLAM did not compute a valid map point here.");
                    }
                }
''')
        continue
        
    if in_block:
        if '{' in line:
            brace_count += line.count('{')
        if '}' in line:
            brace_count -= line.count('}')
        
        if brace_count == 0:
            in_block = False
            out_lines.append(line)
        continue
        
    out_lines.append(line)

with open('/home/zah/ORB_SLAM3-master/landmarkslam/run_landmarkslam_yolo.cc', 'w', encoding='utf-8') as f:
    f.writelines(out_lines)

