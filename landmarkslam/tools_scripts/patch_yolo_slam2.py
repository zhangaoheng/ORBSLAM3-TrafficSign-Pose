import sys
import re

filename = '/home/zah/ORB_SLAM3-master/landmarkslam/run_landmarkslam_yolo.cc'
with open(filename, 'r', encoding='utf-8') as f:
    code = f.read()

new_block = """if (yoloData.find(baseFilename) != yoloData.end()) {
                vector<cv::Point2f> corners = yoloData[baseFilename];
                printAndLog("\\n[Landmark Extractor] Frame " + baseFilename + " has YOLO keypoints!");
                
                // 【修改这里】：获取 SLAM 当前帧跟踪到的所有特征点及其对应的真实 3D 地图点
                vector<cv::KeyPoint> vKeys = SLAM.GetTrackedKeyPointsUn();
                vector<ORB_SLAM3::MapPoint*> vMPs = SLAM.GetTrackedMapPoints();
                
                for (int i = 0; i < 4; ++i) {
                    // 我们在当前帧的 SLAM 特征点里，寻找距离 YOLO 角点最近的一个点
                    float min_dist = 100000.0f;
                    int best_idx = -1;
                    for (size_t j = 0; j < vKeys.size(); ++j) {
                        if (vMPs[j] != nullptr && !vMPs[j]->isBad()) {
                            float dx = vKeys[j].pt.x - corners[i].x;
                            float dy = vKeys[j].pt.y - corners[i].y;
                            float dist = sqrt(dx*dx + dy*dy);
                            if (dist < min_dist) {
                                min_dist = dist;
                                best_idx = j;
                            }
                        }
                    }
                    
                    // 如果找到了并且像素距离在可接受范围内 (比如100像素范围内)
                    // 就直接将该 SLAM 在底层严密求出的世界 3D 坐标赋给这个 YOLO 角点
                    if (best_idx != -1 && min_dist < 100.0f) {
                        Eigen::Vector3f pos3d_eigen = vMPs[best_idx]->GetWorldPos();
                        cv::Point3f P3D(pos3d_eigen(0), pos3d_eigen(1), pos3d_eigen(2));
                        
                        YoloLandmark3D lm;
                        lm.frame_name = baseFilename;
                        lm.corner_index = i;
                        lm.pos3d = P3D;
                        yoloLandmarks.push_back(lm);
                        
                        printAndLog("   Corner " + to_string(i) + " 2D: (" + to_string(corners[i].x) + ", " + to_string(corners[i].y) +
                                    ") -> MapPoint fused (dist=" + to_string(min_dist) + "): [" + to_string(P3D.x) + ", " + to_string(P3D.y) + ", " + to_string(P3D.z) + "]");
                    } else {
                        printAndLog("   Corner " + to_string(i) + " 2D: (" + to_string(corners[i].x) + ", " + to_string(corners[i].y) +
                                    ") -> No nearby SLAM 3D point found (min_dist=" + to_string(min_dist) + ").");
                    }
                }
                printAndLog("-----------------------------------------");
            }"""

code_sub = re.sub(r'if \(yoloData\.find\(baseFilename\).*?-----------------------------------------\"\);\s*\}', new_block, code, flags=re.DOTALL)

with open(filename, 'w', encoding='utf-8') as f:
    f.write(code_sub)
print("Replaced!")
