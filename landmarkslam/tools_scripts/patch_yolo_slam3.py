import sys
import re

filename = '/home/zah/ORB_SLAM3-master/landmarkslam/run_landmarkslam_yolo.cc'
with open(filename, 'r', encoding='utf-8') as f:
    code = f.read()

new_block = """if (yoloData.find(baseFilename) != yoloData.end()) {
                vector<cv::Point2f> corners = yoloData[baseFilename];
                printAndLog("\\n[Landmark Extractor] Frame " + baseFilename + " has YOLO keypoints!");
                
                // 获取 SLAM 当前帧跟踪到的所有特征点及其对应的真实 3D 地图点
                vector<cv::KeyPoint> vKeys = SLAM.GetTrackedKeyPointsUn();
                vector<ORB_SLAM3::MapPoint*> vMPs = SLAM.GetTrackedMapPoints();
                
                // 计算该YOLO物体的像素包围框
                float min_x = 10000.0f, max_x = -10000.0f;
                float min_y = 10000.0f, max_y = -10000.0f;
                for(int i=0; i<4; i++){
                    if(corners[i].x < min_x) min_x = corners[i].x;
                    if(corners[i].x > max_x) max_x = corners[i].x;
                    if(corners[i].y < min_y) min_y = corners[i].y;
                    if(corners[i].y > max_y) max_y = corners[i].y;
                }
                
                // 收集包围框内的SLAM三维点的深度 (相机坐标系下的Z轴分量)
                vector<float> depths;
                cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
                cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
                
                for (size_t j = 0; j < vKeys.size(); ++j) {
                    if (vMPs[j] != nullptr && !vMPs[j]->isBad()) {
                        float k_x = vKeys[j].pt.x;
                        float k_y = vKeys[j].pt.y;
                        // 寻找在 YOLO bounding box 内部的地图点
                        if(k_x >= min_x && k_x <= max_x && k_y >= min_y && k_y <= max_y) {
                            Eigen::Vector3f pos3d_eigen = vMPs[j]->GetWorldPos();
                            cv::Mat Pw = (cv::Mat_<float>(3, 1) << pos3d_eigen(0), pos3d_eigen(1), pos3d_eigen(2));
                            cv::Mat Pc = Rcw * Pw + tcw;
                            float z = Pc.at<float>(2);
                            if(z > 0.1f) depths.push_back(z);
                        }
                    }
                }
                
                float object_depth = -1.0f; 
                bool valid_depth = false;
                if(!depths.empty()) {
                    std::sort(depths.begin(), depths.end());
                    object_depth = depths[depths.size() / 2]; // 使用中位数深度，排查极端噪点
                    valid_depth = true;
                } else {
                    // 如果包围框内部没有点，则找距离包围框中心最近的特征点作为深度的参考
                    float min_dist = 100000.0f;
                    float nearest_depth = -1.0f;
                    float cx_box = (min_x + max_x) / 2.0f;
                    float cy_box = (min_y + max_y) / 2.0f;
                    for (size_t j = 0; j < vKeys.size(); ++j) {
                        if (vMPs[j] != nullptr && !vMPs[j]->isBad()) {
                            float dx = vKeys[j].pt.x - cx_box;
                            float dy = vKeys[j].pt.y - cy_box;
                            float dist = sqrt(dx*dx + dy*dy);
                            if(dist < min_dist) {
                                min_dist = dist;
                                Eigen::Vector3f pos3d_eigen = vMPs[j]->GetWorldPos();
                                cv::Mat Pw = (cv::Mat_<float>(3, 1) << pos3d_eigen(0), pos3d_eigen(1), pos3d_eigen(2));
                                cv::Mat Pc = Rcw * Pw + tcw;
                                nearest_depth = Pc.at<float>(2);
                            }
                        }
                    }
                    if(nearest_depth > 0.1f && min_dist < 300.0f) {
                        object_depth = nearest_depth;
                        valid_depth = true;
                    }
                }
                
                for (int i = 0; i < 4; ++i) {
                    if (valid_depth) {
                        // 使用目标物体的深度，结合相机模型，直接将角点反投影到3D空间
                        cv::Point3f P3D = unprojectToWorld(corners[i], Tcw, fx, fy, cx, cy, object_depth);
                        
                        YoloLandmark3D lm;
                        lm.frame_name = baseFilename;
                        lm.corner_index = i;
                        lm.pos3d = P3D;
                        yoloLandmarks.push_back(lm);
                        
                        printAndLog("   Corner " + to_string(i) + " 2D: (" + to_string(corners[i].x) + ", " + to_string(corners[i].y) + ") -> Unprojected 3D (z_cam=" + to_string(object_depth) + "): [" + to_string(P3D.x) + ", " + to_string(P3D.y) + ", " + to_string(P3D.z) + "]");
                    } else {
                        printAndLog("   Corner " + to_string(i) + " 2D: (" + to_string(corners[i].x) + ", " + to_string(corners[i].y) + ") -> No SLAM MapPoints nearby to estimate depth.");
                    }
                }
                printAndLog("-----------------------------------------");
            }"""

code_sub = re.sub(r'if \(yoloData\.find\(baseFilename\).*?-----------------------------------------\"\);\s*\}', new_block, code, flags=re.DOTALL)
with open(filename, 'w', encoding='utf-8') as f:
    f.write(code_sub)
print("Replaced!")
