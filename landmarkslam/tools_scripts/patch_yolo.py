import sys

filepath = "/home/zah/ORB_SLAM3-master/landmarkslam/run_landmarkslam_yolo.cc"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update the includes and structs definition at the beginning
new_struct = """using namespace std;

// 结构体：保存融合了YOLO框估计出的3D路标点
struct YoloLandmark3D {
    string frame_name;
    int corner_index;
    cv::Point3f pos3d;
};

// 辅助函数：保存YOLO计算出的点云为PLY格式
void SaveYoloLandmarksPLY(const vector<YoloLandmark3D>& landmarks, const string& filename) {
    ofstream f(filename);
    if (!f.is_open()) return;

    f << "ply\\n";
    f << "format ascii 1.0\\n";
    f << "element vertex " << landmarks.size() << "\\n";
    f << "property float x\\n";
    f << "property float y\\n";
    f << "property float z\\n";
    f << "property uchar red\\n";
    f << "property uchar green\\n";
    f << "property uchar blue\\n";
    f << "end_header\\n";

    for (const auto& lm : landmarks) {
        // YOLO路标点我们给它标记为红色 (255, 0, 0)
        f << fixed << setprecision(5) << lm.pos3d.x << " " << lm.pos3d.y << " " << lm.pos3d.z << " 255 0 0\\n";
    }
    f.close();
}

// 保存当前地图的点云为标准的PLY格式，供三维可视化使用"""

content = content.replace("using namespace std;\n\n// 保存当前地图的点云为标准的PLY格式", new_struct)


# 2. Add tracking container before main loop
old_main_vars = """    ORB_SLAM3::System SLAM(strVocFile, strSettingsFile, ORB_SLAM3::System::MONOCULAR, true);

    cv::Mat im;
    double t_frame = 0.0;
    const double t_step = 1.0 / 30.0;

    printAndLog("\\n------- Start processing sequence -------");"""

new_main_vars = """    ORB_SLAM3::System SLAM(strVocFile, strSettingsFile, ORB_SLAM3::System::MONOCULAR, true);

    cv::Mat im;
    double t_frame = 0.0;
    const double t_step = 1.0 / 30.0;
    
    vector<YoloLandmark3D> yoloLandmarks;

    printAndLog("\\n------- Start processing sequence -------");"""

content = content.replace(old_main_vars, new_main_vars)


# 3. Save to tracking container inside loop
old_loop_logic = """                    // 计算空间射线/坐标。由于单目无绝对深度，这里假设深度为1.0
                    cv::Point3f P3D = unprojectToWorld(corners[i], Tcw, fx, fy, cx, cy, 1.0f);
                    printAndLog("   Corner " + to_string(i) + " 2D: (" + to_string(corners[i].x) + ", " + to_string(corners[i].y) + 
                                ") -> 3D(假设d=1): [" + to_string(P3D.x) + ", " + to_string(P3D.y) + ", " + to_string(P3D.z) + "]");
                }"""

new_loop_logic = """                    // 计算空间射线/坐标。由于单目无绝对深度，这里假设深度为1.0
                    cv::Point3f P3D = unprojectToWorld(corners[i], Tcw, fx, fy, cx, cy, 1.0f);
                    
                    YoloLandmark3D lm;
                    lm.frame_name = baseFilename;
                    lm.corner_index = i;
                    lm.pos3d = P3D;
                    yoloLandmarks.push_back(lm);
                    
                    printAndLog("   Corner " + to_string(i) + " 2D: (" + to_string(corners[i].x) + ", " + to_string(corners[i].y) + 
                                ") -> 3D(假设d=1): [" + to_string(P3D.x) + ", " + to_string(P3D.y) + ", " + to_string(P3D.z) + "]");
                }"""

content = content.replace(old_loop_logic, new_loop_logic)

# 4. output to output2 folder
old_end = """    string plyFilename = "output/PointCloud_" + timestamp + ".ply";
    string trajFilename = "output/KeyFrameTrajectory_" + timestamp + ".txt";
    
    // 导出文件供Python 3D环境可视化读取
    SavePointCloudPLY(SLAM, plyFilename);
    printAndLog("Saved 3D Map PointCloud to: " + plyFilename);
    
    SLAM.SaveKeyFrameTrajectoryTUM(trajFilename);
    printAndLog("Saved KeyFrame Trajectory to: " + trajFilename);"""

new_end = """    // 创建 output2 文件夹
    system("mkdir -p output2");

    string plyFilename = "output2/SLAM_PointCloud_" + timestamp + ".ply";
    string trajFilename = "output2/SLAM_KeyFrameTrajectory_" + timestamp + ".txt";
    string yoloFilename = "output2/YOLO_Landmarks_" + timestamp + ".ply";
    
    // 导出常规 SLAM 点云和轨迹
    SavePointCloudPLY(SLAM, plyFilename);
    printAndLog("Saved SLAM 3D Map PointCloud to: " + plyFilename);
    
    SLAM.SaveKeyFrameTrajectoryTUM(trajFilename);
    printAndLog("Saved SLAM KeyFrame Trajectory to: " + trajFilename);
    
    // 导出 YOLO 专有的点云
    SaveYoloLandmarksPLY(yoloLandmarks, yoloFilename);
    printAndLog("Saved YOLO Landmarks PointCloud to: " + yoloFilename);"""

content = content.replace(old_end, new_end)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)
print("Patched YOLO landmarks script successfully!")

