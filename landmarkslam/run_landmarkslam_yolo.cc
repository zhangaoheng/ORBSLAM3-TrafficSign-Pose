#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <iomanip>
#include <unistd.h>
#include <map>

#include <opencv2/opencv.hpp>
#include "System.h"
#include "Atlas.h"
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <Eigen/Core>

using namespace std;

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

    f << "ply\n";
    f << "format ascii 1.0\n";
    f << "element vertex " << landmarks.size() << "\n";
    f << "property float x\n";
    f << "property float y\n";
    f << "property float z\n";
    f << "property uchar red\n";
    f << "property uchar green\n";
    f << "property uchar blue\n";
    f << "end_header\n";

    for (const auto& lm : landmarks) {
        // YOLO路标点我们给它标记为红色 (255, 0, 0)
        f << fixed << setprecision(5) << lm.pos3d.x << " " << lm.pos3d.y << " " << lm.pos3d.z << " 255 0 0\n";
    }
    f.close();
}

// 保存当前地图的点云为标准的PLY格式，供三维可视化使用，供三维可视化使用
void SavePointCloudPLY(ORB_SLAM3::System& SLAM, const string& filename) {
    ORB_SLAM3::Map* pMap = SLAM.GetAtlas()->GetCurrentMap();
    if (!pMap) return;
    vector<ORB_SLAM3::MapPoint*> vpMPs = pMap->GetAllMapPoints();
    
    int valid_points = 0;
    for (auto pMP : vpMPs) {
        if (!pMP || pMP->isBad()) continue;
        valid_points++;
    }

    ofstream f(filename);
    if (!f.is_open()) return;

    f << "ply\n";
    f << "format ascii 1.0\n";
    f << "element vertex " << valid_points << "\n";
    f << "property float x\n";
    f << "property float y\n";
    f << "property float z\n";
    f << "end_header\n";

    for (auto pMP : vpMPs) {
        if (!pMP || pMP->isBad()) continue;
        Eigen::Vector3f pos = pMP->GetWorldPos();
        f << fixed << setprecision(5) << pos(0) << " " << pos(1) << " " << pos(2) << "\n";
    }
    f.close();
}

// 辅助函数：将 Tracking 的内部状态码转换为直观的字符串
string getTrackingStateStr(int state) {
    switch(state) {
        case -1: return "SYSTEM_NOT_READY";
        case 0:  return "NO_IMAGES_YET";
        case 1:  return "NOT_INITIALIZED (尝试初始化中)";
        case 2:  return "OK (稳定跟踪)";
        case 3:  return "RECENTLY_LOST (刚刚丢失)";
        case 4:  return "LOST (彻底丢失)";
        case 5:  return "OK_KLT (KLT跟踪)";
        default: return "UNKNOWN_STATE (" + to_string(state) + ")";
    }
}

// 加载YOLO检测结果，格式假设为: 帧名.png x1 y1 x2 y2 x3 y3 x4 y4
map<string, vector<cv::Point2f>> loadYoloDetections(const string& filename) {
    map<string, vector<cv::Point2f>> detections;
    ifstream f(filename);
    if (!f.is_open()) {
        cerr << "[Warning] Could not open YOLO detection file: " << filename << endl;
        return detections;
    }
    
    string line;
    while (getline(f, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string img_name;
        ss >> img_name;
        
        vector<cv::Point2f> pts;
        float x, y;
        for (int i = 0; i < 4; ++i) {
            if (ss >> x >> y) {
                pts.push_back(cv::Point2f(x, y));
            }
        }
        if (pts.size() == 4) {
            detections[img_name] = pts;
        }
    }
    return detections;
}

// 辅助函数：根据相机位姿 Tcw 和内参将像素坐标转换为世界坐标系下的射线方向
// 注意：由于单目SLAM尺度不确定且缺少深度信息，这里计算的是归一化平面上的射线，或者假设固定深度的三维点
cv::Point3f unprojectToWorld(const cv::Point2f& p, const cv::Mat& Tcw, float fx, float fy, float cx, float cy, float assumed_depth = 1.0) {
    if (Tcw.empty()) return cv::Point3f(0,0,0);
    
    // 1. 像素坐标转相机归一化坐标
    float u = (p.x - cx) / fx;
    float v = (p.y - cy) / fy;
    
    cv::Mat X_c = (cv::Mat_<float>(3, 1) << u * assumed_depth, v * assumed_depth, assumed_depth);
    
    // 2. 取出旋转平移矩阵 (Tcw 是世界到相机的变换)
    cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
    
    // Rwc = Rcw^T, twc = -Rcw^T * tcw
    cv::Mat Rwc = Rcw.t();
    cv::Mat twc = -Rwc * tcw;
    
    // 3. 转世界坐标 P_w = Rwc * P_c + twc
    cv::Mat X_w = Rwc * X_c + twc;
    
    return cv::Point3f(X_w.at<float>(0), X_w.at<float>(1), X_w.at<float>(2));
}

int main(int argc, char **argv)
{
    // 需要传入 YOLO 检测结果文件作为第 4 个参数
    if(argc != 5)
    {
        cerr << endl << "Usage: ./run_landmarkslam_yolo path_to_vocabulary path_to_settings path_to_image_folder path_to_yolo_results.txt" << endl;
        return 1;
    }

    string strVocFile = argv[1];
    string strSettingsFile = argv[2];
    string strImageFolder = argv[3];
    string strYoloFile = argv[4];

    // ============================================
    // 1. 初始化日志文件与输出目录
    // ============================================
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss_ts;
    oss_ts << std::put_time(&tm, "%Y%m%d_%H%M%S");
    string timestamp = oss_ts.str();
    
    int ret = system("mkdir -p log output");
    (void)ret;

    string logFilename = "log/landmarkslam_yolo_log_" + timestamp + ".txt";
    ofstream logFile(logFilename);

    auto printAndLog = [&](const string& msg) {
        cout << msg << endl;
        if(logFile.is_open()) logFile << msg << endl;
    };

    printAndLog("\n=========================================");
    printAndLog("\n    ORB-SLAM3 Yolo Fusion Landmark       ");
    printAndLog("\n=========================================");

    // 读取相机内参 (需要根据 Settings 读取，这里先占位或者自己从 FileStorage 读取)
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened()) {
        cerr << "Failed to open settings file at: " << strSettingsFile << endl;
        return -1;
    }
    float fx = fsSettings["Camera1.fx"];
    float fy = fsSettings["Camera1.fy"];
    float cx = fsSettings["Camera1.cx"];
    float cy = fsSettings["Camera1.cy"];
    printAndLog("\nLoaded camera intrinsics: fx=" + to_string(fx) + " fy=" + to_string(fy));

    // 加载 YOLO 检测数据
    map<string, vector<cv::Point2f>> yoloData = loadYoloDetections(strYoloFile);
    printAndLog("\nLoaded YOLO detections for " + to_string(yoloData.size()) + " frames.");

    vector<cv::String> imageFilePaths;
    cv::glob(strImageFolder + "/*.png", imageFilePaths, false);
    if(imageFilePaths.empty()) {
        cv::glob(strImageFolder + "/*.jpg", imageFilePaths, false);
    }
    sort(imageFilePaths.begin(), imageFilePaths.end());
    int nImages = imageFilePaths.size();

    // ============================================
    // 2. 初始化 ORB_SLAM3 系统结构
    // ============================================
    ORB_SLAM3::System SLAM(strVocFile, strSettingsFile, ORB_SLAM3::System::MONOCULAR, true);

    cv::Mat im;
    double t_frame = 0.0;
    const double t_step = 1.0 / 30.0;
    
    vector<YoloLandmark3D> yoloLandmarks;

    printAndLog("\n\\n------- Start processing sequence -------");

    // ============================================
    // 3. 主循环
    // ============================================
    for(int ni=0; ni<nImages; ni++)
    {
        im = cv::imread(imageFilePaths[ni], cv::IMREAD_UNCHANGED);
        if(im.empty()) continue;

        string baseFilename = imageFilePaths[ni].substr(imageFilePaths[ni].find_last_of("\\/") + 1);

        // 核心：调用 TrackMonocular
        Sophus::SE3f Tcw_sophus = SLAM.TrackMonocular(im, t_frame);
        int currentState = SLAM.GetTrackingState();

        // 检查这一帧是否有 YOLO 的检测信息
        if (currentState == 2) { // 如果SLAM稳定跟踪(OK)且有位姿
            Eigen::Matrix4f Tcw_eigen = Tcw_sophus.matrix();
            cv::Mat Tcw(4, 4, CV_32F);
            for(int i=0;i<4;i++)
                for(int j=0;j<4;j++)
                    Tcw.at<float>(i,j) = Tcw_eigen(i,j);

            if (yoloData.find(baseFilename) != yoloData.end()) {
                vector<cv::Point2f> corners = yoloData[baseFilename];
                printAndLog("\n[Landmark Extractor] Frame " + baseFilename + " has YOLO keypoints!");
                
                // 获取 SLAM 当前帧跟踪到的所有特征点及其对应的真实 3D 地图点
                vector<cv::KeyPoint> vKeys = SLAM.GetTrackedKeyPointsUn();
                vector<ORB_SLAM3::MapPoint*> vMPs = SLAM.GetTrackedMapPoints();
                
                // 【完全遵循您的要求】不进行任何反投影、深度估计等额外计算。
                // 仅记录SLAM已经完成的内部计算过程，将4个角点对应的SLAM 3D MapPoint提取出来
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
                        
                        YoloLandmark3D lm;
                        lm.frame_name = baseFilename;
                        lm.corner_index = i;
                        lm.pos3d = cv::Point3f(pos3d(0), pos3d(1), pos3d(2));
                        yoloLandmarks.push_back(lm);
                    } else {
                        printAndLog("   Corner " + to_string(i) + " [" + to_string(corners[i].x) + ", " + to_string(corners[i].y) + "] -> SLAM did not compute a valid map point here.");
                    }
                }
                }
            }
        usleep(80000);
        t_frame += t_step;
    }

    printAndLog("\n\\n------- Sequence processing finished -------");

    // 创建 output2 文件夹
    system("mkdir -p output2");

    string plyFilename = "output2/SLAM_PointCloud_" + timestamp + ".ply";
    string trajFilename = "output2/SLAM_KeyFrameTrajectory_" + timestamp + ".txt";
    string yoloFilename = "output2/YOLO_Landmarks_" + timestamp + ".ply";
    
    // 导出常规 SLAM 点云和轨迹
    SavePointCloudPLY(SLAM, plyFilename);
    printAndLog("\nSaved SLAM 3D Map PointCloud to: " + plyFilename);
    
    SLAM.SaveKeyFrameTrajectoryTUM(trajFilename);
    printAndLog("\nSaved SLAM KeyFrame Trajectory to: " + trajFilename);
    
    // 导出 YOLO 专有的点云
    SaveYoloLandmarksPLY(yoloLandmarks, yoloFilename);
    printAndLog("\nSaved YOLO Landmarks PointCloud to: " + yoloFilename);

    SLAM.Shutdown();
    
    if(logFile.is_open()) logFile.close();
    return 0;
}
