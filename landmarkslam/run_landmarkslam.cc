#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <iomanip>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "System.h"
#include "Atlas.h"
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"

using namespace std;

string getTrackingStateStr(int state) {
    switch(state) {
        case -1: return "SYSTEM_NOT_READY";
        case 0:  return "NO_IMAGES_YET";
        case 1:  return "NOT_INITIALIZED";
        case 2:  return "OK";
        case 3:  return "RECENTLY_LOST";
        case 4:  return "LOST";
        case 5:  return "OK_KLT";
        default: return "UNKNOWN_STATE (" + to_string(state) + ")";
    }
}

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./run_landmarkslam path_to_vocabulary path_to_settings path_to_image_folder" << endl;
        return 1;
    }

    string strVocFile = argv[1];
    string strSettingsFile = argv[2];
    string strImageFolder = argv[3];

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss_ts;
    oss_ts << std::put_time(&tm, "%Y%m%d_%H%M%S");
    string timestamp = oss_ts.str();
    
    int ret = system("mkdir -p log output");
    (void)ret;

    string logFilename = "log/landmarkslam_log_" + timestamp + ".txt";
    ofstream logFile(logFilename);
    
    // 图片名与时间戳映射文件 
    string mappingFilename = "output/Filename_Mapping_" + timestamp + ".csv";
    ofstream mappingFile(mappingFilename);
    mappingFile << "filename,timestamp_s" << endl;

    // 手动创建的全帧连续轨迹文件
    string myFullTrajFilename = "output/FrameTrajectory_TUM_" + timestamp + ".txt";
    ofstream myFullTrajFile(myFullTrajFilename);

    auto printAndLog = [&](const string& msg) {
        cout << msg << endl;
        logFile << msg << endl;
    };

    printAndLog("=========================================");
    printAndLog("        ORB-SLAM3 Landmark Logger        ");
    printAndLog("=========================================");

    vector<cv::String> imageFilePaths;
    cv::glob(strImageFolder + "/*.png", imageFilePaths, false);
    if(imageFilePaths.empty()) {
        cv::glob(strImageFolder + "/*.jpg", imageFilePaths, false);
    }
    
    sort(imageFilePaths.begin(), imageFilePaths.end(), [](const cv::String& a, const cv::String& b) {
        string nameA = a.substr(a.find_last_of("\\/") + 1);
        string nameB = b.substr(b.find_last_of("\\/") + 1);
        try {
            size_t firstA = nameA.find('_');
            size_t secondA = nameA.find('_', firstA + 1);
            int idxA = std::stoi(nameA.substr(firstA + 1, secondA - firstA - 1));

            size_t firstB = nameB.find('_');
            size_t secondB = nameB.find('_', firstB + 1);
            int idxB = std::stoi(nameB.substr(firstB + 1, secondB - firstB - 1));

            return idxA < idxB;
        } catch (...) {
            return a < b;
        }
    });

    int nImages = imageFilePaths.size();
    printAndLog("Found " + to_string(nImages) + " images. Starting SLAM system...");

    ORB_SLAM3::System SLAM(strVocFile, strSettingsFile, ORB_SLAM3::System::MONOCULAR, true);

    cv::Mat im;
    int prevState = -2;

    printAndLog("\n------- Start processing sequence -------");

    for(int ni=0; ni<nImages; ni++)
    {
        im = cv::imread(imageFilePaths[ni], cv::IMREAD_UNCHANGED);
        if(im.empty()) continue;

        string baseFilename = imageFilePaths[ni].substr(imageFilePaths[ni].find_last_of("\\/") + 1);
        
        double t_frame = 0.0;
        try {
            size_t firstUnderscore = baseFilename.find('_');
            size_t secondUnderscore = baseFilename.find('_', firstUnderscore + 1);
            size_t dotPos = baseFilename.find_last_of('.');
            string tsStr = baseFilename.substr(secondUnderscore + 1, dotPos - secondUnderscore - 1);
            
            t_frame = std::stod(tsStr) / 1e9;
        } catch (...) {
            t_frame = ni * (1.0/30.0); 
        }

        mappingFile << baseFilename << "," << fixed << setprecision(6) << t_frame << endl;

        // 🌟 修复：使用 Sophus::SE3f 接收位姿
        Sophus::SE3f Tcw = SLAM.TrackMonocular(im, t_frame);

        int currentState = SLAM.GetTrackingState();
        std::ostringstream frameLog;
        frameLog << "[Frame " << std::setw(4) << std::setfill('0') << ni << "] "
                 << "Img: " << baseFilename
                 << " | Tracking State: " << getTrackingStateStr(currentState);

        if (currentState != prevState) {
            frameLog << " <--- [STATE CHANGE: " << getTrackingStateStr(prevState) 
                     << " -> " << getTrackingStateStr(currentState) << "]";
            cout << frameLog.str() << endl; 
            prevState = currentState;
        } 

        // 🌟 修复：极其优雅地提取 Sophus 的位姿数据
        if (currentState == 2) {
            // Sophus 求逆极其方便，直接得到相机在世界坐标系的变换 Twc
            Sophus::SE3f Twc = Tcw.inverse();
            
            // 直接提取平移向量 (tx, ty, tz)
            Eigen::Vector3f twc = Twc.translation();
            
            // 直接提取旋转四元数 (qx, qy, qz, qw)
            Eigen::Quaternionf qwc = Twc.unit_quaternion();

            // 以 TUM 标准格式写入
            myFullTrajFile << fixed << setprecision(6) << t_frame << " "
                           << twc.x() << " " << twc.y() << " " << twc.z() << " "
                           << qwc.x() << " " << qwc.y() << " " << qwc.z() << " " << qwc.w() << "\n";
        }

        logFile << frameLog.str() << endl;
        usleep(80000); 
    }

    printAndLog("\n------- Sequence processing finished -------");
    SLAM.Shutdown();

    printAndLog("Saving trajectories and maps...");

    string kfTrajFilename = "output/KeyFrameTrajectory_TUM_" + timestamp + ".txt";
    SLAM.SaveKeyFrameTrajectoryTUM(kfTrajFilename);

    ORB_SLAM3::Atlas* pAtlas = SLAM.GetAtlas();
    if(pAtlas) {
        vector<ORB_SLAM3::Map*> maps = pAtlas->GetAllMaps();
        
        string pcFilename = "output/PointCloud_" + timestamp + ".ply";
        ofstream fPC(pcFilename);
        vector<Eigen::Vector3f> allPoints;
        for (auto pMap : maps) {
            auto mps = pMap->GetAllMapPoints();
            for (auto pMP : mps) {
                if(!pMP || pMP->isBad()) continue;
                allPoints.push_back(pMP->GetWorldPos());
            }
        }
        if(!allPoints.empty()) {
            fPC << "ply\nformat ascii 1.0\nelement vertex " << allPoints.size() << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
            for(const auto& pt : allPoints) {
                fPC << pt.x() << " " << pt.y() << " " << pt.z() << "\n";
            }
        }
        fPC.close();

        string kfFilename = "output/KeyFrames_Poses_" + timestamp + ".ply";
        ofstream fKF(kfFilename);
        vector<Eigen::Vector3f> allKFs;
        for (auto pMap : maps) {
            auto kfs = pMap->GetAllKeyFrames();
            for (auto pKF : kfs) {
                if(!pKF || pKF->isBad()) continue;
                allKFs.push_back(pKF->GetCameraCenter());
            }
        }
        if(!allKFs.empty()) {
            fKF << "ply\nformat ascii 1.0\nelement vertex " << allKFs.size() << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
            for(const auto& pt : allKFs) {
                fKF << pt.x() << " " << pt.y() << " " << pt.z() << "\n";
            }
        }
        fKF.close();

        printAndLog("Save complete! Saved " + to_string(allPoints.size()) + " MapPoints and " + to_string(allKFs.size()) + " KeyFrames.");
    }

    printAndLog("SLAM framework finished cleanly. Outputs saved.");
    
    logFile.close();
    mappingFile.close();
    myFullTrajFile.close();

    return 0;
}