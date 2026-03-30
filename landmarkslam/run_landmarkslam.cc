#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <iomanip>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include "System.h"
#include "Atlas.h"
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <Eigen/Core>


using namespace std;

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

int main(int argc, char **argv)
{
    // 强制要求用户严格传入 3 个基础参数，贴合官方 Examples 的使用规范
    if(argc != 4)
    {
        cerr << endl << "Usage: ./run_landmarkslam path_to_vocabulary path_to_settings path_to_image_folder" << endl;
        return 1;
    }

    string strVocFile = argv[1];
    string strSettingsFile = argv[2];
    string strImageFolder = argv[3];

    // ============================================
    // 1. 初始化日志文件与输出目录 (以当前时间戳命名)
    // ============================================
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss_ts;
    oss_ts << std::put_time(&tm, "%Y%m%d_%H%M%S");
    string timestamp = oss_ts.str();
    
    // 确保存在 log 和 output 文件夹
    int ret = system("mkdir -p log output");
    (void)ret; // 忽略返回值警告

    string logFilename = "log/landmarkslam_log_" + timestamp + ".txt";
    
    ofstream logFile(logFilename);
    if(!logFile.is_open()) {
        cerr << "Error: Failed to open log file: " << logFilename << endl;
        return -1;
    }

    // 统一向终端和日志文件输出的 Lambda 辅助函数
    auto printAndLog = [&](const string& msg) {
        cout << msg << endl;
        logFile << msg << endl;
    };

    printAndLog("=========================================");
    printAndLog("        ORB-SLAM3 Landmark Logger        ");
    printAndLog("=========================================");
    printAndLog("Log file  : " + logFilename);
    printAndLog("Vocab     : " + strVocFile);
    printAndLog("Settings  : " + strSettingsFile);
    printAndLog("Image dir : " + strImageFolder);
    printAndLog("=========================================");

    // 读取并按字典序排序图像
    vector<cv::String> imageFilePaths;
    cv::glob(strImageFolder + "/*.png", imageFilePaths, false);
    if(imageFilePaths.empty()) {
        cv::glob(strImageFolder + "/*.jpg", imageFilePaths, false);
    }
    if(imageFilePaths.empty()) {
        printAndLog("Error: No images found in " + strImageFolder);
        return -1;
    }
    
    // 官方的 glob 可能不会保证完全的字典序，尤其是在不同平台上，所以我们手动排序一下
    // sort(imageFilePaths.begin(), imageFilePaths.end());

    // 🌟 修改：使用自定义排序，提取 capture_N_... 中的数字 N 进行排序
    sort(imageFilePaths.begin(), imageFilePaths.end(), [](const cv::String& a, const cv::String& b) {
        // 提取文件名部分
        string nameA = a.substr(a.find_last_of("\\/") + 1);
        string nameB = b.substr(b.find_last_of("\\/") + 1);

        try {
            // 假设格式为 capture_帧数_时间戳.jpg，跳过 "capture_" (8个字符)
            size_t firstA = nameA.find('_');
            size_t secondA = nameA.find('_', firstA + 1);
            int idxA = std::stoi(nameA.substr(firstA + 1, secondA - firstA - 1));

            size_t firstB = nameB.find('_');
            size_t secondB = nameB.find('_', firstB + 1);
            int idxB = std::stoi(nameB.substr(firstB + 1, secondB - firstB - 1));

            return idxA < idxB;
        } catch (...) {
            // 如果解析失败，回退到普通排序
            return a < b;
        }
    });

    int nImages = imageFilePaths.size();
    printAndLog("Found " + to_string(nImages) + " images. Starting SLAM system...");

    // ============================================
    // 2. 初始化纯粹的官方 ORB_SLAM3 系统结构（单目）
    // ============================================
    ORB_SLAM3::System SLAM(strVocFile, strSettingsFile, ORB_SLAM3::System::MONOCULAR, true);

    cv::Mat im;
    double t_frame = 0.0;
    const double fps = 30.0; 
    const double t_step = 1.0 / fps;

    printAndLog("\n------- Start processing sequence -------");

    int prevState = -2; // 初始设为不存在的状态，以确保第一帧被记录

    // ============================================
    // 3. 主循环：向系统输入图像并执行日志记录
    // ============================================
    for(int ni=0; ni<nImages; ni++)
    {
        im = cv::imread(imageFilePaths[ni], cv::IMREAD_UNCHANGED);
        if(im.empty()) {
            printAndLog("[WARNING] Failed to load image at index " + to_string(ni) + ": " + imageFilePaths[ni]);
            continue;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // 核心：调用 TrackMonocular 并计算位姿
        SLAM.TrackMonocular(im, t_frame);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        // 核心：提取 ORB_SLAM3 当前的底层跟踪状态
        int currentState = SLAM.GetTrackingState();
        
        // 构建当前帧的具体日志信息 (提取文件名)
        string baseFilename = imageFilePaths[ni].substr(imageFilePaths[ni].find_last_of("\\/") + 1);
        std::ostringstream frameLog;
        frameLog << "[Frame " << std::setw(4) << std::setfill('0') << ni << "] "
                 << "Img: " << baseFilename
                 << " | Tracking State: " << getTrackingStateStr(currentState);

        // 如果追踪状态发生变化（比如从 OK 变成了 LOST，或从 NOT_INITIALIZED 变成了 OK）
        if (currentState != prevState) {
            frameLog << " <--- [STATE CHANGE: " << getTrackingStateStr(prevState) 
                     << " -> " << getTrackingStateStr(currentState) << "]";
            
            // 状态改变极其关键，同时打印到控制台
            cout << frameLog.str() << endl; 
            prevState = currentState;
        } 
        else if (currentState == 4 || currentState == 3) {
            // 如果持续处于丢失状态，不要每帧在终端刷屏，每 30 帧 (约1秒) 提醒一次
            if(ni % 30 == 0) cout << "  ... Still LOST at frame " << ni << " (" << baseFilename << ")" << endl;
        }

        // 把这一帧的极其详尽的数据写入 log 文件中 (不论是否发生了状态改变)
        logFile << frameLog.str() << endl;

        // ===================================
        // ===================================
        // 【核心修改】针对 WSL/慢速设备的降速回放策略
        // 我们在数学时间上（t_frame）仍然假装是 30 FPS，以保证运动学模型的准确性。
        // 但在“现实物理时间”中，我们强制主线程每一帧都睡足够久（80 毫秒）。
        // 这给了后台（Local Mapping）极其充裕的时间来建图，专门应对转弯处的特征爆炸。
        // ===================================
        
        usleep(80000);

        t_frame += t_step;
    }

    printAndLog("\n------- Sequence processing finished -------");
    
    // ============================================
    // 4. 关闭系统，保存结果
    // ============================================
    SLAM.Shutdown();
    string trajFilename = "output/KeyFrameTrajectory_" + timestamp + ".txt";
    SLAM.SaveKeyFrameTrajectoryTUM(trajFilename);

    // ============================================
    // 5. 纯静态保存：建图追踪结束后再遍历保存全部点云和关键帧
    // ============================================
    printAndLog("Saving complete PointCloud and KeyFrames...");
    ORB_SLAM3::Atlas* pAtlas = SLAM.GetAtlas();
    if(pAtlas) {
        vector<ORB_SLAM3::Map*> maps = pAtlas->GetAllMaps();
        
        // 保存点云 (PLY格式方便 MeshLab 查看)
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

        // 保存关键帧轨迹位姿 (同样存为点云格式方便直接同时拖入MeshLab查看轨迹)
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

        printAndLog("Save complete! Saved " + to_string(allPoints.size()) + " points and " + to_string(allKFs.size()) + " keyframes.");
    }


    printAndLog("SLAM framework finished cleanly. Outputs saved.");
    logFile.close();

    return 0;
}
