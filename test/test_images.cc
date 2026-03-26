#include <opencv2/opencv.hpp>
#include "System.h"
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <string>
#include <chrono>
#include <iostream>
#include <vector>
#include <map>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <experimental/filesystem>
#include <sstream>
#include <thread>

namespace fs = std::experimental::filesystem;
using namespace std;

string vocFile = "../Vocabulary/ORBvoc.txt";
// 默认路径，如果命令行未指定则使用此路径
string imageDirVariable = "/home/zah/ORB_SLAM3-master/markslam/data/20260127_163648/predictions_with_boxes"; 
string outputBaseDir = "./outputs";
string parameterFile = "./test.yaml";

// 创建输出目录（含时间戳）
string CreateOutputRunDir(const string& base_dir = "./outputs") {
    auto now = chrono::system_clock::now();
    time_t t = chrono::system_clock::to_time_t(now);
    std::tm tm_now{};
#ifdef _WIN32
    localtime_s(&tm_now, &t);
#else
    localtime_r(&t, &tm_now);
#endif
    std::ostringstream oss;
    oss << base_dir << "/run_" << std::put_time(&tm_now, "%Y%m%d_%H%M%S");
    string run_path = oss.str();
    
    // 如果目录已存在，追加序号
    if (fs::exists(run_path)) {
        int i = 1;
        while (fs::exists(run_path + "_" + to_string(i))) {
            i++;
        }
        run_path += "_" + to_string(i);
    }

    try {
        fs::create_directories(run_path);
    } catch (const std::exception& e) {
        cerr << "创建输出目录失败: " << run_path << " error: " << e.what() << endl;
    }
    return run_path;
}

// 保存点云到 CSV 文件
void SavePointCloud(ORB_SLAM3::System& SLAM, const string& filename) {
    cout << "\n正在保存点云 (CSV)..." << endl;
    ORB_SLAM3::Map* pMap = SLAM.GetAtlas()->GetCurrentMap();
    vector<ORB_SLAM3::MapPoint*> vpMPs = pMap ? pMap->GetAllMapPoints() : vector<ORB_SLAM3::MapPoint*>();
    ofstream ofs(filename);
    if (!ofs.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return;
    }
    ofs << "X,Y,Z,R,G,B\n";
    int saved_count = 0;
    for (ORB_SLAM3::MapPoint* pMP : vpMPs) {
        if (!pMP || pMP->isBad()) continue;
        Eigen::Vector3f pos = pMP->GetWorldPos();
        unsigned char r = 255, g = 0, b = 0;
        ofs << std::fixed << std::setprecision(6)
            << pos[0] << "," << pos[1] << "," << pos[2] << ","
            << static_cast<int>(r) << "," << static_cast<int>(g) << "," << static_cast<int>(b) << "\n";
        saved_count++;
    }
    ofs.close();
    cout << "✓ 点云已保存: " << filename << endl;
    cout << "  总点数: " << vpMPs.size() << ", 有效点数: " << saved_count << endl;
}

// 保存所有帧的相机轨迹
void SaveCameraTrajectory(const vector<pair<int, Sophus::SE3f>>& trajectory, const string& filename) {
    cout << "\n正在保存相机位姿 (CSV)..." << endl;
    ofstream ofs(filename);
    if (!ofs.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return;
    }
    ofs << "frame,tx,ty,tz,qx,qy,qz,qw\n";
    int saved_count = 0;
    for (const auto& item : trajectory) {
        int frame_id = item.first;
        Sophus::SE3f Tcw = item.second;
        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Vector3f t = Twc.translation();
        Eigen::Quaternionf q = Twc.unit_quaternion();
        ofs << frame_id << ","
            << std::fixed << std::setprecision(6)
            << t[0] << "," << t[1] << "," << t[2] << ","
            << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\n";
        saved_count++;
    }
    ofs.close();
    cout << "✓ 相机轨迹已保存: " << filename << endl;
    cout << "  总帧数: " << trajectory.size() << ", 有效轨迹: " << saved_count << endl;
}

int main(int argc, char **argv) {
    cout << "Starting ORB-SLAM3 Image Sequence Mode..." << endl;
    
    // 1. 处理输入参数
    string imageDir = imageDirVariable;
    if (argc > 1) {
        imageDir = string(argv[1]);
    }
    cout << "Target Image Directory: " << imageDir << endl;
    
    // 2. 验证图片目录
    if (!fs::exists(imageDir) || !fs::is_directory(imageDir)) {
        cerr << "[Error] Invalid directory path: " << imageDir << endl;
        cerr << "Usage: ./test_slam_images <path_to_images>" << endl;
        return -1;
    }

    // 3. 读取图片文件列表
    vector<string> filenames;
    for (const auto& entry : fs::directory_iterator(imageDir)) {
        string ext = entry.path().extension().string();
        // 转小写比较
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
            filenames.push_back(entry.path().string());
        }
    }
    
    // 4. 排序
    sort(filenames.begin(), filenames.end());
    
    if (filenames.empty()) {
        cerr << "[Error] No images found in: " << imageDir << endl;
        return -1;
    }
    
    int total_frames = filenames.size();
    cout << "Found " << total_frames << " images." << endl;

    // 5. 初始化 SLAM
    if (!fs::exists(vocFile)) {
        cerr << "[Error] Vocabulary file not found: " << vocFile << endl;
        return -1;
    }
    if (!fs::exists(parameterFile)) {
        cerr << "[Error] Parameter file not found: " << parameterFile << endl;
        return -1;
    }

    ORB_SLAM3::System SLAM(vocFile, parameterFile, ORB_SLAM3::System::MONOCULAR, true);
    
    string output_dir = CreateOutputRunDir(outputBaseDir);
    string pointcloud_file = (fs::path(output_dir) / "pointcloud.csv").string();
    string trajectory_file = (fs::path(output_dir) / "camera_trajectory.csv").string();
    cout << "Output Directory: " << output_dir << endl;
    
    vector<pair<int, Sophus::SE3f>> trajectory_data;

    cv::namedWindow("Frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("Frame", 1280, 720);
    
    int frame_count = 0;
    
    // 6. 主循环
    for (const string& filename : filenames) {
        frame_count++;
        cv::Mat frame = cv::imread(filename, cv::IMREAD_UNCHANGED);
        if (frame.empty()) {
            cerr << "Warning: Failed to load image " << filename << endl;
            continue;
        }

        double timestamp = (double)frame_count;
        Sophus::SE3f Tcw;
        
        // 跟踪
        try { 
            Tcw = SLAM.TrackMonocular(frame, timestamp); 
        } catch (const exception& e) {
            cerr << "SLAM Exception: " << e.what() << endl;
        }
        
        // 记录轨迹
        if (!Tcw.translation().isZero()) { 
            trajectory_data.push_back({frame_count, Tcw}); 
        }
        
        // 显示
        cv::Mat display = frame.clone();
        char text[256];
        snprintf(text, sizeof(text), "Frame %d/%d: %s", frame_count, total_frames, fs::path(filename).filename().string().c_str());
        cv::putText(display, text, cv::Point(10, frame.rows - 20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        cv::imshow("Frame", display);
        if (cv::waitKey(1) == 27) break; // ESC to exit
    }
    
    // 7. 保存结果
    SavePointCloud(SLAM, pointcloud_file);
    SaveCameraTrajectory(trajectory_data, trajectory_file);
    
    SLAM.Shutdown();
    cout << "\n==========================" << endl;
    cout << "Processing Complete!" << endl;
    cout << "Trajectory stored in: " << trajectory_file << endl;
    cout << "==========================" << endl;
    return 0;
}
