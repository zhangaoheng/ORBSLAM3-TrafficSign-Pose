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
#include <experimental/filesystem>
#include <sstream>
#include <thread>
namespace fs = std::experimental::filesystem;
namespace fs = std::experimental::filesystem;
string vocFile = "../Vocabulary/ORBvoc.txt";
string mapviewerVideoFile = "/tmp/mapviewer_recording.mp4"; // MapViewer 录屏临时路径
string videoFile = "/home/zah/ORB_SLAM3-master/test/videos/runway_cut.mp4";
string outputBaseDir = "./outputs";
string npy_dir = "/home/zah/ORB_SLAM3-master/test/videos/roi_landmark/roi1";

string parameterFile = "./test.yaml";
// 存储每帧的 ROI 角点（缓存）
map<int, vector<cv::Point2f>> roi_cache;

// 从npy文件读取四个角点坐标
bool loadCornerPointsFromNpyPython(const string& npy_file, vector<cv::Point2f>& corners) {
    string python_cmd = "python3 -c \"import numpy as np; data = np.load('" + npy_file + 
                       "', allow_pickle=True).item(); kpts = data.get('refined_kpts', data.get('raw_kpts')); "
                       "[print(f'{pt[0]:.6f},{pt[1]:.6f}') for pt in kpts]\" 2>/dev/null";
    
    FILE* pipe = popen(python_cmd.c_str(), "r");
    if (!pipe) {
        return false;
    }
    
    corners.clear();
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        float x, y;
        if (sscanf(buffer, "%f,%f", &x, &y) == 2) {
            corners.push_back(cv::Point2f(x, y));
        }
    }
    
    int result = pclose(pipe);
    return (result == 0 && corners.size() == 4);
}

// 检查点是否在四边形内（用于显示过滤）
bool pointInQuadrilateral(const cv::Point2f& point, const vector<cv::Point2f>& quad) {
    if (quad.size() != 4) return true;
    
    auto sign = [](float val) -> int {
        if (val > 1e-6) return 1;
        if (val < -1e-6) return -1;
        return 0;
    };
    
    auto crossProduct = [](const cv::Point2f& O, const cv::Point2f& A, const cv::Point2f& B) -> float {
        return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
    };
    
    int sign_val = 0;
    for (int i = 0; i < 4; i++) {
        float cp = crossProduct(quad[i], quad[(i+1)%4], point);
        int curr_sign = sign(cp);
        if (curr_sign != 0) {
            if (sign_val == 0) {
                sign_val = curr_sign;
            } else if (sign_val != curr_sign) {
                return false;
            }
        }
    }
    
    return true;
}

// 在帧上绘制 ROI 区域
void drawROI(cv::Mat& frame, const vector<cv::Point2f>& roi_quad) {
    if (roi_quad.size() != 4) return;
    
    for (int i = 0; i < 4; i++) {
        cv::line(frame, roi_quad[i], roi_quad[(i+1)%4], cv::Scalar(0, 255, 0), 2);
    }
    
    for (const auto& pt : roi_quad) {
        cv::circle(frame, pt, 5, cv::Scalar(0, 0, 255), -1);
    }
}

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
    string dir = oss.str();
    try {
        fs::create_directories(dir);
    } catch (const std::exception& e) {
        cerr << "创建输出目录失败: " << dir << " error: " << e.what() << endl;
    }
    return dir;
}

// 若源文件存在则复制
bool CopyIfExists(const string& src, const string& dst, const string& label) {
    if (!fs::exists(src)) {
        cerr << "未找到" << label << ": " << src << endl;
        return false;
    }
    try {
        fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
        cout << "✓ 已复制" << label << ": " << dst << endl;
        return true;
    } catch (const std::exception& e) {
        cerr << "复制" << label << "失败: " << e.what() << endl;
        return false;
    }
}
// 保存点云到文件
// 保存点云到文件
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
void SaveCameraTrajectory(ORB_SLAM3::System& SLAM, const string& filename) {
    cout << "\n正在保存相机位姿 (CSV)..." << endl;
    
    ORB_SLAM3::Map* pMap = SLAM.GetAtlas()->GetCurrentMap();
    vector<ORB_SLAM3::KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    
    // 按时间戳排序
    sort(vpKFs.begin(), vpKFs.end(), [](ORB_SLAM3::KeyFrame* a, ORB_SLAM3::KeyFrame* b) {
        return a->mnFrameId < b->mnFrameId;
    });
    
    ofstream ofs(filename);
    if (!ofs.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return;
    }
    // 写入文件头 (CSV: frame,tx,ty,tz,qx,qy,qz,qw)
    ofs << "frame,tx,ty,tz,qx,qy,qz,qw\n";
    
    int saved_count = 0;
    for (ORB_SLAM3::KeyFrame* pKF : vpKFs) {
        if (!pKF || pKF->isBad()) continue;
        Sophus::SE3f Twc = pKF->GetPoseInverse();
        Eigen::Vector3f t = Twc.translation();
        Eigen::Quaternionf q = Twc.unit_quaternion();
        double timestamp = pKF->mTimeStamp;
        ofs << (long long)timestamp << ","
            << std::fixed << std::setprecision(6)
            << t[0] << "," << t[1] << "," << t[2] << ","
            << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\n";
        saved_count++;
    }

    ofs.close();
    cout << "✓ 相机轨迹已保存: " << filename << endl;
    cout << "  总关键帧数: " << vpKFs.size() << ", 有效轨迹: " << saved_count << endl;
}

int main(int argc, char **argv) {
    cout << "Starting ORB-SLAM3..." << endl;
    cout << "Mode: Full frame extraction, ROI-filtered display" << endl;
    cout << "Video: " << videoFile << endl;
    
    // 初始化 SLAM 系统
    ORB_SLAM3::System SLAM(vocFile, parameterFile, ORB_SLAM3::System::MONOCULAR, true);
    
    // 等待 MapViewer 窗口启动并开始录制
    cout << "等待 MapViewer 窗口启动..." << endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    string ffmpeg_cmd = "ffmpeg -y -f x11grab -video_size 1241x829 -i :0.0 "
                       "-r 30 -vcodec libx264 -preset ultrafast -crf 23 " + mapviewerVideoFile + " > /dev/null 2>&1 &";
    
    cout << "✓ 启动 MapViewer 窗口录制" << endl;
    system(ffmpeg_cmd.c_str());

    // 打开视频
    cv::VideoCapture cap(videoFile);
    if (!cap.isOpened()) {
        cerr << "Failed to open video!" << endl;
        return -1;
    }
    
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    cout << "Video: " << frame_width << "x" << frame_height << ", " << total_frames << " frames" << endl;
    string output_dir = CreateOutputRunDir(outputBaseDir);
    string pointcloud_file = (fs::path(output_dir) / "pointcloud.csv").string();
    string trajectory_file = (fs::path(output_dir) / "camera_trajectory.csv").string();
    string original_video_copy = (fs::path(output_dir) / fs::path(videoFile).filename()).string();
    string mapviewer_video_copy = (fs::path(output_dir) / fs::path(mapviewerVideoFile).filename()).string();
    cout << "输出目录: " << output_dir << endl;

    
    auto start = chrono::system_clock::now();
    int frame_count = 0;
    int roi_success = 0;
    
    cv::namedWindow("Frame with ROI", cv::WINDOW_NORMAL);
    cv::resizeWindow("Frame with ROI", 1280, 720);
    
    cout << "Preloading ROI data..." << endl;
    for (int i = 1; i <= min(100, total_frames); i++) {
        char npy_file[256];
        snprintf(npy_file, sizeof(npy_file), "%s/frame_%06d.npy", npy_dir.c_str(), i);
        vector<cv::Point2f> corners;
        if (loadCornerPointsFromNpyPython(npy_file, corners)) {
            roi_cache[i] = corners;
        }
        if (i % 20 == 0) cout << "  Loaded " << i << " frames..." << endl;
    }
    cout << "Preloaded " << roi_cache.size() << " ROI regions" << endl;
    
    while (true) {
        cv::Mat frame;
        cap >> frame;
        
        if (frame.empty()) {
            break;
        }
        
        frame_count++;
        
        vector<cv::Point2f> roi_quad;
        bool has_roi = false;
        
        if (roi_cache.find(frame_count) != roi_cache.end()) {
            roi_quad = roi_cache[frame_count];
            has_roi = true;
            roi_success++;
        } else {
            char npy_file[256];
            snprintf(npy_file, sizeof(npy_file), "%s/frame_%06d.npy", npy_dir.c_str(), frame_count);
            if (loadCornerPointsFromNpyPython(npy_file, roi_quad)) {
                roi_cache[frame_count] = roi_quad;
                has_roi = true;
                roi_success++;
            }
        }
        
        cv::Mat slam_frame = frame;
        
        // auto now = chrono::system_clock::now();
        // auto timestamp = chrono::duration_cast<chrono::milliseconds>(now - start);
        
        try {
            SLAM.TrackMonocular(slam_frame, (double)frame_count);
        } catch (const exception& e) {
            cerr << "SLAM error: " << e.what() << endl;
            break;
        }
        
        cv::Mat display = frame.clone();
        if (has_roi) {
            drawROI(display, roi_quad);
            cv::putText(display, "ROI: ON", cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(display, "ROI: OFF", cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        }
        
        char text[64];
        snprintf(text, sizeof(text), "Frame %d/%d", frame_count, total_frames);
        cv::putText(display, text, cv::Point(10, frame_height - 20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        cv::imshow("Frame with ROI", display);
        
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            break;
        }
    
        
        if (frame_count % 50 == 0) {
            cout << "Frame " << frame_count << "/" << total_frames 
                 << " (ROI: " << roi_success << ")" << endl;
        }
    }
    
    cout << "\n==========================" << endl;
    
    cout << "Processing complete!" << endl;
    cout << "Processed: " << frame_count << " frames" << endl;
    cout << "ROI loaded: " << roi_success << " ("
         << (100.0 * roi_success / frame_count) << "%)" << endl;
    cout << "==========================" << endl;

    // ========================================
    // 保存点云和相机位姿
    // ========================================
    SavePointCloud(SLAM, pointcloud_file);
    SaveCameraTrajectory(SLAM, trajectory_file);
    
    // 停止 MapViewer 录制
    cout << "\n停止 MapViewer 录制..." << endl;
    system("pkill -INT ffmpeg");
    std::this_thread::sleep_for(std::chrono::seconds(2)); // 等待 ffmpeg 完成写入

    CopyIfExists(videoFile, original_video_copy, "原视频");
    CopyIfExists(mapviewerVideoFile, mapviewer_video_copy, "MapViewer 视频");

    cout << "\n输出目录: " << output_dir << endl;
    cout << "已保存文件:" << endl;
    cout << "  • 点云: " << pointcloud_file << endl;
    cout << "  • 轨迹: " << trajectory_file << endl;
    cout << "  • 原视频: " << original_video_copy << endl;
    cout << "  • MapViewer 视频: " << mapviewer_video_copy << endl;

    SLAM.Shutdown();
    return 0;
}
