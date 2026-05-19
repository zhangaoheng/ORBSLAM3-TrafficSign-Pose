/**
 * run_real.cc — RealSense D456 实采数据 SLAM 处理 (单目 / RGB-D)
 *
 * 用法:
 *   ./run_real mono  vocab config seq_path times.txt  [traj_name]
 *   ./run_real rgbd  vocab config seq_path assoc.txt  [traj_name]
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <sstream>

#include <opencv2/core/core.hpp>

#include <System.h>

using namespace std;

static bool startsWith(const string &s, const string &p) {
    return s.size() >= p.size() && s.substr(0, p.size()) == p;
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        cerr << "Usage:\n"
             << "  mono: ./run_real mono  vocab config seq_path times.txt [traj_name]\n"
             << "  rgbd: ./run_real rgbd  vocab config seq_path assoc.txt [traj_name]\n";
        return 1;
    }

    string mode = argv[1];  // "mono" or "rgbd"
    string traj_name = (argc >= 7) ? argv[6] : "CameraTrajectory";
    string seq_dir = argv[4];

    // ---- 加载图像列表 ----
    ifstream fs(argv[5]);
    if (!fs.is_open()) { cerr << "Can't open: " << argv[5] << endl; return 1; }

    vector<string> img_files_rgb, img_files_depth;
    vector<double> timestamps;
    string line;
    while (getline(fs, line)) {
        if (line.empty() || startsWith(line, "#")) continue;
        stringstream ss(line);
        double t1, t2;
        string f1, f2;
        if (mode == "mono") {
            ss >> t1 >> f1;
            timestamps.push_back(t1);
            img_files_rgb.push_back(f1);
        } else { // rgbd
            ss >> t1 >> f1 >> t2 >> f2;
            timestamps.push_back(t1);
            img_files_rgb.push_back(f1);
            img_files_depth.push_back(f2);
        }
    }
    int n = img_files_rgb.size();
    if (n == 0) { cerr << "No images found." << endl; return 1; }

    // ---- 初始化 SLAM ----
    auto sensor = (mode == "mono") ? ORB_SLAM3::System::MONOCULAR : ORB_SLAM3::System::RGBD;
    ORB_SLAM3::System SLAM(argv[2], argv[3], sensor, true);
    float scale = SLAM.GetImageScale();
    cout << "Sensor mode: " << (mode == "mono" ? "Monocular" : "RGB-D") << endl;
    cout << "Frames: " << n << endl;

    vector<float> track_times;
    track_times.reserve(n);

    for (int i = 0; i < n; ++i) {
        double t = timestamps[i];

        cv::Mat rgb = cv::imread(seq_dir + "/" + img_files_rgb[i], cv::IMREAD_UNCHANGED);
        if (rgb.empty()) {
            cerr << "⚠ Skipping corrupted frame [" << i << "]: " << img_files_rgb[i] << endl;
            continue;
        }

        auto t1 = chrono::steady_clock::now();

        if (mode == "mono") {
            if (scale != 1.f)
                cv::resize(rgb, rgb, cv::Size(rgb.cols * scale, rgb.rows * scale));
            SLAM.TrackMonocular(rgb, t);
        } else {
            cv::Mat depth = cv::imread(seq_dir + "/" + img_files_depth[i], cv::IMREAD_UNCHANGED);
            if (scale != 1.f) {
                cv::resize(rgb, rgb, cv::Size(rgb.cols * scale, rgb.rows * scale));
                cv::resize(depth, depth, cv::Size(depth.cols * scale, depth.rows * scale));
            }
            SLAM.TrackRGBD(rgb, depth, t);
        }

        auto t2 = chrono::steady_clock::now();
        track_times.push_back(chrono::duration_cast<chrono::duration<double>>(t2 - t1).count());
    }

    SLAM.Shutdown();

    // ---- 保存轨迹 ----
    SLAM.SaveTrajectoryTUM("AllFrames_" + traj_name + ".txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrames_" + traj_name + ".txt");
    cout << "Done. Trajectories saved." << endl;

    // ---- 耗时统计 ----
    sort(track_times.begin(), track_times.end());
    double total = 0;
    for (double d : track_times) total += d;
    cout << "Frames: " << n
         << "  median: " << track_times[n / 2] * 1000 << " ms"
         << "  mean: " << (total / n) * 1000 << " ms" << endl;

    return 0;
}
