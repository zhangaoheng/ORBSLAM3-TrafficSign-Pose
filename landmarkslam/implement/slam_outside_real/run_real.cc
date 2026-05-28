/**
 * run_real.cc — SLAM 处理 (单目 / RGB-D / RGB-D + IMU)
 *
 * 用法:
 *   ./run_real mono          vocab config seq_path times.txt     [traj_name]
 *   ./run_real rgbd          vocab config seq_path assoc.txt     [traj_name]
 *   ./run_real rgbd_inertial vocab config seq_path assoc.txt imu.txt [traj_name]
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <sstream>

#include <opencv2/core/core.hpp>

#include <System.h>
#include <ImuTypes.h>

using namespace std;

static bool startsWith(const string &s, const string &p) {
    return s.size() >= p.size() && s.substr(0, p.size()) == p;
}

static void LoadAssociations(const string &path,
    vector<string> &rgb, vector<string> &depth, vector<double> &ts)
{
    ifstream fs(path);
    if (!fs.is_open()) { cerr << "Can't open: " << path << endl; exit(-1); }
    string line;
    while (getline(fs, line)) {
        if (line.empty() || startsWith(line, "#")) continue;
        stringstream ss(line);
        double t1, t2;
        string f1, f2;
        ss >> t1 >> f1 >> t2 >> f2;
        ts.push_back(t1);
        rgb.push_back(f1);
        depth.push_back(f2);
    }
}

static void LoadIMU(const string &path,
    vector<double> &ts, vector<cv::Point3f> &acc, vector<cv::Point3f> &gyro)
{
    ifstream fs(path);
    if (!fs.is_open()) { cerr << "Can't open: " << path << endl; exit(-1); }
    ts.reserve(5000);
    acc.reserve(5000);
    gyro.reserve(5000);

    // 判断格式：CSV (timestamp_ms,sensor_type,x,y,z) 或 旧格式 (空格分隔)
    string first;
    getline(fs, first);
    bool csv_format = (first.find(',') != string::npos);

    if (csv_format) {
        // CSV 格式: timestamp_ms,sensor_type,x,y,z
        string line;
        while (getline(fs, line)) {
            if (line.empty()) continue;
            stringstream ss(line);
            string ts_str, type, x_str, y_str, z_str;
            getline(ss, ts_str, ',');
            getline(ss, type, ',');
            getline(ss, x_str, ',');
            getline(ss, y_str, ',');
            getline(ss, z_str, ',');
            double t = stod(ts_str) / 1000.0;  // ms → s
            cv::Point3f val(stof(x_str), stof(y_str), stof(z_str));
            if (type == "accel")      { ts.push_back(t); acc.push_back(val); }
            else if (type == "gyro")  { ts.push_back(t); gyro.push_back(val); }
        }
    } else {
        // 旧格式: timestamp_ns w_x w_y w_z a_x a_y a_z (空格分隔)
        auto parse = [&](const string &l) {
            if (l.empty() || l[0] == '#') return;
            stringstream ss(l);
            double raw[7];
            for (auto &v : raw) ss >> v;
            ts.push_back(raw[0] / 1e9);
            gyro.emplace_back(raw[1], raw[2], raw[3]);
            acc.emplace_back(raw[4], raw[5], raw[6]);
        };
        parse(first);
        string line;
        while (getline(fs, line)) parse(line);
    }
    cout << "IMU measurements: " << ts.size() << " (" << acc.size() << " accel, " << gyro.size() << " gyro)" << endl;
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        cerr << "Usage:\n"
             << "  mono:          ./run_real mono          vocab config seq_path times.txt     [traj_name]\n"
             << "  rgbd:          ./run_real rgbd          vocab config seq_path assoc.txt     [traj_name]\n"
             << "  rgbd_inertial: ./run_real rgbd_inertial vocab config seq_path assoc.txt imu.txt [traj_name]\n";
        return 1;
    }

    string mode = argv[1];
    string traj_name = (argc >= 7 && mode != "rgbd_inertial") ? argv[6] : "CameraTrajectory";
    if (mode == "rgbd_inertial" && argc >= 8) traj_name = argv[7];
    string seq_dir = argv[4];

    ORB_SLAM3::System::eSensor sensor;
    if (mode == "mono")            sensor = ORB_SLAM3::System::MONOCULAR;
    else if (mode == "rgbd")       sensor = ORB_SLAM3::System::RGBD;
    else if (mode == "rgbd_inertial") sensor = ORB_SLAM3::System::IMU_RGBD;
    else { cerr << "Unknown mode: " << mode << endl; return 1; }

    // ---- 加载图像 ----
    vector<string> rgb_files, depth_files;
    vector<double> cam_ts;

    if (mode == "mono") {
        ifstream fs(argv[5]);
        if (!fs.is_open()) { cerr << "Can't open: " << argv[5] << endl; return 1; }
        string line, f1;
        double t1;
        while (getline(fs, line)) {
            if (line.empty() || startsWith(line, "#")) continue;
            stringstream ss(line);
            ss >> t1 >> f1;
            cam_ts.push_back(t1);
            rgb_files.push_back(f1);
        }
    } else {
        LoadAssociations(argv[5], rgb_files, depth_files, cam_ts);
    }

    // ---- 加载 IMU (rgbd_inertial) ----
    vector<double> imu_ts;
    vector<cv::Point3f> acc, gyro;
    int imu_idx = 0;
    if (mode == "rgbd_inertial") {
        LoadIMU(argv[6], imu_ts, acc, gyro);
    }

    int n = rgb_files.size();
    if (n == 0) { cerr << "No images found." << endl; return 1; }

    // ---- 初始化 SLAM ----
    ORB_SLAM3::System SLAM(argv[2], argv[3], sensor, true);
    float scale = SLAM.GetImageScale();
    cout << "Sensor mode: " << mode << endl;
    cout << "Frames: " << n << endl;

    vector<float> track_times;
    track_times.reserve(n);

    for (int i = 0; i < n; ++i) {
        double t = cam_ts[i];

        cv::Mat rgb = cv::imread(seq_dir + "/" + rgb_files[i], cv::IMREAD_UNCHANGED);
        if (rgb.empty()) {
            cerr << "⚠ Skipping corrupted frame [" << i << "]: " << rgb_files[i] << endl;
            continue;
        }

        auto t1 = chrono::steady_clock::now();

        if (mode == "mono") {
            if (scale != 1.f)
                cv::resize(rgb, rgb, cv::Size(rgb.cols * scale, rgb.rows * scale));
            SLAM.TrackMonocular(rgb, t);
        } else {
            cv::Mat depth = cv::imread(seq_dir + "/" + depth_files[i], cv::IMREAD_UNCHANGED);
            if (scale != 1.f) {
                cv::resize(rgb,   rgb,   cv::Size(rgb.cols   * scale, rgb.rows   * scale));
                cv::resize(depth, depth, cv::Size(depth.cols * scale, depth.rows * scale));
            }

            if (mode == "rgbd_inertial") {
                vector<ORB_SLAM3::IMU::Point> vImuMeas;
                while (imu_idx < (int)imu_ts.size() && imu_ts[imu_idx] <= t) {
                    vImuMeas.emplace_back(
                        acc[imu_idx].x, acc[imu_idx].y, acc[imu_idx].z,
                        gyro[imu_idx].x, gyro[imu_idx].y, gyro[imu_idx].z,
                        imu_ts[imu_idx]);
                    ++imu_idx;
                }
                SLAM.TrackRGBD(rgb, depth, t, vImuMeas);
            } else {
                SLAM.TrackRGBD(rgb, depth, t);
            }
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