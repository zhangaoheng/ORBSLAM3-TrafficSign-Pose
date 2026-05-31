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
#include <ctime>
#include <iomanip>
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <opencv2/core/core.hpp>

#include <System.h>
#include <Atlas.h>
#include <MapPoint.h>
#include <ImuTypes.h>

using namespace std;

static ofstream g_log;  // 全局日志文件（每次运行覆盖）
static string timestamp_now();  // 前向声明

// 保存当前轨迹（崩溃前调用）
static void save_crash_trajectory(const string& run_dir, const string& traj_name,
                                  ORB_SLAM3::System* slam, bool is_mono) {
    if (!slam) return;
    string path = run_dir + "/crash_trajectory.txt";
    if (is_mono) {
        // KITTI/TUM 格式均对单目报错；EuRoC 和 KeyFrame 格式可工作
        slam->SaveTrajectoryEuRoC(path);
    } else {
        slam->SaveTrajectoryTUM(path);
    }
}

static ORB_SLAM3::System* g_slam = nullptr;
static string g_run_dir;
static string g_traj_name;
static bool g_is_mono;

// 崩溃信号处理器（SIGSEGV / SIGABRT）
static void crash_handler(int sig) {
    const char* name = (sig == SIGSEGV) ? "SIGSEGV" : "SIGABRT";
    cerr << "\n[FATAL] " << name << " at " << timestamp_now() << endl;
    void* buf[32];
    int n = backtrace(buf, 32);
    backtrace_symbols_fd(buf, n, STDERR_FILENO);
    if (!g_run_dir.empty() && g_slam) {
        cerr << "[FATAL] 正在保存崩溃前轨迹..." << endl;
        save_crash_trajectory(g_run_dir, g_traj_name, g_slam, g_is_mono);
        cerr << "[FATAL] 轨迹已保存: crash_trajectory.txt" << endl;
    }
    if (g_log.is_open()) {
        g_log << "[FATAL] " << name << " at " << timestamp_now() << endl;
        g_log.close();
    }
    _exit(sig == SIGSEGV ? 139 : 134);
}

static string timestamp_now() {
    auto t = chrono::system_clock::to_time_t(chrono::system_clock::now());
    stringstream ss;
    ss << put_time(localtime(&t), "%H:%M:%S");
    return ss.str();
}

#define LOG(msg) do { \
    string _ts = timestamp_now(); \
    cout << "[" << _ts << "] " << msg << endl; \
    if (g_log.is_open()) { g_log << "[" << _ts << "] " << msg << endl; g_log.flush(); } \
} while(0)

#define LOG_RAW(msg) do { \
    cout << msg << endl; \
    if (g_log.is_open()) { g_log << msg << endl; g_log.flush(); } \
} while(0)

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
    LOG("IMU: " << ts.size() << " (" << acc.size() << " accel, " << gyro.size() << " gyro)");
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        cerr << "Usage:\n"
             << "  mono:               ./run_real mono               vocab config seq_path times.txt         [traj_name]\n"
             << "  imu_monocular:      ./run_real imu_monocular      vocab config seq_path times.txt imu.txt [traj_name]\n"
             << "  rgbd:               ./run_real rgbd               vocab config seq_path assoc.txt         [traj_name]\n"
             << "  rgbd_inertial:      ./run_real rgbd_inertial      vocab config seq_path assoc.txt imu.txt [traj_name]\n";
        return 1;
    }

    string mode = argv[1];
    string traj_name = (argc >= 2) ? argv[argc - 1] : "CameraTrajectory";
    string seq_dir = argv[4];

    // ---- 创建本次运行的输出目录 (runs/YYYY-MM-DD_HH-MM-SS_mode/) ----
    time_t now = time(nullptr);
    struct tm* timeinfo = localtime(&now);
    char run_dir_buf[64];
    strftime(run_dir_buf, sizeof(run_dir_buf), "%Y-%m-%d_%H-%M-%S", timeinfo);
    string run_dir = seq_dir + "/runs/" + string(run_dir_buf) + "_" + mode;
    mkdir((seq_dir + "/runs").c_str(), 0755);
    mkdir(run_dir.c_str(), 0755);

    // ---- 注册崩溃信号处理器（segfault + Sophus abort） ----
    signal(SIGSEGV, crash_handler);
    signal(SIGABRT, crash_handler);

    // ---- 打开日志文件（保存到 runs 目录） ----
    string log_path = run_dir + "/" + traj_name + "_run.log";
    g_log.open(log_path, ios::trunc);
    LOG("========================================");
    LOG("run_real 启动  模式: " << mode);
    LOG("数据目录: " << seq_dir);
    LOG("输出目录: " << run_dir);
    LOG("========================================");

    ORB_SLAM3::System::eSensor sensor;
    if (mode == "mono")              sensor = ORB_SLAM3::System::MONOCULAR;
    else if (mode == "imu_monocular") sensor = ORB_SLAM3::System::IMU_MONOCULAR;
    else if (mode == "rgbd")         sensor = ORB_SLAM3::System::RGBD;
    else if (mode == "rgbd_inertial") sensor = ORB_SLAM3::System::IMU_RGBD;
    else { LOG("❌ Unknown mode: " << mode); return 1; }

    // ---- 加载图像 ----
    vector<string> rgb_files, depth_files;
    vector<double> cam_ts;

    if (mode == "mono" || mode == "imu_monocular") {
        ifstream fs(argv[5]);
        if (!fs.is_open()) { LOG("❌ Can't open: " << argv[5]); return 1; }
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

    // ---- 加载 IMU (imu_monocular / rgbd_inertial) ----
    vector<double> imu_ts;
    vector<cv::Point3f> acc, gyro;
    int imu_idx = 0;
    if (mode == "rgbd_inertial") {
        LoadIMU(argv[6], imu_ts, acc, gyro);
    } else if (mode == "imu_monocular") {
        LoadIMU(argv[6], imu_ts, acc, gyro);
    }

    int n = rgb_files.size();
    if (n == 0) { LOG("❌ No images found."); return 1; }
    LOG("总帧数: " << n << "   IMU: " << imu_ts.size() << "条");

    // ---- 初始化 SLAM ----
    // true = 打开 Pangolin 可视化窗口
    LOG("Viewer: disabled");
    ORB_SLAM3::System SLAM(argv[2], argv[3], sensor, false);
    // 全局指针供崩溃处理器使用
    g_slam = &SLAM;
    g_run_dir = run_dir;
    g_traj_name = traj_name;
    g_is_mono = (sensor == ORB_SLAM3::System::MONOCULAR || sensor == ORB_SLAM3::System::IMU_MONOCULAR);
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
        } else if (mode == "imu_monocular") {
            if (scale != 1.f)
                cv::resize(rgb, rgb, cv::Size(rgb.cols * scale, rgb.rows * scale));
            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            while (imu_idx < (int)imu_ts.size() && imu_ts[imu_idx] <= t) {
                vImuMeas.emplace_back(
                    acc[imu_idx].x, acc[imu_idx].y, acc[imu_idx].z,
                    gyro[imu_idx].x, gyro[imu_idx].y, gyro[imu_idx].z,
                    imu_ts[imu_idx]);
                ++imu_idx;
            }
            // IMU 数据还没到就跳过此帧
            if (vImuMeas.empty()) {
                LOG("⚠ 跳过帧[" << i << "]: 无 IMU 数据 (t=" << t << ")");
                continue;
            }
            SLAM.TrackMonocular(rgb, t, vImuMeas);
            // 检测IMU是否已初始化: 首次达到500帧进度时标记初始化完成
            if (i == 201) {
                LOG("🚀 IMU初始化窗口已过, 当前地图KF数=" << SLAM.GetAtlas()->GetCurrentMap()->GetAllKeyFrames().size());
            }
        } else {
            cv::Mat depth = cv::imread(seq_dir + "/" + depth_files[i], cv::IMREAD_UNCHANGED);
            if (scale != 1.f) {
                cv::resize(rgb,   rgb,   cv::Size(rgb.cols   * scale, rgb.rows   * scale));
                cv::resize(depth, depth, cv::Size(depth.cols * scale, depth.rows * scale));
            }
            // RealSense D456: 保持原始深度值，由 SLAM 内部处理

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

        // 每 500 帧打一次进度 + checkpoint
        if ((i + 1) % 500 == 0) {
            LOG("进度: " << (i + 1) << "/" << n << " 帧  ("
                << (100.0 * (i + 1) / n) << "%)");
            // 每 1000 帧存 checkpoint（崩了也有数据）
            if ((i + 1) % 1000 == 0) {
                string ckpt = run_dir + "/trajectory.txt";
                if (g_is_mono) SLAM.SaveTrajectoryEuRoC(ckpt);
                else           SLAM.SaveTrajectoryTUM(ckpt);
            }
            // 打印地图状态
            auto atlas = SLAM.GetAtlas();
            if (atlas) {
                int nm = atlas->CountMaps();
                LOG("  地图数=" << nm);
            }
        }
    }

    LOG("所有帧处理完毕，共 " << n << " 帧");

    // ---- 保存地图点云 (PLY) ----
    {
        string ply_path = run_dir + "/map_points.ply";
        LOG("正在保存地图点云...");
        auto atlas = SLAM.GetAtlas();
        if (atlas) {
            auto allMPs = atlas->GetAllMapPoints();
            ofstream ply(ply_path);
            vector<Eigen::Vector3f> valid_pts;
            for (auto pMP : allMPs) {
                if (pMP && !pMP->isBad()) {
                    valid_pts.push_back(pMP->GetWorldPos());
                }
            }
            ply << "ply\nformat ascii 1.0\nelement vertex " << valid_pts.size()
                << "\nproperty float x\nproperty float y\nproperty float z\n"
                << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
            for (auto& p : valid_pts) {
                ply << p(0) << " " << p(1) << " " << p(2) << " 200 200 200\n";
            }
            ply.close();
            LOG("✅ 点云(PLY): " << ply_path << "  (" << valid_pts.size() << " 个点)");
        }
    }

    SLAM.Shutdown();

    // ---- 保存轨迹到 runs 目录 ----
    string traj_file = run_dir + "/trajectory.txt";
    string traj_kf   = run_dir + "/trajectory_kf.txt";
    LOG("正在保存轨迹文件...");

    // 保存每个地图的独立轨迹，用于后处理拼接
    {
        auto atlas = SLAM.GetAtlas();
        if (atlas) {
            auto allMaps = atlas->GetAllMaps();
            int map_count = 0;
            for (auto pMap : allMaps) {
                if (!pMap) continue;
                auto kfs = pMap->GetAllKeyFrames();
                if (kfs.empty()) continue;
                sort(kfs.begin(), kfs.end(), ORB_SLAM3::KeyFrame::lId);

                string map_file = run_dir + "/map_" + to_string(pMap->GetId()) + "_trajectory.txt";
                ofstream f(map_file);
                f << fixed;
                Sophus::SE3f Twb = kfs[0]->GetPoseInverse();
                for (auto pKF : kfs) {
                    if (pKF->isBad()) continue;
                    Sophus::SE3f Twc = pKF->GetPoseInverse();
                    Eigen::Quaternionf q = Twc.unit_quaternion();
                    Eigen::Vector3f t = Twc.translation();
                    f << setprecision(6) << pKF->mTimeStamp << " "
                      << setprecision(7) << t(0) << " " << t(1) << " " << t(2) << " "
                      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
                }
                f.close();
                LOG("✅ 地图" << pMap->GetId() << "轨迹: " << map_file << "  (" << kfs.size() << " KFs)");
                map_count++;
            }
            LOG("✅ 共保存 " << map_count << " 个地图的轨迹");
        }
    }

    if (sensor == ORB_SLAM3::System::MONOCULAR || sensor == ORB_SLAM3::System::IMU_MONOCULAR) {
        SLAM.SaveTrajectoryEuRoC(traj_file);
        SLAM.SaveKeyFrameTrajectoryTUM(traj_kf);
        LOG("✅ 轨迹(EuRoC): " << traj_file);
    } else {
        SLAM.SaveTrajectoryTUM(traj_file);
        SLAM.SaveKeyFrameTrajectoryTUM(traj_kf);
        LOG("✅ 轨迹(TUM): " << traj_file);
    }
    LOG("✅ 关键帧轨迹: " << traj_kf);

    // ---- 复制配置文件到 runs 目录（方便复现） ----
    ifstream cfg_src(argv[3], ios::binary);
    ofstream cfg_dst(run_dir + "/config.yaml", ios::binary);
    if (cfg_src && cfg_dst) { cfg_dst << cfg_src.rdbuf(); }
    sort(track_times.begin(), track_times.end());
    double total = 0;
    for (double d : track_times) total += d;
    LOG("📊 耗时统计: 总帧数=" << n
        << "  中位数=" << track_times[n / 2] * 1000 << " ms"
        << "  均值=" << (total / n) * 1000 << " ms");

    // ---- 关闭日志 ----
    LOG("========================================");
    LOG("🏁 run_real 结束");
    LOG("========================================");
    g_log.close();

    return 0;
}