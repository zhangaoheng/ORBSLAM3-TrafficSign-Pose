/**
 * run_mono_imu_140500.cc
 *
 * ORB-SLAM3 单目+IMU 运行程序（EuRoC 格式）
 *
 * 用法:
 *   ./run_mono_imu_140500 vocab.yaml config.yaml euroc_path times.txt trajectory_name
 *
 * 依赖:
 *   - ORB-SLAM3 (System.h, ImuTypes.h)
 *   - OpenCV, Eigen3, Pangolin
 *
 * 数据集: 20260613_140500 (Intel RealSense D456, 848x480, 单目+IMU)
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>

#include <opencv2/core/core.hpp>

#include <System.h>
#include <ImuTypes.h>

using namespace std;

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps,
             vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

int main(int argc, char *argv[])
{
    // ./run_mono_imu_140500 vocab config euroc_path times.txt [trajectory_name]
    if (argc < 5)
    {
        cerr << endl
             << "Usage: ./run_mono_imu_140500 path_to_vocabulary path_to_settings "
             << "path_to_euroc_sequence path_to_times_file [trajectory_output_name]"
             << endl;
        return 1;
    }

    string trajectory_name = "trajectory_140500";
    if (argc >= 6)
        trajectory_name = string(argv[5]);

    // ---- 加载数据 ----
    string pathSeq = string(argv[3]);
    string pathTimes = string(argv[4]);
    string pathCam0 = pathSeq + "/mav0/cam0/data";
    string pathImu = pathSeq + "/mav0/imu0/data.csv";

    vector<string> vstrImages;
    vector<double> vTimestampsCam;
    LoadImages(pathCam0, pathTimes, vstrImages, vTimestampsCam);

    vector<double> vTimestampsImu;
    vector<cv::Point3f> vAcc, vGyro;
    LoadIMU(pathImu, vTimestampsImu, vAcc, vGyro);

    int nImages = (int)vstrImages.size();
    int nImu = (int)vTimestampsImu.size();

    if (nImages <= 0 || nImu <= 0)
    {
        cerr << "ERROR: Failed to load data. Images=" << nImages
             << ", IMU measurements=" << nImu << endl;
        return 1;
    }

    // 找到第一个不早于第一帧图像的 IMU 测量
    int first_imu = 0;
    while (first_imu < nImu && vTimestampsImu[first_imu] <= vTimestampsCam[0])
        first_imu++;
    if (first_imu > 0)
        first_imu--; // 包含前一帧

    cout << endl
         << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl;
    cout << "IMU measurements: " << nImu << endl;
    cout << "First camera timestamp: " << vTimestampsCam[0] << " s" << endl;
    cout << "Last camera timestamp: " << vTimestampsCam[nImages - 1] << " s" << endl;
    cout << "-------" << endl;

    // ---- 创建 SLAM 系统 (单目+IMU，禁用 Viewer 以支持无 GUI 环境) ----
    ORB_SLAM3::System SLAM(argv[1], argv[2],
                           ORB_SLAM3::System::IMU_MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    // ---- 主循环 ----
    vector<float> vTimesTrack(nImages, 0);
    double ttrack_tot = 0;

    for (int ni = 0; ni < nImages; ni++)
    {
        // 读取图像
        cv::Mat im = cv::imread(vstrImages[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestampsCam[ni];

        if (im.empty())
        {
            cerr << "Failed to load image: " << vstrImages[ni] << endl;
            return 1;
        }

        // 缩放（如果需要）
        if (imageScale != 1.f)
        {
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
        }

        // 收集当前帧之前的 IMU 测量
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        if (ni > 0)
        {
            while (first_imu < nImu && vTimestampsImu[first_imu] <= tframe)
            {
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(
                    vAcc[first_imu].x, vAcc[first_imu].y, vAcc[first_imu].z,
                    vGyro[first_imu].x, vGyro[first_imu].y, vGyro[first_imu].z,
                    vTimestampsImu[first_imu]));
                first_imu++;
            }
        }

        // 跟踪
        auto t1 = chrono::steady_clock::now();
        SLAM.TrackMonocular(im, tframe, vImuMeas);
        auto t2 = chrono::steady_clock::now();

        double ttrack = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;
        ttrack_tot += ttrack;

        // 等待以维持实时
        if (ni < nImages - 1)
        {
            double T = vTimestampsCam[ni + 1] - tframe;
            if (ttrack < T)
                usleep((T - ttrack) * 1e6);
        }

        if ((ni + 1) % 500 == 0)
            cout << "  Processed " << (ni + 1) << "/" << nImages << " images" << endl;
    }

    // ---- 关闭并保存 ----
    SLAM.Shutdown();

    // 保存轨迹
    string allframes_path = "AllFrames_" + trajectory_name + ".txt";
    string keyframes_path = "KeyFrames_" + trajectory_name + ".txt";

    SLAM.SaveTrajectoryTUM(allframes_path);
    cout << "✅ All frames trajectory saved to: " << allframes_path << endl;

    SLAM.SaveKeyFrameTrajectoryTUM(keyframes_path);
    cout << "✅ Keyframe trajectory saved to: " << keyframes_path << endl;

    // 统计
    sort(vTimesTrack.begin(), vTimesTrack.end());
    cout << "-------" << endl;
    cout << "Total tracking time: " << ttrack_tot << " s" << endl;
    cout << "Median tracking time: " << vTimesTrack[nImages / 2] << " s" << endl;
    cout << "Mean tracking time: " << ttrack_tot / nImages << " s" << endl;
    cout << "\n[INFO] Processing finished." << endl;

    return 0;
}

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);

    while (!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            // times.txt 中的每行是一个纳秒时间戳
            double t_ns;
            ss >> t_ns;

            // 构建图像路径: path/timestamp_ns.png
            ostringstream oss;
            oss << strImagePath << "/" << (long long)t_ns << ".png";
            vstrImages.push_back(oss.str());

            // 转换为秒
            vTimeStamps.push_back(t_ns / 1e9);
        }
    }
    cout << "Loaded " << vTimeStamps.size() << " images from " << strImagePath << endl;
}

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps,
             vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);

    string line;
    int line_count = 0;
    while (getline(fImu, line))
    {
        line_count++;
        if (line.empty() || line[0] == '#')
            continue;

        // CSV 格式: timestamp_ns,wx,wy,wz,ax,ay,az
        stringstream ss(line);
        string item;
        double data[7];
        int count = 0;
        while (getline(ss, item, ',') && count < 7)
        {
            data[count++] = stod(item);
        }
        if (count < 7)
        {
            cerr << "WARNING: Malformed IMU line " << line_count << ": " << line << endl;
            continue;
        }

        // 时间戳转为秒
        vTimeStamps.push_back(data[0] / 1e9);
        vGyro.push_back(cv::Point3f((float)data[1], (float)data[2], (float)data[3]));
        vAcc.push_back(cv::Point3f((float)data[4], (float)data[5], (float)data[6]));
    }
    cout << "Loaded " << vTimeStamps.size() << " IMU measurements from " << strImuPath << endl;
}
