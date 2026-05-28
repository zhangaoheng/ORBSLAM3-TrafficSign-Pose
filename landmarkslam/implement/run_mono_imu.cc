/**
* This file is part of ORB-SLAM3
* (Modified: IMU_RGBD mode, CSV/space IMU format support, custom trajectory path)
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <ctime>
#include <sstream>

#include<opencv2/core/core.hpp>

#include<System.h>
#include<ImuTypes.h>

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

int main(int argc, char **argv)
{
    // ./run_mono_imu vocab config seq_path associations imu_path [trajectory_name]
    if(argc < 6)
    {
        cerr << endl << "Usage: ./run_mono_imu path_to_vocabulary path_to_settings path_to_sequence path_to_associations path_to_imu [trajectory_output_name]" << endl;
        return 1;
    }

    string trajectory_file = "CameraTrajectory";
    if(argc == 7)
        trajectory_file = string(argv[6]);

    // 加载图像名称和时间戳
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestampsCam;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestampsCam);

    // 加载 IMU 数据
    vector<double> vTimestampsImu;
    vector<cv::Point3f> vAcc;
    vector<cv::Point3f> vGyro;
    string strImuPath = string(argv[5]);
    LoadIMU(strImuPath, vTimestampsImu, vAcc, vGyro);

    int nImages = vstrImageFilenamesRGB.size();
    if(nImages<=0)
    {
        cerr << "ERROR: No images found." << endl;
        return 1;
    }

    // 创建 SLAM 系统 (IMU_RGBD 模式)
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_RGBD, true);
    float imageScale = SLAM.GetImageScale();

    cv::Mat imRGB, imD;
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl;
    cout << "IMU measurements: " << vTimestampsImu.size() << endl << endl;

    int first_imu = 0;

    for(int ni=0; ni<nImages; ni++)
    {
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED);
        imD   = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestampsCam[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

        // 收集当前帧之前的 IMU 测量
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        if(ni > 0)
        {
            while(first_imu < (int)vTimestampsImu.size() && vTimestampsImu[first_imu] <= tframe)
            {
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(
                    vAcc[first_imu].x, vAcc[first_imu].y, vAcc[first_imu].z,
                    vGyro[first_imu].x, vGyro[first_imu].y, vGyro[first_imu].z,
                    vTimestampsImu[first_imu]));
                first_imu++;
            }
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        SLAM.TrackRGBD(imRGB, imD, tframe, vImuMeas);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        vTimesTrack[ni]=ttrack;
    }

    SLAM.Shutdown();

    // 保存轨迹
    string output_dir = "./";
    string base_name = trajectory_file;
    size_t pos = base_name.find_last_of("/\\");
    if(pos != string::npos)
    {
        output_dir = base_name.substr(0, pos+1);
        base_name = base_name.substr(pos+1);
    }

    string allframes_path = output_dir + "AllFrames_" + base_name + ".txt";
    SLAM.SaveTrajectoryTUM(allframes_path);
    cout << "✅ All frames trajectory saved to: " << allframes_path << endl;

    string keyframes_path = output_dir + "KeyFrames_" + base_name + ".txt";
    SLAM.SaveKeyFrameTrajectoryTUM(keyframes_path);
    cout << "✅ Keyframe trajectory saved to: " << keyframes_path << endl;

    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    cout << "\n[INFO] Program finished cleanly." << endl;
    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    if(!fAssociation.is_open())
    {
        cerr << "Failed to open association file: " << strAssociationFilename << endl;
        exit(-1);
    }
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
}

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    if(!fImu.is_open())
    {
        cerr << "Failed to open IMU file: " << strImuPath << endl;
        exit(-1);
    }

    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);

    string first;
    getline(fImu, first);
    bool csv_format = (first.find("timestamp") != string::npos);

    if(csv_format)
    {
        // CSV 格式: timestamp_ms,sensor_type,x,y,z (accel/gyro 分两行)
        string line;
        while(getline(fImu, line))
        {
            if(line.empty()) continue;
            stringstream ss(line);
            string ts_str, type, x_str, y_str, z_str;
            getline(ss, ts_str, ',');
            getline(ss, type, ',');
            getline(ss, x_str, ',');
            getline(ss, y_str, ',');
            getline(ss, z_str, ',');
            double t = stod(ts_str) / 1000.0;
            cv::Point3f val(stof(x_str), stof(y_str), stof(z_str));
            if(type == "accel")
            {
                vTimeStamps.push_back(t);
                vAcc.push_back(val);
            }
            else if(type == "gyro")
            {
                vTimeStamps.push_back(t);
                vGyro.push_back(val);
            }
        }
        cout << "Loaded " << vAcc.size() << " accel, " << vGyro.size() << " gyro measurements." << endl;
    }
    else
    {
        // 旧格式: timestamp_ns w_x w_y w_z a_x a_y a_z (空格分隔)
        auto parse = [&](const string &s) {
            if(s.empty() || s[0] == '#') return;
            stringstream ss;
            ss << s;
            double data[7];
            for(int i = 0; i < 7; i++)
                ss >> data[i];
            vTimeStamps.push_back(data[0] / 1e9);
            vGyro.push_back(cv::Point3f(data[1], data[2], data[3]));
            vAcc.push_back(cv::Point3f(data[4], data[5], data[6]));
        };
        parse(first);
        string line;
        while(getline(fImu, line)) parse(line);
        cout << "Loaded " << vTimeStamps.size() << " IMU measurements." << endl;
    }
}