/**
* This file is part of ORB-SLAM3
* (Modified: Custom trajectory file path, save all frames trajectory, safe shutdown)
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <ctime>
#include <sstream>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc < 5)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_associations [trajectory_output_name]" << endl;
        return 1;
    }

    // 判断是否提供了自定义输出文件名
    string trajectory_file = "CameraTrajectory"; // 默认基础名
    if(argc == 6)
        trajectory_file = string(argv[5]);

    // 加载图像名称和时间戳
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    int nImages = vstrImageFilenamesRGB.size();
    if(nImages<=0)
    {
        cerr << "ERROR: No images found." << endl;
        return 1;
    }

    // 创建 SLAM 系统
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, true);
    float imageScale = SLAM.GetImageScale();

    cv::Mat imRGB, imD;
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    for(int ni=0; ni<nImages; ni++)
    {
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED);
        imD   = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

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

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        SLAM.TrackRGBD(imRGB, imD, tframe);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        vTimesTrack[ni]=ttrack;

        // 根据时间戳延时等待（保持实时播放效果，可注释掉以全速运行）
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];
        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // 停止所有线程
    SLAM.Shutdown();

    // =======================================================
    // 🌟 主动保存所有帧轨迹（使用自定义路径）
    // =======================================================
    // 提取目录与基础文件名
    string output_dir = "./";
    string base_name = trajectory_file;
    size_t pos = base_name.find_last_of("/\\");
    if(pos != string::npos)
    {
        output_dir = base_name.substr(0, pos+1);
        base_name = base_name.substr(pos+1);
    }

    // 保存所有帧的 TUM 轨迹
    string allframes_path = output_dir + "AllFrames_" + base_name + ".txt";
    SLAM.SaveTrajectoryTUM(allframes_path);
    cout << "✅ All frames trajectory saved to: " << allframes_path << endl;

    // 同时保存关键帧轨迹（保留原始行为）
    string keyframes_path = output_dir + "KeyFrames_" + base_name + ".txt";
    SLAM.SaveKeyFrameTrajectoryTUM(keyframes_path);
    cout << "✅ Keyframe trajectory saved to: " << keyframes_path << endl;
    // =======================================================

    // 跟踪时间统计
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