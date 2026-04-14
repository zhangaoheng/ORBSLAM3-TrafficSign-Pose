/**
* This file is part of ORB-SLAM3
* (Modified for Full Frame TUM Trajectory Output & Robust Data Loading)
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <ctime>
#include <sstream>

#include<opencv2/core/core.hpp>

#include<System.h>
#include "ImuTypes.h"

using namespace std;

void LoadImagesTUMVI(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

double ttrack_tot = 0;

int main(int argc, char **argv)
{
    const int num_seq = (argc-3)/3;
    cout << "num_seq = " << num_seq << endl;
    bool bFileName= ((argc % 3) == 1);

    string file_name;
    if (bFileName)
        file_name = string(argv[argc-1]);

    cout << "file name: " << file_name << endl;

    if(argc < 6)
    {
        cerr << endl << "Usage: ./mono_inertial_tum_vi path_to_vocabulary path_to_settings path_to_image_folder_1 path_to_times_file_1 path_to_imu_data_1 [trajectory_file_name]" << endl;
        return 1;
    }

    // Load all sequences:
    int seq;
    vector< vector<string> > vstrImageFilenames;
    vector< vector<double> > vTimestampsCam;
    vector< vector<cv::Point3f> > vAcc, vGyro;
    vector< vector<double> > vTimestampsImu;
    vector<int> nImages;
    vector<int> nImu;
    vector<int> first_imu(num_seq,0);

    vstrImageFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    vAcc.resize(num_seq);
    vGyro.resize(num_seq);
    vTimestampsImu.resize(num_seq);
    nImages.resize(num_seq);
    nImu.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
        cout << "Loading images for sequence " << seq << "...";
        LoadImagesTUMVI(string(argv[3*(seq+1)]), string(argv[3*(seq+1)+1]), vstrImageFilenames[seq], vTimestampsCam[seq]);
        cout << "LOADED!" << endl;

        cout << "Loading IMU for sequence " << seq << "...";
        LoadIMU(string(argv[3*(seq+1)+2]), vTimestampsImu[seq], vAcc[seq], vGyro[seq]);
        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageFilenames[seq].size();
        tot_images += nImages[seq];
        nImu[seq] = vTimestampsImu[seq].size();

        if((nImages[seq]<=0)||(nImu[seq]<=0))
        {
            cerr << "ERROR: Failed to load images or IMU for sequence" << seq << endl;
            return 1;
        }

        // Find first imu to be considered
        while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][0])
            first_imu[seq]++;
        first_imu[seq]--; 
    }

    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    cout.precision(17);

    // Create SLAM system.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_MONOCULAR, true, 0, file_name);
    float imageScale = SLAM.GetImageScale();

    double t_resize = 0.f;
    double t_track = 0.f;

    int proccIm = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
        cv::Mat im;
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        proccIm = 0;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {
            im = cv::imread(vstrImageFilenames[seq][ni],cv::IMREAD_GRAYSCALE);
            clahe->apply(im,im);
            
            double tframe = vTimestampsCam[seq][ni];

            if(im.empty())
            {
                cerr << endl << "Failed to load image at: " <<  vstrImageFilenames[seq][ni] << endl;
                return 1;
            }

            vImuMeas.clear();
            if(ni>0)
            {
                while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][ni])
                {
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[seq][first_imu[seq]].x,vAcc[seq][first_imu[seq]].y,vAcc[seq][first_imu[seq]].z,
                                                             vGyro[seq][first_imu[seq]].x,vGyro[seq][first_imu[seq]].y,vGyro[seq][first_imu[seq]].z,
                                                             vTimestampsImu[seq][first_imu[seq]]));
                    first_imu[seq]++;
                }
            }

            if(imageScale != 1.f)
            {
                int width = im.cols * imageScale;
                int height = im.rows * imageScale;
                cv::resize(im, im, cv::Size(width, height));
            }

            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

            // Pass the image to the SLAM system
            SLAM.TrackMonocular(im,tframe,vImuMeas); 

            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            ttrack_tot += ttrack;
            vTimesTrack[ni]=ttrack;

            // Wait to load the next frame
            double T=0;
            if(ni<nImages[seq]-1)
                T = vTimestampsCam[seq][ni+1]-tframe;
            else if(ni>0)
                T = tframe-vTimestampsCam[seq][ni-1];

            if(ttrack<T)
                usleep((T-ttrack)*1e6); 
        }
        if(seq < num_seq - 1)
        {
            cout << "Changing the dataset" << endl;
            SLAM.ChangeDataset();
        }
    }

    // Stop all threads
    SLAM.Shutdown();

    // =================================================================
    // 【核心优化】：强行保存“所有帧”的 TUM 格式轨迹
    // =================================================================
    cout << "\n==================================================" << endl;
    cout << ">>> 正在保存计算结果..." << endl;

    if (bFileName)
    {
        const string all_frames_file =  "all_frames_" + string(argv[argc-1]) + ".txt";
        const string keyframes_file =  "keyframes_" + string(argv[argc-1]) + ".txt";
        
        // SaveTrajectoryTUM 会遍历所有追踪成功的帧，并以 TUM 格式保存
        SLAM.SaveTrajectoryTUM(all_frames_file);
        SLAM.SaveKeyFrameTrajectoryTUM(keyframes_file);

        cout << "✅ [所有帧轨迹] 已安全保存至: " << all_frames_file << endl;
        cout << "   (请使用这个文件配合 Python 脚本提取 FOE 和去旋矩阵)" << endl;
        cout << "✅ [关键帧轨迹] 已安全保存至: " << keyframes_file << endl;
    }
    else
    {
        SLAM.SaveTrajectoryTUM("AllFramesTrajectory_TUM.txt");
        SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_TUM.txt");
        cout << "✅ [所有帧轨迹] 已安全保存至: AllFramesTrajectory_TUM.txt" << endl;
    }
    cout << "==================================================\n" << endl;

    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages[0]; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl;
    cout << "median tracking time: " << vTimesTrack[nImages[0]/2] << endl;
    cout << "mean tracking time: " << totaltime/proccIm << endl;

    return 0;
}

// =================================================================
// 【防卡死护城河】：安全的数据读取逻辑
// =================================================================
void LoadImagesTUMVI(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    ifstream fTimes(strPathTimes.c_str());
    if (!fTimes.is_open()) {
        cerr << "\n[致命错误 ❌] 无法打开图像时间戳文件: " << strPathTimes << endl;
        exit(-1);
    }

    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    string s;
    
    while(getline(fTimes, s))
    {
        if(s.empty() || s[0] == '#') continue;

        replace(s.begin(), s.end(), ',', ' ');
        stringstream ss(s);
        
        string timestamp_str;
        ss >> timestamp_str;
        if(timestamp_str.empty()) continue;

        double t = stod(timestamp_str);
        if(t > 1e14) t /= 1e9;
        
        string filename;
        if (ss >> filename) {
            vstrImages.push_back(strImagePath + "/" + filename);
        } else {
            vstrImages.push_back(strImagePath + "/" + timestamp_str + ".png");
        }
        vTimeStamps.push_back(t);
    }
    fTimes.close();
}

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream fImu(strImuPath.c_str());
    if (!fImu.is_open()) {
        cerr << "\n[致命错误 ❌] 无法打开 IMU 数据文件: " << strImuPath << endl;
        exit(-1);
    }

    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);
    string s;
    
    while(getline(fImu, s))
    {
        if(s.empty() || s[0] == '#') continue;

        replace(s.begin(), s.end(), ',', ' ');
        stringstream ss(s);
        
        double t, wx, wy, wz, ax, ay, az;
        if (!(ss >> t >> wx >> wy >> wz >> ax >> ay >> az)) continue;

        if(t > 1e14) t /= 1e9;

        vTimeStamps.push_back(t);
        vAcc.push_back(cv::Point3f(ax, ay, az));
        vGyro.push_back(cv::Point3f(wx, wy, wz));
    }
    fImu.close();
}