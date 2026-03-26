/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

// 引入标准输入输出库
#include<iostream>
// 引入算法库，比如 std::sort
#include<algorithm>
// 引入文件流库，用于读取文件
#include<fstream>
// 引入时间库，用于计算处理时间
#include<chrono>

// 引入 OpenCV 核心库，用于处理图像
#include<opencv2/core/core.hpp>

// 引入 ORB-SLAM3 系统类的头文件
#include<System.h>

// 使用标准命名空间，方便直接使用 string, vector 等
using namespace std;

// 声明一个辅助函数，用于加载图像路径和对应的时间戳
// 参数：图像文件夹路径，时间戳文件路径，输出的图像文件名列表，输出的时间戳列表
void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

// 主函数入口
int main(int argc, char **argv)
{  
    // 检查命令行参数是否足够
    // 程序至少需要 4 个固定参数 + 至少 1 组数据路径
    // 如果参数少于 5 个，打印使用说明并退出
    if(argc < 5)
    {
        cerr << endl << "Usage: ./mono_euroc path_to_vocabulary path_to_settings path_to_sequence_folder_1 path_to_times_file_1 (path_to_image_folder_2 path_to_times_file_2 ... path_to_image_folder_N path_to_times_file_N) (trajectory_file_name)" << endl;
        return 1;
    }

    // 计算要处理的序列数量
    // 前3个参数是程序名、词袋文件、设置文件，剩下的参数每2个一组（图片目录+时间文件）
    const int num_seq = (argc-3)/2;
    cout << "num_seq = " << num_seq << endl;

    // 检查是否有指定输出轨迹的文件名（参数个数是否使得最后剩下一个文件名）
    bool bFileName= (((argc-3) % 2) == 1);
    string file_name;
    if (bFileName)
    {
        // 如果有，获取最后一个参数作为文件名
        file_name = string(argv[argc-1]);
        cout << "file name: " << file_name << endl;
    }

    // 准备容器来加载所有序列的数据：
    int seq;
    // 存储每个序列的所有图片路径
    vector< vector<string> > vstrImageFilenames;
    // 存储每个序列的所有时间戳
    vector< vector<double> > vTimestampsCam;
    // 存储每个序列的图片数量
    vector<int> nImages;

    // 根据序列数量调整容器大小
    vstrImageFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    nImages.resize(num_seq);

    int tot_images = 0; // 记录总图片数
    // 遍历所有序列进行加载
    for (seq = 0; seq<num_seq; seq++)
    {
        cout << "Loading images for sequence " << seq << "...";
        
        // 调用 LoadImages 加载第 seq 个序列的数据
        // 参数索引计算：argv[3+] 是数据部分，每组 2 个，所以是 2*seq + 3 和 +4
        // 注意：EuRoC 格式通常在 mav0/cam0/data 下放图片，所以这里拼接了路径
        LoadImages(string(argv[(2*seq)+3]) + "/mav0/cam0/data", string(argv[(2*seq)+4]), vstrImageFilenames[seq], vTimestampsCam[seq]);
        cout << "LOADED!" << endl;

        // 记录该序列的图片数量
        nImages[seq] = vstrImageFilenames[seq].size();
        tot_images += nImages[seq];
    }

    // 用于统计每一帧跟踪耗时的向量，大小设为所有图片的总数
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    cout.precision(17); // 设置输出精度，用于显示精确的时间戳


    // 假设输入视频的帧率为 20 FPS，计算每帧的时间间隔 dT
    int fps = 20;
    float dT = 1.f/fps;
    
    // 创建 SLAM 系统对象
    // 参数 1: 词袋文件路径
    // 参数 2: 参数设置文件路径
    // 参数 3: 传感器类型（单目 MONOCULAR）
    // 参数 4: 是否启用可视化界面 (true)
    // 这一步会初始化所有系统线程（Tracking, LocalMapping, LoopClosing）
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR, true);
    
    // 获取配置文件中设定的图像缩放比例
    float imageScale = SLAM.GetImageScale();

    // 用于记录时间的变量
    double t_resize = 0.f; // 缩放耗时
    double t_track = 0.f;  // 跟踪耗时

    // 开始遍历处理每一个序列
    for (seq = 0; seq<num_seq; seq++)
    {

        // 主循环：遍历序列中的每一张图片
        cv::Mat im;
        int proccIm = 0;
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {

            // 从文件中读取图片，以不变的方式读取（保持原有的通道数等）
            im = cv::imread(vstrImageFilenames[seq][ni],cv::IMREAD_UNCHANGED); //,CV_LOAD_IMAGE_UNCHANGED);
            // 获取当前图片的时间戳
            double tframe = vTimestampsCam[seq][ni];

            // 如果图片读取失败，打印错误并退出
            if(im.empty())
            {
                cerr << endl << "Failed to load image at: "
                     <<  vstrImageFilenames[seq][ni] << endl;
                return 1;
            }

            // 如果需要缩放图片（配置文件中 imageScale不为1）
            if(imageScale != 1.f)
            {
// 如果定义了注册时间宏，记录开始时间
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #else
                std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #endif
#endif
                // 计算新的宽高并执行缩放
                int width = im.cols * imageScale;
                int height = im.rows * imageScale;
                cv::resize(im, im, cv::Size(width, height));
// 记录缩放结束时间并存入SLAM系统
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #else
                std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #endif
                t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
                SLAM.InsertResizeTime(t_resize);
#endif
            }

    // 记录跟踪开始的时间点
    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #else
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #endif

            // 【核心步骤】将图像传递给 SLAM 系统进行处理
            // 参数：当前图像，当前时间戳
            // 返回值是当前帧的相机位姿（虽然这里没接收返回值）
            // TODO 注释提到：如果使用 Monocular-Inertial 模式需要更改函数调用
            SLAM.TrackMonocular(im,tframe); 

    // 记录跟踪结束的时间点
    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    #else
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    #endif

// 如果定义了注册时间宏，记录总跟踪时间（包含缩放耗时）
#ifdef REGISTER_TIMES
            t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif

            // 计算本次跟踪实际消耗的时间（秒）
            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

            // 保存这次的时间统计
            vTimesTrack[ni]=ttrack;

            // 等待加载下一帧的时间控制逻辑
            // 我们的目的是模拟实时的帧率，如果处理太快了，需要在这里等待一会
            double T=0;
            // 计算当前帧到下一帧的时间间隔 T
            if(ni<nImages[seq]-1)
                T = vTimestampsCam[seq][ni+1]-tframe;
            else if(ni>0)
                T = tframe-vTimestampsCam[seq][ni-1]; // 最后一帧用前一帧的间隔代替

            //std::cout << "T: " << T << std::endl;
            //std::cout << "ttrack: " << ttrack << std::endl;

            // 如果实际处理时间小于帧间隔时间，说明处理得太快，需要延时
            if(ttrack<T) {
                //std::cout << "usleep: " << (dT-ttrack) << std::endl;
                // usleep 接受微秒，所以乘以 1e6
                usleep((T-ttrack)*1e6); 
            }
        }

        // 如果还没处理完最后一个序列
        if(seq < num_seq - 1)
        {
            // 构造保存子地图文件名的路径
            string kf_file_submap =  "./SubMaps/kf_SubMap_" + std::to_string(seq) + ".txt";
            string f_file_submap =  "./SubMaps/f_SubMap_" + std::to_string(seq) + ".txt";
            // 保存当前序列的轨迹
            SLAM.SaveTrajectoryEuRoC(f_file_submap);
            SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file_submap);

            cout << "Changing the dataset" << endl;

            // 告诉 SLAM 系统切换数据集（重置状态但保留地图用于后续合并）
            SLAM.ChangeDataset();
        }

    }
    // 停止所有系统线程 (Tracking, LocalMapping, LoopClosing 等)
    SLAM.Shutdown();

    // 保存相机轨迹结果
    if (bFileName)
    {
        // 如果命令行指定了文件名，使用指定的名字
        const string kf_file =  "kf_" + string(argv[argc-1]) + ".txt";
        const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else
    {
        // 否则使用默认文件名
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    return 0; // 程序正常退出
}

// 辅助函数的实现：读取图像列表和时间戳
void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    // 打开时间戳文件
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    // 预分配内存，避免频繁重新分配，提高效率
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    
    // 逐行读取文件
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s); // 读取一行
        if(!s.empty()) // 如果行不为空
        {
            stringstream ss;
            ss << s; // 将行内容放入 stringstream
            
            // 假设文件格式是：时间戳(也是文件名部分)
            // 拼接完整的图片路径：路径 + 时间戳字符串 + .png
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
            
            // 读取时间戳数值
            double t;
            ss >> t;
            // EuRoC 时间戳是纳秒，乘以 1e-9 转换为秒
            vTimeStamps.push_back(t*1e-9);

        }
    }
}
