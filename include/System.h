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


#ifndef SYSTEM_H
#define SYSTEM_H

// 引入标准库和系统库
#include <unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

// 引入项目内部模块的头文件
#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"
#include "ImuTypes.h"
#include "Settings.h"


namespace ORB_SLAM3
{

// 日志详细程度控制类
class Verbose
{
public:
    enum eLevel
    {
        VERBOSITY_QUIET=0,      // 安静模式，不输出
        VERBOSITY_NORMAL=1,     // 普通模式
        VERBOSITY_VERBOSE=2,    // 详细模式
        VERBOSITY_VERY_VERBOSE=3, // 非常详细
        VERBOSITY_DEBUG=4       // 调试模式
    };

    static eLevel th; // 当前的日志阈值

public:
    // 打印消息函数，如果级别允许则输出
    static void PrintMess(std::string str, eLevel lev)
    {
        if(lev <= th)
            cout << str << endl;
    }

    // 设置日志级别
    static void SetTh(eLevel _th)
    {
        th = _th;
    }
};

// 前向声明，减少头文件依赖，加快编译
class Viewer;
class FrameDrawer;
class MapDrawer;
class Atlas;
class Tracking;
class LocalMapping;
class LoopClosing;
class Settings;

// ORB-SLAM3 的主系统类
// 也就是我们在 main 函数中调用的那个 "System"
class System
{
public:
    // 输入传感器类型枚举
    enum eSensor{
        MONOCULAR=0,        // 单目
        STEREO=1,           // 双目
        RGBD=2,             // RGB-D (深度相机)
        IMU_MONOCULAR=3,    // 单目 + IMU
        IMU_STEREO=4,       // 双目 + IMU
        IMU_RGBD=5,         // RGB-D + IMU
    };

    // 文件类型枚举
    enum FileType{
        TEXT_FILE=0,
        BINARY_FILE=1,
    };

public:
    // 保证 Eigen库 内存对齐的宏，防止段错误
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // 构造函数：初始化 SLAM 系统
    // 1. 加载词袋文件 (Vocabulary)
    // 2. 加载配置文件 (Settings)
    // 3. 初始化三大线程：Local Mapping (局部建图), Loop Closing (回环检测), Viewer (可视化)
    System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer = true, const int initFr = 0, const string &strSequence = std::string());

    // 处理双目帧
    // 输入：左图，右图，时间戳，(可选)IMU数据
    // 返回：相机位姿 SE3 (如果跟踪失败则为空)
    Sophus::SE3f TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");

    // 处理 RGB-D 帧
    // 输入：RGB图，深度图，时间戳，(可选)IMU数据
    // 返回：相机位姿
    Sophus::SE3f TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");

    // 处理单目帧 (TrackMonocular)
    // -------------------------------------------------------------------------------------
    // 参数详解：
    // 1. im (const cv::Mat &):       
    //      当前时刻摄像头拍摄到的图像。可以是 RGB 或灰度图（系统内部会自动转灰度）。
    //      确保传入前图像不为空 (!im.empty())。
    //
    // 2. timestamp (const double &): 
    //      当前帧对应的时间戳，单位通常是秒(s)。
    //      用于计算相机速度和积分 IMU。必须严格随时间递增。
    //
    // 3. vImuMeas (const vector<IMU::Point>&): [可选]
    //      从上一帧到当前帧期间采集到的所有 IMU 数据列表。
    //      - 纯单目模式 (MONOCULAR): 不需要传，使用默认空向量即可。
    //      - 单目惯性模式 (IMU_MONOCULAR): 必须传入，否则无法初始化。
    //
    // 4. filename (string): [可选]
    //      图像文件名。仅用于调试输出或结果标记，对算法核心逻辑无影响。
    //
    // 返回值 (Sophus::SE3f):
    //      即时计算出的相机位姿 Tcw (World -> Camera)。
    //      如果跟踪失败（丢失），返回的位姿可能无效，建议配合 isLost() 检查状态。
    Sophus::SE3f TrackMonocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");


    // 激活定位模式
    // 停止局部建图线程，只进行相机跟踪（不插入新关键帧，不更新地图）
    // 也就是常说的 "纯定位模式"
    void ActivateLocalizationMode();
    // 停用定位模式
    // 恢复局部建图线程，恢复正常的 SLAM 过程
    void DeactivateLocalizationMode();

    // 检查是否有大的地图变动（如回环检测、全局BA）
    bool MapChanged();

    // 重置系统（清除 Atlas 或当前活动地图）
    void Reset();
    void ResetActiveMap();

    // 系统关闭
    // 请求所有线程结束，并等待它们完全退出
    // 在保存轨迹之前必须调用此函数
    void Shutdown();
    bool isShutDown();

    // 保存相机轨迹 (TUM 格式)
    // 仅适用于 Stereo 和 RGB-D
    void SaveTrajectoryTUM(const string &filename);

    // 保存关键帧轨迹 (TUM 格式)
    // 适用于所有传感器类型
    void SaveKeyFrameTrajectoryTUM(const string &filename);

    // 保存相机轨迹 (EuRoC 格式)
    // 也就是 timestamp tx ty tz qx qy qz qw
    void SaveTrajectoryEuRoC(const string &filename);
    void SaveKeyFrameTrajectoryEuRoC(const string &filename);

    // 保存指定地图的轨迹
    void SaveTrajectoryEuRoC(const string &filename, Map* pMap);
    void SaveKeyFrameTrajectoryEuRoC(const string &filename, Map* pMap);

    // 保存用于调试初始化的数据
    void SaveDebugData(const int &iniIdx);

    // 保存相机轨迹 (KITTI 格式)
    // 也就是 r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
    void SaveTrajectoryKITTI(const string &filename);

    // TODO: Save/Load functions
    // SaveMap(const string &filename);
    // LoadMap(const string &filename);

    // 获取最近一帧的信息
    // 可以在 TrackMonocular 等函数之后立即调用
    int GetTrackingState();
    std::vector<MapPoint*> GetTrackedMapPoints();
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

    // 调试用
    double GetTimeFromIMUInit();
    bool isLost();
    bool isFinished();

    // 切换数据集（用于处理多段序列）
    void ChangeDataset();

    // 获取图像缩放比例
    float GetImageScale();

      // 获取 Atlas 指针（用于导出地图/点云）
      Atlas* GetAtlas();

#ifdef REGISTER_TIMES
    void InsertRectTime(double& time);
    void InsertResizeTime(double& time);
    void InsertTrackTime(double& time);
#endif

private:

    void SaveAtlas(int type);
    bool LoadAtlas(int type);

    string CalculateCheckSum(string filename, int type);

    // 输入传感器类型
    eSensor mSensor;

    // ORB 词袋库，用于位置识别和特征匹配
    ORBVocabulary* mpVocabulary;

    // 关键帧数据库，用于重定位和回环检测
    KeyFrameDatabase* mpKeyFrameDatabase;

    // 地图结构，管理所有的地图（Atlas 包含多个 Map）
    //Map* mpMap;
    Atlas* mpAtlas;

    // 跟踪器 (Tracker)
    // 它是主线程的一部分，接收图像并计算相机位姿
    // 决定何时插入关键帧，何时创建地图点
    Tracking* mpTracker;

    // 局部建图线程 (Local Mapper)
    // 管理局部地图，进行局部 Bundle Adjustment 优化
    LocalMapping* mpLocalMapper;

    // 回环检测线程 (Loop Closer)
    // 对每个新关键帧进行回环搜索，如果发现回环，执行位姿图优化和全局 BA
    LoopClosing* mpLoopCloser;

    // 可视化器 (Viewer)
    // 使用 Pangolin 绘制地图和相机轨迹
    Viewer* mpViewer;

    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    // 系统线程句柄：局部建图、回环检测、可视化
    // 跟踪线程 (Tracking) 没有单独的线程句柄，它运行在调用 System 的主线程中
    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;
    std::thread* mptViewer;

    // 重置标志
    std::mutex mMutexReset;
    bool mbReset;
    bool mbResetActiveMap;

    // 模式切换标志
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;

    // 关闭标志
    bool mbShutDown;

    // 跟踪状态
    int mTrackingState;
    std::vector<MapPoint*> mTrackedMapPoints;
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
    std::mutex mMutexState;

    //
    string mStrLoadAtlasFromFile;
    string mStrSaveAtlasToFile;

    string mStrVocabularyFilePath;

    Settings* settings_;
};

}// namespace ORB_SLAM

#endif // SYSTEM_H
