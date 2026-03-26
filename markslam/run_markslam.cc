#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // 用于 sort 排序
#include <opencv2/opencv.hpp>
#include "MarkSLAM.h" // 引用我们的 SLAM 系统头文件

using namespace std;

// 1. 在这里硬编码你的路径参数，就像 test.cc 那样
//    请确保这些路径是真实存在的
string strVocFile = "/home/zah/ORB_SLAM3-master/Vocabulary/ORBvoc.txt";
string strSettingsFile = "/home/zah/ORB_SLAM3-master/markslam/markslam.yaml"; 
string strImageFolder = "/home/zah/ORB_SLAM3-master/markslam/data/20260127_163648/roi_extracted"; // 比如你的图片在这里

// 主函数入口
int main(int argc, char **argv) {
    
    // 2. 如果提供了额外的 YAML 参数文件（像 test.cc 里的 parameterFile），就读取它去覆盖上面的默认值
    //    假设用法是: ./run_markslam [optional_custom_config.yaml]
    string parameterFile = "./test.yaml"; // 默认的额外参数文件
    if (argc >= 2) {
        parameterFile = argv[1];
    }
    
    // 尝试读取额外的配置文件
    cv::FileStorage fs(parameterFile, cv::FileStorage::READ);
    if(fs.isOpened()) {
        cout << "Loading parameters from " << parameterFile << endl;
        // 如果yaml里有定义这些字段，就覆盖默认值
        if (!fs["VocFile"].empty()) fs["VocFile"] >> strVocFile;
        if (!fs["SettingsFile"].empty()) fs["SettingsFile"] >> strSettingsFile;
        // if (!fs["ImageFolder"].empty()) fs["ImageFolder"] >> strImageFolder; // 可以按需添加
    } else {
        cout << "Parameter file " << parameterFile << " not found or invalid. Using hardcoded defaults." << endl;
    }

    std::cout << "---------------------------------" << std::endl;
    std::cout << "Vocabulary: " << strVocFile << std::endl;
    std::cout << "Settings:   " << strSettingsFile << std::endl;
    std::cout << "Images:     " << strImageFolder << std::endl;
    std::cout << "---------------------------------" << std::endl;

    // 使用 OpenCV 的 glob 函数查找文件夹下的所有图片
    vector<cv::String> imageFilePaths; // 存储找到的文件路径
    
    // 优先查找 .png 格式
    string pattern = strImageFolder + "/*.png";
    cv::glob(pattern, imageFilePaths, false);
    
    // 如果没找到 png，尝试查找 .jpg
    if(imageFilePaths.empty()) {
        pattern = strImageFolder + "/*.jpg";
        cv::glob(pattern, imageFilePaths, false);
    }

    // 如果还是没找到，报错退出
    if(imageFilePaths.empty()) {
        cerr << "Error: No .png or .jpg files found in " << strImageFolder << endl;
        return -1;
    }

    // 对文件名进行排序，确保按顺序播放 (例如 001.png, 002.png...)
    sort(imageFilePaths.begin(), imageFilePaths.end());

    std::cout << "Found " << imageFilePaths.size() << " images." << std::endl;

    // 实例化 SLAM 系统
    MarkSLAM slam(strVocFile, strSettingsFile);

    // 定义图像容器
    cv::Mat im;
    
    // 遍历所有图片文件
    for(size_t i = 0; i < imageFilePaths.size(); ++i) {
        // 读取图片，保持原格式 (灰度或彩色)
        im = cv::imread(imageFilePaths[i], cv::IMREAD_UNCHANGED);
        
        // 校验图片是否有效
        if(im.empty()) {
            cerr << "Failed to load image: " << imageFilePaths[i] << endl;
            continue;
        }

        // 模拟时间戳 (假设 30 FPS, 每帧 +0.033s)
        // 解析真实帧号逻辑
        string filename = imageFilePaths[i];
        size_t lastSlash = filename.find_last_of("/");
        string name = filename.substr(lastSlash + 1);
        size_t startPos = name.find("frame_");
        int frameNum = i;
        if (startPos != string::npos) {
            startPos += 6;
            size_t endPos = name.find("_roi", startPos);
            if (endPos != string::npos) {
                try { frameNum = stoi(name.substr(startPos, endPos - startPos)); } catch(...) {}
            }
        }
        double timestamp = frameNum * 0.033333;


        // 【核心】将图片传入 SLAM 系统进行跟踪
        Sophus::SE3f pose = slam.TrackMonocular(im, timestamp);

        // --- 以下是可视化代码 ---
        
        // 获取当前状态
        MarkSLAM::State state = slam.GetState();
        string stateName = "UNKNOWN";
        if (state == MarkSLAM::NOT_INITIALIZED) stateName = "NOT_INITIALIZED";
        else if (state == MarkSLAM::INITIALIZING) stateName = "INITIALIZING";
        else if (state == MarkSLAM::TRACKING) stateName = "TRACKING";
        else if (state == MarkSLAM::LOST) stateName = "LOST";

        // 在图片上绘制状态文字
        cv::putText(im, "State: " + stateName, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        // 如果跟踪成功，绘制当前位姿坐标
        if (state == MarkSLAM::TRACKING) {
             // 从位姿 se3 中提取平移向量 (x, y, z)
             Eigen::Vector3f t = pose.translation();
             // 拼接字符串显示坐标
             string pStr = "Pos: " + to_string(t.x()) + ", " + to_string(t.y()) + ", " + to_string(t.z());
             // 在图像下方 (10, 60) 绘制坐标信息，颜色为黄色 (B:0, G:255, R:255)
             cv::putText(im, pStr, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        }

        // 显示图片
        cv::imshow("MarkSLAM", im);
        
        // 延时 1ms，如果有按键按下则退出 (ESC 键 ASCII 27)
        char c = (char)cv::waitKey(1); 
        if (c == 27) break; 
    }

    return 0;
}
