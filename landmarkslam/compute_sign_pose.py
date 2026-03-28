#ifndef COMPUTE_SIGN_POSE_CC
#define COMPUTE_SIGN_POSE_CC

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <opencv2/opencv.hpp>

// ORB-SLAM3 相关头文件
#include "Frame.h"
#include "MapPoint.h"

using namespace std;

// YOLO检测数据结构
struct YoloDetection {
    int frame_id;
    cv::Point2f center;       // YOLO 框的中心点 (2D 像素)
    cv::Rect2f bounding_box;  // 提取点云用的外接矩形框
};

// 最终位姿结果结构
struct SignPoseResult {
    int frame_id;
    cv::Point3f center_3d;    // 标识牌在世界坐标系下的 3D 坐标
    cv::Vec3f normal_3d;      // 标识牌的法向量 (代表朝向)
    float distance;           // 标识牌到相机光心的物理距离
    bool is_valid = false;
};

// 步骤一：读取手动清洗后的 YOLO 关键点文件
std::map<int, YoloDetection> LoadYoloKeypoints(const std::string& filepath) {
    std::map<int, YoloDetection> yolo_data;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "错误：无法打开 YOLO 数据文件 " << filepath << endl;
        return yolo_data;
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        string filename;
        float x1, y1, x2, y2, x3, y3, x4, y4;
        // 格式: frame_00183.png x1 y1 x2 y2 x3 y3 x4 y4
        if (iss >> filename >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4) {
            YoloDetection det;
            sscanf(filename.c_str(), "frame_%d.png", &det.frame_id);
            
            // 计算中心点：利用四个角点均值，比单个检测点更稳
            det.center = cv::Point2f((x1+x2+x3+x4)/4.0f, (y1+y2+y3+y4)/4.0f);
            
            // 构建语义掩码框：用于从 SLAM 场景中筛选属于标识牌的点云
            float min_x = min({x1, x2, x3, x4});
            float max_x = max({x1, x2, x3, x4});
            float min_y = min({y1, y2, y3, y4});
            float max_y = max({y1, y2, y3, y4});
            det.bounding_box = cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
            
            yolo_data[det.frame_id] = det;
        }
    }
    return yolo_data;
}

// 步骤二：RANSAC + SVD 平面拟合 (核心：利用周围物体点云恢复尺度)
bool FitSignPlane(const vector<cv::Point3f>& points, cv::Vec3f& normal, float& D) {
    if (points.size() < 5) return false; 

    int iterations = 50;
    float threshold = 0.05f; // 允许 5cm 的点云平面误差
    vector<cv::Point3f> best_inliers;

    for(int i=0; i<iterations; i++) {
        // 随机采样 3 个点构建平面模型
        cv::Point3f p1 = points[rand()%points.size()];
        cv::Point3f p2 = points[rand()%points.size()];
        cv::Point3f p3 = points[rand()%points.size()];
        
        cv::Vec3f v12 = cv::Vec3f(p2-p1);
        cv::Vec3f v13 = cv::Vec3f(p3-p1);
        cv::Vec3f n = v12.cross(v13);
        float mag = cv::norm(n);
        if(mag < 1e-6) continue;
        n /= mag;
        float d = -(n[0]*p1.x + n[1]*p1.y + n[2]*p1.z);

        vector<cv::Point3f> inliers;
        for(const auto& p : points) {
            if(abs(n[0]*p.x + n[1]*p.y + n[2]*p.z + d) < threshold) inliers.push_back(p);
        }
        if(inliers.size() > best_inliers.size()) best_inliers = inliers;
    }

    if(best_inliers.size() < 5) return false;

    // 使用 SVD 对所有内点进行最小二乘精拟合
    cv::Mat data(best_inliers.size(), 3, CV_32F);
    cv::Point3f mean(0,0,0);
    for(size_t i=0; i<best_inliers.size(); i++) {
        mean += best_inliers[i];
        data.at<float>(i,0) = best_inliers[i].x;
        data.at<float>(i,1) = best_inliers[i].y;
        data.at<float>(i,2) = best_inliers[i].z;
    }
    mean *= (1.0f/best_inliers.size());
    for(int i=0; i<data.rows; i++) {
        data.at<float>(i,0) -= mean.x;
        data.at<float>(i,1) -= mean.y;
        data.at<float>(i,2) -= mean.z;
    }
    cv::SVD svd(data);
    // 最小奇异值对应的向量即为平面法向量
    normal = cv::Vec3f(svd.vt.at<float>(2,0), svd.vt.at<float>(2,1), svd.vt.at<float>(2,2));
    if (normal[2] > 0) normal = -normal; // 确保法向量朝向相机侧
    D = -(normal[0]*mean.x + normal[1]*mean.y + normal[2]*mean.z);
    return true;
}

// 步骤三 & 四：执行计算
SignPoseResult ComputeSignPose(ORB_SLAM3::Frame &frame, const YoloDetection &yolo) {
    SignPoseResult res;
    res.frame_id = yolo.frame_id;

    // 1. 筛选：提取当前帧落在 YOLO 框内且被 SLAM 优化过的 3D 地图点
    vector<cv::Point3f> pts3d;
    for(int i=0; i<frame.N; i++) {
        if(yolo.bounding_box.contains(frame.mvKeysUn[i].pt)) {
            ORB_SLAM3::MapPoint* pMP = frame.mvpMapPoints[i];
            if(pMP && !pMP->isBad()) {
                cv::Mat p = pMP->GetWorldPos();
                pts3d.push_back(cv::Point3f(p.at<float>(0), p.at<float>(1), p.at<float>(2)));
            }
        }
    }

    // 2. 拟合平面并传递尺度
    cv::Vec3f normal; float D;
    if(FitSignPlane(pts3d, normal, D)) {
        // 3. 射线求交：将 2D 语义中心投影到 3D 约束平面上
        cv::Mat ray_c = (cv::Mat_<float>(3,1) << (yolo.center.x - frame.cx)/frame.fx, (yolo.center.y - frame.cy)/frame.fy, 1.0);
        cv::Mat ray_w = frame.mRwc * ray_c; 
        cv::Vec3f dir(ray_w.at<float>(0), ray_w.at<float>(1), ray_w.at<float>(2));
        cv::Vec3f pos(frame.mOw.at<float>(0), frame.mOw.at<float>(1), frame.mOw.at<float>(2));

        // 射线方程: P = pos + t*dir -> 带入平面方程求解 t
        float t = -(normal.dot(pos) + D) / normal.dot(dir);
        if(t > 0) {
            res.center_3d = cv::Point3f(pos[0] + t*dir[0], pos[1] + t*dir[1], pos[2] + t*dir[2]);
            res.normal_3d = normal;
            res.distance = t * cv::norm(dir);
            res.is_valid = true;
        }
    }
    return res;
}

#endif