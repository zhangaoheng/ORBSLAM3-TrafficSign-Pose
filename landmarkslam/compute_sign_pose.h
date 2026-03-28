#ifndef COMPUTE_SIGN_POSE_H
#define COMPUTE_SIGN_POSE_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace std;

// 数据结构定义
struct YoloDetection {
    int frame_id;
    cv::Point2f center;       
    cv::Rect2f bounding_box;  
};

struct SignPoseResult {
    int frame_id;
    cv::Point3f center_3d;          // 标识牌 3D 中心
    cv::Vec3f normal_3d;            // 标识牌法向量
    float distance;                 // 到相机距离
    vector<cv::Point3f> inlier_points; // 属于标识牌的 3D 点云簇
    bool is_valid = false;
};

// 步骤一：读取清洗后的 YOLO 关键点
static std::map<int, YoloDetection> LoadYoloKeypoints(const std::string& filepath) {
    std::map<int, YoloDetection> yolo_data;
    std::ifstream file(filepath);
    if (!file.is_open()) return yolo_data;

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        string filename;
        float x1, y1, x2, y2, x3, y3, x4, y4;
        if (iss >> filename >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4) {
            YoloDetection det;
            // 处理 frame_00183.png 这种格式提取数字
            size_t start = filename.find('_') + 1;
            size_t end = filename.find('.');
            if(start != string::npos && end != string::npos) {
                det.frame_id = stoi(filename.substr(start, end - start));
            } else continue;
            
            det.center = cv::Point2f((x1 + x2 + x3 + x4) / 4.0f, (y1 + y2 + y3 + y4) / 4.0f);
            float min_x = min({x1, x2, x3, x4}); float max_x = max({x1, x2, x3, x4});
            float min_y = min({y1, y2, y3, y4}); float max_y = max({y1, y2, y3, y4});
            det.bounding_box = cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
            yolo_data[det.frame_id] = det;
        }
    }
    return yolo_data;
}

// 步骤二：RANSAC 平面拟合（返回平面方程和内点簇）
static bool FitPlaneWithInliers(const vector<cv::Point3f>& points, cv::Vec3f& normal, float& D_val, vector<cv::Point3f>& out_inliers) {
    if (points.size() < 8) return false; // 增加点数阈值保证分析质量

    int iterations = 100;
    float threshold = 0.03f; // 3厘米误差，更严格的约束
    vector<cv::Point3f> best_inliers;

    for (int i = 0; i < iterations; i++) {
        cv::Point3f p1 = points[rand() % points.size()];
        cv::Point3f p2 = points[rand() % points.size()];
        cv::Point3f p3 = points[rand() % points.size()];
        cv::Vec3f n = cv::Vec3f(p2 - p1).cross(cv::Vec3f(p3 - p1));
        float mag = cv::norm(n);
        if (mag < 1e-6) continue;
        n /= mag;
        float d = -(n[0] * p1.x + n[1] * p1.y + n[2] * p1.z);

        vector<cv::Point3f> inliers;
        for (const auto& p : points) {
            if (abs(n[0] * p.x + n[1] * p.y + n[2] * p.z + d) < threshold) inliers.push_back(p);
        }
        if (inliers.size() > best_inliers.size()) {
            best_inliers = inliers;
            normal = n;
            D_val = d;
        }
    }

    if (best_inliers.size() < 8) return false;
    
    // 强制让法向量指向相机侧
    if (normal[2] > 0) { normal = -normal; D_val = -D_val; }
    out_inliers = best_inliers;
    return true;
}

#endif