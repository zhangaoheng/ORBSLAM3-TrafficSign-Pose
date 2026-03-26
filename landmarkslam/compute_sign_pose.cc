#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <map>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;

// 保存为PLY文件
void SaveSignPointCloudPLY(const vector<cv::Point3f>& points, const vector<cv::Vec3b>& colors, const string& filename) {
    if (points.empty()) return;
    ofstream f(filename);
    if (!f.is_open()) return;

    f << "ply\n";
    f << "format ascii 1.0\n";
    f << "element vertex " << points.size() << "\n";
    f << "property float x\n";
    f << "property float y\n";
    f << "property float z\n";
    f << "property uchar red\n";
    f << "property uchar green\n";
    f << "property uchar blue\n";
    f << "end_header\n";

    for (size_t i = 0; i < points.size(); ++i) {
        f << fixed << setprecision(5) << points[i].x << " " << points[i].y << " " << points[i].z << " "
          << (int)colors[i][2] << " " << (int)colors[i][1] << " " << (int)colors[i][0] << "\n"; // BGR to RGB
    }
    f.close();
}

// 检查一个点是否在多边形内
bool isPointInsidePolygon(const cv::Point2f& pt, const vector<cv::Point2f>& polygon) {
    return cv::pointPolygonTest(polygon, pt, false) >= 0;
}

// 加载YOLO检测结果，格式假设为: 帧名 x1 y1 x2 y2 x3 y3 x4 y4
map<string, vector<cv::Point2f>> loadYoloDetections(const string& filename) {
    map<string, vector<cv::Point2f>> detections;
    ifstream f(filename);
    if (!f.is_open()) {
        cerr << "[Warning] Could not open YOLO detection file: " << filename << endl;
        return detections;
    }
    
    string line;
    while (getline(f, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string img_name;
        ss >> img_name;
        
        vector<cv::Point2f> pts;
        float x, y;
        for (int i = 0; i < 4; ++i) {
            if (ss >> x >> y) {
                pts.push_back(cv::Point2f(x, y));
            }
        }
        if (pts.size() == 4) {
            detections[img_name] = pts;
        }
    }
    return detections;
}

// 通过相机内参、平面法向量和距离，将2D图像像素反投影到3D平面上
cv::Point3f backprojectToPlane(const cv::Point2f& p, const cv::Mat& K_inv, const cv::Mat& R, const cv::Mat& t) {
    cv::Mat pt_c = (cv::Mat_<double>(3, 1) << p.x, p.y, 1.0);
    cv::Mat ray_c = K_inv * pt_c;
    
    cv::Mat R_T = R.t();
    cv::Mat r_row2 = R_T.row(2);
    
    double num = r_row2.dot(t.t());
    double den = r_row2.dot(ray_c.t());
    
    double s = num / den;
    
    cv::Mat P_c = s * ray_c;
    cv::Mat P_w = R_T * (P_c - t);
    
    return cv::Point3f(P_w.at<double>(0), P_w.at<double>(1), P_w.at<double>(2));
}

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./compute_sign_pose path_to_settings path_to_image_folder path_to_yolo_results.txt output.ply" << endl;
        return 1;
    }

    string strSettingsFile = argv[1];
    string strImageFolder = argv[2];
    string strYoloFile = argv[3];
    string outputPly = argv[4];

    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened()) {
        cerr << "Failed to open settings file." << endl;
        return -1;
    }
    
    double fx = fsSettings["Camera1.fx"];
    double fy = fsSettings["Camera1.fy"];
    double cx = fsSettings["Camera1.cx"];
    double cy = fsSettings["Camera1.cy"];
    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat K_inv = K.inv();
    
    map<string, vector<cv::Point2f>> yoloData = loadYoloDetections(strYoloFile);

    vector<cv::String> imageFilePaths;
    cv::glob(strImageFolder + "/*.png", imageFilePaths, false);
    if (imageFilePaths.empty()) {
        cv::glob(strImageFolder + "/*.jpg", imageFilePaths, false);
    }
    sort(imageFilePaths.begin(), imageFilePaths.end());

    cv::Ptr<cv::ORB> orb = cv::ORB::create(3000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

    vector<cv::Point3f> all_sign_points;
    vector<cv::Vec3b> all_sign_colors;

    double sign_width = 1.0; 
    double sign_height = 1.0;
    vector<cv::Point2f> obj_pts;
    obj_pts.push_back(cv::Point2f(-sign_width/2, -sign_height/2));
    obj_pts.push_back(cv::Point2f(sign_width/2, -sign_height/2));
    obj_pts.push_back(cv::Point2f(sign_width/2, sign_height/2));
    obj_pts.push_back(cv::Point2f(-sign_width/2, sign_height/2));

    for(size_t i=0; i<imageFilePaths.size(); i++)
    {
        string baseFilename = imageFilePaths[i].substr(imageFilePaths[i].find_last_of("\\/") + 1);
        if (yoloData.find(baseFilename) == yoloData.end()) continue;

        vector<cv::Point2f> corners = yoloData[baseFilename];
        if (corners.size() != 4) continue;

        cv::Mat im = cv::imread(imageFilePaths[i]);
        if (im.empty()) continue;

        cv::Mat H = cv::findHomography(obj_pts, corners, 0);
        if (H.empty()) continue;

        vector<cv::Mat> Rs, ts, normals;
        int solutions = cv::decomposeHomographyMat(H, K, Rs, ts, normals);
        if (solutions == 0) continue;

        int best_idx = 0;
        for (int j = 0; j < solutions; j++) {
            if (normals[j].at<double>(2) < 0 && ts[j].at<double>(2) > 0) {
                best_idx = j;
                break;
            }
        }

        cv::Mat R = Rs[best_idx];
        cv::Mat t = ts[best_idx];

        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb->detectAndCompute(im, cv::noArray(), keypoints, descriptors);

        for (const auto& kp : keypoints) {
            if (isPointInsidePolygon(kp.pt, corners)) {
                cv::Point3f pt3d = backprojectToPlane(kp.pt, K_inv, R, t);
                all_sign_points.push_back(pt3d);
                cv::Vec3b color = im.at<cv::Vec3b>(kp.pt.y, kp.pt.x);
                all_sign_colors.push_back(color);
            }
        }
    }

    SaveSignPointCloudPLY(all_sign_points, all_sign_colors, outputPly);

    return 0;
}
