#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <iomanip>
#include <unistd.h>
#include <map>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "System.h"
#include "Frame.h"
#include "MapPoint.h"
#include "Converter.h"

#include "compute_sign_pose.h"

using namespace std;

// 保存任意点集为 PLY
void SavePointsToPLY(const vector<cv::Point3f>& pts, const string& filename, cv::Vec3b color) {
    ofstream f(filename);
    f << "ply\nformat ascii 1.0\nelement vertex " << pts.size() << "\nproperty float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (const auto& p : pts) {
        f << p.x << " " << p.y << " " << p.z << " " << (int)color[2] << " " << (int)color[1] << " " << (int)color[0] << "\n";
    }
    f.close();
}

int main(int argc, char **argv)
{
    if(argc != 5) {
        cerr << "Usage: ./run_landmarkslam_yolo voc settings img_folder yolo_txt" << endl;
        return 1;
    }

    string strVocFile = argv[1];
    string strSettingsFile = argv[2];
    string strImageFolder = argv[3];
    string strYoloFile = argv[4];

    // 初始化参数
    cv::FileStorage fs(strSettingsFile, cv::FileStorage::READ);
    float fx = fs["Camera1.fx"]; float fy = fs["Camera1.fy"];
    float cx = fs["Camera1.cx"]; float cy = fs["Camera1.cy"];

    // 1. 加载 YOLO
    std::map<int, YoloDetection> yolo_map = LoadYoloKeypoints(strYoloFile);
    cout << "Loaded " << yolo_map.size() << " frames of sign data." << endl;

    vector<cv::String> imageFilePaths;
    cv::glob(strImageFolder + "/*.png", imageFilePaths, false);
    sort(imageFilePaths.begin(), imageFilePaths.end());
    int nImages = imageFilePaths.size();

    // 2. 启动 SLAM
    ORB_SLAM3::System SLAM(strVocFile, strSettingsFile, ORB_SLAM3::System::MONOCULAR, true);
    
    vector<SignPoseResult> results_list;
    vector<cv::Point3f> total_sign_cloud; // 汇总所有帧算出的标识牌点云
    double t_frame = 0.0;

    cout << "\n>>>> Start Analysis (Target: Frame 183 to 266) <<<<" << endl;

    for(int ni=0; ni<nImages; ni++)
    {
        cv::Mat im = cv::imread(imageFilePaths[ni], cv::IMREAD_UNCHANGED);
        if(im.empty()) continue;

        Sophus::SE3f Tcw = SLAM.TrackMonocular(im, t_frame);
        int state = SLAM.GetTrackingState();

        // 🌟 核心逻辑：执行到 266 帧强制停止
        if (ni > 266) {
            cout << "\n[INFO] Target Frame 266 reached. Stopping and Saving Data..." << endl;
            break;
        }

        if (state == 2 && yolo_map.count(ni)) {
            vector<cv::KeyPoint> vKeys = SLAM.GetTrackedKeyPointsUn();
            vector<ORB_SLAM3::MapPoint*> vMPs = SLAM.GetTrackedMapPoints();
            const YoloDetection& yolo = yolo_map[ni];

            // A. 提取框内原始点
            vector<cv::Point3f> raw_pts;
            for (size_t i = 0; i < vKeys.size(); i++) {
                if (yolo.bounding_box.contains(vKeys[i].pt)) {
                    if (vMPs[i] && !vMPs[i]->isBad()) {
                        Eigen::Vector3f p_e = vMPs[i]->GetWorldPos();
                        raw_pts.push_back(cv::Point3f(p_e(0), p_e(1), p_e(2)));
                    }
                }
            }

            // B. 平面拟合分析 (核心方法)
            cv::Vec3f normal; float D; vector<cv::Point3f> inliers;
            if (FitPlaneWithInliers(raw_pts, normal, D, inliers)) {
                Sophus::SE3f Twc = Tcw.inverse();
                cv::Mat Rwc = ORB_SLAM3::Converter::toCvMat(Twc.rotationMatrix());
                cv::Mat mOw = ORB_SLAM3::Converter::toCvMat(Twc.translation());

                cv::Mat ray_c = (cv::Mat_<float>(3,1) << (yolo.center.x - cx)/fx, (yolo.center.y - cy)/fy, 1.0f);
                cv::Mat ray_w = Rwc * ray_c;
                cv::Vec3f dir(ray_w.at<float>(0), ray_w.at<float>(1), ray_w.at<float>(2));
                cv::Vec3f pos(mOw.at<float>(0), mOw.at<float>(1), mOw.at<float>(2));

                float t_val = -(normal.dot(pos) + D) / normal.dot(dir);
                if (t_val > 0) {
                    SignPoseResult res;
                    res.frame_id = ni;
                    res.center_3d = cv::Point3f(pos[0] + t_val*dir[0], pos[1] + t_val*dir[1], pos[2] + t_val*dir[2]);
                    res.normal_3d = normal;
                    res.distance = t_val * cv::norm(dir);
                    results_list.push_back(res);
                    
                    // 汇总点云用于保存
                    total_sign_cloud.insert(total_sign_cloud.end(), inliers.begin(), inliers.end());
                    cout << ">>> Analyzed Frame " << ni << " | Dist: " << res.distance << "m | Inliers: " << inliers.size() << endl;
                }
            }
        }
        t_frame += (1.0/30.0);
    }

    // 3. 最终分析数据保存
    system("mkdir -p analysis_results");
    
    // 保存标识牌轨迹 CSV
    ofstream csv("analysis_results/Sign_Poses_Analysis.csv");
    csv << "frame,x,y,z,nx,ny,nz,dist\n";
    for(auto& r : results_list) 
        csv << r.frame_id << "," << r.center_3d.x << "," << r.center_3d.y << "," << r.center_3d.z << ","
            << r.normal_3d[0] << "," << r.normal_3d[1] << "," << r.normal_3d[2] << "," << r.distance << "\n";
    csv.close();

    // 保存标识牌点云簇 (红色)
    SavePointsToPLY(total_sign_cloud, "analysis_results/RoadSign_Points.ply", cv::Vec3b(0,0,255));

    // 保存全局场景点云
    vector<ORB_SLAM3::MapPoint*> allMPs = SLAM.GetAtlas()->GetCurrentMap()->GetAllMapPoints();
    vector<cv::Point3f> scene_pts;
    for(auto pMP : allMPs) if(pMP && !pMP->isBad()) {
        Eigen::Vector3f p = pMP->GetWorldPos();
        scene_pts.push_back(cv::Point3f(p(0), p(1), p(2)));
    }
    SavePointsToPLY(scene_pts, "analysis_results/Global_Scene.ply", cv::Vec3b(200,200,200));

    // 保存相机轨迹
    SLAM.SaveTrajectoryTUM("analysis_results/Camera_Trajectory.txt");

    cout << "\n✅ Analysis Finished. Please check 'analysis_results' folder." << endl;
    SLAM.Shutdown();
    return 0;
}