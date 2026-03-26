#include "MarkSLAM.h"
#include <iostream>
#include <opencv2/core/eigen.hpp>

using namespace ORB_SLAM3;

MarkSLAM::MarkSLAM(string strVocFile, string strSettingsFile) : mState(NOT_INITIALIZED) {
    std::cout << "MarkSLAM: Initializing specialized planar SLAM..." << std::endl;
    // 1. Load Settings (Camera Calibration)
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened()) {
        cerr << "Failed to open settings file at: " << strSettingsFile << endl;
        exit(-1);
    }

    float fx = fsSettings["Camera1.fx"];
    float fy = fsSettings["Camera1.fy"];
    float cx = fsSettings["Camera1.cx"];
    float cy = fsSettings["Camera1.cy"];

    std::cout << "Camera Parameters: " << fx << ", " << fy << ", " << cx << ", " << cy << std::endl;

    mK = cv::Mat::eye(3,3,CV_32F);
    mK.at<float>(0,0) = fx;
    mK.at<float>(1,1) = fy;
    mK.at<float>(0,2) = cx;
    mK.at<float>(1,2) = cy;

    // Load distortion coefficients manually
    float k1 = fsSettings["Camera1.k1"];
    float k2 = fsSettings["Camera1.k2"];
    float p1 = fsSettings["Camera1.p1"];
    float p2 = fsSettings["Camera1.p2"];
    
    // Create 4x1 matrix for dist coeffs
    mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    mDistCoef.at<float>(0) = k1;
    mDistCoef.at<float>(1) = k2;
    mDistCoef.at<float>(2) = p1;
    mDistCoef.at<float>(3) = p2;

    std::cout << "Distortion Coefficients: " << mDistCoef.t() << std::endl;

    mnFeatures = fsSettings["ORBextractor.nFeatures"];
    mScaleFactor = fsSettings["ORBextractor.scaleFactor"];
    mnLevels = fsSettings["ORBextractor.nLevels"];
    int iniThFAST = fsSettings["ORBextractor.iniThFAST"];
    int minThFAST = fsSettings["ORBextractor.minThFAST"];
    
    // 2. Initialize ORB Extractor
    mpORBextractor = new ORBextractor(mnFeatures, mScaleFactor, mnLevels, iniThFAST, minThFAST);
    std::cout << "MarkSLAM: Ready." << std::endl;
}

Sophus::SE3f MarkSLAM::TrackMonocular(const cv::Mat &im, const double &timestamp) {
    // 1. Feature Extraction
    std::vector<int> vLapping = {0, 1000}; 
    int nMonoLeft = (*mpORBextractor)(im, cv::Mat(), mCurrentKeys, mCurrentDescriptors, vLapping);
    
    if (mCurrentKeys.empty()) {
        return Sophus::SE3f();
    }

    if (mState == NOT_INITIALIZED) {
        mLastImage = im.clone();
        mLastKeys = mCurrentKeys;
        mLastDescriptors = mCurrentDescriptors.clone();
        mState = INITIALIZING;
        std::cout << "MarkSLAM: First frame captured. Waiting for movement..." << std::endl;
        return Sophus::SE3f();
    }
    else if (mState == INITIALIZING) {
        if (InitializeFromHomography(mLastKeys, mLastDescriptors, mCurrentKeys, mCurrentDescriptors)) {
            mState = TRACKING;
            std::cout << "MarkSLAM: Planar Initialization SUCCESS!" << std::endl;
        } else {
             mLastImage = im.clone();
             mLastKeys = mCurrentKeys;
             mLastDescriptors = mCurrentDescriptors.clone();
        }
    }
    else if (mState == TRACKING) {
        bool bOK = TrackLocalMap();
        if (!bOK) {
             std::cout << "MarkSLAM: Lost tracking! Resetting..." << std::endl;
             mLocalMap.clear();
             mState = NOT_INITIALIZED;
        }
    }

    return mCurrentPose;
}

bool MarkSLAM::InitializeFromHomography(const std::vector<cv::KeyPoint>& kps1, const cv::Mat& desc1, 
                                        const std::vector<cv::KeyPoint>& kps2, const cv::Mat& desc2) {
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);

    if (matches.size() < 20) return false;

    std::vector<cv::Point2f> pts1, pts2;
    for(auto m : matches) {
        pts1.push_back(kps1[m.queryIdx].pt);
        pts2.push_back(kps2[m.trainIdx].pt);
    }

    cv::Mat mask;
    cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, mask);
    
    if (H.empty()) return false;
    
    int inliers = cv::countNonZero(mask);
    if (inliers < 15) return false;

    std::vector<cv::Mat> Rs, ts, normals;
    int solutions = cv::decomposeHomographyMat(H, mK, Rs, ts, normals);
    
    if (solutions > 0) {
       // Assume the first solution is correct for now (naive)
       cv::Mat R = Rs[0];
       cv::Mat t = ts[0];
       
       Eigen::Matrix3f R_eigen;
       Eigen::Vector3f t_eigen;
       cv::cv2eigen(R, R_eigen);
       cv::cv2eigen(t, t_eigen);
       
       mCurrentPose = Sophus::SE3f(R_eigen, t_eigen);
       
       // Construct Projection Matrices P1 (Identity) and P2 (R,t)
       cv::Mat P1 = cv::Mat::eye(3, 4, CV_32F);
       cv::Mat K_32F; mK.convertTo(K_32F, CV_32F);
       P1 = K_32F * P1;
       
       cv::Mat Rt = cv::Mat::zeros(3, 4, CV_32F);
       R.convertTo(Rt(cv::Rect(0,0,3,3)), CV_32F);
       t.convertTo(Rt(cv::Rect(3,0,1,3)), CV_32F);
       cv::Mat P2 = K_32F * Rt;

       // Triangulate points
       for (int i = 0; i < matches.size(); i++) {
           if (mask.at<uchar>(i)) {
               cv::Point2f p1 = pts1[i];
               cv::Point2f p2 = pts2[i];
               
               cv::Mat p4D;
               std::vector<cv::Point2f> vec_p1; vec_p1.push_back(p1);
               std::vector<cv::Point2f> vec_p2; vec_p2.push_back(p2);
               
               cv::triangulatePoints(P1, P2, vec_p1, vec_p2, p4D);
               
               PlanarMapPoint mp;
               float w = p4D.at<float>(3);
               if(abs(w) < 0.0001) continue;
               mp.pos3D.x = p4D.at<float>(0) / w;
               mp.pos3D.y = p4D.at<float>(1) / w;
               mp.pos3D.z = p4D.at<float>(2) / w;
               
               // Basic check: depth positive
               if (mp.pos3D.z > 0) {
                   mp.descriptor = desc2.row(matches[i].trainIdx).clone();
                   mp.lastSeenFrameId = 0; 
                   mLocalMap.push_back(mp);
               }
           }
       }
       
       if (mLocalMap.size() < 10) return false;
       return true;
    }

    return false;
}

bool MarkSLAM::TrackLocalMap() {
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    
    cv::Mat mapDesc;
    for(const auto& mp : mLocalMap) {
        mapDesc.push_back(mp.descriptor);
    }
    
    if (mapDesc.empty()) return false;

    std::vector<cv::DMatch> matches;
    matcher.match(mapDesc, mCurrentDescriptors, matches);
    
    for(auto m : matches) {
        if(m.distance < 50) { // Hamming distance threshold
            objectPoints.push_back(mLocalMap[m.queryIdx].pos3D);
            imagePoints.push_back(mCurrentKeys[m.trainIdx].pt);
        }
    }
    
    if (objectPoints.size() < 6) return false;
    
    cv::Mat rvec, tvec;
    Eigen::Vector3f t_eig = mCurrentPose.translation();
    Eigen::Matrix3f R_eig = mCurrentPose.rotationMatrix();
    
    cv::Mat R_cv, t_cv;
    cv::eigen2cv(R_eig, R_cv);
    cv::eigen2cv(t_eig, t_cv);
    cv::Rodrigues(R_cv, rvec);
    
    bool pnpSuccess = cv::solvePnP(objectPoints, imagePoints, mK, mDistCoef, rvec, t_cv, true);
    
    if(pnpSuccess) {
        cv::Rodrigues(rvec, R_cv);
        cv::cv2eigen(R_cv, R_eig);
        cv::cv2eigen(t_cv, t_eig);
        mCurrentPose = Sophus::SE3f(R_eig, t_eig);
        return true;
    }
    
    return false;
}
