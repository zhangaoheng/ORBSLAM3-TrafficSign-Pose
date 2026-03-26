#ifndef MARKSLAM_H
#define MARKSLAM_H

#include <vector>
#include <string>
#include <thread>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>

// Reuse ORB-SLAM3 components
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "Frame.h"
#include "Converter.h"

using namespace std;

class MarkSLAM {
public:
    enum State {
        NOT_INITIALIZED,
        INITIALIZING,
        TRACKING,
        LOST
    };

    MarkSLAM(string strVocFile, string strSettingsFile);

    // Main interface
    Sophus::SE3f TrackMonocular(const cv::Mat &im, const double &timestamp);

    State GetState() { return mState; }

private:
    // Core components from ORB-SLAM3
    ORB_SLAM3::ORBextractor* mpORBextractor;
    
    // Structure to hold a simple MapPoint for planar tracking
    struct PlanarMapPoint {
        cv::Point3f pos3D; // In World Frame (Plane Frame)
        cv::Mat descriptor;
        int lastSeenFrameId;
    };

    // Main storage
    std::vector<PlanarMapPoint> mLocalMap;
    
    // Tracking State
    State mState;
    
    // Frames
    cv::Mat mLastImage;
    std::vector<cv::KeyPoint> mLastKeys;
    cv::Mat mLastDescriptors;
    
    cv::Mat mCurrentDescriptors;
    std::vector<cv::KeyPoint> mCurrentKeys;
    
    // Pose: World (Plane) -> Camera
    Sophus::SE3f mCurrentPose;

    // Initialization helpers
    bool InitializeFromHomography(const std::vector<cv::KeyPoint>& kps1, const cv::Mat& desc1,
                                  const std::vector<cv::KeyPoint>& kps2, const cv::Mat& desc2);

    // Tracking helpers (PnP)
    bool TrackLocalMap();

    // Configuration
    cv::Mat mK; // Intrinsic matrix
    cv::Mat mDistCoef;
    int mnFeatures;
    float mScaleFactor;
    int mnLevels;
};

#endif // MARKSLAM_H
