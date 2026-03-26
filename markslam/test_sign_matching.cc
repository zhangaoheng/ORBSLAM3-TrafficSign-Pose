#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "ORBextractor.h"

using namespace std;
using namespace ORB_SLAM3;

// Recognition function prototype
bool recognizeSign(const cv::Mat &im1, const cv::Mat &im2, string outDebugPath = "") {
    // 1. Initialize ORB Extractor
    int nFeatures = 2000; 
    float scaleFactor = 1.2f;
    int nLevels = 8;
    int iniThFAST = 20; 
    int minThFAST = 7;

    ORBextractor extractor(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);

    // 2. Extract Features
    vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    vector<int> vLapping = {0, 1000}; 

    extractor(im1, cv::Mat(), kp1, desc1, vLapping);
    extractor(im2, cv::Mat(), kp2, desc2, vLapping);

    cout << "Features in Image 1: " << kp1.size() << endl;
    cout << "Features in Image 2: " << kp2.size() << endl;

    if(kp1.empty() || kp2.empty()) {
        cerr << "Not enough features to match." << endl;
        return false;
    }

    // 3. Match Features (Brute Force Hamming)
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    vector<vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);

    // 4. Filter Matches (Ratio Test)
    const float ratio_thresh = 0.70f;
    vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    cout << "Good Matches found: " << good_matches.size() << endl;

    // 5. Draw Debug Image
    if(!outDebugPath.empty()) {
        cv::Mat img_matches;
        cv::drawMatches(im1, kp1, im2, kp2, good_matches, img_matches, 
                        cv::Scalar::all(-1), cv::Scalar::all(-1), 
                        vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imwrite(outDebugPath, img_matches);
    }

    // 6. Decision Logic
    return good_matches.size() >= 8; 
}

int main(int argc, char** argv) {
    if(argc != 4) {
        cerr << "Usage: ./test_sign_matching <image1_path> <image2_path> <output_image_path>" << endl;
        return 1;
    }

    string img1Path = argv[1];
    string img2Path = argv[2];
    string outPath = argv[3];

    cv::Mat im1 = cv::imread(img1Path, cv::IMREAD_GRAYSCALE);
    cv::Mat im2 = cv::imread(img2Path, cv::IMREAD_GRAYSCALE);
    
    if(im1.empty() || im2.empty()) {
        cerr << "Failed to load images" << endl;
        return 1;
    }

    bool isSame = recognizeSign(im1, im2, outPath);

    if(isSame) {
        cout << "\n>>> RESULT: MATCH CONFIRMED (Same Sign) <<<" << endl;
    } else {
        cout << "\n>>> RESULT: NO MATCH <<<" << endl;
    }

    return 0;
}
