#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <sys/stat.h> // For mkdir
#include "ORBextractor.h"

using namespace std;
using namespace ORB_SLAM3;

int main(int argc, char** argv) {
    // Hardcoded paths
    string strImageFolder = "/home/zah/ORB_SLAM3-master/markslam/data/20260127_163648/predictions_with_boxes";
    string strOutputFolder = "/home/zah/ORB_SLAM3-master/markslam/data/20260127_163648/box_orb";

    // Create output directory
    string command = "mkdir -p " + strOutputFolder;
    int result = system(command.c_str());
    if (result != 0) {
        cerr << "Failed to create output directory: " << strOutputFolder << endl;
        return 1;
    }

    vector<string> vImageFilenames;
    // Try different extensions
    vector<string> vPatterns;
    vPatterns.push_back(strImageFolder + "/*.png");
    vPatterns.push_back(strImageFolder + "/*.jpg");
    vPatterns.push_back(strImageFolder + "/*.jpeg");

    for(auto& pattern : vPatterns) {
        vector<string> temp;
        cv::glob(pattern, temp, false);
        vImageFilenames.insert(vImageFilenames.end(), temp.begin(), temp.end());
    }

    if (vImageFilenames.empty()) {
        cerr << "No images found in " << strImageFolder << endl;
        return 1;
    }
    
    // ORB Parameters
    int nFeatures = 1000;
    float scaleFactor = 1.2f;
    int nLevels = 8;
    int iniThFAST = 20;
    int minThFAST = 7;

    ORBextractor* mpORBextractor = new ORBextractor(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);

    cout << "Found " << vImageFilenames.size() << " images." << endl;
    cout << "Saving results to " << strOutputFolder << endl;

    for(const string &filename : vImageFilenames) {
        cv::Mat im = cv::imread(filename, cv::IMREAD_GRAYSCALE); // ORB needs grayscale
        if(im.empty()) {
            cerr << "Failed to load " << filename << endl;
            continue;
        }

        vector<cv::KeyPoint> vKeys;
        cv::Mat mDescriptors;
        vector<int> vLapping = {0, 1000}; 

        (*mpORBextractor)(im, cv::Mat(), vKeys, mDescriptors, vLapping);

        // Draw keypoints
        cv::Mat imColor;
        cv::cvtColor(im, imColor, cv::COLOR_GRAY2BGR);
        cv::drawKeypoints(imColor, vKeys, imColor, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);

        // Extract filename from path
        size_t lastSlash = filename.find_last_of("/\\");
        string baseFilename = (lastSlash == string::npos) ? filename : filename.substr(lastSlash + 1);
        string outputFilename = strOutputFolder + "/" + baseFilename;

        cv::imwrite(outputFilename, imColor);
        
        cout << "Processed " << filename << ", features: " << vKeys.size() << " -> Saved to " << outputFilename << endl;
    }

    cout << "Done." << endl;

    delete mpORBextractor;

    return 0;
}
