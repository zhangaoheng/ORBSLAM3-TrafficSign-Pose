/**
 * Pure monocular runner (fast, no usleep)
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <System.h>

using namespace std;

int main(int argc, char *argv[]) {
    if (argc < 5) {
        cerr << "Usage: ./run_mono_fast vocab config euroc_path times.txt [name]" << endl;
        return 1;
    }
    string name = (argc >= 6) ? argv[5] : "trajectory";
    string pathCam0 = string(argv[3]) + "/mav0/cam0/data";

    // Load times
    ifstream fTimes(argv[4]);
    vector<double> vTimes;
    vector<string> vImages;
    string line;
    while (getline(fTimes, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        double t_ns; ss >> t_ns;
        vImages.push_back(pathCam0 + "/" + to_string((long long)t_ns) + ".png");
        vTimes.push_back(t_ns / 1e9);
    }
    cout << "Loaded " << vTimes.size() << " images" << endl;

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);

    for (size_t i = 0; i < vTimes.size(); i++) {
        cv::Mat im = cv::imread(vImages[i], cv::IMREAD_UNCHANGED);
        if (im.empty()) {
            cerr << "Failed: " << vImages[i] << endl;
            continue;
        }
        SLAM.TrackMonocular(im, vTimes[i]);
        if ((i+1) % 500 == 0) cout << "  Processed " << (i+1) << "/" << vTimes.size() << endl;
    }

    SLAM.Shutdown();
    SLAM.SaveTrajectoryEuRoC("AllFrames_" + name + ".txt");
    SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrames_" + name + ".txt");
    cout << "Saved: AllFrames_" + name + ".txt, KeyFrames_" + name + ".txt" << endl;
    return 0;
}
