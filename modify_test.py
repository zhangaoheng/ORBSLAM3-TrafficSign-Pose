import os

file_path = 'test/test.cc'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replacement 1
old_str_1 = '''    // 写入文件头 (CSV: timestamp,tx,ty,tz,qx,qy,qz,qw)
    ofs << "timestamp,tx,ty,tz,qx,qy,qz,qw\\n";
    
    int saved_count = 0;
    for (ORB_SLAM3::KeyFrame* pKF : vpKFs) {
        if (!pKF || pKF->isBad()) continue;
        Sophus::SE3f Twc = pKF->GetPoseInverse();
        Eigen::Vector3f t = Twc.translation();
        Eigen::Quaternionf q = Twc.unit_quaternion();
        double timestamp = pKF->mTimeStamp;
        ofs << std::fixed << std::setprecision(6)
            << timestamp << ","
            << t[0] << "," << t[1] << "," << t[2] << ","
            << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\\n";
        saved_count++;
    }'''

new_str_1 = '''    // 写入文件头 (CSV: frame,tx,ty,tz,qx,qy,qz,qw)
    ofs << "frame,tx,ty,tz,qx,qy,qz,qw\\n";
    
    int saved_count = 0;
    for (ORB_SLAM3::KeyFrame* pKF : vpKFs) {
        if (!pKF || pKF->isBad()) continue;
        Sophus::SE3f Twc = pKF->GetPoseInverse();
        Eigen::Vector3f t = Twc.translation();
        Eigen::Quaternionf q = Twc.unit_quaternion();
        double timestamp = pKF->mTimeStamp;
        ofs << (long long)timestamp << ","
            << std::fixed << std::setprecision(6)
            << t[0] << "," << t[1] << "," << t[2] << ","
            << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\\n";
        saved_count++;
    }'''

# Replacement 2
old_str_2 = '''        auto now = chrono::system_clock::now();
        auto timestamp = chrono::duration_cast<chrono::milliseconds>(now - start);
        
        try {
            SLAM.TrackMonocular(slam_frame, double(timestamp.count()) / 1000.0);
        } catch (const exception& e) {'''

new_str_2 = '''        // auto now = chrono::system_clock::now();
        // auto timestamp = chrono::duration_cast<chrono::milliseconds>(now - start);
        
        try {
            SLAM.TrackMonocular(slam_frame, (double)frame_count);
        } catch (const exception& e) {'''

if old_str_1 in content:
    content = content.replace(old_str_1, new_str_1)
    print("Replacement 1 success")
else:
    print("Replacement 1 failed: string not found")

if old_str_2 in content:
    content = content.replace(old_str_2, new_str_2)
    print("Replacement 2 success")
else:
    print("Replacement 2 failed: string not found")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
