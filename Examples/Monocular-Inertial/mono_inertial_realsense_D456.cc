/**
 * Live D456 monocular-inertial odometry and lossless dataset recorder.
 *
 * One RealSense pipeline feeds both ORB-SLAM3 and an asynchronous EuRoC writer,
 * so online and offline experiments see exactly the same image/IMU timestamps.
 */

#include <System.h>

#include <librealsense2/rs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

volatile std::sig_atomic_t g_stop = 0;

void signalHandler(int) { g_stop = 1; }

struct ImagePacket {
    std::int64_t timestamp_ns;
    std::int64_t host_utc_ns;
    double timestamp_s;
    cv::Mat image;
};

struct ImuPacket {
    std::int64_t timestamp_ns;
    double timestamp_s;
    rs2_vector gyro;
    rs2_vector accel;
};

struct TimedMotion {
    std::int64_t timestamp_ns;
    double timestamp_s;
    rs2_vector value;
};

bool makeDirectory(const std::string &path)
{
    if (path.empty()) return false;
    std::string current;
    if (path[0] == '/') current = "/";
    std::stringstream ss(path);
    std::string part;
    while (std::getline(ss, part, '/')) {
        if (part.empty()) continue;
        if (!current.empty() && current.back() != '/') current += '/';
        current += part;
        if (::mkdir(current.c_str(), 0775) != 0 && errno != EEXIST) {
            std::cerr << "Cannot create directory " << current << ": "
                      << std::strerror(errno) << std::endl;
            return false;
        }
    }
    return true;
}

bool pathExists(const std::string &path)
{
    struct stat info {};
    return ::stat(path.c_str(), &info) == 0;
}

std::string absolutePath(const std::string &path)
{
    char resolved[PATH_MAX];
    return ::realpath(path.c_str(), resolved) ? std::string(resolved) : path;
}

std::string dirname(std::string path)
{
    const std::size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? "." : path.substr(0, slash);
}

rs2_vector interpolate(const TimedMotion &previous,
                       const TimedMotion &current,
                       double target_s)
{
    if (previous.timestamp_s <= 0.0 || current.timestamp_s <= previous.timestamp_s)
        return current.value;
    if (target_s <= previous.timestamp_s) return previous.value;
    if (target_s >= current.timestamp_s) return current.value;
    const double alpha = (target_s - previous.timestamp_s) /
                         (current.timestamp_s - previous.timestamp_s);
    rs2_vector result;
    result.x = previous.value.x + (current.value.x - previous.value.x) * alpha;
    result.y = previous.value.y + (current.value.y - previous.value.y) * alpha;
    result.z = previous.value.z + (current.value.z - previous.value.z) * alpha;
    return result;
}

void writeSettings(const std::string &path,
                   const rs2_intrinsics &intr,
                   const rs2_extrinsics &extr,
                   float fps,
                   float imu_fps)
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write ORB settings: " + path);
    out << std::setprecision(10);
    out << "%YAML:1.0\n\n"
        << "File.version: \"1.0\"\n"
        << "Camera.type: \"PinHole\"\n\n"
        << "Camera1.fx: " << intr.fx << "\n"
        << "Camera1.fy: " << intr.fy << "\n"
        << "Camera1.cx: " << intr.ppx << "\n"
        << "Camera1.cy: " << intr.ppy << "\n"
        << "Camera1.k1: " << intr.coeffs[0] << "\n"
        << "Camera1.k2: " << intr.coeffs[1] << "\n"
        << "Camera1.p1: " << intr.coeffs[2] << "\n"
        << "Camera1.p2: " << intr.coeffs[3] << "\n"
        << "Camera1.k3: " << intr.coeffs[4] << "\n"
        << "Camera.width: " << intr.width << "\n"
        << "Camera.height: " << intr.height << "\n"
        << "Camera.fps: " << static_cast<int>(std::lround(fps)) << "\n"
        << "Camera.RGB: 0\n\n"
        << "IMU.T_b_c1: !!opencv-matrix\n"
        << "   rows: 4\n   cols: 4\n   dt: f\n   data: [";
    out << std::setprecision(10);
    for (int row = 0; row < 3; ++row) {
        if (row) out << "          ";
        // librealsense stores rs2_extrinsics::rotation column-major.
        out << extr.rotation[row] << ", "
            << extr.rotation[row + 3] << ", "
            << extr.rotation[row + 6] << ", "
            << extr.translation[row] << ",\n";
    }
    out << "          0.0, 0.0, 0.0, 1.0]\n"
        << "IMU.Frequency: " << std::fixed << std::setprecision(1) << imu_fps << "\n"
        << "IMU.NoiseGyro: 1.5e-1\n"
        << "IMU.NoiseAcc: 1.5e0\n"
        << "IMU.GyroWalk: 1e-4\n"
        << "IMU.AccWalk: 1e-3\n"
        << "IMU.InsertKFsWhenLost: 1\n\n"
        << "ORBextractor.nFeatures: 2000\n"
        << "ORBextractor.scaleFactor: 1.2\n"
        << "ORBextractor.nLevels: 8\n"
        << "ORBextractor.iniThFAST: 10\n"
        << "ORBextractor.minThFAST: 5\n\n"
        << "Viewer.KeyFrameSize: 0.05\n"
        << "Viewer.KeyFrameLineWidth: 1.0\n"
        << "Viewer.GraphLineWidth: 0.9\n"
        << "Viewer.PointSize: 2.0\n"
        << "Viewer.CameraSize: 0.08\n"
        << "Viewer.CameraLineWidth: 3.0\n"
        << "Viewer.ViewpointX: 0.0\n"
        << "Viewer.ViewpointY: -0.7\n"
        << "Viewer.ViewpointZ: -1.8\n"
        << "Viewer.ViewpointF: 500.0\n";
}

void writeOfflineRunner(const std::string &path,
                        const std::string &project_root,
                        const std::string &dataset)
{
    std::ofstream out(path);
    out << "#!/usr/bin/env bash\nset -euo pipefail\n"
        << "PROJECT_ROOT=\"" << project_root << "\"\n"
        << "DATASET=\"" << dataset << "\"\n"
        << "export LD_LIBRARY_PATH=/opt/ros/lyrical/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}\n"
        << "cd \"$DATASET\"\n"
        << "\"$PROJECT_ROOT/Examples/Monocular-Inertial/mono_inertial_euroc\" \\\n"
        << "  \"$PROJECT_ROOT/Vocabulary/ORBvoc.txt\" \\\n"
        << "  \"$DATASET/orbslam_d456_mono_imu.yaml\" \\\n"
        << "  \"$DATASET\" \"$DATASET/times.txt\" offline_replay\n";
    out.close();
    ::chmod(path.c_str(), 0755);
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " vocabulary output_directory [duration_seconds] [viewer:0|1]"
                     " [preview:0|1]\n";
        return 2;
    }

    const std::string vocabulary = absolutePath(argv[1]);
    const std::string output = absolutePath(argv[2]);
    const double duration_s = argc >= 4 ? std::stod(argv[3]) : 0.0;
    const bool use_viewer = argc < 5 || std::string(argv[4]) != "0";
    const bool use_preview = argc < 6 ? use_viewer : std::string(argv[5]) != "0";
    if (!pathExists(vocabulary)) {
        std::cerr << "Vocabulary not found: " << vocabulary << std::endl;
        return 2;
    }
    if (pathExists(output)) {
        std::cerr << "Refusing to overwrite existing output: " << output << std::endl;
        return 2;
    }
    if (!makeDirectory(output + "/mav0/cam0/data") ||
        !makeDirectory(output + "/mav0/imu0")) return 2;
    if (::symlink("mav0/cam0/data", (output + "/rgb").c_str()) != 0) {
        std::cerr << "Cannot create RGB compatibility link: " << std::strerror(errno)
                  << std::endl;
        return 2;
    }

    const std::string incomplete = output + "/.incomplete";
    { std::ofstream marker(incomplete); marker << "recording\n"; }

    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    try {
        std::unique_ptr<rs2::pipeline> pipe(new rs2::pipeline());
        rs2::config config;
        config.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 30);
        config.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 100);
        config.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 200);

        rs2::pipeline_profile resolved = config.resolve(*pipe);
        rs2::device device = resolved.get_device();
        const std::string serial = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
        for (rs2::sensor sensor : device.query_sensors()) {
            if (sensor.supports(RS2_OPTION_GLOBAL_TIME_ENABLED))
                sensor.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 1.f);
            if (sensor.supports(RS2_OPTION_ENABLE_MOTION_CORRECTION))
                sensor.set_option(RS2_OPTION_ENABLE_MOTION_CORRECTION, 0.f);
        }

        auto color_profile = resolved.get_stream(RS2_STREAM_COLOR)
                                      .as<rs2::video_stream_profile>();
        auto gyro_profile = resolved.get_stream(RS2_STREAM_GYRO);
        const rs2_intrinsics intr = color_profile.get_intrinsics();
        const rs2_extrinsics extr = color_profile.get_extrinsics_to(gyro_profile);
        const std::string settings = output + "/orbslam_d456_mono_imu.yaml";
        writeSettings(settings, intr, extr,
                      static_cast<float>(color_profile.fps()),
                      static_cast<float>(gyro_profile.fps()));

        const std::string executable = absolutePath(argv[0]);
        const std::string project_root = dirname(dirname(dirname(executable)));
        writeOfflineRunner(output + "/run_offline.sh", project_root, output);

        std::cout << "D456 S/N " << serial << "  " << intr.width << "x"
                  << intr.height << "@" << color_profile.fps() << std::endl;
        std::cout << "Loading ORB-SLAM3 vocabulary before opening the stream..." << std::endl;
        ORB_SLAM3::System slam(vocabulary, settings,
                              ORB_SLAM3::System::IMU_MONOCULAR, use_viewer);

        std::mutex mutex;
        std::condition_variable sensor_cv;
        std::condition_variable writer_cv;
        std::deque<ImagePacket> slam_images;
        std::deque<ImuPacket> slam_imus;
        std::deque<ImagePacket> write_images;
        std::deque<ImuPacket> write_imus;
        std::deque<TimedMotion> pending_gyros;
        TimedMotion previous_accel{0, 0.0, {0.f, 0.f, 0.f}};
        TimedMotion current_accel{0, 0.0, {0.f, 0.f, 0.f}};
        std::atomic<bool> recording{false};
        std::atomic<bool> writer_stop{false};
        std::atomic<bool> writer_failed{false};
        std::atomic<bool> timestamp_domain_mismatch{false};
        std::atomic<std::size_t> image_count{0};
        std::atomic<std::size_t> imu_count{0};
        std::atomic<std::size_t> discarded_color_frames{0};
        std::atomic<std::size_t> max_image_backlog{0};
        std::int64_t latest_slam_imu_ns = 0;
        std::int64_t last_color_timestamp_ns = 0;
        bool have_timestamp_domain = false;
        rs2_timestamp_domain timestamp_domain = RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK;

        auto publishImuLocked = [&](bool force) {
            while (!pending_gyros.empty() && current_accel.timestamp_ns != 0 &&
                   (force || pending_gyros.front().timestamp_s <= current_accel.timestamp_s)) {
                const TimedMotion gyro = pending_gyros.front();
                pending_gyros.pop_front();
                ImuPacket packet{gyro.timestamp_ns, gyro.timestamp_s, gyro.value,
                                 interpolate(previous_accel, current_accel,
                                             gyro.timestamp_s)};
                slam_imus.push_back(packet);
                write_imus.push_back(packet);
                latest_slam_imu_ns = std::max(latest_slam_imu_ns, packet.timestamp_ns);
            }
        };

        auto callback = [&](const rs2::frame &frame) {
            try {
                std::unique_lock<std::mutex> lock(mutex);
                const rs2_timestamp_domain frame_domain = frame.get_frame_timestamp_domain();
                if (!have_timestamp_domain) {
                    timestamp_domain = frame_domain;
                    have_timestamp_domain = true;
                } else if (frame_domain != timestamp_domain) {
                    timestamp_domain_mismatch = true;
                    std::cerr << "Camera and IMU timestamp domains differ; stopping."
                              << std::endl;
                    g_stop = 1;
                    return;
                }
                if (rs2::motion_frame motion = frame.as<rs2::motion_frame>()) {
                    const double timestamp_ms = motion.get_timestamp();
                    const std::int64_t timestamp_ns =
                        static_cast<std::int64_t>(std::llround(timestamp_ms * 1e6));
                    const TimedMotion sample{timestamp_ns, timestamp_ns * 1e-9,
                                             motion.get_motion_data()};
                    if (motion.get_profile().stream_type() == RS2_STREAM_ACCEL) {
                        previous_accel = current_accel;
                        current_accel = sample;
                        if (recording.load()) publishImuLocked(false);
                    } else if (recording.load() &&
                               motion.get_profile().stream_type() == RS2_STREAM_GYRO) {
                        pending_gyros.push_back(sample);
                        publishImuLocked(false);
                    }
                    lock.unlock();
                    sensor_cv.notify_one();
                    writer_cv.notify_one();
                    return;
                }

                rs2::video_frame color(nullptr);
                if (rs2::frameset frames = frame.as<rs2::frameset>()) {
                    color = frames.get_color_frame();
                } else {
                    color = frame.as<rs2::video_frame>();
                }
                if (!recording.load() || !color ||
                    color.get_profile().stream_type() != RS2_STREAM_COLOR) return;

                publishImuLocked(true);
                const std::int64_t timestamp_ns = static_cast<std::int64_t>(
                    std::llround(color.get_timestamp() * 1e6));
                // A pipeline callback can occasionally deliver a composite
                // frame containing an already-seen color frame. Never write
                // duplicate filenames or feed repeated timestamps to SLAM.
                if (timestamp_ns <= last_color_timestamp_ns) {
                    ++discarded_color_frames;
                    return;
                }
                last_color_timestamp_ns = timestamp_ns;
                const std::int64_t host_utc_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
                cv::Mat view(cv::Size(intr.width, intr.height), CV_8UC3,
                             const_cast<void *>(color.get_data()), cv::Mat::AUTO_STEP);
                ImagePacket packet{timestamp_ns, host_utc_ns,
                                   timestamp_ns * 1e-9, view.clone()};
                slam_images.push_back(packet);
                write_images.push_back(packet);
                const std::size_t backlog = slam_images.size();
                std::size_t old_max = max_image_backlog.load();
                while (backlog > old_max &&
                       !max_image_backlog.compare_exchange_weak(old_max, backlog)) {}
                lock.unlock();
                sensor_cv.notify_one();
                writer_cv.notify_one();
            } catch (const std::exception &error) {
                std::cerr << "RealSense callback error: " << error.what() << std::endl;
                g_stop = 1;
            }
        };

        auto writerFunction = [&] {
            std::ofstream camera_csv(output + "/mav0/cam0/data.csv");
            std::ofstream imu_csv(output + "/mav0/imu0/data.csv");
            std::ofstream times(output + "/times.txt");
            std::ofstream clock_csv(output + "/device_host_clock.csv");
            camera_csv << "#timestamp [ns],filename\n";
            imu_csv << "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
                       "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],"
                       "a_RS_S_z [m s^-2]\n";
            clock_csv << "device_ns,host_utc_ns\n";
            const std::vector<int> png_params{cv::IMWRITE_PNG_COMPRESSION, 3};
            while (true) {
                ImagePacket image;
                ImuPacket imu;
                bool has_image = false, has_imu = false;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    writer_cv.wait(lock, [&] {
                        return writer_stop.load() || !write_images.empty() ||
                               !write_imus.empty();
                    });
                    if (!write_imus.empty()) {
                        imu = write_imus.front();
                        write_imus.pop_front();
                        has_imu = true;
                    }
                    if (!write_images.empty()) {
                        image = write_images.front();
                        write_images.pop_front();
                        has_image = true;
                    }
                    if (!has_image && !has_imu && writer_stop.load()) break;
                }
                if (has_imu) {
                    imu_csv << imu.timestamp_ns << ',' << std::setprecision(9)
                            << imu.gyro.x << ',' << imu.gyro.y << ',' << imu.gyro.z << ','
                            << imu.accel.x << ',' << imu.accel.y << ',' << imu.accel.z << '\n';
                    ++imu_count;
                }
                if (has_image) {
                    const std::string filename = std::to_string(image.timestamp_ns) + ".png";
                    const std::string color_path = output + "/mav0/cam0/data/" + filename;
                    const std::string color_tmp = color_path + ".tmp.png";
                    // Publish each PNG with an atomic rename so any concurrent
                    // consumer can never observe a partially-written file.
                    if (!cv::imwrite(color_tmp, image.image, png_params) ||
                        std::rename(color_tmp.c_str(), color_path.c_str()) != 0) {
                        writer_failed = true;
                        g_stop = 1;
                    } else {
                        camera_csv << image.timestamp_ns << ',' << filename << '\n';
                        times << image.timestamp_ns << '\n';
                        clock_csv << image.timestamp_ns << ',' << image.host_utc_ns << '\n';
                        const std::size_t saved = ++image_count;
                        // Preserve a usable index even if ORB-SLAM or a display
                        // driver fails later in a long recording.
                        if (saved % 30 == 0) {
                            camera_csv.flush();
                            imu_csv.flush();
                            times.flush();
                            clock_csv.flush();
                        }
                    }
                }
            }
            camera_csv.flush();
            imu_csv.flush();
            times.flush();
            clock_csv.flush();
        };

        rs2::pipeline_profile active;
        for (int attempt = 1; attempt <= 3; ++attempt) {
            try {
                active = pipe->start(config, callback);
                break;
            } catch (const rs2::error &error) {
                const std::string message = error.what();
                const bool busy = message.find("resource busy") != std::string::npos ||
                                  message.find("errno=16") != std::string::npos;
                if (!busy || attempt == 3) {
                    if (busy) {
                        std::cerr
                            << "\nD456 video interfaces are still busy after 3 attempts.\n"
                            << "Close realsense-viewer/other camera programs, then run:\n"
                            << "  sudo fuser -v /dev/video*\n"
                            << "If no Linux process owns them under WSL, detach and attach the "
                               "RealSense again with usbipd on Windows.\n";
                    }
                    throw;
                }
                std::cerr << "D456 is busy; retrying pipeline start (" << attempt
                          << "/3) in 2 seconds..." << std::endl;
                // A failed librealsense start may already have opened the HID
                // interface. Destroy the whole pipeline before retrying;
                // retrying the same object produces "Hid device is already opened".
                try {
                    pipe->stop();
                } catch (...) {
                }
                pipe.reset();
                std::this_thread::sleep_for(std::chrono::seconds(2));
                pipe.reset(new rs2::pipeline());
            }
        }
        std::thread writer(writerFunction);
        std::cout << "Warming up D456 for 2 seconds..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
        recording = true;
        const auto start = std::chrono::steady_clock::now();
        std::cout << "Recording and live visual-inertial tracking: " << output << std::endl;
        std::cout << "Keep still for about 2 s, then translate and rotate the camera. Ctrl+C to stop."
                  << std::endl;

        std::ofstream realtime_pose(output + "/realtime_odometry.csv");
        realtime_pose << "#timestamp_ns,tracking_state,valid,tx,ty,tz,qx,qy,qz,qw\n";
        if (use_preview)
            cv::namedWindow("D456 Live Preview", cv::WINDOW_NORMAL);
        std::size_t processed = 0;
        std::size_t valid_poses = 0;
        std::size_t dropped_slam_frames = 0;
        bool ever_tracking = false;
        int final_tracking_state = 0;
        while (!g_stop && !slam.isShutDown()) {
            if (duration_s > 0.0) {
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();
                if (elapsed >= duration_s) break;
            }

            ImagePacket image;
            std::vector<ORB_SLAM3::IMU::Point> measurements;
            std::size_t backlog_after_pop = 0;
            {
                std::unique_lock<std::mutex> lock(mutex);
                sensor_cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                    return g_stop || (!slam_images.empty() &&
                           latest_slam_imu_ns >= slam_images.front().timestamp_ns);
                });
                if (slam_images.empty() ||
                    latest_slam_imu_ns < slam_images.front().timestamp_ns) continue;
                // Software-rendered Pangolin can temporarily consume most CPU.
                // Drop only stale image frames (never IMU) until the queue is
                // short; the next TrackMonocular call receives every IMU sample
                // spanning the larger image interval.
                while (slam_images.size() > 3 &&
                       latest_slam_imu_ns >= slam_images[1].timestamp_ns) {
                    slam_images.pop_front();
                    ++dropped_slam_frames;
                }
                image = slam_images.front();
                slam_images.pop_front();
                backlog_after_pop = slam_images.size();
                while (!slam_imus.empty() &&
                       slam_imus.front().timestamp_ns <= image.timestamp_ns) {
                    const ImuPacket packet = slam_imus.front();
                    slam_imus.pop_front();
                    measurements.emplace_back(packet.accel.x, packet.accel.y, packet.accel.z,
                                              packet.gyro.x, packet.gyro.y, packet.gyro.z,
                                              packet.timestamp_s);
                }
            }

            if (processed > 0 && measurements.empty()) {
                ++dropped_slam_frames;
                continue;
            }

            const Sophus::SE3f Tcw = slam.TrackMonocular(
                image.image, image.timestamp_s, measurements,
                std::to_string(image.timestamp_ns) + ".png");
            const int state = slam.GetTrackingState();
            final_tracking_state = state;
            // RECENTLY_LOST can return a predicted/stale pose; keep it in the
            // state log but never label it as an experiment-grade observation.
            const bool valid = state == 2 || state == 5;
            realtime_pose << image.timestamp_ns << ',' << state << ',' << (valid ? 1 : 0);
            if (valid) {
                const Sophus::SE3f Twc = Tcw.inverse();
                const Eigen::Vector3f t = Twc.translation();
                const Eigen::Quaternionf q = Twc.unit_quaternion();
                realtime_pose << ',' << std::setprecision(9) << t.x() << ',' << t.y() << ','
                              << t.z() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ','
                              << q.w();
                ever_tracking = true;
                ++valid_poses;
            } else {
                realtime_pose << ",,,,,,,";
            }
            realtime_pose << '\n';
            if (use_preview) {
                cv::Mat preview = image.image.clone();
                const std::string state_name =
                    valid ? "TRACKING" : (state == 1 ? "INITIALIZING" : "LOST");
                const std::string label = "ORB-SLAM3: " + state_name +
                                          "  frames=" + std::to_string(processed + 1);
                cv::putText(preview, label, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX,
                            0.7, valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 200, 255),
                            2, cv::LINE_AA);
                cv::imshow("D456 Live Preview", preview);
                const int key = cv::waitKey(1) & 0xff;
                if (key == 27 || key == 'q' || key == 'Q') g_stop = 1;
            }
            ++processed;
            if (processed % 30 == 0) {
                realtime_pose.flush();
                std::cout << "frames=" << processed << " state=" << state
                          << " imu_batch=" << measurements.size()
                          << " backlog=" << backlog_after_pop << std::endl;
            }
        }

        recording = false;
        try {
            pipe->stop();
        } catch (const rs2::error &error) {
            // Continue flushing disk queues and shutting SLAM down even when a
            // disconnected USB device makes pipeline::stop report an error.
            std::cerr << "Warning: D456 stop failed during shutdown: "
                      << error.what() << std::endl;
            writer_failed = true;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            publishImuLocked(true);
        }
        writer_stop = true;
        writer_cv.notify_all();
        writer.join();
        realtime_pose.close();
        if (use_preview)
            cv::destroyWindow("D456 Live Preview");

        std::cout << "Shutting down ORB-SLAM3..." << std::endl;
        slam.Shutdown();
        // ORB-SLAM3's EuRoC exporter dereferences the first keyframe. A run can
        // briefly reach OK and then reset its active map, leaving zero KFs;
        // exporting that empty map segfaults. Only export when tracking is
        // still valid at shutdown. realtime_odometry.csv remains available for
        // every run, including failed initialization attempts.
        const bool trajectory_saved =
            ever_tracking && (final_tracking_state == 2 || final_tracking_state == 5);
        if (trajectory_saved) {
            slam.SaveTrajectoryEuRoC(output + "/trajectory_euroc.txt");
            slam.SaveKeyFrameTrajectoryEuRoC(output + "/keyframes_euroc.txt");
        } else {
            std::cerr << "Warning: no valid final map; optimized trajectory not saved."
                      << std::endl;
        }

        std::ofstream metadata(output + "/live_metadata.json");
        metadata << "{\n"
                 << "  \"device_serial\": \"" << serial << "\",\n"
                 << "  \"recording_mode\": \"monocular_imu\",\n"
                 << "  \"depth_recorded\": false,\n"
                 << "  \"image_count\": " << image_count.load() << ",\n"
                 << "  \"imu_count\": " << imu_count.load() << ",\n"
                 << "  \"discarded_color_frames\": "
                 << discarded_color_frames.load() << ",\n"
                 << "  \"online_processed_frames\": " << processed << ",\n"
                 << "  \"online_valid_poses\": " << valid_poses << ",\n"
                 << "  \"dropped_slam_frames\": " << dropped_slam_frames << ",\n"
                 << "  \"final_tracking_state\": " << final_tracking_state << ",\n"
                 << "  \"optimized_trajectory_saved\": " << (trajectory_saved ? "true" : "false") << ",\n"
                 << "  \"max_slam_image_backlog\": " << max_image_backlog.load() << ",\n"
                 << "  \"slam_initialized\": " << (ever_tracking ? "true" : "false") << ",\n"
                 << "  \"writer_failed\": " << (writer_failed.load() ? "true" : "false") << ",\n"
                 << "  \"timestamp_domain\": \""
                 << (have_timestamp_domain ? rs2_timestamp_domain_to_string(timestamp_domain) : "unknown")
                 << "\",\n"
                 << "  \"timestamp_domain_mismatch\": "
                 << (timestamp_domain_mismatch.load() ? "true" : "false") << "\n"
                 << "}\n";
        metadata.close();
        ::unlink(incomplete.c_str());

        std::cout << "Saved " << image_count.load() << " images, " << imu_count.load()
                  << " IMU samples and " << valid_poses << " valid online poses." << std::endl;
        std::cout << "Offline replay: " << output << "/run_offline.sh" << std::endl;
        const int exit_code =
            writer_failed.load() || timestamp_domain_mismatch.load() || !trajectory_saved ? 1 : 0;
        // ORB-SLAM3's Viewer owns an OpenCV/Qt window from a worker thread.
        // Qt may segfault while destroying its global GUI objects at normal
        // process teardown (QObject::killTimer). All sensors, writers and SLAM
        // threads are already stopped and files are closed here, so bypass only
        // those unsafe static destructors.
        std::cout.flush();
        std::cerr.flush();
        std::_Exit(exit_code);
    } catch (const rs2::error &error) {
        std::cerr << "RealSense error: " << error.what() << std::endl;
    } catch (const std::exception &error) {
        std::cerr << "Fatal error: " << error.what() << std::endl;
    }
    return 1;
}
