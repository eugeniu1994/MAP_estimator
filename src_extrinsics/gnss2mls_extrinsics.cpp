

#include "DataHandler_vux.hpp"

#include "IMU.hpp"
#include "GNSS.hpp"

#ifndef USE_EKF
#include "PoseGraph.hpp"
#endif

#ifdef USE_ALS
#include "ALS.hpp"
#endif

#include <chrono>

#include "Vux_reader.hpp"
#include <GeographicLib/UTMUPS.hpp>
#include <liblas/liblas.hpp>

#include <visualization_msgs/Marker.h>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

volatile bool flg_exit = false;
void SigHandle(int sig)
{
    ROS_WARN("Caught signal %d, stopping...", sig);
    flg_exit = true; // Set the flag to stop the loop
}

void publish_refined_ppk_gnss(const Sophus::SE3 &_pose, const double &msg_time)
{
    tf::Transform transformGPS;
    tf::Quaternion q;
    static tf::TransformBroadcaster br;

    auto t = _pose.translation();
    auto R_yaw = _pose.so3().matrix();

    transformGPS.setOrigin(tf::Vector3(t[0], t[1], t[2]));
    Eigen::Quaterniond quat_(R_yaw);
    q = tf::Quaternion(quat_.x(), quat_.y(), quat_.z(), quat_.w());
    transformGPS.setRotation(q);
    br.sendTransform(tf::StampedTransform(transformGPS, ros::Time().fromSec(msg_time), "world", "VUX_B"));
}

void publish_ppk_gnss(const Sophus::SE3 &_pose, const double &msg_time)
{
    tf::Transform transformGPS;
    tf::Quaternion q;
    static tf::TransformBroadcaster br;

    auto t = _pose.translation();
    auto R_yaw = _pose.so3().matrix();

    transformGPS.setOrigin(tf::Vector3(t[0], t[1], t[2]));
    Eigen::Quaterniond quat_(R_yaw);
    q = tf::Quaternion(quat_.x(), quat_.y(), quat_.z(), quat_.w());
    transformGPS.setRotation(q);
    br.sendTransform(tf::StampedTransform(transformGPS, ros::Time().fromSec(msg_time), "world", "PPK_GNSS"));

    // tf::Transform transform_inv = transformGPS.inverse();
    // static tf::TransformBroadcaster br2;
    // br2.sendTransform(tf::StampedTransform(transform_inv, ros::Time().fromSec(msg_time), "PPK_GNSS", "world"));
}

// Estimate rotation + translation from point sets
Sophus::SE3 estimateRigidTransformFromPoints(const std::vector<V3D> &lidar_positions,
                                             const std::vector<V3D> &gnss_positions)
{
    assert(lidar_positions.size() == gnss_positions.size());
    size_t N = lidar_positions.size();

    // Compute centroids
    V3D centroid_lidar = V3D::Zero();
    V3D centroid_gnss = V3D::Zero();
    for (size_t i = 0; i < N; ++i)
    {
        centroid_lidar += lidar_positions[i];
        centroid_gnss += gnss_positions[i];
    }
    centroid_lidar /= N;
    centroid_gnss /= N;

    // Center the points
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 3);
    double sum_errors_init = 0., sum_errors_after = 0.;
    for (size_t i = 0; i < N; ++i)
    {
        V3D p_L = lidar_positions[i] - centroid_lidar;
        V3D p_G = gnss_positions[i] - centroid_gnss;
        H += p_G * p_L.transpose(); // Note the order: from GNSS to LiDAR

        sum_errors_init += (lidar_positions[i] - gnss_positions[i]).norm();
    }
    sum_errors_init /= N;

    // Compute rotation via SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R = V * U.transpose();

    if (R.determinant() < 0)
    {
        V.col(2) *= -1;        // Reflect V to correct for reflection
        R = V * U.transpose(); // Recompute proper rotation
    }

    // Compute translation
    Eigen::Vector3d t = centroid_lidar - R * centroid_gnss;

    for (size_t i = 0; i < N; ++i)
    {
        V3D p_L = lidar_positions[i];
        V3D p_G = R * gnss_positions[i] + t;
        sum_errors_after += (p_L - p_G).norm();
    }
    sum_errors_after /= N;

    std::cout << "sum_errors_init:" << sum_errors_init << ", sum_errors_after:" << sum_errors_after << std::endl;

    return Sophus::SE3(R, t);
}

struct vux_gnss_post
{
    double gps_tod;
    double gps_tow;
    double easting, northing, h_ell;
    double omega, phi, kappa;
    Sophus::SE3 se3;

    V3D acc_no_gravity, acc, gyro, velocity;

    // V3D pos_cov;
    // V3D rot_cov;
    double stdev;
};

std::vector<vux_gnss_post> readMeasurements(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }

    std::vector<vux_gnss_post> measurements;
    std::string line;

    while (std::getline(file, line))
    {
        if (line.find("GPSTime") != std::string::npos)
        {
            break;
        }
    }
    std::getline(file, line); // Skip the units line
    double skip_val;          // Used to discard unnecessary columns

    double Heading, SDNorth, SDEast, SDHeight;
    double Roll, Pitch, RollSD, PitchSD, AzStDev, E_Sep, N_Sep, H_Sep, RollSep, PtchSep, HdngSep, Azimuth;

    double VelBdyZ, VelBdyY, VelBdyX;    // velocity
    double AccBdyZ, AccBdyY, AccBdyX;    // acceleration no gravity
    double AngRateZ, AngRateY, AngRateX; // gyroscope values deg/s

    while (std::getline(file, line))
    {
        std::istringstream iss(line);

        // Ignore empty line
        if (line.empty() || line.find_first_not_of(" \t") == std::string::npos)
        {
            continue;
        }

        double gpstime_double; // Read as double first
        vux_gnss_post m;

        // Read fields: GPSTime, Easting, Northing, H-Ell, skip 14 fields, then Phi, Omega, Kappa
        if (!(iss >> gpstime_double >> m.easting >> m.northing >> m.h_ell >>
              Heading >> skip_val >> SDEast >> SDHeight >> skip_val >> skip_val >> AngRateZ >> skip_val >> skip_val >> AngRateY >> skip_val >>
              AngRateX >> skip_val >> Roll >> Pitch >> m.phi >> m.omega >> m.kappa >>
              skip_val >> Azimuth >> RollSD >> VelBdyZ >> AccBdyZ >> VelBdyY >> AccBdyY >> VelBdyX >> AccBdyX >> PitchSD >> AzStDev >> E_Sep >> N_Sep >> H_Sep >> RollSep >> PtchSep >> HdngSep))
        {
            // !(iss >> gpstime_double >> m.easting >> m.northing >> m.h_ell >>
            // skip_val >> skip_val >> skip_val >> skip_val >> skip_val >>
            // skip_val >> skip_val >> skip_val >> skip_val >> skip_val >> skip_val >> skip_val >> skip_val >>
            // skip_val >> skip_val >> m.phi >> m.omega >> m.kappa))
            std::cerr << "Warning: Could not parse line: " << line << std::endl;
            continue;
        }
        if (measurements.empty())
        {
            std::cout << "first measurement" << std::endl;
            std::cout << "Heading:" << Heading << std::endl;
            std::cout << "SDNorth:" << SDNorth << std::endl;
            std::cout << "SDEast:" << SDEast << std::endl;
            std::cout << "SDHeight:" << SDHeight << std::endl;
            std::cout << "AngRateZ:" << AngRateZ << std::endl;
            std::cout << "AngRateY:" << AngRateY << std::endl;
            std::cout << "AngRateX:" << AngRateX << std::endl;
            std::cout << "Roll:" << Roll << std::endl;
            std::cout << "Pitch:" << Pitch << std::endl;
            std::cout << "Azimuth:" << Azimuth << std::endl;
            std::cout << "RollSD:" << RollSD << std::endl;
            std::cout << "VelBdyZ:" << VelBdyZ << std::endl;
            std::cout << "AccBdyZ:" << AccBdyZ << std::endl;
            std::cout << "VelBdyY:" << VelBdyY << std::endl;
            std::cout << "AccBdyY:" << AccBdyY << std::endl;
            std::cout << "VelBdyX:" << VelBdyX << std::endl;
            std::cout << "AccBdyX:" << AccBdyX << std::endl;
        }

        gpstime_double -= 18.; // convert to UTC

        m.gps_tow = gpstime_double;
        m.gps_tod = std::fmod(gpstime_double, 86400.0); // Get the time of the day from time of the week;

        tf::Matrix3x3 rotation;
        // these are degrees - convert to radians
        double omega = m.omega * (M_PI / 180.0);
        double phi = m.phi * (M_PI / 180.0);
        double kappa = m.kappa * (M_PI / 180.0);

        rotation.setEulerYPR(kappa, phi, omega);
        // rotation.setEulerYPR(0, 0, 0);

        V3D translation(m.easting, m.northing, m.h_ell);

        M3D R;
        // for (int i = 0; i < 3; ++i)
        //     for (int j = 0; j < 3; ++j)
        //         R(i, j) = rotation[i][j];

        M3D R_heikki;
        R_heikki << cos(phi) * cos(kappa), -cos(phi) * sin(kappa), sin(phi),
            cos(omega) * sin(kappa) + cos(kappa) * sin(omega) * sin(phi), cos(omega) * cos(kappa) - sin(omega) * sin(phi) * sin(kappa), -cos(phi) * sin(omega),
            sin(omega) * sin(kappa) - cos(omega) * cos(kappa) * sin(phi), cos(kappa) * sin(omega) + cos(omega) * sin(phi) * sin(kappa), cos(omega) * cos(phi);

        R = R_heikki;
        m.se3 = Sophus::SE3(Eigen::Quaterniond(R), translation);

        m.gyro = V3D(AngRateX, AngRateY, AngRateZ) * (M_PI / 180.0); // gyroscope values in radians
        m.acc_no_gravity = V3D(AccBdyX, AccBdyY, AccBdyZ);           // acceleration no gravity
        m.velocity = V3D(VelBdyX, VelBdyY, VelBdyZ);                 // velocity
        // defined somewhere G_m_s2 = 9.81
        m.acc = m.se3.so3().inverse() * ((m.se3.so3() * m.acc_no_gravity) + V3D(0, 0, G_m_s2));

        measurements.push_back(m);
    }

    return measurements;
}

// Find the two closest SE3 poses and interpolate
Sophus::SE3 interpolateSE3(const std::vector<vux_gnss_post> &gnss_vux_data, double point_time, int &initial_index, bool is_ned = false)
{
    if (initial_index == 0 || initial_index >= gnss_vux_data.size())
    {
        throw std::runtime_error("GNSS data is empty.");
    }

    int size = gnss_vux_data.size();

    int idx = std::max(0, std::min(initial_index, size - 1));

    // Adjust idx to find the closest timestamps
    if (gnss_vux_data[idx].gps_tod < point_time)
    {
        while (idx + 1 < size && gnss_vux_data[idx + 1].gps_tod < point_time)
        {
            idx++;
        }
    }
    else
    {
        while (idx > 0 && gnss_vux_data[idx].gps_tod > point_time)
        {
            idx--;
        }
    }

    // Now idx is the lower bound (t1), and idx + 1 is the upper bound (t2)
    if (idx + 1 >= size)
    {
        return gnss_vux_data[idx].se3;
    }

    double t1 = gnss_vux_data[idx].gps_tod;
    double t2 = gnss_vux_data[idx + 1].gps_tod;
    // std::cout << "\nt1:" << t1 << ", t2:" << t2 << " cloud time:" << point_time << std::endl;

    Sophus::SE3 se3_1, se3_2;
    se3_1 = gnss_vux_data[idx].se3;
    se3_2 = gnss_vux_data[idx + 1].se3;

    // Compute interpolation weight
    double alpha = (point_time - t1) / (t2 - t1);

    // Interpolate using SE3 logarithm map
    Sophus::SE3 delta = se3_1.inverse() * se3_2;
    Eigen::Matrix<double, 6, 1> log_delta = delta.log();
    Sophus::SE3 interpolated = se3_1 * Sophus::SE3::exp(alpha * log_delta);

    return interpolated;
}

Sophus::SE3 interpolateSE3(const Sophus::SE3 &pose1, const double time1,
                           const Sophus::SE3 &pose2, const double time2,
                           const double t)
{
    Sophus::SE3 delta = pose1.inverse() * pose2;

    // Compute interpolation weight
    double alpha = (t - time1) / (time2 - time1); // when t will be bigger thatn time 2 it will extrapolate

    // exponential map to interpolate
    Sophus::SE3 interpolated = pose1 * Sophus::SE3::exp(alpha * delta.log());

    return interpolated;
}

Sophus::SE3 interpolateSE3Log(
    const std::vector<Sophus::SE3> &poses,
    const std::vector<double> &times,
    double query_time)
{
    if (poses.size() != times.size() || poses.size() < 2)
    {
        throw std::runtime_error("Invalid input: poses and times must be same size and >= 2");
    }

    // Find the interval [i, i+1] such that times[i] <= query_time < times[i+1]
    size_t i = 0;
    while (i + 1 < times.size() && query_time >= times[i + 1])
    {
        ++i;
    }

    if (i + 1 >= times.size())
    {
        throw std::out_of_range("Query time is out of bounds.");
    }

    double t1 = times[i];
    double t2 = times[i + 1];
    double alpha = (query_time - t1) / (t2 - t1);

    const Sophus::SE3 &T1 = poses[i];
    const Sophus::SE3 &T2 = poses[i + 1];

    // Relative transform from T1 to T2
    Sophus::SE3 T_rel = T1.inverse() * T2;

    // Interpolate in tangent space
    Eigen::Matrix<double, 6, 1> xi = T_rel.log();            // twist vector
    Eigen::Matrix<double, 6, 1> xi_scaled = alpha * xi;      // scaled twist
    Sophus::SE3 T_interp = T1 * Sophus::SE3::exp(xi_scaled); // apply scaled motion to T1

    return T_interp;
}

bool readSE3FromFile(const std::string &filename, Sophus::SE3 &transform_out)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file " << filename << " for reading.\n";
        return false;
    }

    std::string line;
    Eigen::Matrix4d mat;
    int row = 0;

    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        if (line.find("T_als2mls") != std::string::npos)
            continue;

        std::istringstream iss(line);
        for (int col = 0; col < 4; ++col)
        {
            iss >> mat(row, col);
        }
        ++row;
        if (row == 4)
            break;
    }

    if (row != 4)
    {
        std::cerr << "Failed to read full 4x4 matrix from " << filename << "\n";
        return false;
    }

    transform_out = Sophus::SE3(mat.block<3, 3>(0, 0), mat.block<3, 1>(0, 3));
    return true;
}

void DataHandler::Subscribe()
{
    std::cout << "Run test" << std::endl;
    std::cout << std::fixed << std::setprecision(12);
    std::cerr << std::fixed << std::setprecision(12);

    // ros::TransportHints().tcpNoDelay()
    std::cout << "\n=============================== BagHandler VUX ===============================" << std::endl;
#ifdef MP_EN
    std::cout << "Open_MP is available " << std::endl;
#else
    std::cout << "Open_MP is not available" << std::endl;
#endif

    std::shared_ptr<IMU_Class> imu_obj(new IMU_Class());
    std::shared_ptr<GNSS> gnss_obj(new GNSS());

    imu_obj->set_param(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, 10. * acc_cov),
                       V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov), V3D(b_acc_cov, b_acc_cov, 10. * b_acc_cov));

    gnss_obj->set_param(GNSS_T_wrt_IMU, GNSS_IMU_calibration_distance, postprocessed_gnss_path);

#define USE_ALS
#ifdef USE_ALS
    std::shared_ptr<ALS_Handler> als_obj = std::make_shared<ALS_Handler>(folder_root, downsample, closest_N_files, filter_size_surf_min);

    ros::Publisher pubLaserALSMap = nh.advertise<sensor_msgs::PointCloud2>("/ALS_map", 1000);
#endif

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudLocal = nh.advertise<sensor_msgs::PointCloud2>("/cloud_local", 100000);

    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100);

    ros::Publisher pubLaserCloudDebug = nh.advertise<sensor_msgs::PointCloud2>("/cloud_debug", 10);
    ros::Publisher point_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/vux_data", 10000);
    ros::Publisher pubOptimizedVUX = nh.advertise<sensor_msgs::PointCloud2>("/vux_optimized", 10);

    std::ifstream file(bag_file);
    if (!file)
    {
        std::cerr << "Error: Bag file does not exist or is not accessible: " << bag_file << std::endl;
        return;
    }

    rosbag::Bag bag;
    bag.open(bag_file, rosbag::bagmode::Read);

    std::vector<std::string> topics;
    topics.push_back(lid_topic);
    topics.push_back(imu_topic);
    topics.push_back(gnss_topic);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    signal(SIGINT, SigHandle); // Handle Ctrl+C (SIGINT)
    flg_exit = false;

    std::cout << "Start reading the data..." << std::endl;

    // for the car used so far
    std::string post_processed_gnss_imu_vux_file = "/media/eugeniu/T7/roamer/evo_20240725_reCalculated2_poqmodouEugent.txt";
    std::vector<vux_gnss_post> gnss_vux_data = readMeasurements(post_processed_gnss_imu_vux_file);

    // test for drone data
    // std::string post_processed_gnss_imu_vux_file = "/home/eugeniu/Desktop/Evo_HesaiALS_20250709/Hesai_ALS_20250709_Evo_1014.txt";
    // std::vector<vux_gnss_post> gnss_vux_data = readMeasurements2(post_processed_gnss_imu_vux_file);

    std::cout << "gnss_vux_data:" << gnss_vux_data.size() << std::endl;
    auto first_m = gnss_vux_data[0];
    V3D first_t(first_m.easting, first_m.northing, first_m.h_ell), first_t_ned;
    std::cout << "\n gps_tod:" << first_m.gps_tod << ", easting:" << first_m.easting << ", northing:" << first_m.northing << ", h_ell:" << first_m.h_ell << "\n"
              << std::endl;
    std::cout << "gps_tow:" << first_m.gps_tow << ", omega:" << first_m.omega << ", phi:" << first_m.phi << ", kappa:" << first_m.kappa << std::endl;
    int tmp_index = 0, init_guess_index = 0;

    ros::Rate rate(500);

    int scan_id = 0, vux_scan_id = 0;

    Sophus::SE3 prev_mls, curr_mls;
    double prev_mls_time, curr_mls_time;
    cov prev_P, curr_P;

    V3D prev_enu, curr_enu;
    double prev_raw_gnss_diff_time, curr_raw_gnss_diff_time;

    bool perform_mls_registration = true;
    bool als_integrated = false;
    Sophus::SE3 als2mls = Sophus::SE3();

    //----------------------------------------------------------------------------

    // the next 2 will be used for extrinsics estimation
    bool use_als = true;
    std::vector<Sophus::SE3> mls_poses, gnss_poses, gnss_poses_original;
    std::vector<double> mls_times, gnss_times;
    std::vector<Eigen::Vector6d> log_gnss_lidar_relative, log_gnss_lidar_relative2;

    std::vector<pcl::PointCloud<PointType>::Ptr> mls_clouds;

    bool ppk_gnss_init = false;
    Sophus::SE3 first_ppk_gnss_pose_inverse = Sophus::SE3();

    bool gnss_mls_extrinsic_init1 = false;
    bool gnss_mls_extrinsic_init2 = false;

    Sophus::SE3 rough_gnss2mls = Sophus::SE3();
    Sophus::SE3 gnss2mls = Sophus::SE3();

#define integrate_ppk_gnss

    using namespace std::chrono;

    for (const rosbag::MessageInstance &m : view)
    {
        ros::spinOnce();
        if (flg_exit || !ros::ok())
            break;

        std::string topic = m.getTopic();
        if (topic == imu_topic)
        {
            sensor_msgs::Imu::ConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
            if (imu_msg)
            {
                imu_cbk(imu_msg);
                continue;
            }
        }
        else if (topic == gnss_topic)
        {
            gps_common::GPSFix::ConstPtr gps_msg = m.instantiate<gps_common::GPSFix>();
            if (gps_msg)
            {
                gps_cbk(gps_msg);
                continue;
            }
        }
        else if (topic == lid_topic)
        {
            sensor_msgs::PointCloud2::ConstPtr pcl_msg = m.instantiate<sensor_msgs::PointCloud2>();
            if (pcl_msg)
            {
                pcl_cbk(pcl_msg);
            }
        }

        if (sync_packages(Measures))
        {
            scan_id++;
            std::cout << "scan_id:" << scan_id << std::endl;
            if (scan_id > 8000) // 500 1050 used for data before
            {
                std::cout << "Stop here... enough data 8000 scans" << std::endl;
                break;
            }

            perform_mls_registration = true;
            if (perform_mls_registration)
            {
                double t00 = omp_get_wtime();

                if (flg_first_scan)
                {
                    first_lidar_time = Measures.lidar_beg_time;
                    flg_first_scan = false;
                    curr_mls = Sophus::SE3(state_point.rot, state_point.pos);
                    prev_mls = curr_mls;
                    curr_mls_time = Measures.lidar_end_time;
                    prev_mls_time = curr_mls_time;
                    curr_P = estimator_.get_P();
                    prev_P = curr_P;

                    curr_enu = gnss_obj->curr_enu;
                    prev_enu = curr_enu;
                    curr_raw_gnss_diff_time = gnss_obj->diff_curr_gnss2mls;
                    prev_raw_gnss_diff_time = curr_raw_gnss_diff_time;

                    continue;
                }

                // undistort and provide initial guess
                imu_obj->Process(Measures, estimator_, feats_undistort);

                double t_IMU_process = omp_get_wtime();

                // publish_frame_debug(pubLaserCloudDebug, feats_undistort);

                if (imu_obj->imu_need_init_)
                {
                    std::cout << "IMU was not initialised " << std::endl;
                    continue;
                }

                if (feats_undistort->empty() || (feats_undistort == NULL))
                {
                    ROS_WARN("No feats_undistort point, skip this scan!\n");
                    std::cout << "feats_undistort:" << feats_undistort->size() << std::endl;
                    // throw std::runtime_error("NO points -> ERROR: check your data");
                    continue;
                }
                state_point = estimator_.get_x();

                flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(*feats_down_body);

                feats_down_size = feats_down_body->points.size();
                if (feats_down_size < 5)
                {
                    ROS_WARN("No feats_down_body point, skip this scan!\n");
                    std::cout << "feats_undistort:" << feats_undistort->size() << std::endl;
                    std::cout << "feats_down_body:" << feats_down_size << std::endl;
                    throw std::runtime_error("NO feats_down_body points -> ERROR");
                }

                double t_cloud_voxelization = omp_get_wtime();

                if (!map_init)
                {
                    feats_down_size = feats_undistort->size();
                    feats_down_world->resize(feats_down_size);

                    tbb::parallel_for(tbb::blocked_range<int>(0, feats_down_size),
                                      [&](tbb::blocked_range<int> r)
                                      {
                                          for (int i = r.begin(); i < r.end(); i++)
                                          // for (int i = 0; i < feats_down_size; i++)
                                          {
                                              // pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i])); // transform to world coordinates
                                              pointBodyToWorld(&(feats_undistort->points[i]), &(feats_down_world->points[i])); // transform to world coordinates
                                          }
                                      });

                    *laserCloudSurfMap += *feats_down_world;
                    map_init = true;
                    continue;
                }

                // register to MLS map  ----------------------------------------------
                Nearest_Points.resize(feats_down_size);

                if (use_als)
                {
                    if (!als_obj->refine_als) // als was not setup
                    {
                        estimator_.update(LASER_POINT_COV, feats_down_body, laserCloudSurfMap, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en);
                        if (gnss_obj->GNSS_extrinsic_init)
                        {
                            //*featsFromMap = *laserCloudSurfMap;

                            // only for now - remove this later
                            std::string als2mls_filename = "/home/eugeniu/x_vux-georeferenced-final/_Hesai/als2mls.txt";
                            Sophus::SE3 known_als2mls;
                            readSE3FromFile(als2mls_filename, known_als2mls);
                            std::cout << "Read the known transformation" << std::endl;

                            als_obj->init(known_als2mls);

                            // if (!this->downsample) // if it is sparse ALS data from NLS
                            // {
                            //     V3D t = known_als2mls.translation();
                            //     t.z() += 20.;
                            //     Sophus::SE3 als2mls_for_sparse_ALS = Sophus::SE3(known_als2mls.so3().matrix(), t);
                            //     std::cout << "Init ALS from known T map refinement" << std::endl;
                            //     als_obj->init(als2mls_for_sparse_ALS, laserCloudSurfMap); // with refinement
                            // }
                            // else
                            // {
                            //     std::cout << "Init ALS from known T" << std::endl;
                            //     als_obj->init(known_als2mls);
                            // }

                            // als_obj->init(gnss_obj->origin_enu, gnss_obj->R_GNSS_to_MLS, featsFromMap);

                            gnss_obj->updateExtrinsic(als_obj->R_to_mls); // use the scan-based refined rotation for GNSS
                            als_obj->Update(Sophus::SE3(state_point.rot, state_point.pos));
                            gnss_obj->als2mls_T = als_obj->als_to_mls;

                            // reset local map
                            // laserCloudSurfMap.reset(new PointCloudXYZI());

                            als2mls = als_obj->als_to_mls;
                        }
                    }
                    else // als was set up
                    {
                        als_obj->Update(Sophus::SE3(state_point.rot, state_point.pos));

                        // update only MLS
                        // if (!estimator_.update(LASER_POINT_COV, feats_down_body, laserCloudSurfMap, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en))
                        // {
                        //     std::cout << "\n------------------MLS update failed--------------------------------" << std::endl;
                        // }

                        // update only ALS
                        // if (!estimator_.update(LASER_POINT_COV / 4, feats_down_body, als_obj->als_cloud, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en))
                        // {
                        //     std::cout << "\n------------------ ALS update failed --------------------------------" << std::endl;
                        //     // TODO check here why -  there is not enough als data
                        // }

                        // update tighly fusion from MLS and ALS
                        // if (!estimator_.update_tighly_fused(LASER_POINT_COV, feats_down_body, laserCloudSurfMap, als_obj->als_cloud, als_obj->localKdTree_map_als, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en))
                        // {
                        //     std::cout << "\n------------------FUSION ALS-MLS update failed--------------------------------" << std::endl;
                        // }

                        // update tighly fusion from MLS and ALS
                        double R_gps_cov = .0001; // GNSS_VAR * GNSS_VAR;
                        Sophus::SE3 gnss_pose = (gnss_obj->use_postprocessed_gnss) ? gnss_obj->postprocessed_gps_pose : gnss_obj->gps_pose;
                        const V3D &gnss_in_mls = gnss_pose.translation();

                        bool tightly_coupled = true;
                        bool use_gnss = false;
                        bool use_als = false; // true;

                        if (!estimator_.update_final(
                                LASER_POINT_COV, R_gps_cov, feats_down_body, gnss_in_mls, laserCloudSurfMap, als_obj->als_cloud, als_obj->localKdTree_map_als,
                                Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en, use_gnss, use_als, tightly_coupled))
                        {
                            std::cout << "\n------------------FUSION ALS-MLS update failed--------------------------------" << std::endl;
                        }

                        als_integrated = true;
                    }

                    if (pubLaserALSMap.getNumSubscribers() != 0)
                    {
                        als_obj->getCloud(featsFromMap);
                        publish_map(pubLaserALSMap);
                    }
                }
                else if (!estimator_.update(LASER_POINT_COV, feats_down_body, laserCloudSurfMap, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en))
                {
                    std::cout << "\n------------------MLS update failed--------------------------------" << std::endl;
                }

                // use_als = false;
                als_integrated = true; // remove this later

                // double t_LiDAR_update = omp_get_wtime();
                // std::cout << "\nIMU_process time(ms):  " << (t_IMU_process - t00) * 1000 <<
                //", cloud_voxelization (ms): " << (t_cloud_voxelization - t_IMU_process) * 1000 <<
                //", LiDAR_update (ms): " << (t_LiDAR_update - t_cloud_voxelization) * 1000 << std::endl;

                // Crop the local map------
                state_point = estimator_.get_x();
                // get and publish the GNSS pose--------
                gnss_obj->Process(gps_buffer, lidar_end_time, state_point.pos);

                Sophus::SE3 gnss_pose = (gnss_obj->use_postprocessed_gnss) ? gnss_obj->postprocessed_gps_pose : gnss_obj->gps_pose;

                curr_mls = Sophus::SE3(state_point.rot, state_point.pos);
                curr_mls_time = Measures.lidar_end_time;
                curr_P = estimator_.get_P();
                curr_enu = gnss_obj->curr_enu;
                curr_raw_gnss_diff_time = gnss_obj->diff_curr_gnss2mls;

                double time_of_day_sec = gnss_obj->tod;

#ifdef integrate_ppk_gnss
                while (tmp_index < gnss_vux_data.size() - 1 && als_integrated) //
                {
                    tmp_index++;

                    // std::cout<<"GPS Time of the week :"<<gnss_vux_data[tmp_index].gps_tow<<std::endl;

                    double time_diff_curr = fabs(time_of_day_sec - gnss_vux_data[tmp_index].gps_tod);
                    double time_diff_next = fabs(time_of_day_sec - gnss_vux_data[tmp_index + 1].gps_tod);

                    // std::cout << "time_diff_curr:" << time_diff_curr << ", time_diff_next:" << time_diff_next << std::endl;
                    if (time_diff_curr > time_diff_next) // get to the closest message on time
                    {
                        continue; // continue to go to the next message
                    }

                    const auto &msg_time = gnss_vux_data[tmp_index].gps_tod;
                    std::cout << "Closest GNSS time diff = " << time_diff_curr << std::endl;
                    // std::cout<<"als2mls:\n"<<als2mls.matrix()<<std::endl;

                    auto interpolated_pose = interpolateSE3(
                        gnss_vux_data[tmp_index].se3, gnss_vux_data[tmp_index].gps_tod,
                        gnss_vux_data[tmp_index + 1].se3, gnss_vux_data[tmp_index + 1].gps_tod,
                        time_of_day_sec);

                    // auto ppk_gnss_imu = als2mls * gnss_vux_data[tmp_index].se3;
                    // publish_ppk_gnss(ppk_gnss_imu, msg_time);

                    // auto ppk_gnss_imu = als2mls * gnss_vux_data[tmp_index].se3; // in mls frame

                    auto ppk_gnss_imu = als2mls * interpolated_pose;

                    // put it in the IMU position
                    V3D ext(0.042, 0.193, 0.326); // X, Y, Z in meters

                    // auto t_ = ppk_gnss_imu.translation() - ext;
                    // ppk_gnss_imu = Sophus::SE3(ppk_gnss_imu.so3().matrix(), t_);
                    {
                        if (!ppk_gnss_init)
                        {
                            ppk_gnss_init = true;
                            first_ppk_gnss_pose_inverse = interpolated_pose;
                        }

                        std::cout << "first_ppk_gnss_pose_inverse:\n"
                                  << first_ppk_gnss_pose_inverse.matrix() << std::endl;

                        auto first_t = first_ppk_gnss_pose_inverse.translation();
                        auto curr_t = interpolated_pose.translation();

                        ppk_gnss_imu = Sophus::SE3(interpolated_pose.so3().matrix(), curr_t - first_t);
                    }
                    publish_ppk_gnss(ppk_gnss_imu, msg_time);

                    // gnss - lidar extrinsic estimation method

                    {
                        // save the original gnss poses
                        // save the 3d mls

                        // find a rough alignment from gnss to mls
                        //     plot it

                        // after the rough alignment
                        //     use the log difference refinement
                        //     plot it

                        //                     std::vector<Sophus::SE3> mls_poses, gnss_poses, gnss_poses_original;
                        // std::vector<double> mls_times, gnss_times;
                        // std::vector<Eigen::Vector6d> log_gnss_lidar_relative;

                        // Sophus::SE3 gnss2mls = Sophus::SE3();


                        auto gnss_in_mls_rough = rough_gnss2mls * interpolated_pose;
                        publish_gnss_odometry(gnss_in_mls_rough);

                        auto gnss_in_mls = gnss_in_mls_rough * gnss2mls;
                        publish_refined_ppk_gnss(gnss_in_mls, msg_time);

                        std::cout<<"\n\nTravelled: "<<curr_mls.translation()<<std::endl;

                        std::cout << "rough_gnss2mls:\n"
                                  << rough_gnss2mls.matrix() << std::endl;

                        std::cout << "gnss2mls:\n"
                                  << gnss2mls.matrix() << std::endl;

                        if (!gnss_mls_extrinsic_init1)
                        {
                            mls_poses.push_back(curr_mls);
                            gnss_poses.push_back(interpolated_pose);

                            if (curr_mls.translation() > 15)
                            {
                                gnss_mls_extrinsic_init1 = true;

                                std::vector<V3D> lidar_positions, gnss_positions;
                                for (size_t i = 0; i < mls_poses.size(); ++i)
                                {
                                    lidar_positions.push_back(mls_poses[i].translation());
                                    gnss_positions.push_back(gnss_poses[i].translation());
                                }

                                rough_gnss2mls = estimateRigidTransformFromPoints(lidar_positions, gnss_positions);
                                
                                for(size_t i=0; i< mls_poses.size();++i)
                                {
                                    auto g = rough_gnss2mls * gnss_poses[i];
                                    log_gnss_lidar_relative.push_back((mls_poses[i].inverse() * g).log());
                                    log_gnss_lidar_relative2.push_back((mls_poses[i].inverse() * gnss_poses[i]).log());
                                }
                                
                                Sophus::SE3 lidar2gnss_extrinsic = averageSE3Log(log_gnss_lidar_relative);
                                std::cout<<"lidar2gnss_extrinsic:\n"<<lidar2gnss_extrinsic.matrix()<<std::endl;

                                gnss2mls = lidar2gnss_extrinsic.inverse();
                            }
                        }


                        // bool gnss_mls_extrinsic_init2 = false;
                    }

                    break;
                }

#endif

                // Update the local map--------------------------------------------------
                feats_down_world->resize(feats_down_size);
                local_map_update(); // this will update local map with curr measurements and crop the map

                prev_mls = curr_mls;           // Sophus::SE3(state_point.rot, state_point.pos);
                prev_mls_time = curr_mls_time; // Measures.lidar_end_time;
                prev_P = curr_P;
                prev_enu = curr_enu;
                prev_raw_gnss_diff_time = curr_raw_gnss_diff_time;

                // Publish odometry and point clouds------------------------------------
                publish_odometry(pubOdomAftMapped);
                if (scan_pub_en)
                {
                    if (pubLaserCloudFull.getNumSubscribers() != 0)
                        publish_frame_world(pubLaserCloudFull);

                    if (pubLaserCloudLocal.getNumSubscribers() != 0)
                    {
                        publish_frame_body(pubLaserCloudLocal);
                    }
                }

                if (pubLaserCloudMap.getNumSubscribers() != 0)
                {
                    *featsFromMap = *laserCloudSurfMap;
                    publish_map(pubLaserCloudMap);
                }

                double t11 = omp_get_wtime();
                std::cout << "Mapping time(ms):  " << (t11 - t00) * 1000 << ", feats_down_size: " << feats_down_size << ", lidar_end_time:" << lidar_end_time << "\n"
                          << std::endl;

                // std::cout<<"Debug als2mls:\n"<<als2mls.matrix()<<std::endl;
            }
        }
    }
    bag.close();
}
