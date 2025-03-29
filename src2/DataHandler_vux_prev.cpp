

#include "DataHandler_vux.hpp"

#include "IMU.hpp"
#include "GNSS.hpp"

#ifndef USE_EKF
#include "PoseGraph.hpp"
#endif

#ifdef USE_ALS
#include "ALS.hpp"
#endif

#include "Vux_reader.hpp"
#include <GeographicLib/UTMUPS.hpp>
#include <liblas/liblas.hpp>

volatile bool flg_exit = false;
void SigHandle(int sig)
{
    ROS_WARN("Caught signal %d, stopping...", sig);
    flg_exit = true; // Set the flag to stop the loop
}

void publishPointCloud(pcl::PointCloud<PointType>::Ptr &cloud, const ros::Publisher &point_cloud_pub)
{
    if (cloud->empty())
    {
        std::cerr << "VUX Point cloud is empty. Skipping publish.\n";
        return;
    }

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);

    // Get the first point's GPS time and convert to ROS time
    ros::Time first_point_time_ros(cloud->points[0].time);

    cloud_msg.header.stamp = first_point_time_ros; // ros::Time::now();
    cloud_msg.header.frame_id = "VUX";

    point_cloud_pub.publish(cloud_msg);

    std::cout << "\nPublished " << cloud->size() << " points" << ", Header time: " << first_point_time_ros << std::endl;
}

void publishPointCloud_vux(pcl::PointCloud<VUX_PointType>::Ptr &cloud, const ros::Publisher &point_cloud_pub)
{
    if (cloud->empty())
    {
        // std::cerr << "VUX Point cloud is empty. Skipping publish.\n";
        return;
    }

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);

    // Get the first point's GPS time and convert to ROS time
    ros::Time first_point_time_ros(cloud->points[0].time);

    cloud_msg.header.stamp = first_point_time_ros; // ros::Time::now();
    // cloud_msg.header.frame_id = "VUX";  //change the frame id
    // cloud_msg.header.frame_id = "PPK_GNSS"; // Publish in GNSS frame
    cloud_msg.header.frame_id = "world"; // Publish in GNSS frame

    // publish transformation for this

    point_cloud_pub.publish(cloud_msg);

    // std::cout << "\nPublished " << cloud->size() << " points" << ", Header time: " << first_point_time_ros << std::endl;
}

void publish_raw_gnss(const V3D &t, const double &msg_time)
{
    tf::Transform transformGPS;
    tf::Quaternion q;
    static tf::TransformBroadcaster br;

    auto R_yaw = M3D::Identity();

    transformGPS.setOrigin(tf::Vector3(t[0], t[1], t[2]));
    Eigen::Quaterniond quat_(R_yaw);
    q = tf::Quaternion(quat_.x(), quat_.y(), quat_.z(), quat_.w());
    transformGPS.setRotation(q);
    br.sendTransform(tf::StampedTransform(transformGPS, ros::Time().fromSec(msg_time), "world", "GPSFix"));
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
}

void publish_ppk_gnss_ned(const Sophus::SE3 &_pose, const double &msg_time)
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
    br.sendTransform(tf::StampedTransform(transformGPS, ros::Time().fromSec(msg_time), "world", "MLS"));
}

struct vux_gnss_post
{
    double gpstime;
    double original_gpstime;
    double easting, northing, h_ell;
    double omega, phi, kappa;
    Sophus::SE3 se3;
    M3D R;

    double roll, pitch;

    V3D t_ned;
    Sophus::SE3 se3_ned;
};

Sophus::SE3 createSE3(const V3D &t, const tf::Matrix3x3 &rotation)
{
    M3D R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R(i, j) = rotation[i][j];

    return Sophus::SE3(Eigen::Quaterniond(R), t);
}

// #include "matplotlibcpp.h"
// namespace plt = matplotlibcpp;

// Create rotation matrix from Omega, Phi, Kappa in ENU format
M3D createRotationENU(double omega, double phi, double kappa)
{
    // Individual rotation matrices
    M3D Rx, Ry, Rz;
    Rx << 1, 0, 0,
        0, cos(omega), -sin(omega),
        0, sin(omega), cos(omega);

    Ry << cos(phi), 0, sin(phi),
        0, 1, 0,
        -sin(phi), 0, cos(phi);

    Rz << cos(kappa), -sin(kappa), 0,
        sin(kappa), cos(kappa), 0,
        0, 0, 1;

    // Combined rotation (ground-to-air)
    Eigen::Matrix3d R = Rz * Ry * Rx;

    return R;
}

M3D convertENUtoNED(const Eigen::Matrix3d &R_ENU)
{
    // ENU to NED transformation matrix
    M3D R_enu_to_ned;
    R_enu_to_ned << 0, 1, 0,
        1, 0, 0,
        0, 0, -1;

    // Convert rotation matrix to NED
    return R_enu_to_ned * R_ENU * R_enu_to_ned.transpose();
}

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
        if (!(iss >> gpstime_double >> m.easting >> m.northing >> m.h_ell >> skip_val >> skip_val >> skip_val >> skip_val >> skip_val >> skip_val >> skip_val >> skip_val >> skip_val >> skip_val >> skip_val >> skip_val >> skip_val >> m.roll >> m.pitch >> m.phi >> m.omega >> m.kappa))
        {
            std::cerr << "Warning: Could not parse line: " << line << std::endl;
            continue;
        }

        gpstime_double -= 18.; // convert to UTC

        m.original_gpstime = gpstime_double;

        m.gpstime = std::fmod(gpstime_double, 86400.0); // Get the time of the day from time of the week;

        tf::Matrix3x3 rotation;
        // these are degrees - convert to radians
        double omega = m.omega * (M_PI / 180.0);
        double phi = m.phi * (M_PI / 180.0);
        double kappa = m.kappa * (M_PI / 180.0);

        // rotation.setEulerYPR(kappa, phi, omega);
        //  rotation.setEulerYPR(0, 0, 0);

        double roll = m.roll * (M_PI / 180.0);
        double pitch = m.pitch * (M_PI / 180.0);
        // rotation.setEulerYPR(kappa, pitch, roll);

        rotation.setEulerYPR(kappa, pitch, omega); // this works

        M3D R_heikki;
        R_heikki << cos(phi) * cos(kappa), -cos(phi) * sin(kappa), sin(phi),
            cos(omega) * sin(kappa) + cos(kappa) * sin(omega) * sin(phi), cos(omega) * cos(kappa) - sin(omega) * sin(phi) * sin(kappa), -cos(phi) * sin(omega),
            sin(omega) * sin(kappa) - cos(omega) * cos(kappa) * sin(phi), cos(kappa) * sin(omega) + cos(omega) * sin(phi) * sin(kappa), cos(omega) * cos(phi);

        V3D translation(m.easting, m.northing, m.h_ell);

        M3D R;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R(i, j) = rotation[i][j];

        // UTM (NEU) x - East, y - North, z - Up
        // m.se3 = Sophus::SE3(Eigen::Quaterniond(R), translation); // createSE3(translation, rotation);
        m.se3 = Sophus::SE3(Eigen::Quaterniond(R_heikki), translation); // createSE3(translation, rotation);

        m.R = R;

        if (false)
        {
            // In the NED frame, the convention is: x - North y -  East z - Down
            //  Define the NEU -> NED transformation matrix
            Eigen::Matrix3d R_neu_to_ned;
            R_neu_to_ned << 0, 1, 0,
                1, 0, 0,
                0, 0, -1;
            // Transform rotation: R_NED = R_neu_to_ned * R * R_neu_to_ned.transpose()
            M3D R_NED = R_neu_to_ned * R * R_neu_to_ned.transpose();
            // Transform translation: (swap North/East, negate height)
            V3D t_NED(m.northing, m.easting, -m.h_ell);
            m.t_ned = t_NED;
            // Convert to Sophus SE3
            m.se3_ned = Sophus::SE3(Eigen::Quaterniond(R_NED), t_NED);
        }

        // Translation in NED convention: North (x), East (y), Down (z)
        translation = V3D(m.northing, m.easting, -m.h_ell);
        m.t_ned = translation;
        m.se3_ned = Sophus::SE3(Eigen::Quaterniond(R), translation);

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
    if (gnss_vux_data[idx].gpstime < point_time)
    {
        while (idx + 1 < size && gnss_vux_data[idx + 1].gpstime < point_time)
        {
            idx++;
        }
    }
    else
    {
        while (idx > 0 && gnss_vux_data[idx].gpstime > point_time)
        {
            idx--;
        }
    }

    // Now idx is the lower bound (t1), and idx + 1 is the upper bound (t2)
    if (idx + 1 >= size)
    {
        return gnss_vux_data[idx].se3;
    }

    double t1 = gnss_vux_data[idx].gpstime;
    double t2 = gnss_vux_data[idx + 1].gpstime;
    // std::cout << "\nt1:" << t1 << ", t2:" << t2 << " cloud time:" << point_time << std::endl;

    Sophus::SE3 se3_1, se3_2;
    if (is_ned)
    {
        se3_1 = gnss_vux_data[idx].se3_ned;
        se3_2 = gnss_vux_data[idx + 1].se3_ned;
    }
    else
    {
        se3_1 = gnss_vux_data[idx].se3;
        se3_2 = gnss_vux_data[idx + 1].se3;
    }

    // Compute interpolation weight
    double alpha = (point_time - t1) / (t2 - t1);

    // Interpolate using SE3 logarithm map
    Sophus::SE3 delta = se3_1.inverse() * se3_2;
    Eigen::Matrix<double, 6, 1> log_delta = delta.log();
    Sophus::SE3 interpolated = se3_1 * Sophus::SE3::exp(alpha * log_delta);

    return interpolated;
}

// Convert Euler angles to a 4×4 transformation matrix
Eigen::Matrix4d eulerToTransformationMatrix(double roll, double pitch, double yaw)
{
    Eigen::Matrix3d R;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

    Eigen::Matrix3d Rx, Ry, Rz;
    Rx << 1, 0, 0,
        0, cos(roll), -sin(roll),
        0, sin(roll), cos(roll);

    Ry << cos(pitch), 0, sin(pitch),
        0, 1, 0,
        -sin(pitch), 0, cos(pitch);

    Rz << cos(yaw), -sin(yaw), 0,
        sin(yaw), cos(yaw), 0,
        0, 0, 1;

    R = Rz * Ry * Rx; // ZYX intrinsic rotation
    T.block<3, 3>(0, 0) = R;

    return T;
}

Eigen::Vector3d rotationMatrixToEulerAngles(const Eigen::Matrix3d &R)
{
    double roll, pitch, yaw;

    // Compute Euler angles (Intrinsic ZYX rotation)
    pitch = atan2(-R(2, 0), sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0)));

    if (fabs(pitch - M_PI / 2) < 1e-6)
    {
        yaw = 0;
        roll = atan2(R(0, 1), R(1, 1));
    }
    else if (fabs(pitch + M_PI / 2) < 1e-6)
    {
        yaw = 0;
        roll = -atan2(R(0, 1), R(1, 1));
    }
    else
    {
        yaw = atan2(R(1, 0), R(0, 0));
        roll = atan2(R(2, 1), R(2, 2));
    }

    // Convert to degrees
    return Eigen::Vector3d(roll, pitch, yaw) * (180.0 / M_PI);
}

#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct LidarDistanceFactor
{
    LidarDistanceFactor(const V3D &curr_point_, const V3D &closest_point_, const Sophus::SE3 &gnss_pose_)
        : curr_point(curr_point_), closest_point(closest_point_), gnss_pose(gnss_pose_) {}

    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {
        // Convert extrinsic transformation (Scanner to IMU)
        Eigen::Quaternion<T> q_scanner_to_imu(q[3], q[0], q[1], q[2]);
        Eigen::Matrix<T, 3, 1> t_scanner_to_imu(t[0], t[1], t[2]);

        // Transform raw scanner point to GNSS-IMU frame
        Eigen::Matrix<T, 3, 1> point_in_imu = q_scanner_to_imu * curr_point.template cast<T>() + t_scanner_to_imu;

        // Convert GNSS-IMU pose to appropriate type
        Eigen::Matrix<T, 3, 3> R_gnss_imu = gnss_pose.rotation_matrix().template cast<T>();
        Eigen::Matrix<T, 3, 1> t_gnss_imu = gnss_pose.translation().template cast<T>();

        // Georeference point
        Eigen::Matrix<T, 3, 1> point_world = R_gnss_imu * point_in_imu + t_gnss_imu;

        // Compute residual as difference to the closest map point
        residual[0] = point_world.x() - T(closest_point.x());
        residual[1] = point_world.y() - T(closest_point.y());
        residual[2] = point_world.z() - T(closest_point.z());

        return true;
    }

    static ceres::CostFunction *Create(const V3D &curr_point_, const V3D &closest_point_, const Sophus::SE3 &gnss_pose_)
    {
        return new ceres::AutoDiffCostFunction<LidarDistanceFactor, 3, 4, 3>(
            new LidarDistanceFactor(curr_point_, closest_point_, gnss_pose_));
    }

    V3D curr_point;
    V3D closest_point;
    Sophus::SE3 gnss_pose;
};

struct LidarPlaneNormFactor
{

    LidarPlaneNormFactor(const V3D &curr_point_, const V3D &plane_unit_norm_,
                         double negative_OA_dot_norm_, const Sophus::SE3 &gnss_pose_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
                                                                                        negative_OA_dot_norm(negative_OA_dot_norm_), gnss_pose(gnss_pose_) {}

    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {
        // Convert extrinsic transformation (Scanner to IMU)
        Eigen::Quaternion<T> q_scanner_to_imu(q[3], q[0], q[1], q[2]);
        Eigen::Matrix<T, 3, 1> t_scanner_to_imu(t[0], t[1], t[2]);

        // Convert GNSS-IMU pose to appropriate type
        Eigen::Matrix<T, 3, 3> R_gnss_imu = gnss_pose.rotation_matrix().template cast<T>();
        Eigen::Matrix<T, 3, 1> t_gnss_imu = gnss_pose.translation().template cast<T>();

        // Transform raw scanner point to GNSS-IMU frame
        Eigen::Matrix<T, 3, 1> point_in_imu = q_scanner_to_imu * curr_point.template cast<T>() + t_scanner_to_imu;

        // Georeference point
        Eigen::Matrix<T, 3, 1> point_world = R_gnss_imu * point_in_imu + t_gnss_imu;

        Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
        residual[0] = norm.dot(point_world) + T(negative_OA_dot_norm);

        return true;
    }

    static ceres::CostFunction *Create(const V3D &curr_point_, const V3D &plane_unit_norm_,
                                       const double negative_OA_dot_norm_, const Sophus::SE3 &gnss_pose_)
    {
        return (new ceres::AutoDiffCostFunction<
                LidarPlaneNormFactor, 1, 4, 3>(
            new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_, gnss_pose_)));
    }

    V3D curr_point;
    V3D plane_unit_norm;
    double negative_OA_dot_norm;
    Sophus::SE3 gnss_pose;
};

#include <pcl/kdtree/kdtree_flann.h>

void DataHandler::Subscribe()
{
    std::cout << "Subscribe" << std::endl;
    std::cout << std::fixed << std::setprecision(12);

    // ros::TransportHints().tcpNoDelay()
    std::cout << "\n=============================== BagHandler VUX ===============================" << std::endl;
#ifdef MP_EN
    std::cout << "Open_MP is available " << std::endl;
#else
    std::cout << "Open_MP is not available" << std::endl;
#endif

#ifdef USE_EKF
    std::shared_ptr<IMU_Class> imu_obj(new IMU_Class());
#else
    std::shared_ptr<Graph> imu_obj(new Graph());
#endif

    std::shared_ptr<GNSS> gnss_obj(new GNSS());

    imu_obj->set_param(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, acc_cov),
                       V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov), V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    gnss_obj->set_param(GNSS_T_wrt_IMU, GNSS_IMU_calibration_distance, postprocessed_gnss_path);

#ifdef USE_ALS
    std::shared_ptr<ALS_Handler> als_obj = std::make_shared<ALS_Handler>(folder_root, downsample, closest_N_files, filter_size_surf_min);

    ros::Publisher pubLaserALSMap = nh.advertise<sensor_msgs::PointCloud2>("/ALS_map", 1000);
// #ifndef USE_EKF
//     std::shared_ptr<GlobalGraph> global_graph_obj(new GlobalGraph());
// #endif
#endif

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100);

    ros::Publisher pubLaserCloudDebug = nh.advertise<sensor_msgs::PointCloud2>("/cloud_debug", 10);
    ros::Publisher point_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/vux_data", 10000);

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

    // VUX subscriber
    // ros::Subscriber sub_vux = nh.subscribe(vux_topic, 200000, &DataHandler::vux_cbk, this);

    // get this as param
    std::string folder_path = "/media/eugeniu/T7/roamer/03_RIEGL_RAW/02_RXP/VUX-1HA-22-2022-B/";
    // folder_path = "/media/eugeniu/T7/roamer/03_RIEGL_RAW/02_RXP/VUX-1HA-22-2022-A/";

    vux::VuxAdaptor readVUX(std::cout);
    if (!readVUX.setUpReader(folder_path)) // get all the rxp files
    {
        std::cerr << "Cannot set up the VUX reader" << std::endl;
        return;
    }

    pcl::PointCloud<VUX_PointType>::Ptr next_line(new pcl::PointCloud<VUX_PointType>);
    auto gps_init_origin = false;

    std::vector<V3D> gps_measurements;
    V3D origin_enu, carthesian, curr_enu;
    GeographicLib::LocalCartesian geo_converter;

    // std::string post_processed_gnss_imu_vux_file = "/media/eugeniu/T7/roamer/evo_20240725_reCalculated2_poqmodout.txt";
    std::string post_processed_gnss_imu_vux_file = "/media/eugeniu/T7/roamer/evo_20240725_reCalculated2_poqmodouEugent.txt";
    std::vector<vux_gnss_post> gnss_vux_data = readMeasurements(post_processed_gnss_imu_vux_file);
    std::cout << "gnss_vux_data:" << gnss_vux_data.size() << std::endl;
    auto first_m = gnss_vux_data[0];

    V3D first_t(first_m.easting, first_m.northing, first_m.h_ell), first_t_ned;
    std::cout << "\n gpstime:" << first_m.gpstime << ", easting:" << first_m.easting << ", northing:" << first_m.northing << ", h_ell:" << first_m.h_ell << "\n"
              << std::endl;
    std::cout << "original_gpstime:" << first_m.original_gpstime << ", omega:" << first_m.omega << ", phi:" << first_m.phi << ", kappa:" << first_m.kappa << std::endl;
    Sophus::SE3 first_pose = first_m.se3.inverse(), first_pose_ned;
    int tmp_index = 0, init_guess_index = 0;

    // ros::Rate rate(250);
    ros::Rate rate(500);

    bool time_aligned = false;

    int current_counter = 0;

    bool do_once = true;

    int some_index = 0;

    Eigen::Affine3d transform_socs_to_tocs; // scanner to tilded

    M3D rotation_socs_to_tocs;
    {
        // double roll = -0.02651 * M_PI / 180.0; // Convert to radians
        // double pitch = 0.04036 * M_PI / 180.0; // Convert to radians
        // double yaw = 30.00981 * M_PI / 180.0;  // Convert to radians

        double roll = -0.02651 * M_PI / 180.0; // Convert to radians
        double pitch = 0.04036 * M_PI / 180.0; // Convert to radians
        double yaw = 30.00981 * M_PI / 180.0;  // Convert to radians

        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

        Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
        rotation_socs_to_tocs = q.matrix();

        transform_socs_to_tocs = Eigen::Translation3d(0, 0, 0) * rotation_socs_to_tocs;

        std::cout << "Init transform_socs_to_tocs Matrix:\n"
                  << transform_socs_to_tocs.matrix() << std::endl;
        // Compute Euler angles
        V3D euler_angles = rotationMatrixToEulerAngles(rotation_socs_to_tocs);
        std::cout << "Init Extracted Euler Angles (Roll, Pitch, Yaw) in degrees:\n"
                  << euler_angles.transpose() << std::endl;

        Eigen::Matrix4d T_SOCS_TO_TOCS = eulerToTransformationMatrix(roll, pitch, yaw);
        // transform_socs_to_tocs.matrix() = T_SOCS_TO_TOCS;

        std::cout << "Second transform_socs_to_tocs Matrix:\n"
                  << transform_socs_to_tocs.matrix() << std::endl;
        euler_angles = rotationMatrixToEulerAngles(T_SOCS_TO_TOCS.block<3, 3>(0, 0));
        std::cout << "Second Extracted Euler Angles (Roll, Pitch, Yaw) in degrees:\n"
                  << euler_angles.transpose() << std::endl;

        rotation_socs_to_tocs = T_SOCS_TO_TOCS.block<3, 3>(0, 0);
    }
    Eigen::Affine3d transform_tocs_to_imu; // tilded to gnss-imu antena
    M3D R_tocs_to_imu;
    V3D t_tocs_to_imu;
    {
        Eigen::Matrix4d tocs_to_imu_matrix;
        tocs_to_imu_matrix << 0., 0., -1., -0.227000,
            0., -1., 0., -0.184000,
            -1., 0., 0., -0.110000,
            0., 0., 0., 1.;
        transform_tocs_to_imu.matrix() = tocs_to_imu_matrix;
        R_tocs_to_imu << 0, 0, -1,
            0, -1, 0,
            -1, 0, 0;
        t_tocs_to_imu = V3D(-0.227, -0.184, -0.11);
    }

    pcl::PointCloud<PointType>::Ptr original_als_cloud(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr downsampled_als_cloud(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr downsampled_als_cloud2(new pcl::PointCloud<PointType>);

    // Eigen::Affine3d transform_socs_to_imu = transform_tocs_to_imu * transform_socs_to_tocs.inverse();
    Eigen::Affine3d transform_socs_to_imu = transform_tocs_to_imu * transform_socs_to_tocs;

    // Eigen::Affine3d transform_socs_to_imu = transform_tocs_to_imu.inverse() * transform_socs_to_tocs;
    // Eigen::Affine3d transform_socs_to_imu = transform_socs_to_tocs.inverse() * transform_tocs_to_imu;
    // Eigen::Affine3d transform_socs_to_imu = transform_socs_to_tocs * transform_tocs_to_imu;

    M3D R_90; // Rotation Matrix for -90 Degrees Around Y-Axis
    R_90 << 0, 0, -1,
        0, 1, 0,
        1, 0, 0;

    // how about co-registering every line independently to the map

    M3D R_scanner_to_IMU;
    // R_scanner_to_IMU <<
    //     0,  0, -1,
    //     0, -1,  0,
    //     1,  0,  0;
    R_scanner_to_IMU << 
        0, 0, -1,
        0, 1, 0,
        1, 0, 0;

    // given angles
    double roll = -0.02651 * M_PI / 180.0; // Convert to radians
    double pitch = 0.04036 * M_PI / 180.0; // Convert to radians
    double yaw = 30.00981 * M_PI / 180.0;  // Convert to radians

    M3D vux_euler;
    tf::Matrix3x3 rotation;
    rotation.setEulerYPR(yaw, pitch, roll);
    // rotation.setEulerYPR(roll, pitch, -yaw);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            vux_euler(i, j) = rotation[i][j];

    std::cout << "vux_euler:\n"
              << vux_euler << std::endl;

    M3D R_minus_30;
    double theta = -30.0 * M_PI / 180.0;
    R_minus_30 << 1, 0, 0,
        0, std::cos(theta), -std::sin(theta),
        0, std::sin(theta), std::cos(theta);

    M3D R_body_to_imu; // given a rotation matrix of -90° around the Y-axis
    R_body_to_imu << 
        0, 0, -1,
        0, 1, 0,
        1, 0, 0;

    double x_off = -0.227;
    double y_off = -0.184; // 0.184 for scanner A
    double z_off = -0.11;

    V3D to_imu(x_off, -y_off, z_off);

    M3D init_guess_R = vux_euler.transpose() * R_scanner_to_IMU;
    V3D init_guess_t = to_imu;
    pcl::KdTreeFLANN<PointType> kdtree;

    std::vector<pcl::PointCloud<VUX_PointType>::Ptr> lines;
    std::vector<Sophus::SE3> line_poses;

    M3D p2plane_ref_R;
    p2plane_ref_R << -0.0001486877, 0.4998181193, -0.8661303744,
        0.0006285201, 0.8661302597, 0.4998179451,
        0.9999997914, -0.0004700636, -0.0004429287;
    V3D p2plane_ref_t(-0.5922161686, 0.1854945762, 0.7806042559);
    
    //values estimated via point to plane 
    // init_guess_R = p2plane_ref_R;
    // init_guess_t = p2plane_ref_t;

    while (1)
    {
        ros::spinOnce();
        if (flg_exit || !ros::ok())
            break;
        if (readVUX.next(next_line)) // read one line
        {
            some_index++;
            if (time_aligned && some_index % 10 == 0)
            {
                auto cloud_time = next_line->points[0].time;
                auto pose = interpolateSE3(gnss_vux_data, cloud_time, tmp_index, false);
                Sophus::SE3 pose_local = Sophus::SE3(pose.so3(), pose.translation() - first_t);
                // Sophus::SE3 pose_local = Sophus::SE3(pose.so3().matrix()*R_body_to_imu, pose.translation() - first_t);

                // if(pose_local.translation().norm() > 45)
                //     break;

                publish_ppk_gnss_ned(pose_local, cloud_time); // publish interpolated pose

                // Sophus::SE3 pose_local = first_pose * pose_local;
                // Sophus::SE3 pose_local = Sophus::SE3(first_pose.so3() * pose.so3(), pose.translation() - first_t);

                // auto pose_ned = interpolateSE3(gnss_vux_data, cloud_time, tmp_index, true);
                // Sophus::SE3 pose_local_ned = Sophus::SE3(pose_ned.so3(), pose_ned.translation() - first_t_ned);

                pcl::PointCloud<VUX_PointType>::Ptr transformed_cloud(new pcl::PointCloud<VUX_PointType>);
                pcl::PointCloud<VUX_PointType>::Ptr downsampled_cloud(new pcl::PointCloud<VUX_PointType>);

                pcl::VoxelGrid<VUX_PointType> downSizeFilter_;
                downSizeFilter_.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
                downSizeFilter_.setInputCloud(next_line);
                downSizeFilter_.filter(*downsampled_cloud);

                *transformed_cloud = *downsampled_cloud;
                // pcl::transformPointCloud(*downsampled_cloud, *transformed_cloud, transform_socs_to_imu.cast<float>());

                M3D R_refined = init_guess_R;
                V3D t_refined = init_guess_t;
                if (false) // registration here
                {
                    // Initial guess for extrinsic transformation (Scanner -> IMU)
                    Eigen::Quaterniond q_extrinsic(R_refined);
                    V3D t_extrinsic = t_refined;

                    // we have to do the registration for few iterations - so that it gets closer and closer
                    double prev_cost = std::numeric_limits<double>::max(); // Initialize with a large value
                    double current_cost = prev_cost;
                    double cost_threshold = .001; // Threshold for stopping criterion
                    for (int iter_num = 0; iter_num < 50; iter_num++)
                    {
                        if (flg_exit || !ros::ok())
                            break;

                        // if (iter_num > 3)
                        //     break;

                        // find extrinsics------------------------------
                        ceres::Problem problem;
                        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                        // Ensure the quaternion stays valid during optimization
                        ceres::LocalParameterization *q_parameterization =
                            new ceres::EigenQuaternionParameterization();
                        int points_used_for_registration = 0;

                        double q_param[4] = {q_extrinsic.x(), q_extrinsic.y(), q_extrinsic.z(), q_extrinsic.w()};
                        double t_param[3] = {t_extrinsic.x(), t_extrinsic.y(), t_extrinsic.z()};

                        // Add the quaternion parameter block with the local parameterization
                        // to Ensure the quaternion stays valid during optimization
                        problem.AddParameterBlock(q_param, 4, q_parameterization);
                        problem.AddParameterBlock(t_param, 3); // Add the translation parameter block

                        for (const auto &raw_point : downsampled_cloud->points) // for each scanner raw point
                        {
                            V3D p_src(raw_point.x, raw_point.y, raw_point.z);
                            // transform to global with best T for NN search
                            V3D p_transformed = pose_local * (R_refined * p_src + t_refined);
                            // Nearest neighbor search
                            PointType search_point;
                            search_point.x = p_transformed.x();
                            search_point.y = p_transformed.y();
                            search_point.z = p_transformed.z();

                            bool p2plane = true;
                            if (p2plane)
                            {
                                std::vector<int> point_idx(5);
                                std::vector<float> point_dist(5);

                                if (kdtree.nearestKSearch(search_point, 5, point_idx, point_dist) > 0) // there are neighbours
                                {
                                    if (point_dist[4] < 1.) // not too far
                                    {
                                        Eigen::Matrix<double, 5, 3> matA0;
                                        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                                        for (int j = 0; j < 5; j++)
                                        {
                                            matA0(j, 0) = downsampled_als_cloud->points[point_idx[j]].x;
                                            matA0(j, 1) = downsampled_als_cloud->points[point_idx[j]].y;
                                            matA0(j, 2) = downsampled_als_cloud->points[point_idx[j]].z;
                                        }

                                        // find the norm of plane
                                        V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                                        double negative_OA_dot_norm = 1 / norm.norm();
                                        norm.normalize();

                                        bool planeValid = true;
                                        for (int j = 0; j < 5; j++)
                                        {
                                            if (fabs(norm(0) * downsampled_als_cloud->points[point_idx[j]].x +
                                                     norm(1) * downsampled_als_cloud->points[point_idx[j]].y +
                                                     norm(2) * downsampled_als_cloud->points[point_idx[j]].z + negative_OA_dot_norm) > 0.1)
                                            {
                                                planeValid = false;
                                                break;
                                            }
                                        }

                                        if (planeValid)
                                        {
                                            ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(p_src, norm, negative_OA_dot_norm, pose_local);
                                            problem.AddResidualBlock(cost_function, loss_function, q_param, t_param);
                                            points_used_for_registration++;
                                        }
                                    }
                                }
                           
                           
                            }
                            else // p2p
                            {
                                std::vector<int> point_idx(1);
                                std::vector<float> point_dist(1);

                                if (kdtree.nearestKSearch(search_point, 1, point_idx, point_dist) > 0) // there are neighbours
                                {
                                    if (point_dist[0] < 1.) // not too far
                                    {
                                        const PointType &nearest_neighbor = downsampled_als_cloud->points[point_idx[0]];
                                        points_used_for_registration++;
                                        V3D target_map(nearest_neighbor.x, nearest_neighbor.y, nearest_neighbor.z);

                                        ceres::CostFunction *cost_function = LidarDistanceFactor::Create(p_src, target_map, pose_local);

                                        // problem.AddResidualBlock(cost_function, nullptr, q_param, t_param);
                                        problem.AddResidualBlock(cost_function, loss_function, q_param, t_param);
                                    }
                                }
                            }
                        }

                        // Solve the problem
                        ceres::Solver::Options options;
                        options.linear_solver_type = ceres::DENSE_QR; // options.max_num_iterations = 100;
                        options.minimizer_progress_to_stdout = true;
                        ceres::Solver::Summary summary;
                        ceres::Solve(options, &problem, &summary);

                        // std::cout << summary.FullReport() << std::endl;
                        // std::cout << summary.BriefReport() << std::endl;

                        // Output the refined extrinsic transformation
                        Eigen::Quaterniond refined_q(q_param[3], q_param[0], q_param[1], q_param[2]);
                        t_refined = V3D(t_param[0], t_param[1], t_param[2]);
                        R_refined = refined_q.toRotationMatrix();
                        std::cout << "Registration done with " << points_used_for_registration << "/" << downsampled_cloud->size() << " points" << std::endl;
                        // std::cout << "Refined Rotation (Quaternion): " << refined_q.coeffs().transpose() << std::endl;
                        std::cout << "Refined Translation: " << t_refined.transpose() << ", prev t:" << init_guess_t.transpose() << "\n\n"
                                  << std::endl;

                        current_cost = summary.final_cost;
                        std::cout << "Iteration " << iter_num << " - Cost: " << current_cost << std::endl;

                        // Check if the cost function change is small enough to stop
                        if (std::abs(prev_cost - current_cost) < cost_threshold)
                        {
                            std::cout << "Stopping optimization: Cost change below threshold.\n";
                            break;
                        }

                        prev_cost = current_cost;
                    }
                }

                // copy this code somewhere
                // save a buffer of line clouds
                // save a buffer of gnss-imu poses
                // pass them to this function tahat will add everything in a common ceres problem for extrinsic refinement
                if (true)
                {
                    lines.push_back(downsampled_cloud);
                    line_poses.push_back(pose_local);
                    if (pose_local.translation().norm() > 50) // drove enough
                    {
                        std::cout << "Start registration with all the lines====================================" << std::endl;
                        // Initial guess for extrinsic transformation (Scanner -> IMU)
                        Eigen::Quaterniond q_extrinsic(R_refined);
                        V3D t_extrinsic = t_refined;

                        // we have to do the registration for few iterations - so that it gets closer and closer
                        double prev_cost = std::numeric_limits<double>::max(); // Initialize with a large value
                        double current_cost = prev_cost;
                        double cost_threshold = .001; // Threshold for stopping criterion
                        for (int iter_num = 0; iter_num < 50; iter_num++)
                        {
                            if (flg_exit || !ros::ok())
                                break;

                            // find extrinsics------------------------------
                            ceres::Problem problem;
                            ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                            // Ensure the quaternion stays valid during optimization
                            ceres::LocalParameterization *q_parameterization =
                                new ceres::EigenQuaternionParameterization();
                            int points_used_for_registration = 0;

                            double q_param[4] = {q_extrinsic.x(), q_extrinsic.y(), q_extrinsic.z(), q_extrinsic.w()};
                            double t_param[3] = {t_extrinsic.x(), t_extrinsic.y(), t_extrinsic.z()};

                            // Add the quaternion parameter block with the local parameterization
                            // to Ensure the quaternion stays valid during optimization
                            problem.AddParameterBlock(q_param, 4, q_parameterization);
                            problem.AddParameterBlock(t_param, 3); // Add the translation parameter block

                            for (int l = 0; l < lines.size(); l++)
                            {
                                const auto &line_cloud = lines[l];
                                const auto &line_pose = line_poses[l];
                                for (const auto &raw_point : line_cloud->points) // for each scanner raw point
                                {
                                    V3D p_src(raw_point.x, raw_point.y, raw_point.z);
                                    // transform to global with best T for NN search
                                    V3D p_transformed = line_pose * (R_refined * p_src + t_refined);
                                    // Nearest neighbor search
                                    PointType search_point;
                                    search_point.x = p_transformed.x();
                                    search_point.y = p_transformed.y();
                                    search_point.z = p_transformed.z();

                                    bool p2plane = false;// true;
                                    if (p2plane)
                                    {
                                        std::vector<int> point_idx(5);
                                        std::vector<float> point_dist(5);

                                        if (kdtree.nearestKSearch(search_point, 5, point_idx, point_dist) > 0) // there are neighbours
                                        {
                                            if (point_dist[4] < 1.) // not too far
                                            {
                                                Eigen::Matrix<double, 5, 3> matA0;
                                                Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                                                for (int j = 0; j < 5; j++)
                                                {
                                                    matA0(j, 0) = downsampled_als_cloud->points[point_idx[j]].x;
                                                    matA0(j, 1) = downsampled_als_cloud->points[point_idx[j]].y;
                                                    matA0(j, 2) = downsampled_als_cloud->points[point_idx[j]].z;
                                                }

                                                // find the norm of plane
                                                V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                                                double negative_OA_dot_norm = 1 / norm.norm();
                                                norm.normalize();

                                                bool planeValid = true;
                                                for (int j = 0; j < 5; j++)
                                                {
                                                    if (fabs(norm(0) * downsampled_als_cloud->points[point_idx[j]].x +
                                                             norm(1) * downsampled_als_cloud->points[point_idx[j]].y +
                                                             norm(2) * downsampled_als_cloud->points[point_idx[j]].z + negative_OA_dot_norm) > 0.1)
                                                    {
                                                        planeValid = false;
                                                        break;
                                                    }
                                                }

                                                if (planeValid)
                                                {
                                                    ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(p_src, norm, negative_OA_dot_norm, line_pose);
                                                    problem.AddResidualBlock(cost_function, loss_function, q_param, t_param);
                                                    points_used_for_registration++;
                                                }
                                            }
                                        }
                                    }
                                    else // p2p
                                    {
                                        std::vector<int> point_idx(1);
                                        std::vector<float> point_dist(1);

                                        if (kdtree.nearestKSearch(search_point, 1, point_idx, point_dist) > 0) // there are neighbours
                                        {
                                            if (point_dist[0] < 1.) // not too far
                                            {
                                                const PointType &nearest_neighbor = downsampled_als_cloud->points[point_idx[0]];
                                                points_used_for_registration++;
                                                V3D target_map(nearest_neighbor.x, nearest_neighbor.y, nearest_neighbor.z);

                                                ceres::CostFunction *cost_function = LidarDistanceFactor::Create(p_src, target_map, line_pose);

                                                // problem.AddResidualBlock(cost_function, nullptr, q_param, t_param);
                                                problem.AddResidualBlock(cost_function, loss_function, q_param, t_param);
                                            }
                                        }
                                    }
                                }
                            }
                            // Solve the problem
                            ceres::Solver::Options options;
                            options.linear_solver_type = ceres::DENSE_QR; // options.max_num_iterations = 100;
                            options.minimizer_progress_to_stdout = true;
                            ceres::Solver::Summary summary;
                            ceres::Solve(options, &problem, &summary);

                            // std::cout << summary.FullReport() << std::endl;
                            // std::cout << summary.BriefReport() << std::endl;

                            // Output the refined extrinsic transformation
                            Eigen::Quaterniond refined_q(q_param[3], q_param[0], q_param[1], q_param[2]);
                            t_refined = V3D(t_param[0], t_param[1], t_param[2]);
                            R_refined = refined_q.toRotationMatrix();
                            std::cout << "Registration done with " << points_used_for_registration << "/" << downsampled_cloud->size() << " points" << std::endl;
                            // std::cout << "Refined Rotation (Quaternion): " << refined_q.coeffs().transpose() << std::endl;
                            std::cout << "Refined Translation: " << t_refined.transpose() << ", prev t:" << init_guess_t.transpose() << "\n\n"
                                      << std::endl;

                            current_cost = summary.final_cost;
                            std::cout << "Iteration " << iter_num << " - Cost: " << current_cost << std::endl;

                            // Check if the cost function change is small enough to stop
                            if (std::abs(prev_cost - current_cost) < cost_threshold)
                            {
                                std::cout << "Stopping optimization: Cost change below threshold.\n";
                                break;
                            }

                            prev_cost = current_cost;
                        }

                        std::cout << "Refined solution:==========================================" << std::endl;
                        std::cout << "T:" << t_refined.transpose() << std::endl;
                        std::cout << "R:\n"
                                  << R_refined << std::endl;
                        break;
                    }
                }

                feats_undistort->clear();
                // transform to global
                for (size_t i = 0; i < transformed_cloud->size(); i++)
                {
                    V3D point_scanner(transformed_cloud->points[i].x, transformed_cloud->points[i].y, transformed_cloud->points[i].z);

                    // manual and given extrinsics
                    if (false)
                    {
                        V3D tmp = rotation_socs_to_tocs * point_scanner; // to tocs

                        // tmp = point_scanner;
                        tmp = R_tocs_to_imu * tmp; // + t_tocs_to_imu;   // to imu

                        V3D tmp_global = pose_local * tmp;

                        PointType po;
                        po.x = tmp_global[0];
                        po.y = tmp_global[1];
                        po.z = tmp_global[2];
                        feats_undistort->push_back(po);
                    }

                    if (true)
                    {
                        // V3D tmp_global_ned = pose_local * ((vux_euler.transpose() * R_scanner_to_IMU * point_scanner) + to_imu);

                        V3D tmp_global_ned = pose_local * (R_refined * point_scanner + t_refined);
                        // V3D tmp_global_ned = pose_local * (R_scanner_to_IMU * (point_scanner));

                        PointType po;
                        po.x = tmp_global_ned[0];
                        po.y = tmp_global_ned[1];
                        po.z = tmp_global_ned[2];
                        feats_undistort->push_back(po);
                    }

                    // V3D point_global = pose_local * (R_scanner_to_IMU * (R_minus_30 * point_scanner));
                    // V3D point_global = pose_local * (R_scanner_to_IMU * (point_scanner));

                    // V3D point_global = pose_local * (vux_euler.transpose() * R_scanner_to_IMU * point_scanner);

                    // point_global = point_scanner;

                    V3D point_global = pose_local * (vux_euler.transpose() * (R_scanner_to_IMU * point_scanner + to_imu));

                    // V3D point_global = pose_local * (init_guess_R * point_scanner + init_guess_t);

                    transformed_cloud->points[i].x = point_global.x();
                    transformed_cloud->points[i].y = point_global.y();
                    transformed_cloud->points[i].z = point_global.z();

                    // if (point_global.y() > 0)
                    // {
                    //     transformed_cloud->points[i].time = 100 + i * i;
                    // }
                    // else
                    // {
                    //     transformed_cloud->points[i].time = 0;
                    // }
                }

                // publishPointCloud_vux(next_line, point_cloud_pub);
                publishPointCloud_vux(transformed_cloud, point_cloud_pub);
                publish_frame_debug(pubLaserCloudDebug, feats_undistort);
            }
        }

        // raw gps from vux
        Sophus::SE3 vux_pose_global;
        {
            V3D lla;
            uint32_t raw_gnss_tod;
            if (readVUX.nextGNSS(lla, raw_gnss_tod)) // get the raw GNSS into lla - if there is raw gnss
            {
                if (!time_aligned)
                {
                    std::cout << "Start time alignment" << std::endl;
                    std::cout << "Post time:" << gnss_vux_data[tmp_index].gpstime << " raw time:" << raw_gnss_tod << std::endl;
                    // skip the measurements based on time untill aligned properly
                    while (tmp_index < gnss_vux_data.size() && gnss_vux_data[tmp_index].gpstime < raw_gnss_tod)
                    {
                        // std::cout<<"Post time:"<<gnss_vux_data[tmp_index].gpstime<<" raw time:"<<raw_gnss_tod<<std::endl;
                        // get the current pose
                        // first_pose = gnss_vux_data[tmp_index].se3;
                        tmp_index++;
                    }
                    first_pose = gnss_vux_data[tmp_index].se3;
                    first_pose_ned = gnss_vux_data[tmp_index].se3_ned;

                    first_t = first_pose.translation();
                    first_t_ned = first_pose_ned.translation();

                    // Convert to Euler angles (ZYX convention: yaw, pitch, roll)
                    V3D euler = first_pose.so3().matrix().eulerAngles(2, 1, 0) * (180.0 / M_PI);
                    std::cout << "Yaw (Z): " << euler[0] << " degrees\n";
                    std::cout << "Pitch (Y): " << euler[1] << " degrees\n";
                    std::cout << "Roll (X): " << euler[2] << " degrees\n";

                    first_pose = first_pose.inverse();

                    std::cout << "tmp_index:" << tmp_index << std::endl;
                    init_guess_index = tmp_index;
                    std::cout << "Post time:" << gnss_vux_data[tmp_index].gpstime << " raw time:" << raw_gnss_tod << std::endl;

                    std::cout << "Finished time alignment" << std::endl;
                    time_aligned = true;
                }
                // std::cout<<"lla:"<<lla.transpose()<<", raw_gnss_tod:"<<raw_gnss_tod<<std::endl;
                //  publish the gnss now - or convert to x y z first
                double easting, northing, height = lla[2];
                int zone;
                bool northp;

                if (!gps_init_origin)
                {
                    auto ref_gps_point_lla = lla;
                    GeographicLib::UTMUPS::Forward(ref_gps_point_lla[0], ref_gps_point_lla[1], zone, northp, easting, northing);

                    origin_enu = V3D(easting, northing, height);
                    std::cout << "origin_enu:" << origin_enu.transpose() << std::endl;
                    geo_converter.Reset(ref_gps_point_lla[0], ref_gps_point_lla[1], ref_gps_point_lla[2]); // set origin the current point

                    gps_init_origin = true;
                }

                // Convert Latitude and Longitude to UTM
                GeographicLib::UTMUPS::Forward(lla[0], lla[1], zone, northp, easting, northing);
                curr_enu = V3D(easting, northing, height);

                double x, y, z; // LLA->carthesian
                geo_converter.Forward(lla[0], lla[1], lla[2], x, y, z);
                carthesian = V3D(x, y, z); // this will be relative w.r.t. zero origin
                if (abs(carthesian.x()) > 10000 || abs(carthesian.x()) > 10000 || abs(carthesian.x()) > 10000)
                {
                    ROS_INFO("Error origin : %f, %f, %f", carthesian(0), carthesian(1), carthesian(2));
                    return;
                }

                std::cout << "raw gnss carthesian:" << carthesian.transpose() << " at time:         " << raw_gnss_tod << std::endl;

                // auto raw_gnss_pose = Sophus::SE3(first_pose.so3().matrix().transpose(), carthesian);
                // raw_gnss_pose = first_pose*raw_gnss_pose;
                // carthesian = raw_gnss_pose.translation();

                publish_raw_gnss(carthesian, raw_gnss_tod);
            }

            if (time_aligned)
            {
                // VUX data - georeferenced as reference
                if (do_once)
                {
                    do_once = false;
                    // work with sensor B
                    std::string f1 = "/media/eugeniu/T7/roamer/09_Export/VUX-1HA-22-2022-B/240725_092351_v12.las";
                    std::istream *ifs = liblas::Open(f1, std::ios::in | std::ios::binary);
                    if (!ifs)
                    {
                        std::cerr << "Cannot open " << f1 << " for read.  Exiting..." << std::endl;
                        throw std::invalid_argument("Cannot open the las/laz file");
                    }

                    if (!ifs->good())
                        throw std::runtime_error("Reading went wrong!");

                    liblas::ReaderFactory readerFactory;
                    liblas::Reader reader = readerFactory.CreateWithStream(*ifs);
                    liblas::Header const &header = reader.GetHeader();

                    std::cout << "Compressed: " << (header.Compressed() == true) ? "true" : "false";
                    std::cout << " Signature: " << header.GetFileSignature() << '\n';
                    std::cout << "Start reading the data" << std::endl;
                    V3D offset_ = V3D(header.GetOffsetX(), header.GetOffsetY(), header.GetOffsetZ());
                    std::cout << "offset_:" << offset_.transpose() << std::endl;
                    std::cout << "GetPointRecordsCount:" << header.GetPointRecordsCount() << std::endl;

                    M3D R_; // estimated from mls-als its from ALS 2 MLS rotation
                    R_ << -8.231261e-01, -5.678501e-01, -3.097665e-03,
                        5.675232e-01, -8.224404e-01, -3.884915e-02,
                        1.951285e-02, -3.373575e-02, 9.992403e-01;
                    Sophus::SO3 R(R_);
                    V3D als2mls_translation(4.181350e+06, 5.355192e+06, 2.210141e+05);

                    PointType point;
                    // pcl::PointCloud<PointType>::Ptr original_als_cloud(new pcl::PointCloud<PointType>);
                    original_als_cloud->resize(header.GetPointRecordsCount());
                    size_t index = 0;
                    double max_length = 75 * 75; // 150 * 150;
                    auto als_to_mls = als_obj->als_to_mls;
                    als_to_mls = Sophus::SE3(R, als2mls_translation);
                    // std::cout << "als_to_mls:" << als_to_mls.translation().transpose() << std::endl;

                    offset_ = R * offset_;         // also transformed to MLS frame
                    while (reader.ReadNextPoint()) // this is in ENU frame
                    {
                        liblas::Point const &p = reader.GetPoint();
                        // substract the postprocessed GNSS-IMU position
                        V3D cloudPoint = V3D(p.GetX(), p.GetY(), p.GetZ()) - first_t;

                        // shift with the inverse of the first pose
                        // V3D cloudPoint = first_pose * V3D(p.GetX(), p.GetY(), p.GetZ());

                        // rotate to mls frame
                        // V3D cloudPoint = R * V3D(p.GetX(), p.GetY(), p.GetZ()) - offset_;
                        if (cloudPoint.squaredNorm() > max_length)
                            continue;

                        point.x = cloudPoint.x();
                        point.y = cloudPoint.y();
                        point.z = cloudPoint.z();

                        point.time = p.GetTime();
                        // point.reflectance = index; // p.GetReflectance();

                        original_als_cloud->points[index] = point;
                        index++;
                        // if (index > 500000)
                        // {
                        //      break;
                        // }
                    }
                    original_als_cloud->resize(index);

                    std::cout << "original_als_cloud:" << original_als_cloud->points.size() << std::endl;

                    pcl::VoxelGrid<PointType> downSizeFilter_;
                    downSizeFilter_.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
                    // downSizeFilter_.setLeafSize(1, 1, 1);

                    downSizeFilter_.setInputCloud(original_als_cloud);
                    downSizeFilter_.filter(*downsampled_als_cloud);

                    //*downsampled_als_cloud = *original_als_cloud;

                    std::cout << "downsampled_als_cloud:" << downsampled_als_cloud->size() << std::endl;
                    // add the second sensor too
                    bool add_second_scanner = false;
                    if (add_second_scanner)
                    {
                        std::cout << "\nstart adding point from the second sensor\n"
                                  << std::endl;
                        f1 = "/media/eugeniu/T7/roamer/09_Export/VUX-1HA-22-2022-A/240725_093813_v12.las";

                        std::istream *ifs = liblas::Open(f1, std::ios::in | std::ios::binary);
                        if (!ifs)
                        {
                            std::cerr << "Cannot open " << f1 << " for read.  Exiting..." << std::endl;
                            throw std::invalid_argument("Cannot open the las/laz file");
                        }

                        if (!ifs->good())
                            throw std::runtime_error("Reading went wrong!");

                        liblas::ReaderFactory readerFactory;
                        liblas::Reader reader = readerFactory.CreateWithStream(*ifs);
                        liblas::Header const &header = reader.GetHeader();

                        std::cout << "Compressed: " << (header.Compressed() == true) ? "true" : "false";
                        std::cout << " Signature: " << header.GetFileSignature() << '\n';
                        std::cout << "Start reading the data" << std::endl;
                        std::cout << "offset_:" << offset_.transpose() << std::endl;
                        std::cout << "GetPointRecordsCount:" << header.GetPointRecordsCount() << std::endl;

                        PointType point;
                        pcl::PointCloud<PointType>::Ptr original_als_cloud2(new pcl::PointCloud<PointType>);
                        original_als_cloud2->resize(header.GetPointRecordsCount());
                        index = 0;
                        while (reader.ReadNextPoint())
                        {
                            liblas::Point const &p = reader.GetPoint();
                            // substract the postprocessed GNSS-IMU position
                            V3D cloudPoint = V3D(p.GetX(), p.GetY(), p.GetZ()) - first_t;

                            if (cloudPoint.squaredNorm() > max_length)
                                continue;

                            point.x = cloudPoint.x();
                            point.y = cloudPoint.y();
                            point.z = cloudPoint.z();

                            point.time = p.GetTime();
                            // point.reflectance = index; // p.GetReflectance();

                            original_als_cloud2->points[index] = point;
                            index++;
                            // if (index > 500000)
                            // {
                            //      break;
                            // }
                        }
                        original_als_cloud2->resize(index);

                        std::cout << "original_als_cloud2:" << original_als_cloud2->points.size() << std::endl;

                        downSizeFilter_.setInputCloud(original_als_cloud2);
                        downSizeFilter_.filter(*downsampled_als_cloud2);

                        //*downsampled_als_cloud = *original_als_cloud;

                        std::cout << "downsampled_als_cloud2:" << downsampled_als_cloud2->size() << std::endl;
                        *downsampled_als_cloud += *downsampled_als_cloud2;
                    }

                    publish_frame_debug(pubLaserALSMap, downsampled_als_cloud);
                    // publishPointCloud(downsampled_als_cloud, pubLaserALSMap);

                    // Build KDTree for the reference map
                    kdtree.setInputCloud(downsampled_als_cloud);
                }

                // PPK gnss-imu data
                while (tmp_index < gnss_vux_data.size() && gnss_vux_data[tmp_index].gpstime < raw_gnss_tod) // if (tmp_index < gnss_vux_data.size())
                {
                    const double &msg_time = gnss_vux_data[tmp_index].gpstime;

                    auto p = Sophus::SE3(gnss_vux_data[tmp_index].se3.so3(), gnss_vux_data[tmp_index].se3.translation() - first_t);
                    // auto p = first_pose * gnss_vux_data[tmp_index].se3; //shift with the inverse of the first pose
                    //  auto p = Sophus::SE3(first_pose.so3() * gnss_vux_data[tmp_index].se3.so3(), gnss_vux_data[tmp_index].se3.translation() - first_t);

                    publish_ppk_gnss(p, msg_time);

                    // auto p_ned = Sophus::SE3(gnss_vux_data[tmp_index].se3_ned.so3(), gnss_vux_data[tmp_index].se3_ned.translation() - first_t_ned);
                    // publish_ppk_gnss_ned(p_ned, msg_time);

                    vux_pose_global = p;
                    tmp_index++;
                    rate.sleep();
                }
            }
        }

        rate.sleep();
        if (flg_exit || !ros::ok())
            break;
    }

    return; //-------------------------------------------------------------

    // #define compile_this  //this will be the MLS
#ifdef compile_this

    // ros::Rate rate(5);
    for (const rosbag::MessageInstance &m : view)
    {
        ros::spinOnce();
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

        if (flg_exit || !ros::ok())
            break;

        // std::cout << "\nIMU:" << imu_buffer.size() << ", GPS:" << gps_buffer.size() << ", LiDAR:" << lidar_buffer.size() << std::endl;

        if (sync_packages(Measures))
        {
            // hesai-mls registration
            if (true)
            {
                double t00 = omp_get_wtime();

                if (flg_first_scan)
                {
                    first_lidar_time = Measures.lidar_beg_time;
                    flg_first_scan = false;
                    continue;
                }

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

                pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;
                flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

#define USE_EKF
#ifdef USE_EKF
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

#if USE_STATIC_KDTREE == 0
                if (ikdtree.Root_Node == nullptr)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
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
                    ikdtree.Build(feats_down_world->points);
                    continue;
                }
#else
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
#endif

                Nearest_Points.resize(feats_down_size);

#if USE_STATIC_KDTREE == 0
                if (!estimator_.update(LASER_POINT_COV, feats_down_body, ikdtree, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en))
#else
                if (!estimator_.update(LASER_POINT_COV, feats_down_body, laserCloudSurfMap, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en))
#endif
                {
                    std::cout << "\n------------------MLS update failed--------------------------------" << std::endl;
                }
                // double t_LiDAR_update = omp_get_wtime();
                // std::cout << "\nIMU_process time(ms):  " << (t_IMU_process - t00) * 1000 <<
                //", cloud_voxelization (ms): " << (t_cloud_voxelization - t_IMU_process) * 1000 <<
                //", LiDAR_update (ms): " << (t_LiDAR_update - t_cloud_voxelization) * 1000 << std::endl;

                // Crop the local map----------------------------------------------------
                state_point = estimator_.get_x();
                pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;
                RemovePointsFarFromLocation();

                // get and publish the GNSS pose-----------------------------------------
                gnss_obj->Process(gps_buffer, lidar_end_time, state_point.pos);

                Sophus::SE3 gnss_pose = (gnss_obj->use_postprocessed_gnss) ? gnss_obj->postprocessed_gps_pose : gnss_obj->gps_pose;

                // gnss_pose.so3() = state_point.rot; // use the MLS orientation
                publish_gnss_odometry(gnss_pose);

                if (gnss_obj->GNSS_extrinsic_init && use_gnss) // if gnss aligned
                {
                    const bool global_error = false; // set this true for global error of gps
                    // auto gps_cov_ = V3D(GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR);
                    // auto gps_cov_ = gnss_obj->gps_cov;
                    auto gps_cov_ = Eigen::Vector3d(GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR);
                    if (global_error)
                    {
                        const V3D &gnss_in_enu = gnss_obj->carthesian;
                        estimator_.update(gnss_in_enu, gps_cov_, NUM_MAX_ITERATIONS, global_error, gnss_obj->R_GNSS_to_MLS.transpose());
                    }
                    else
                    {
                        const V3D &gnss_in_mls = gnss_pose.translation();
                        estimator_.update(gnss_in_mls, gps_cov_, NUM_MAX_ITERATIONS, global_error);
                    }
                }

#ifdef USE_ALS
                if (!als_obj->refine_als)
                {                                      // not initialized
                    if (gnss_obj->GNSS_extrinsic_init) // if gnss aligned
                    {
#if USE_STATIC_KDTREE == 0
                        PointVector().swap(ikdtree.PCL_Storage);
                        ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                        featsFromMap->clear();
                        featsFromMap->points = ikdtree.PCL_Storage;
#else
                        *featsFromMap = *laserCloudSurfMap;
#endif
                        als_obj->init(gnss_obj->origin_enu, gnss_obj->R_GNSS_to_MLS, featsFromMap);
                        gnss_obj->updateExtrinsic(als_obj->R_to_mls); // use the scan-based refined rotation for GNSS
                    }
                }
                else
                {
                    als_obj->Update(Sophus::SE3(state_point.rot, state_point.pos));

                    Nearest_Points.resize(feats_down_size);
#if USE_STATIC_KDTREE == 0
                    if (!estimator_.update(LASER_POINT_COV / 4, feats_down_body, als_obj->ikdtree, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en))
#else
                    if (!estimator_.update(LASER_POINT_COV / 4, feats_down_body, als_obj->als_cloud, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en))
#endif
                    {
                        std::cout << "\n------------------ ALS update failed --------------------------------" << std::endl;
                        // TODO check here why -  there is not enough als data
                    }
                    else
                    {
                        // add the ALS points to MLS local map
                        if (saveALS_NN_2_MLS)
                        {
                            local_map_update_from_ALS(Nearest_Points);
                        }
                    }
                }

                if (pubLaserALSMap.getNumSubscribers() != 0)
                {
                    als_obj->getCloud(featsFromMap);
                    publish_map(pubLaserALSMap);
                }
#endif

                // Update the local map--------------------------------------------------
                feats_down_world->resize(feats_down_size);
                local_map_update();

                // Publish odometry and point clouds------------------------------------
                publish_odometry(pubOdomAftMapped);
                if (scan_pub_en)
                {
                    if (pubLaserCloudFull.getNumSubscribers() != 0)
                        publish_frame_world(pubLaserCloudFull);
                }

                if (pubLaserCloudMap.getNumSubscribers() != 0)
                {
#if USE_STATIC_KDTREE == 0
                    PointVector().swap(ikdtree.PCL_Storage);
                    ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                    featsFromMap->clear();
                    featsFromMap->points = ikdtree.PCL_Storage;
#else
                    *featsFromMap = *laserCloudSurfMap;
#endif
                    publish_map(pubLaserCloudMap);
                }

#endif
                double t11 = omp_get_wtime();
                std::cout << "Mapping time(ms):  " << (t11 - t00) * 1000 << ", feats_down_size: " << feats_down_size << ", lidar_end_time:" << lidar_end_time << "\n"
                          << std::endl;
            }

            // publishPointCloud(downsampled_als_cloud, point_cloud_pub);

            // if (false)
            // {
            //     while (!readVUX.timeAlign(lidar_end_time))
            //     {
            //         if (flg_exit || !ros::ok())
            //             break;
            //         std::cout << "still inside the while loop" << std::endl;
            //     }
            //     std::cout << "out of the while loop" << std::endl;
            // }

            // VUX data
            if (do_once && false) //&& gnss_obj->GNSS_extrinsic_init
            {
                do_once = false;
                // work with sensor B
                std::string f1 = "/media/eugeniu/T7/roamer/09_Export/VUX-1HA-22-2022-B/240725_092351_v12.las";

                std::istream *ifs = liblas::Open(f1, std::ios::in | std::ios::binary);
                if (!ifs)
                {
                    std::cerr << "Cannot open " << f1 << " for read.  Exiting..." << std::endl;
                    throw std::invalid_argument("Cannot open the las/laz file");
                }

                if (!ifs->good())
                    throw std::runtime_error("Reading went wrong!");

                liblas::ReaderFactory readerFactory;
                liblas::Reader reader = readerFactory.CreateWithStream(*ifs);
                liblas::Header const &header = reader.GetHeader();

                std::cout << "Compressed: " << (header.Compressed() == true) ? "true" : "false";
                std::cout << " Signature: " << header.GetFileSignature() << '\n';
                std::cout << "Start reading the data" << std::endl;
                V3D offset_ = V3D(header.GetOffsetX(), header.GetOffsetY(), header.GetOffsetZ());
                std::cout << "offset_:" << offset_.transpose() << std::endl;
                std::cout << "GetPointRecordsCount:" << header.GetPointRecordsCount() << std::endl;

                //            398285.228  6786325.401  199.714  //the original offset
                // from als2mls
                offset_ = V3D(398266.295, 6786185.313, 151.899); // from the postprocessed gnss - imu of the vux

                // the utc and gps time from postprocessed gnss-imu hesai
                // 1721899390.00 379408.00 - these correspond to first lidar cloud

                offset_ = V3D(398265.461, 6786168.301, 152.220);
                //+18 s
                // offset_ = V3D(398267.105,  6786166.119, 152.241);

                Eigen::Matrix3d R_; // estimated from mls-als its from ALS 2 MLS rotation
                R_ << -8.231261e-01, -5.678501e-01, -3.097665e-03,
                    5.675232e-01, -8.224404e-01, -3.884915e-02,
                    1.951285e-02, -3.373575e-02, 9.992403e-01;
                Sophus::SO3 R(R_);
                V3D als2mls_translation(4.181350e+06, 5.355192e+06, 2.210141e+05);

                PointType point;
                pcl::PointCloud<PointType>::Ptr original_als_cloud(new pcl::PointCloud<PointType>);
                original_als_cloud->resize(header.GetPointRecordsCount());
                size_t index = 0;
                double max_length = 50 * 50; // 150 * 150;
                auto als_to_mls = als_obj->als_to_mls;

                als_to_mls = Sophus::SE3(R, als2mls_translation);

                // std::cout << "als_to_mls:" << als_to_mls.translation().transpose() << std::endl;

                offset_ = R * offset_; // also transformed to MLS frame
                while (reader.ReadNextPoint())
                {
                    liblas::Point const &p = reader.GetPoint();

                    // V3D cloudPoint = V3D(p.GetX(), p.GetY(), p.GetZ()) - offset_;
                    // V3D cloudPoint = als_to_mls * V3D(p.GetX(), p.GetY(), p.GetZ()); // align to als

                    // substract the postprocessed GNSS-IMU position - rotate to MLS frame
                    // V3D cloudPoint = R * (V3D(p.GetX(), p.GetY(), p.GetZ()) - offset_);

                    // rotate to mls frame
                    V3D cloudPoint = R * V3D(p.GetX(), p.GetY(), p.GetZ()) - offset_;

                    if (cloudPoint.squaredNorm() > max_length)
                        continue;

                    point.x = cloudPoint.x();
                    point.y = cloudPoint.y();
                    point.z = cloudPoint.z();

                    point.time = p.GetTime();
                    // point.reflectance = index; // p.GetReflectance();

                    original_als_cloud->points[index] = point;
                    index++;
                    // if (index > 500000)
                    // {
                    //      break;
                    // }
                }
                original_als_cloud->resize(index);

                std::cout << "original_als_cloud:" << original_als_cloud->points.size() << std::endl;

                pcl::VoxelGrid<PointType> downSizeFilter_;
                downSizeFilter_.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
                downSizeFilter_.setInputCloud(original_als_cloud);
                downSizeFilter_.filter(*downsampled_als_cloud);

                //*downsampled_als_cloud = *original_als_cloud;

                std::cout << "downsampled_als_cloud:" << downsampled_als_cloud->size() << std::endl;

                publishPointCloud(downsampled_als_cloud, point_cloud_pub);
            }
        }

        // rate.sleep();
    }
    bag.close();
#endif
}