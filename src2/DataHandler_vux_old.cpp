

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

#include <visualization_msgs/Marker.h>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

volatile bool flg_exit = false;
void SigHandle(int sig)
{
    ROS_WARN("Caught signal %d, stopping...", sig);
    flg_exit = true; // Set the flag to stop the loop
}

void publishPointCloud(pcl::PointCloud<PointType>::Ptr &cloud, const ros::Publisher &point_cloud_pub)
{
    if (point_cloud_pub.getNumSubscribers() == 0)
        return;

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
    if (point_cloud_pub.getNumSubscribers() == 0)
        return;

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

void publish_raw_vux_imu_gnss(const Sophus::SE3 &_pose, const double &msg_time)
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
    br.sendTransform(tf::StampedTransform(transformGPS, ros::Time().fromSec(msg_time), "world", "vux_imu"));
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

void publishAccelerationArrow(ros::Publisher &marker_pub, const Eigen::Vector3d &acceleration, const double &msg_time)
{
    visualization_msgs::Marker arrow;

    arrow.header.frame_id = "PPK_GNSS"; // the location of the ppk gnss imu
    arrow.header.stamp = ros::Time().fromSec(msg_time);
    arrow.ns = "acceleration_arrow";
    arrow.id = 0;
    arrow.type = visualization_msgs::Marker::ARROW;
    arrow.action = visualization_msgs::Marker::ADD;

    // Define arrow start (origin) and end (acceleration direction)
    geometry_msgs::Point start, end;
    start.x = 0.0;
    start.y = 0.0;
    start.z = 0.0;

    end.x = acceleration.x();
    end.y = acceleration.y();
    end.z = acceleration.z();

    arrow.points.push_back(start);
    arrow.points.push_back(end);

    // Set arrow properties
    arrow.scale.x = 0.5;
    arrow.scale.y = 0.5;
    arrow.scale.z = 0.5;

    arrow.color.r = 1.0; // Full red
    arrow.color.g = 0.5; // Medium green
    arrow.color.b = 0.0; // No blue
    arrow.color.a = 1.0; // Fully opaque

    marker_pub.publish(arrow);
}

struct vux_gnss_post
{
    double gps_tod;
    double gps_tow;
    double easting, northing, h_ell;
    double omega, phi, kappa;
    Sophus::SE3 se3;

    V3D acc_no_gravity, acc, gyro, velocity;
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

        // Ignore empty lines
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
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R(i, j) = rotation[i][j];

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
                           const Sophus::SE3 &pose2, const double time2, const double t)
{
    // Compute interpolation weight
    double alpha = (t - time1) / (time2 - time1);

    Sophus::SE3 delta = pose1.inverse() * pose2;
    // exponential map to interpolate
    Sophus::SE3 interpolated = pose1 * Sophus::SE3::exp(alpha * delta.log());

    return interpolated;
}

#include <pcl/kdtree/kdtree_flann.h>

// #include "gtsam_part.hpp"

#include "vux_registration.hpp"

#include "VoxelHash.hpp"

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

#define USE_ALS
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

    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("acceleration_marker", 10);

    ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("cloud", 1);
    ros::Publisher normals_pub = nh.advertise<visualization_msgs::Marker>("normals", 1);

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

    // std::string post_processed_gnss_imu_vux_file = "/media/eugeniu/T7/roamer/evo_20240725_reCalculated2_poqmodout.txt";
    std::string post_processed_gnss_imu_vux_file = "/media/eugeniu/T7/roamer/evo_20240725_reCalculated2_poqmodouEugent.txt";

    std::vector<vux_gnss_post> gnss_vux_data = readMeasurements(post_processed_gnss_imu_vux_file);
    std::cout << "gnss_vux_data:" << gnss_vux_data.size() << std::endl;
    auto first_m = gnss_vux_data[0];
    V3D first_t(first_m.easting, first_m.northing, first_m.h_ell), first_t_ned;
    std::cout << "\n gps_tod:" << first_m.gps_tod << ", easting:" << first_m.easting << ", northing:" << first_m.northing << ", h_ell:" << first_m.h_ell << "\n"
              << std::endl;
    std::cout << "gps_tow:" << first_m.gps_tow << ", omega:" << first_m.omega << ", phi:" << first_m.phi << ", kappa:" << first_m.kappa << std::endl;
    int tmp_index = 0, init_guess_index = 0;

    // ros::Rate rate(250);
    ros::Rate rate(500);

    bool time_aligned = false;
    bool do_once = true;
    int some_index = 0;

    pcl::PointCloud<PointType>::Ptr original_als_cloud(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr downsampled_als_cloud(new pcl::PointCloud<PointType>);

    // Reading the mls data - and sync with them too-------------
    int mls_msg_index_start = 0;
    bool vux_mls_time_aligned = false;

    M3D R_; // estimated from mls-als its from ALS 2 MLS rotation
    R_ << -8.231261e-01, -5.678501e-01, -3.097665e-03,
        5.675232e-01, -8.224404e-01, -3.884915e-02,
        1.951285e-02, -3.373575e-02, 9.992403e-01;
    Sophus::SO3 R_als2mls(R_);
    V3D als2mls_translation(4.181350e+06, 5.355192e+06, 2.210141e+05);
    auto als_to_mls = Sophus::SE3(R_als2mls, als2mls_translation);

    pcl::VoxelGrid<VUX_PointType> downSizeFilter_vux;
    downSizeFilter_vux.setLeafSize(filter_size_surf_min / 2, filter_size_surf_min / 2, filter_size_surf_min / 2);

    M3D vux_euler;
    {
        // given angles
        double roll = -0.02651 * M_PI / 180.0; // Convert to radians
        double pitch = 0.04036 * M_PI / 180.0; // Convert to radians
        double yaw = 30.00981 * M_PI / 180.0;  // Convert to radians

        tf::Matrix3x3 rotation;
        rotation.setEulerYPR(yaw, pitch, roll);
        // rotation.setEulerYPR(roll, pitch, -yaw);

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                vux_euler(i, j) = rotation[i][j];

        std::cout << "vux_euler:\n"
                  << vux_euler << std::endl;
    }

    M3D Rz;
    double angle = -M_PI / 2.0; // -90 degrees in radians
    Rz << cos(angle), -sin(angle), 0,
        sin(angle), cos(angle), 0,
        0, 0, 1;

    M3D R_vux2imu;
    R_vux2imu << -0.0001486877, 0.4998181193, -0.8661303744,
        0.0006285201, 0.8661302597, 0.4998179451,
        0.9999997914, -0.0004700636, -0.0004429287;
    V3D t_vux2imu(-0.5922161686, 0.1854945762, 0.7806042559);

    int scan_id = 0, vux_scan_id = 0;
    Sophus::SE3 first_vux_pose;

    Sophus::SE3 prev_mls, curr_mls;
    double prev_mls_time, curr_mls_time;

    std::vector<pcl::PointCloud<VUX_PointType>::Ptr> lines;
    std::vector<Sophus::SE3> line_poses;

    M3D R_vux2mls; // init guess
    R_vux2mls << 0, -1, 0,
        0, 0, -1,
        1, 0, 0;
    V3D t_vux2mls(0, -2.09, 1.02);
    // R_vux2mls << 0.0238753375, -0.7991623924, -0.6006408567,
    //     -0.2655540110, 0.5741582890, -0.7744826185,
    //     0.9638003089, 0.1779936225, -0.1985125558;
    // //V3D t_vux2mls(-2.3673467879, -21.6629767852, -9.7389833445);
    // V3D t_vux2mls(0, 0, 0);

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());

    M3D R_refined = R_vux2mls;
    V3D t_refined = t_vux2mls;
    R_refined = vux_euler.transpose() * R_vux2mls;

    // use init guess from vux ppk-gnss
    // R_refined = R_vux2imu;
    // t_refined = t_vux2imu;

    // start with zero init guess
    // R_refined = Eye3d;
    // t_refined = Zero3d;

    bool perform_registration_refinement = true;
    double threshold_nn = 1; // ;

    Sophus::SE3 T_vux2mls = Sophus::SE3(R_als2mls, V3D(0, 0, 0));

    bool raw_vux_imu_time_aligned = false;

    // cylinder buffer for vux
    std::deque<pcl::PointCloud<VUX_PointType>::Ptr> lines_buffer;
    std::deque<Sophus::SE3> line_poses_buffer;

    // pcl::KdTreeFLANN<PointType> kdtree;
    pcl::KdTreeFLANN<PointType>::Ptr kdtree(new pcl::KdTreeFLANN<PointType>());

    bool vux_imu_init = false, gps_init_origin = false;
    V3D origin_enu, carthesian, curr_enu;
    GeographicLib::LocalCartesian geo_converter;
    RIEKF vux_imu_estimator_;
    state vux_imu_state;

    bool perform_mls_registration = true;
    bool hash_map_built = false;

    // 1m voxel size
    // double voxel_size, double max_distance, int max_points_per_voxel
    VoxelHashMap landmarks_map(1.0, 100, 1000);

    for (const rosbag::MessageInstance &m : view)
    {
        scan_id++;

        // if (scan_id < 45100) // this is only for the 0 bag
        //     continue;

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
            // hesai-mls registration
            if (perform_mls_registration)
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
                if (use_gnss)
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

                local_map_update(); // this will update local map with curr measurements and crop the map

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

            Sophus::SE3 als2mls = als_to_mls;

            // put it back later
            // if (!gnss_obj->GNSS_extrinsic_init)
            //     continue;

            // als2mls = als_obj->als_to_mls;

            // // std::cout << "als_obj->als_to_mls:\n" << als2mls.log().transpose() << std::endl;
            // use only the rotation
            T_vux2mls = Sophus::SE3(als2mls.so3(), V3D(0, 0, 0));

            double timestamp = Measures.lidar_end_time;
            // timestamp+=1.; //this gets vux closer

            std::time_t time_sec = static_cast<std::time_t>(timestamp);
            // Convert to GMT time
            std::tm *gmt_timeinfo = std::gmtime(&time_sec);
            double time_of_day_sec = gmt_timeinfo->tm_hour * 3600 +
                                     gmt_timeinfo->tm_min * 60 +
                                     gmt_timeinfo->tm_sec +
                                     (timestamp - time_sec); // Extract fractional part

            // get the tod from GNSS not hesai
            // time_of_day_sec = gnss_obj->tod;

            if (!vux_mls_time_aligned)
            {
                if (!raw_vux_imu_time_aligned)
                {
                    double first_vux_time = 0;
                    if (!readVUX.timeAlign(time_of_day_sec))
                    {
                        throw std::runtime_error("There is an issue with time aligning of raw vux and hesai mls");
                    }

                    if (readVUX.next(next_line))
                    {
                        if (!next_line->empty())
                        {
                            first_vux_time = next_line->points[0].time;
                            std::cout << "first_vux_time:" << first_vux_time << std::endl;
                        }
                    }

                    // synch raw vux with ppk------------------------------------
                    while (readVUX.next(next_line))
                    {
                        ros::spinOnce();
                        if (flg_exit || !ros::ok())
                            break;

                        V3D lla;
                        uint32_t raw_gnss_tod;
                        if (readVUX.nextGNSS(lla, raw_gnss_tod)) // get the raw GNSS into lla - if there is raw gnss
                        {
                            if (!time_aligned)
                            {
                                const auto &cloud_time = next_line->points[0].time;
                                std::cout << "\nreadVUX.next raw vux cloud time:" << cloud_time << std::endl;

                                std::cout << "Start RAW-PPK VUX time alignment" << std::endl;
                                std::cout << "Post time:" << gnss_vux_data[tmp_index].gps_tod << " raw time:" << raw_gnss_tod << ", first_vux_time:" << first_vux_time << std::endl;
                                // skip the measurements based on time untill aligned properly
                                while (tmp_index < gnss_vux_data.size() && gnss_vux_data[tmp_index].gps_tod < raw_gnss_tod)
                                {
                                    // std::cout<<"Post time:"<<gnss_vux_data[tmp_index].gps_tod<<" raw time:"<<raw_gnss_tod<<std::endl;
                                    tmp_index++;
                                }

                                std::cout << "tmp_index:" << tmp_index << std::endl;
                                init_guess_index = tmp_index;

                                {
                                    int time_of_day_seconds = gnss_vux_data[tmp_index].gps_tod;
                                    // Calculate hours, minutes, and seconds
                                    int hours = time_of_day_seconds / 3600; // 1 hour = 3600 seconds
                                    int remaining_seconds = time_of_day_seconds % 3600;
                                    int minutes = remaining_seconds / 60; // 1 minute = 60 seconds
                                    int seconds = remaining_seconds % 60;

                                    // Print in "h minus seconds" format
                                    std::cout << hours << "h " << minutes << "m " << seconds << "s" << std::endl;
                                }

                                std::cout << "Post time:" << gnss_vux_data[tmp_index].gps_tod << " raw time:" << raw_gnss_tod << ", first_vux_time:" << first_vux_time << std::endl;
                                std::cout << "Finished time alignment\n"
                                          << std::endl;
                                time_aligned = true;
                                break;
                            }
                        }

                        rate.sleep();
                    }

                    raw_vux_imu_time_aligned = true;
                }

                // std::cout << "Hesai GMT Time of the day (seconds): " << time_of_day_sec << " s" << std::endl;
                // char buffer_gmt[9]; // HH:MM:SS
                // std::strftime(buffer_gmt, sizeof(buffer_gmt), "%H:%M:%S", gmt_timeinfo);
                // std::cout << "GMT Message time: " << buffer_gmt << ",  scan_id:" << scan_id << std::endl;
                //---------------------------------------------------------------------------------------------------------
                double diff = fabs(gnss_vux_data[tmp_index].gps_tod - time_of_day_sec);
                std::cout << " vux-gnss:" << gnss_vux_data[tmp_index].gps_tod << ", time_of_day_sec:" << time_of_day_sec << ", diff:" << diff << std::endl;

                prev_mls = Sophus::SE3(state_point.rot, state_point.pos);
                prev_mls_time = time_of_day_sec;

                if (diff < 0.1)
                {
                    std::cout << "\nsynchronised\n"
                              << std::endl;
                    vux_mls_time_aligned = true;
                    first_vux_pose = gnss_vux_data[tmp_index].se3;

                    continue;
                }
                if (gnss_vux_data[tmp_index].gps_tod > time_of_day_sec) // vux is ahead on time - we do not have vux for this hesai data
                {
                    // drop hesai frames
                    std::cout << "Do nothing - wating till we get the VUX data based on time" << std::endl;
                }
                else
                {
                    // drop vux frames
                    while (tmp_index < gnss_vux_data.size() && gnss_vux_data[tmp_index].gps_tod <= time_of_day_sec)
                    {
                        if (flg_exit || !ros::ok())
                            break;

                        diff = fabs(gnss_vux_data[tmp_index].gps_tod - time_of_day_sec);
                        std::cout << "\ndrop vux frames Post time:" << gnss_vux_data[tmp_index].gps_tod << " time_of_day_sec time:" << time_of_day_sec << ", diff:" << diff << std::endl;

                        // get the scanner to this time
                        while (readVUX.next(next_line))
                        {
                            if (flg_exit || !ros::ok())
                                break;

                            if (next_line->empty())
                                break;

                            const auto &cloud_time = next_line->points[0].time;
                            if (cloud_time > gnss_vux_data[tmp_index].gps_tod)
                                break;
                        }
                        if (!next_line->empty())
                        {
                            const auto &cloud_time = next_line->points[0].time;
                            std::cout << "raw vux cloud time:" << cloud_time << std::endl;
                        }

                        tmp_index++;
                    }
                }
            }
            else
            {
                // throw std::runtime_error("\n\nstart printing the data\n");

                {
                    // //this works the other way around - from tod to ROS timestamp
                    // int week_number = 2324;             // this is from the data recording day
                    // int SECONDS_PER_WEEK = 604800;      // Number of seconds in a week
                    // int SECONDS_PER_DAY = 86400;
                    // int GPS_TO_UNIX_OFFSET = 315964800; // Offset between GPS epoch and UNIX epoch
                    // week_number = (timestamp - GPS_TO_UNIX_OFFSET) / SECONDS_PER_WEEK;

                    // // Extract the day of the week (0 = Sunday, 6 = Saturday)
                    // int day_of_week = gmt_timeinfo->tm_wday; //we add one to get thurday
                    // std::cout<<"day_of_week:"<<day_of_week<<", week_number:"<<week_number<<std::endl;

                    // double vux_unix_time = GPS_TO_UNIX_OFFSET + (week_number * SECONDS_PER_WEEK) + (day_of_week * SECONDS_PER_DAY) + gnss_vux_data[tmp_index].gps_tod;
                    // auto dif = fabs(vux_unix_time - timestamp);
                    // std::cout << " vux_unix_time:" << vux_unix_time << ", timestamp:" << timestamp << ", dif:" << dif << std::endl;
                }

                std::cout << "start reading VUX, vux-time:" << gnss_vux_data[tmp_index].gps_tod << ", hesai-time:" << time_of_day_sec << std::endl;

#define second_approach
#ifdef second_approach

                /*CURRENT APPROACH
                keep a buffer of lines and their poses around 25 lines
                when a new line is received with initial guess
                    we drop the last one with its pose
                    we convert the others into the frame of the latest line - to create a small cloud
                    we refine the curent line pose by coregistering the cloud to mls/als
                    then we save the current line with its updated pose to buffer
                    proceed next


                */

                // TODO - modify here - use the kdtree from mls or ALS,  do not recreate it from scratch

                // bool use_mls_ref = true;
                // if (use_mls_ref)
                //{
                std::cout << "kdtree set input MLS points: " << laserCloudSurfMap->size() << std::endl;
                kdtree->setInputCloud(laserCloudSurfMap); // take this from mls
                const auto &refference_kdtree = kdtree;
                const auto &reference_localMap_cloud = laserCloudSurfMap;
                // }
                // else
                // {
                // std::cout << "kdtree set input ALS points: " << als_obj->als_cloud->size() << std::endl;
                // const auto &refference_kdtree = estimator_.localKdTree_map; // we can re-use this, no need to recreate it
                // const auto &reference_localMap_cloud = als_obj->als_cloud;
                // }

                
                if (true)
                {
                    std::cout << "Start normal estimation" << std::endl;

                    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
                    tree->setInputCloud(laserCloudSurfMap);

                    // Compute normals with given k neighbors
                    auto computeNormals = [&](pcl::PointCloud<PointType>::Ptr cloud, int k, double r)
                    {
                        // pcl::NormalEstimation<PointType, pcl::Normal> normal_estimator;
                        pcl::NormalEstimationOMP<PointType, pcl::Normal> normal_estimator;
                        normal_estimator.setNumberOfThreads(4);
                        normal_estimator.setInputCloud(cloud);
                        normal_estimator.setSearchMethod(tree);
                        //normal_estimator.setKSearch(k);
                        normal_estimator.setRadiusSearch(r);   //1m ;

                        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
                        normal_estimator.compute(*normals);
                        return normals;
                    };

                    // Constants
                    double DEGREE_THRESHOLD = 5.0;
                    double RAD_TO_DEG = 180.0 / M_PI;

                    // Compute normals with 10 and 5 neighbors
                    auto normals_10 = computeNormals(reference_localMap_cloud, 10, 2.);
                    auto normals_5 = computeNormals(reference_localMap_cloud, 5, 1.);

                    // Validate stability based on angle
                    pcl::PointCloud<pcl::Normal>::Ptr stable_normals(new pcl::PointCloud<pcl::Normal>);
                    featsFromMap->clear();

                    for (size_t i = 0; i < normals_10->size(); i++)
                    {
                        const auto &n10 = normals_10->at(i);
                        const auto &n5 = normals_5->at(i);

                        auto dot_product = std::max(-1.0f, std::min(1.0f, n10.normal_x * n5.normal_x + n10.normal_y * n5.normal_y + n10.normal_z * n5.normal_z));
                        auto angle = std::acos(dot_product) * RAD_TO_DEG;

                        if (angle <= DEGREE_THRESHOLD && n10.curvature < .1) 
                        {
                            stable_normals->push_back(n5); // Keep the normal with 10 neighbors

                            PointType p = reference_localMap_cloud->points[i];

                            p.intensity = angle;
                            p.time = n10.curvature;

                            featsFromMap->push_back(p);
                        }
                    }

                    publishJustPoints(featsFromMap, cloud_pub);
                    publishPointCloudWithNormals(featsFromMap, stable_normals, normals_pub);
                }


                if (false)
                {
                    if (!hash_map_built)
                    {
                        landmarks_map.Build(reference_localMap_cloud);
                        hash_map_built = true;
                    }

                    int landmarks_map_size = landmarks_map.Pointcloud(featsFromMap);
                    std::cout << "landmarks_map_size:" << landmarks_map_size << std::endl;
                    publishJustPoints(featsFromMap, cloud_pub);

                    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
                    pcl::PointCloud<PointType>::Ptr normals_origin(new pcl::PointCloud<PointType>);

                    landmarks_map_size = landmarks_map.Pointcloud_and_Normals(normals_origin, normals);
                    std::cout << "landmarks_map_size:" << landmarks_map_size << std::endl;

                    publishPointCloudWithNormals(normals_origin, normals, normals_pub);
                }
                ros::spinOnce();
                rate.sleep();

                continue;
                // std::cout << "KD-tree: " << refference_kdtree->getInputCloud()->size() << ",  map cloud size : " << reference_localMap_cloud->size() << std::endl;

                while (gnss_vux_data[tmp_index].gps_tod <= time_of_day_sec && tmp_index < gnss_vux_data.size())
                {
                    ros::spinOnce();
                    if (flg_exit || !ros::ok())
                        break;

                    double diff = fabs(gnss_vux_data[tmp_index].gps_tod - time_of_day_sec);
                    std::cout << " vux-gnss:" << gnss_vux_data[tmp_index].gps_tod << ", time_of_day_sec:" << time_of_day_sec << ", diff:" << diff << std::endl;

                    const double &msg_time = gnss_vux_data[tmp_index].gps_tod;

                    // rotate to ENU but shifted to zero
                    // Sophus::SE3 p_vux_local = Sophus::SE3(gnss_vux_data[tmp_index].se3.so3(), gnss_vux_data[tmp_index].se3.translation() - first_vux_pose.translation());
                    // SHIFT TO MLS using the init guess
                    // p_vux_local = T_vux2mls * p_vux_local;

                    Sophus::SE3 p_vux_local = als2mls * gnss_vux_data[tmp_index].se3;

                    publish_ppk_gnss(p_vux_local, msg_time);
                    publishAccelerationArrow(marker_pub, -gnss_vux_data[tmp_index].acc, msg_time);

                    if (false) // vux imu
                    {
                        if (!vux_imu_init)
                        {
                            cov init_P = vux_imu_estimator_.get_P();
                            init_P.block(Re_ID, Re_ID, 3, 3) = Eye3d * 0.00001;
                            init_P.block(Te_ID, Te_ID, 3, 3) = Eye3d * 0.00001;
                            init_P.block(BG_ID, BG_ID, 3, 3) = Eye3d * 0.0001;
                            init_P.block(BA_ID, BA_ID, 3, 3) = Eye3d * 0.001;
                            init_P.block(G_ID, G_ID, 3, 3) = Eye3d * 0.00001;
                            vux_imu_estimator_.set_P(init_P);

                            auto tmp = als2mls * gnss_vux_data[tmp_index].se3;
                            // auto tmp = gnss_vux_data[tmp_index].se3;

                            vux_imu_state = vux_imu_estimator_.get_x();
                            vux_imu_state.pos = tmp.translation(); // get the initial position
                            // vux_imu_state.rot = tmp.so3();              //get the initial orientation

                            vux_imu_state.grav = V3D(0, 0, 0); // no gravity
                            vux_imu_estimator_.set_x(vux_imu_state);

                            vux_imu_init = true;
                        }
                        else
                        {
                            // V3D acc_no_gravity, acc, gyro, velocity;
                            input in;                                         // IMU frame - X-forward, Y-left, Z-up
                            in.acc = gnss_vux_data[tmp_index].acc_no_gravity; // IMU frame
                            in.gyro = gnss_vux_data[tmp_index].gyro;          // IMU frame

                            // convert to MLS frame: X-right, Y-forward, Z-up
                            //  in.acc = Rz * in.acc;
                            //  in.gyro = Rz * in.gyro;

                            auto dt = gnss_vux_data[tmp_index].gps_tod - gnss_vux_data[tmp_index - 1].gps_tod;
                            vux_imu_state = vux_imu_estimator_.propagete_NO_gravity(dt, in);
                            // Sophus::SE3 predictedPose = Sophus::SE3(vux_imu_state.rot, vux_imu_state.pos);

                            vux_imu_state = vux_imu_estimator_.get_x();
                            Sophus::SE3 updatedPose = Sophus::SE3(vux_imu_state.rot, vux_imu_state.pos);
                            publish_raw_vux_imu_gnss(updatedPose, msg_time);
                        }
                    }

                    tmp_index++;
                    rate.sleep();

                    // continue; //---------------------------------------------------------

                    while (readVUX.next(next_line))
                    {
                        if (next_line->empty())
                            break;

                        const auto &cloud_time = next_line->points[0].time;
                        // from the future w.r.t. curr gnss time
                        if (cloud_time > gnss_vux_data[tmp_index].gps_tod)
                            break;

                        some_index++;
                        if (time_aligned && some_index % 2 == 0) // 10
                        {
                            auto interpolated_pose = interpolateSE3(gnss_vux_data, cloud_time, tmp_index, false); // interpolated from ppk-gnss
                            Sophus::SE3 pose_local = Sophus::SE3(interpolated_pose.so3(), interpolated_pose.translation() - first_vux_pose.translation());
                            pose_local = als2mls * interpolated_pose;

                            pcl::PointCloud<VUX_PointType>::Ptr downsampled_line(new pcl::PointCloud<VUX_PointType>);
                            downSizeFilter_vux.setInputCloud(next_line);
                            downSizeFilter_vux.filter(*downsampled_line);

                            // Sophus::SE3 interpolated_pose_mls = interpolateSE3(prev_mls, prev_mls_time, curr_mls, curr_mls_time, cloud_time);
                            // Sophus::SE3 pose4georeference = interpolated_pose_mls;
                            //  pose4georeference = pose_local;

                            // first extrinsic, then georeference, then transform to als
                            Sophus::SE3 pose4georeference = als2mls * interpolated_pose * Sophus::SE3(R_vux2imu, t_vux2imu);

                            pcl::PointCloud<VUX_PointType>::Ptr transformed_cloud(new pcl::PointCloud<VUX_PointType>);
                            *transformed_cloud = *downsampled_line;

                            Sophus::SE3 T_to_be_refined = pose4georeference;
                            if (false) // prev approach
                            {
                                if (lines_buffer.size() > 25) // or difference between the first and the last line pose is big enough
                                {
                                    // drop old scan and pose
                                    lines_buffer.pop_front();
                                    line_poses_buffer.pop_front();

                                    // we have enough reference local map
                                    feats_undistort->clear(); // for debug

                                    // use the current cloud init guess as reference - put the ref map in the frame of the latest line
                                    Sophus::SE3 ref_inv = T_to_be_refined.inverse(); // Compute T_ref^{-1}
                                    pcl::PointCloud<PointType>::Ptr past_lines_in_latest_line_frame(new pcl::PointCloud<PointType>);
                                    for (int l = 0; l < lines_buffer.size(); l++)
                                    {
                                        const auto &line_cloud = lines_buffer[l];
                                        Sophus::SE3 T_line = line_poses_buffer[l];
                                        T_line = ref_inv * T_line;                       // relative to curr pose
                                        for (const auto &raw_point : line_cloud->points) // for each scanner raw point
                                        {
                                            V3D p_src(raw_point.x, raw_point.y, raw_point.z);
                                            V3D p_transformed = T_line * p_src;

                                            PointType search_point;
                                            search_point.x = p_transformed.x();
                                            search_point.y = p_transformed.y();
                                            search_point.z = p_transformed.z();

                                            // search_point.intensity = 100; // l + vux_scan_id;

                                            past_lines_in_latest_line_frame->push_back(search_point);
                                        }
                                    }
                                    auto T_line = ref_inv * T_to_be_refined;               // this should be the identity
                                    for (const auto &raw_point : downsampled_line->points) // for each scanner raw point
                                    {
                                        V3D p_src(raw_point.x, raw_point.y, raw_point.z);
                                        V3D p_transformed = T_line * p_src;

                                        PointType search_point;
                                        search_point.x = p_transformed.x();
                                        search_point.y = p_transformed.y();
                                        search_point.z = p_transformed.z();

                                        // search_point.intensity = 0;

                                        past_lines_in_latest_line_frame->push_back(search_point);
                                    }

                                    bool debug_msg = true;
                                    // if (true) // registration is going to happen here
                                    {
                                        Eigen::Quaterniond q_init(T_to_be_refined.so3().matrix());
                                        V3D t_init = T_to_be_refined.translation();
                                        double prev_cost = std::numeric_limits<double>::max(); // Initialize with a large value
                                        double current_cost = prev_cost;
                                        double cost_threshold = .01; // Threshold for stopping criterion

                                        if (debug_msg)
                                            std::cout << "Start registration..." << std::endl;

                                        for (int iter_num = 0; iter_num < 5; iter_num++)
                                        {
                                            if (flg_exit || !ros::ok())
                                                break;

                                            feats_undistort->clear();
                                            ceres::Problem problem;
                                            ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                                            // Ensure the quaternion stays valid during optimization
                                            ceres::LocalParameterization *q_parameterization =
                                                new ceres::EigenQuaternionParameterization();
                                            int points_used_for_registration = 0;

                                            double q_param[4] = {q_init.x(), q_init.y(), q_init.z(), q_init.w()};
                                            double t_param[3] = {t_init.x(), t_init.y(), t_init.z()};

                                            // Add the quaternion parameter block with the local parameterization
                                            // to Ensure the quaternion stays valid during optimization
                                            problem.AddParameterBlock(q_param, 4, q_parameterization);
                                            problem.AddParameterBlock(t_param, 3); // Add the translation parameter block

                                            int total_points = past_lines_in_latest_line_frame->size();

                                            for (const auto &raw_point : past_lines_in_latest_line_frame->points)
                                            {
                                                V3D p_src(raw_point.x, raw_point.y, raw_point.z);
                                                // transform to global with best T for NN search
                                                V3D p_transformed = T_to_be_refined * p_src;

                                                // Nearest neighbor search
                                                PointType search_point;
                                                search_point.x = p_transformed.x();
                                                search_point.y = p_transformed.y();
                                                search_point.z = p_transformed.z();

                                                feats_undistort->push_back(search_point);

                                                bool p2plane = true;
                                                if (p2plane)
                                                {
                                                    std::vector<int> point_idx(5);
                                                    std::vector<float> point_dist(5);
                                                    if (refference_kdtree->nearestKSearch(search_point, 5, point_idx, point_dist) > 0) // there are neighbours
                                                    {
                                                        if (point_dist[4] < threshold_nn) // not too far
                                                        {
                                                            Eigen::Matrix<double, 5, 3> matA0;
                                                            Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                                                            for (int j = 0; j < 5; j++)
                                                            {
                                                                matA0(j, 0) = reference_localMap_cloud->points[point_idx[j]].x;
                                                                matA0(j, 1) = reference_localMap_cloud->points[point_idx[j]].y;
                                                                matA0(j, 2) = reference_localMap_cloud->points[point_idx[j]].z;
                                                            }

                                                            // find the norm of plane
                                                            V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                                                            double negative_OA_dot_norm = 1 / norm.norm();
                                                            norm.normalize();

                                                            bool planeValid = true;
                                                            for (int j = 0; j < 5; j++)
                                                            {
                                                                if (fabs(norm(0) * reference_localMap_cloud->points[point_idx[j]].x +
                                                                         norm(1) * reference_localMap_cloud->points[point_idx[j]].y +
                                                                         norm(2) * reference_localMap_cloud->points[point_idx[j]].z + negative_OA_dot_norm) > 0.1)
                                                                {
                                                                    planeValid = false;
                                                                    break;
                                                                }
                                                            }

                                                            if (planeValid)
                                                            {
                                                                ceres::CostFunction *cost_function = VuxPlanes::Create(p_src, norm, negative_OA_dot_norm);
                                                                problem.AddResidualBlock(cost_function, loss_function, q_param, t_param);
                                                                points_used_for_registration++;
                                                            }
                                                        }
                                                    }
                                                }
                                            }

                                            // publish_frame_debug(pubLaserCloudDebug, feats_undistort);
                                            // ros::spinOnce();
                                            // rate.sleep();

                                            // Solve the problem
                                            ceres::Solver::Options options;
                                            options.linear_solver_type = ceres::DENSE_QR; // options.max_num_iterations = 100;
                                            options.minimizer_progress_to_stdout = false; // true; avoid much text interminal
                                            ceres::Solver::Summary summary;
                                            ceres::Solve(options, &problem, &summary);

                                            // std::cout << summary.FullReport() << std::endl;
                                            // std::cout << summary.BriefReport() << std::endl;

                                            // Output the refined extrinsic transformation
                                            Eigen::Quaterniond refined_q(q_param[3], q_param[0], q_param[1], q_param[2]);
                                            q_init = refined_q;
                                            t_init = V3D(t_param[0], t_param[1], t_param[2]);
                                            if (debug_msg)
                                                std::cout << "Registration done with " << points_used_for_registration << "/" << total_points << " points" << std::endl;
                                            T_to_be_refined = Sophus::SE3(refined_q, t_init);

                                            current_cost = summary.final_cost;
                                            if (debug_msg)
                                                std::cout << "Iteration " << iter_num << " - Cost: " << current_cost << " \n\n"
                                                          << std::endl;

                                            if (std::abs(prev_cost - current_cost) < cost_threshold) // Check if the cost function change is small enough to stop
                                            {
                                                std::cout << "Stopping optimization: Cost change below threshold.\n";
                                                std::cout << "Init   :" << pose4georeference.log().transpose() << std::endl;
                                                std::cout << "Refined:" << T_to_be_refined.log().transpose() << std::endl;
                                                break;
                                            }

                                            prev_cost = current_cost;
                                        }
                                    }

                                    lines_buffer.push_back(downsampled_line);
                                    line_poses_buffer.push_back(T_to_be_refined); // this one here should be refined

                                    publish_refined_ppk_gnss(T_to_be_refined, cloud_time);

                                    publish_frame_debug(pubLaserCloudDebug, feats_undistort);

                                    vux_scan_id++;
                                }
                                else
                                {
                                    // saving the data - the initial map will be based on the init guess
                                    lines_buffer.push_back(downsampled_line);
                                    line_poses_buffer.push_back(T_to_be_refined);
                                }
                                std::cout << "lines_buffer:" << lines_buffer.size() << std::endl;
                            }

                            if (true)
                            {
                                // make it more optimal - maybe use only Eigen points
                                // One scan is warking
                                // try several scans
                                if (false) // only 1 scan registration for debug reasons
                                {
                                    auto source = downsampled_line;       // copy of the current scan
                                    auto initial_guess = T_to_be_refined; // copy of the init guess
                                    Sophus::SE3 T_icp = Sophus::SE3();
                                    for (int i = 0; i < source->size(); i++) // apply init guess T
                                    {
                                        V3D p_src(source->points[i].x, source->points[i].y, source->points[i].z);
                                        V3D p_transformed = initial_guess * p_src;
                                        source->points[i].x = p_transformed.x();
                                        source->points[i].y = p_transformed.y();
                                        source->points[i].z = p_transformed.z();
                                    }

                                    int max_iterations = 100;
                                    for (int iter_num = 0; iter_num < max_iterations; iter_num++)
                                    {
                                        std::cout << "\nIteration " << iter_num << std::endl;
                                        if (flg_exit || !ros::ok())
                                            break;

                                        // global variables
                                        Eigen::MatrixXd H;
                                        Eigen::VectorXd g;
                                        int num_scans = 1;
                                        H = Eigen::MatrixXd::Zero(6 * num_scans, 6 * num_scans);
                                        g = Eigen::VectorXd::Zero(6 * num_scans);

                                        // can be done in parallel
                                        // for every point search tgt via NN - compute the residual and hesians

                                        Eigen::Matrix6d JTJ_private; // state_size x state_size  (6x6)
                                        Eigen::Vector6d JTr_private; // state_size x 1           (6x1)

                                        feats_undistort->clear();
                                        for (int i = 0; i < source->size(); i++) // apply init guess T
                                        {
                                            V3D p_transformed(source->points[i].x, source->points[i].y, source->points[i].z);
                                            PointType search_point;
                                            search_point.x = p_transformed.x();
                                            search_point.y = p_transformed.y();
                                            search_point.z = p_transformed.z();
                                            search_point.time = 1;
                                            feats_undistort->push_back(search_point); // this cannot be done in parallel - or make it without push
                                        }
                                        publish_frame_debug(pubLaserCloudDebug, feats_undistort);

                                        int nn_found = computeScanDerivatives(source, reference_localMap_cloud, refference_kdtree,
                                                                              threshold_nn, JTJ_private, JTr_private);

                                        if (nn_found > 0)
                                        {
                                            // Insert into global Hessian
                                            H.block<6, 6>(0, 0) = JTJ_private;
                                            g.segment<6>(0) = JTr_private;

                                            // Solve the system H * deltaT = -g
                                            Eigen::VectorXd deltaT = H.ldlt().solve(-g); // 6*num_scans   X   1
                                            Eigen::VectorXd dx = deltaT.segment<6>(0);   // Eigen::Vector6d

                                            // const Eigen::Vector6d dx = JTJ_private.ldlt().solve(-JTr_private); // translation and rotation perturbations

                                            const Sophus::SE3 estimation = Sophus::SE3::exp(dx); // this is in local-align init guess to map

                                            T_icp = estimation * T_icp;

                                            std::cout << "dx.norm():" << dx.norm() << std::endl;
                                            if (dx.norm() < .001 || iter_num >= max_iterations - 2)
                                            {
                                                std::cout << "Points registered with src:" << source->size() << " in " << iter_num << " iterations " << std::endl;
                                                break;
                                            }

                                            for (int i = 0; i < source->size(); i++) // apply init guess T
                                            {
                                                V3D p_src(source->points[i].x, source->points[i].y, source->points[i].z);
                                                V3D p_transformed = estimation * p_src;
                                                source->points[i].x = p_transformed.x();
                                                source->points[i].y = p_transformed.y();
                                                source->points[i].z = p_transformed.z();
                                            }
                                        }
                                        else
                                        {
                                            break;
                                        }
                                    }

                                    T_to_be_refined = T_icp * T_to_be_refined;
                                }

                                // co-register every line independently
                                if (lines_buffer.size() > 50) // or difference between the first and the last line pose is big enough
                                {
                                    // drop old scan and pose
                                    lines_buffer.pop_front();
                                    line_poses_buffer.pop_front();

                                    auto original_T_to_be_refined = T_to_be_refined;

                                    lines_buffer.push_back(downsampled_line);
                                    line_poses_buffer.push_back(original_T_to_be_refined);

                                    int max_iterations = 100;
                                    // do work here
                                    if (false)
                                    {

                                        double kernel = 2.0;
                                        auto Weight = [&](double residual2)
                                        {
                                            return square(kernel) / square(kernel + residual2);
                                        };

                                        // std::deque<pcl::PointCloud<VUX_PointType>::Ptr> lines_buffer;
                                        std::vector<std::vector<V3D>> eigen_lines;
                                        std::deque<Sophus::SE3> line_poses_buffer_copy;

                                        Sophus::SE3 T_icp = Sophus::SE3();

                                        // apply init guess T for every line in a buffer
                                        for (int l = 0; l < lines_buffer.size(); l++)
                                        {
                                            auto initial_guess = line_poses_buffer[l]; // copy of the init guess
                                            std::vector<V3D> curr_line;
                                            for (int i = 0; i < lines_buffer[l]->size(); i++)
                                            {
                                                V3D p_src(lines_buffer[l]->points[i].x, lines_buffer[l]->points[i].y, lines_buffer[l]->points[i].z);
                                                V3D p_transformed = initial_guess * p_src;
                                                curr_line.push_back(p_transformed);
                                            }

                                            eigen_lines.push_back(curr_line);

                                            Sophus::SE3 T_icp = Sophus::SE3();
                                            line_poses_buffer_copy.push_back(T_icp);
                                        }

                                        if (true)
                                        {
                                            for (int iter_num = 0; iter_num < max_iterations; iter_num++)
                                            {
                                                if (flg_exit || !ros::ok())
                                                    break;

                                                feats_undistort->clear();

                                                // global variables
                                                Eigen::MatrixXd H;
                                                Eigen::VectorXd g;

                                                int num_scans = eigen_lines.size();
                                                H = Eigen::MatrixXd::Zero(6 * num_scans, 6 * num_scans);
                                                g = Eigen::VectorXd::Zero(6 * num_scans);
                                                int total_points_found = 0;
                                                double total_error = 0.0;
                                                // for every line/scan
                                                for (int l = 0; l < eigen_lines.size(); l++)
                                                {
                                                    Eigen::Matrix6d JTJ_private; // state_size x state_size  (6x6)
                                                    Eigen::Vector6d JTr_private; // state_size x 1           (6x1)
                                                    JTJ_private.setZero();
                                                    JTr_private.setZero();

                                                    int nn_found = 0;
                                                    for (const auto &raw_point : eigen_lines[l])
                                                    {
                                                        PointType search_point; // I think this requires to be transformed to global - modify it
                                                        search_point.x = raw_point.x();
                                                        search_point.y = raw_point.y();
                                                        search_point.z = raw_point.z();
                                                        search_point.time = l;

                                                        feats_undistort->push_back(search_point); // this cannot be done in parallel - or make it without push

                                                        bool p2plane = false;
                                                        if (p2plane)
                                                        {
                                                            // to be implemented
                                                        }
                                                        else
                                                        {
                                                            std::vector<int> point_idx(1);
                                                            std::vector<float> point_dist(1);
                                                            if (refference_kdtree->nearestKSearch(search_point, 1, point_idx, point_dist) > 0) // there are neighbours
                                                            {
                                                                if (point_dist[0] < threshold_nn) // 1
                                                                {
                                                                    V3D src = raw_point; //(raw_point.x, raw_point.y, raw_point.z);

                                                                    V3D tgt(reference_localMap_cloud->points[point_idx[0]].x, reference_localMap_cloud->points[point_idx[0]].y, reference_localMap_cloud->points[point_idx[0]].z);

                                                                    const V3D residual = src - tgt;
                                                                    Eigen::Matrix3_6d J_r;
                                                                    J_r.block<3, 3>(0, 0) = Eye3d;                        // df/dt
                                                                    J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(src); // df/dR

                                                                    // double w = 1.; // to be handled later
                                                                    double w = Weight(residual.squaredNorm());

                                                                    JTJ_private.noalias() += J_r.transpose() * w * J_r;
                                                                    JTr_private.noalias() += J_r.transpose() * w * residual;

                                                                    nn_found++;
                                                                    total_error += residual.norm();
                                                                }
                                                            }
                                                        }
                                                    }
                                                    // std::cout << "Found " << nn_found << "/" << lines_buffer_global[l]->size() << " neighbours" << std::endl;
                                                    total_points_found += nn_found;
                                                    if (nn_found > 0)
                                                    {
                                                        // Insert into global Hessian
                                                        H.block<6, 6>(l * 6, l * 6) = JTJ_private;
                                                        g.segment<6>(l * 6) = JTr_private;
                                                    }
                                                    else
                                                    {
                                                        std::cout << "didn;t find any points" << std::endl;
                                                    }
                                                }

                                                std::cout << "\n Iteration " << iter_num << " total_error:" << total_error << std::endl;
                                                std::cout << "num_scans:" << num_scans << ", total_points_found:" << total_points_found << std::endl;
                                                // const Eigen::Vector6d dx = JTJ_private.ldlt().solve(-JTr_private); // translation and rotation perturbations

                                                // Eigen::VectorXd deltaT = -H.ldlt().solve(g);
                                                // Solve the system H * deltaT = -g
                                                Eigen::VectorXd deltaT = H.ldlt().solve(-g); // 6*num_scans   X   1

                                                // std::cout << "deltaT Number of rows: " << deltaT.rows() << std::endl;
                                                // std::cout << "deltaT Number of columns: " << deltaT.cols() << std::endl;

                                                // apply the correction
                                                for (size_t l = 0; l < num_scans; l++)
                                                {
                                                    Eigen::VectorXd dTi = deltaT.segment<6>(l * 6);       // Eigen::Vector6d
                                                    const Sophus::SE3 estimation = Sophus::SE3::exp(dTi); // this is in local-align init guess to map

                                                    // T_icp = estimation * T_icp;
                                                    T_icp = estimation * T_icp;

                                                    line_poses_buffer_copy[l] = estimation * line_poses_buffer_copy[l];

                                                    // SE3 delta = SE3::exp(dTi);
                                                    // scans[i].initial_guess = delta * scans[i].initial_guess;
                                                }

                                                // publish_frame_debug(pubLaserCloudDebug, feats_undistort);

                                                double update_norm = deltaT.norm();

                                                double max_update = 0.0;
                                                for (int l = 0; l < num_scans; l++)
                                                {
                                                    double scan_update = deltaT.segment<6>(l * 6).norm();
                                                    max_update = std::max(max_update, scan_update);
                                                }
                                                // if (max_update < 0.001) break;

                                                if (max_update < .01 || iter_num >= max_iterations - 1)
                                                {
                                                    std::cout << "Points registered with src:" << feats_undistort->size() << " in " << iter_num << " iterations " << std::endl;
                                                    break;
                                                }

                                                // apply the corrections
                                                for (int l = 0; l < num_scans; l++)
                                                {
                                                    Eigen::VectorXd dTi = deltaT.segment<6>(l * 6); // Eigen::Vector6d
                                                    Sophus::SE3 estimation = Sophus::SE3::exp(dTi);

                                                    for (int i = 0; i < eigen_lines[l].size(); i++)
                                                    {
                                                        V3D p_src = eigen_lines[l][i];
                                                        // V3D p_transformed = estimation * initial_guess * p_src;
                                                        eigen_lines[l][i] = estimation * p_src;
                                                    }
                                                }
                                                std::cout << "line_poses_buffer_copy:" << line_poses_buffer_copy.size() << std::endl;

                                                // T_to_be_refined = line_poses_buffer_copy[num_scans-1] * T_to_be_refined;
                                                // T_to_be_refined = T_icp * T_to_be_refined;
                                            }

                                            publish_frame_debug(pubLaserCloudDebug, feats_undistort);
                                        }
                                    }
                                    // auto original_T_to_be_refined = T_to_be_refined;

                                    // lines_buffer.push_back(downsampled_line);
                                    // line_poses_buffer.push_back(original_T_to_be_refined); // this one here should be refined

                                    //     // publish_refined_ppk_gnss(T_to_be_refined, cloud_time);

                                    // ceres approach
                                    if (true)
                                    {
                                        max_iterations = 5; // 10;
                                        threshold_nn = 1;   // 3 * 3; // use 3 m away points

                                        std::vector<std::vector<V3D>> eigen_lines;

                                        bool local_error = false;
                                        bool p2p = true;

                                        // apply init guess T for every line in a buffer
                                        for (int l = 0; l < lines_buffer.size(); l++)
                                        {
                                            auto initial_guess = line_poses_buffer[l]; // copy of the init guess
                                            std::vector<V3D> curr_line;
                                            for (int i = 0; i < lines_buffer[l]->size(); i++)
                                            {
                                                V3D p_src(lines_buffer[l]->points[i].x, lines_buffer[l]->points[i].y, lines_buffer[l]->points[i].z);
                                                V3D p_transformed = initial_guess * p_src;

                                                if (local_error)
                                                    curr_line.push_back(p_transformed); // point transformed to globa via init guess
                                                else
                                                    curr_line.push_back(p_src); // point local
                                            }

                                            eigen_lines.push_back(curr_line); // point is transformed to global already
                                        }

                                        // Initial Poses: Quaternion + Translation per scan
                                        // Create q_params and t_params from SE3 poses
                                        std::vector<Eigen::Quaterniond> q_params;
                                        std::vector<V3D> t_params;
                                        for (const auto &pose : line_poses_buffer)
                                        {
                                            if (local_error)
                                            {
                                                // this will refine locally
                                                q_params.push_back(Eigen::Quaterniond::Identity());
                                                t_params.push_back(V3D::Zero());
                                            }
                                            else
                                            { // this will refine global values
                                                q_params.push_back(pose.unit_quaternion());
                                                t_params.push_back(pose.translation());
                                            }
                                        }

                                        double prev_cost = std::numeric_limits<double>::max(); // Initialize with a large value
                                        double current_cost = prev_cost;
                                        double cost_threshold = .01; // Threshold for stopping criterion

                                        for (int iter_num = 0; iter_num < max_iterations; iter_num++)
                                        {
                                            if (flg_exit || !ros::ok())
                                                break;

                                            feats_undistort->clear();

                                            // current_cost = joint_registration(eigen_lines, q_params, t_params, refference_kdtree, reference_localMap_cloud, threshold_nn, p2p, local_error);

                                            std::vector<valid_tgt> landmarks;
                                            current_cost = BA(eigen_lines, q_params, t_params, refference_kdtree, reference_localMap_cloud, landmarks, threshold_nn, p2p, local_error);

                                            cloud_with_normals->clear();
                                            for (const auto &land : landmarks)
                                            {
                                                pcl::PointNormal pt;
                                                pt.x = reference_localMap_cloud->points[land.map_point_index].x;
                                                pt.y = reference_localMap_cloud->points[land.map_point_index].y;
                                                pt.z = reference_localMap_cloud->points[land.map_point_index].z;

                                                pt.normal_x = land.norm.x();
                                                pt.normal_y = land.norm.y();
                                                pt.normal_z = land.norm.z();

                                                pt.curvature = land.seen;
                                                // pt.curvature = land.line_idx.size();

                                                cloud_with_normals->push_back(pt);

                                                if (land.seen > 1) // seen more than once
                                                {
                                                    for (int i = 0; i < land.seen; i++)
                                                    {
                                                        const auto &l = land.line_idx[i];     // point from line l
                                                        const auto &p_idx = land.scan_idx[i]; // at index p_idx

                                                        Eigen::Quaterniond &q = q_params[l];
                                                        Eigen::Vector3d &t = t_params[l];

                                                        const auto &raw_point = eigen_lines[l][p_idx];
                                                        V3D p_transformed = q * raw_point + t;

                                                        // Nearest neighbor search
                                                        PointType search_point;
                                                        search_point.x = p_transformed.x();
                                                        search_point.y = p_transformed.y();
                                                        search_point.z = p_transformed.z();

                                                        search_point.time = l;

                                                        feats_undistort->push_back(search_point);
                                                    }
                                                }
                                            }

                                            // publishPointCloudWithNormals(cloud_with_normals, cloud_pub, normals_pub);
                                            // publish_frame_debug(pubLaserCloudDebug, feats_undistort);

                                            // break;

                                            std::cout << "Iteration " << iter_num << " - Cost: " << current_cost << " \n\n"
                                                      << std::endl;
                                            if (std::abs(prev_cost - current_cost) < cost_threshold) // Check if the cost function change is small enough to stop
                                            {
                                                std::cout << "Stopping optimization: Cost change below threshold.\n";
                                                break;
                                            }

                                            prev_cost = current_cost;
                                        }

                                        publish_frame_debug(pubLaserCloudDebug, feats_undistort);
                                        ros::spinOnce();
                                    }

                                    vux_scan_id++;
                                }
                                else
                                {
                                    // saving the data - the initial map will be based on the init guess
                                    lines_buffer.push_back(downsampled_line);
                                    line_poses_buffer.push_back(T_to_be_refined);
                                }
                                std::cout << "lines_buffer:" << lines_buffer.size() << std::endl;
                            }

                            for (size_t i = 0; i < transformed_cloud->size(); i++)
                            {
                                V3D point_scanner(transformed_cloud->points[i].x, transformed_cloud->points[i].y, transformed_cloud->points[i].z);

                                V3D point_global;
                                if (true)
                                {
                                    // with our refinement
                                    point_global = T_to_be_refined * point_scanner;
                                }

                                if (false) // georeference with ppk gnss-imu
                                {
                                    // transform to vux gnss-imu  and georeference with ppk data
                                    point_global = R_vux2imu * point_scanner + t_vux2imu;
                                    point_global = pose_local * point_global;

                                    // transform to mls
                                    // Sophus::SE3 T_vux2mls = Sophus::SE3(R_als2mls, V3D(0,0,0));
                                    // point_global = T_vux2mls * point_global;
                                }

                                if (false) // georeference with mls and extrinsic init guess
                                {
                                    // point_global = R_vux2mls * point_scanner; // now in mls frame
                                    // point_global += t_vux2mls;

                                    // transform to mls
                                    // point_global = R_refined * point_scanner + t_refined;

                                    // georeference with mls
                                    // point_global = pose4georeference * point_global;
                                }

                                transformed_cloud->points[i].x = point_global.x();
                                transformed_cloud->points[i].y = point_global.y();
                                transformed_cloud->points[i].z = point_global.z();
                            }

                            publishPointCloud_vux(transformed_cloud, point_cloud_pub);
                        }

                        // ppk gnss = 10Hz same as hesai MLS
                        // imu also 10 Hz

                        V3D lla;
                        uint32_t raw_gnss_tod;
                        bool got_gnss_measurement = false;
                        V3D measurement_in_mls;
                        if (false && readVUX.nextGNSS(lla, raw_gnss_tod)) // get the raw GNSS into lla - if there is raw gnss
                        {
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

                            measurement_in_mls = als2mls.so3() * carthesian; // only rotation
                            measurement_in_mls = als2mls * curr_enu;

                            publish_raw_gnss(measurement_in_mls, raw_gnss_tod);
                            got_gnss_measurement = true;

                            std::cout << "IMU time:" << gnss_vux_data[tmp_index].gps_tod << ", raw gnss time:" << raw_gnss_tod << std::endl;
                            // update here
                            // if (got_gnss_measurement)
                            {
                                // GNSS_VAR (0.05)
                                //  auto gps_cov_ = Eigen::Vector3d(GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR);
                                //  const V3D &measurement_in_mls_ = p_vux_local.translation();
                                //  vux_imu_estimator_.update(measurement_in_mls_, gps_cov_, 10, false);

                                // const V3D &measurement_in_mls = p_vux_local.translation();
                                // if(raw_gnss_tod > gnss_vux_data[tmp_index].gps_tod){
                                auto gps_cov_ = Eigen::Vector3d(10, 10, 10);
                                vux_imu_estimator_.update(measurement_in_mls, gps_cov_, 20, false);
                                //}
                            }
                        }
                    }
                }
#endif
            }
            prev_mls = Sophus::SE3(state_point.rot, state_point.pos);
            prev_mls_time = time_of_day_sec;
        }
    }
    bag.close();
}
