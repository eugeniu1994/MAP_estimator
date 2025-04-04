

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

void publishJustPoints(const pcl::PointCloud<PointType>::Ptr &cloud_,const ros::Publisher &cloud_pub)
{
    // --- 1. Publish Point Cloud ---
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud_, cloud_msg);
    cloud_msg.header.frame_id = "world";
    cloud_pub.publish(cloud_msg);
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

//#include "vux_registration.hpp"
//#include "VoxelHash.hpp"

#include "clean_registration.hpp"

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
    ros::Publisher pubOptimizedVUX = nh.advertise<sensor_msgs::PointCloud2>("/vux_optimized", 10);

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

    vux::VuxAdaptor readVUX(std::cout, 75.);
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

    M3D Rz;
    double angle = M_PI / 2.0; // 90 degrees in radians
    Rz << cos(angle), -sin(angle), 0,
        sin(angle), cos(angle), 0,
        0, 0, 1;

    // THE EXTRINSICS FROM VUX TO VUX-IMU
    M3D R_vux2imu;
    R_vux2imu << -0.0001486877, 0.4998181193, -0.8661303744,
        0.0006285201, 0.8661302597, 0.4998179451,
        0.9999997914, -0.0004700636, -0.0004429287;
    V3D t_vux2imu(-0.5922161686, 0.1854945762, 0.7806042559);

    M3D R_vux2mls; // from vux scanner to mls point cloud
    R_vux2mls << 0.0064031121, -0.8606533346, -0.5091510953,
        -0.2586398121, 0.4904106092, -0.8322276624,
        0.9659526116, 0.1370155907, -0.2194590626;
    V3D t_vux2mls(-0.2238580597, -3.0124498678, -0.8051626709);

    // from vux to mls init guess
    // Sophus::SE3 vux2mls_extrinsics = Sophus::SE3(Rz, V3D::Zero()) * Sophus::SE3(R_vux2imu, t_vux2imu);
    Sophus::SE3 vux2mls_extrinsics = Sophus::SE3(R_vux2mls, t_vux2mls); // refined - vux to mls cloud
    Sophus::SE3 vux2imu_extrinsics = Sophus::SE3(R_vux2imu, t_vux2imu); // refined - vux to back imu

    Sophus::SE3 imu2mls_extrinsics = vux2mls_extrinsics * vux2imu_extrinsics.inverse();
    Sophus::SE3 mls2imu_extrinsics = vux2imu_extrinsics * vux2mls_extrinsics.inverse();

    int scan_id = 0, vux_scan_id = 0;
    Sophus::SE3 first_vux_pose;

    Sophus::SE3 prev_mls, curr_mls, prev_segment_last_pose = Sophus::SE3();
    double prev_mls_time, curr_mls_time;

    std::vector<pcl::PointCloud<VUX_PointType>::Ptr> lines;
    std::vector<Sophus::SE3> line_poses;

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());

    double threshold_nn = 1; // ;

    bool raw_vux_imu_time_aligned = false;
    int segment_id = 0;
    // cylinder buffer for vux
    std::deque<pcl::PointCloud<VUX_PointType>::Ptr> lines_buffer;
    std::deque<Sophus::SE3> line_poses_buffer, refined_line_poses_buffer;

    // pcl::KdTreeFLANN<PointType> kdtree;
    pcl::KdTreeFLANN<PointType>::Ptr kdtree(new pcl::KdTreeFLANN<PointType>());
    pcl::KdTreeFLANN<PointType>::Ptr kdtree_prev_segment(new pcl::KdTreeFLANN<PointType>());
    pcl::PointCloud<PointType>::Ptr prev_segment(new pcl::PointCloud<PointType>);
    bool prev_segment_init = false;

    bool perform_mls_registration = true;
    bool first_segment_aligned = false;

    Sophus::SE3 coarse_delta_T = Sophus::SE3(); // relative correction of the segment
    Sophus::SE3 last_refined_pose;
    for (const rosbag::MessageInstance &m : view)
    {
        scan_id++;

        if (scan_id < 45100) // this is only for the 0 bag
            continue;

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

            // //put it back later
            if (!gnss_obj->GNSS_extrinsic_init)
                continue;

            als2mls = als_obj->als_to_mls;

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
            time_of_day_sec = gnss_obj->tod;

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
                std::cout << "start reading VUX, vux-time:" << gnss_vux_data[tmp_index].gps_tod << ", hesai-time:" << time_of_day_sec << std::endl;
                curr_mls = Sophus::SE3(state_point.rot, state_point.pos);
                curr_mls_time = time_of_day_sec;

                // bool use_mls_ref = true;
                // if (use_mls_ref)
                //{
                // std::cout << "kdtree set input MLS points: " << laserCloudSurfMap->size() << std::endl;
                // kdtree->setInputCloud(laserCloudSurfMap); // take this from mls
                // const auto &refference_kdtree = kdtree;
                // const auto &reference_localMap_cloud = laserCloudSurfMap;
                // }
                // else
                // {
                std::cout << "kdtree set input ALS points: " << als_obj->als_cloud->size() << std::endl;
                const auto &refference_kdtree = estimator_.localKdTree_map; // we can re-use this, no need to recreate it
                const auto &reference_localMap_cloud = als_obj->als_cloud;
                // }

                while (gnss_vux_data[tmp_index].gps_tod <= time_of_day_sec && tmp_index < gnss_vux_data.size())
                {
                    ros::spinOnce();
                    if (flg_exit || !ros::ok())
                        break;

                    double diff = fabs(gnss_vux_data[tmp_index].gps_tod - time_of_day_sec);
                    std::cout << " vux-gnss:" << gnss_vux_data[tmp_index].gps_tod << ", time_of_day_sec:" << time_of_day_sec << ", diff:" << diff << std::endl;

                    const double &msg_time = gnss_vux_data[tmp_index].gps_tod;

                    Sophus::SE3 p_vux_local = als2mls * gnss_vux_data[tmp_index].se3;

                    publish_ppk_gnss(p_vux_local, msg_time);

                    // publishAccelerationArrow(marker_pub, -gnss_vux_data[tmp_index].acc, msg_time);

                    tmp_index++;
                    rate.sleep();

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
                            pcl::PointCloud<VUX_PointType>::Ptr downsampled_line(new pcl::PointCloud<VUX_PointType>);
                            downSizeFilter_vux.setInputCloud(next_line);
                            downSizeFilter_vux.filter(*downsampled_line);

                            pcl::PointCloud<VUX_PointType>::Ptr transformed_cloud(new pcl::PointCloud<VUX_PointType>);
                            *transformed_cloud = *downsampled_line;

                            Sophus::SE3 interpolated_pose_ppk = interpolateSE3(gnss_vux_data, cloud_time, tmp_index, false); // interpolated from ppk-gnss
                            Sophus::SE3 interpolated_pose_mls = interpolateSE3(prev_mls, prev_mls_time, curr_mls, curr_mls_time, cloud_time);

                            // std::cout << "Segment:" << segment_id << " distance:" << (prev_mls.translation() - curr_mls.translation()).norm() << " m" << std::endl;

                            // another way for ppk gnss
                            //  Sophus::SE3 pose_local = Sophus::SE3(interpolated_pose_ppk.so3(), interpolated_pose_ppk.translation() - first_vux_pose.translation());
                            //  pose_local = als2mls * interpolated_pose_ppk;

                            // PPK/GNSS pose as init guess---- first extrinsic, then georeference, then transform to als - //ppk-gnss
                            Sophus::SE3 pose4georeference = als2mls * interpolated_pose_ppk * vux2imu_extrinsics; // this does not have the extrinsics for mls

                            //the above has an issue with sparse ALS data
                            //its about the hight - not the same - so shoft everything in the origin of first pose 

                            // MLS pose as init guess ---- first extrinsics, then georeference  // mls pose
                            pose4georeference = interpolated_pose_mls * vux2mls_extrinsics;

                            publish_refined_ppk_gnss(pose4georeference, cloud_time);


                            Sophus::SE3 T_to_be_refined = pose4georeference; // ppk-gnss

                            if (false) // find the extrinsics vux 2 mls
                            {
                                /*
                                find the extrinsics vux2mls_extrinsics
                                get the current lines
                                define cost function that keeps interpolated_pose_mls fixed
                                refine w.r.t. vux2mls_extrinsics
                                */

                                lines_buffer.push_back(downsampled_line);
                                line_poses_buffer.push_back(interpolated_pose_mls); /// save here the fixed pose

                                std::cout << "lines_buffer:" << lines_buffer.size() << std::endl;
                                if (lines_buffer.size() > 1100) // eonough lines
                                {
                                    // Initial guess for extrinsic transformation (Scanner -> IMU)
                                    Eigen::Quaterniond q_extrinsic(vux2mls_extrinsics.so3().matrix());
                                    V3D t_extrinsic = vux2mls_extrinsics.translation();

                                    double prev_cost = std::numeric_limits<double>::max(); // Initialize with a large value
                                    double current_cost = prev_cost;
                                    double cost_threshold = .01; // Threshold for stopping criterion

                                    for (int iter_num = 0; iter_num < 100; iter_num++)
                                    {
                                        if (flg_exit || !ros::ok())
                                            break;

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

                                        feats_undistort->clear();

                                        // iterate the points,  georeference with init guess and search for NN
                                        for (int l = 0; l < lines_buffer.size(); l++) // for each line
                                        {
                                            const auto &fixed_pose = line_poses_buffer[l];    // copy of the init guess
                                            for (int i = 0; i < lines_buffer[l]->size(); i++) // for each point in the line
                                            {
                                                V3D p_src(lines_buffer[l]->points[i].x, lines_buffer[l]->points[i].y, lines_buffer[l]->points[i].z);
                                                V3D p_transformed = fixed_pose * vux2mls_extrinsics * p_src;

                                                PointType search_point;
                                                search_point.x = p_transformed.x();
                                                search_point.y = p_transformed.y();
                                                search_point.z = p_transformed.z();
                                                feats_undistort->push_back(search_point);

                                                std::vector<int> point_idx(5);
                                                std::vector<float> point_dist(5);

                                                if (refference_kdtree->nearestKSearch(search_point, 5, point_idx, point_dist) > 0) // there are neighbours
                                                {
                                                    if (point_dist[4] < 1.) // not too far
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
                                                            ceres::CostFunction *cost_function = registration::LidarPlaneNormFactor_extrinsics::Create(p_src, norm, negative_OA_dot_norm, fixed_pose);
                                                            problem.AddResidualBlock(cost_function, loss_function, q_param, t_param);
                                                            points_used_for_registration++;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        publish_frame_debug(pubLaserCloudDebug, feats_undistort);
                                        ros::spinOnce();
                                        rate.sleep();

                                        // Solve the problem
                                        ceres::Solver::Options options;
                                        options.linear_solver_type = ceres::DENSE_QR; // options.max_num_iterations = 100;
                                        options.minimizer_progress_to_stdout = false; // true;
                                        ceres::Solver::Summary summary;
                                        ceres::Solve(options, &problem, &summary);

                                        // std::cout << summary.FullReport() << std::endl;
                                        // std::cout << summary.BriefReport() << std::endl;

                                        std::cout << "\nIter: " << iter_num << ", Registration done with " << points_used_for_registration << " points" << std::endl;
                                        current_cost = summary.final_cost;
                                        std::cout << "Iteration " << iter_num << " - Cost: " << current_cost << std::endl;

                                        // Output the refined extrinsic transformation
                                        q_extrinsic = Eigen::Quaterniond(q_param[3], q_param[0], q_param[1], q_param[2]);
                                        t_extrinsic = V3D(t_param[0], t_param[1], t_param[2]);
                                        vux2mls_extrinsics = Sophus::SE3(q_extrinsic, t_extrinsic);
                                        std::cout << "t_extrinsic:" << t_extrinsic.transpose() << std::endl;

                                        // Check if the cost function change is small enough to stop
                                        if (std::abs(prev_cost - current_cost) < cost_threshold)
                                        {
                                            std::cout << "Stopping optimization: Cost change below threshold.\n";
                                            break;
                                        }

                                        prev_cost = current_cost;
                                    }
                                    std::cout << "Final transform is vux2mls_extrinsics log:" << vux2mls_extrinsics.log().transpose() << std::endl;
                                    std::cout << "Final t_extrinsic :" << t_extrinsic.transpose() << std::endl;
                                    std::cout << "Final vux2mls_extrinsics translation  :" << vux2mls_extrinsics.translation().transpose() << std::endl;
                                    std::cout << "Final rotation:\n"
                                              << vux2mls_extrinsics.so3().matrix() << std::endl;

                                    break;
                                }
                            }

                            if (true) // I AM ON THIS PART NOW
                            {
                                //std::cout << "Segment:" << segment_id << " distance:" << (prev_mls.translation() - curr_mls.translation()).norm() << " m" << std::endl;

                                // saving the data - the initial map will be based on the init guess
                                lines_buffer.push_back(downsampled_line);
                                line_poses_buffer.push_back(T_to_be_refined);
                                refined_line_poses_buffer.push_back(T_to_be_refined);

                                //use prev coarse delta as correction 
                                //line_poses_buffer.push_back(coarse_delta_T*T_to_be_refined);
                                //refined_line_poses_buffer.push_back(coarse_delta_T*T_to_be_refined);
                                
                                
                                //  if (!prev_segment_init) // first scan
                                //  {
                                //      total_scans = 500; // use 500
                                //  }
                                int total_scans = 100;
                                int mid_scan = total_scans / 2;
                                //mid_scan = total_scans; //this will not do anything
                                mid_scan = 75;
                                mid_scan = 50;    // if (l >= mid_scan) prev scan

                                total_scans = 200; //also good
                                mid_scan = 150;

                                //total_scans = 60; //also good
                                //mid_scan = 40;
                                // std::cout << "lines_buffer:" << lines_buffer.size() << std::endl;
                                if (lines_buffer.size() >= total_scans) // we have a list of scans
                                {
                                    feats_undistort->clear();
                                    // create the initial segment with init guess
                                    // std::vector<V3D> init_georeferenced_segment;
                                    pcl::PointCloud<PointType>::Ptr init_georeferenced_segment(new pcl::PointCloud<PointType>);

                                    //Sophus::SE3 ref_pose = first_segment_aligned ? last_refined_pose : line_poses_buffer[0];
                                    Sophus::SE3 ref_pose = line_poses_buffer[0];

                                    Sophus::SE3 relative = ref_pose.inverse() * ref_pose; // identity
                                    for (int l = 0; l < lines_buffer.size(); l++)         // for each line
                                    {
                                        const auto &initial_guess = line_poses_buffer[l]; // copy of the init guess

                                        if (l > 0)
                                        {
                                            relative = line_poses_buffer[l - 1].inverse() * line_poses_buffer[l];
                                        }
                                        ref_pose = ref_pose * relative;

                                        for (int i = 0; i < lines_buffer[l]->size(); i++) // for each point in the line
                                        {
                                            V3D p_src(lines_buffer[l]->points[i].x, lines_buffer[l]->points[i].y, lines_buffer[l]->points[i].z);

                                            //V3D p_transformed = initial_guess * p_src;
                                            
                                            V3D p_transformed = ref_pose * p_src;

                                            // p_transformed = coarse_delta_T * p_transformed; //prev relative correction

                                            PointType p;
                                            p.x = p_transformed.x();
                                            p.y = p_transformed.y();
                                            p.z = p_transformed.z();
                                            p.intensity = l;
                                            p.time = segment_id;
                                            init_georeferenced_segment->push_back(p);

                                            //feats_undistort->push_back(p);
                                        }
                                    }

                                    // init_georeferenced_segment is in the frame of the line_poses_buffer[0]
                                    // shift it to the frame of the prev estimated pose

                                    // publish_frame_debug(pubLaserCloudDebug, feats_undistort);
                                    // ros::spinOnce();
                                    // rate.sleep();
                                    // std::cout << "Init guess Press Enter to continue..." << std::endl;
                                    // std::cin.get();

                                    // Coarse register init_georeferenced_segment to map------------------
                                    int max_iterations_ = 20;
                                    double threshold_nn_ = 1.; // set init to 1 for init guess

                                    double prev_cost = std::numeric_limits<double>::max(); // Initialize with a large value
                                    double current_cost = prev_cost;
                                    double cost_threshold = .01; // Threshold for stopping criterion

                                    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
                                    V3D t = V3D::Zero();

                                    Sophus::SE3 T_icp = Sophus::SE3();
                                    bool use_GN = true;
                                    if (use_GN)
                                    {
                                        // with points only
                                        max_iterations_ = 100;
                                        cost_threshold = .001;

                                        max_iterations_ = 200;
                                        // with planes
                                        //max_iterations_ = 50;
                                        //cost_threshold = .02;
                                    }

                                    // debug of prev half segment
                                    publishJustPoints(prev_segment, cloud_pub); // prev segment for debug

                                    //
                                    bool coarse_register = true;
                                    if(coarse_register ){ //&& !first_segment_aligned
                                        //#define debug_clouds
                                        for (int iter_num = 0; iter_num < max_iterations_; iter_num++)
                                        {
                                            // break; // do not perform init guess refinement

                                            if (flg_exit || !ros::ok())
                                                break;

                                            if (use_GN)
                                            {
                                                // current_cost = scan2map_GN(init_georeferenced_segment, refference_kdtree,
                                                //                            reference_localMap_cloud, q, t, T_icp,
                                                //                            true, true, false, threshold_nn_);

                                                current_cost = scan2map_GN_omp(init_georeferenced_segment, refference_kdtree,
                                                                            reference_localMap_cloud, q, t, T_icp,
                                                                            false, prev_segment, kdtree_prev_segment,
                                                                            true, false, false, threshold_nn_);
                                            }
                                            else
                                            {
                                                current_cost = scan2map_ceres(init_georeferenced_segment, refference_kdtree,
                                                                            reference_localMap_cloud, q, t,
                                                                            false, prev_segment, kdtree_prev_segment,
                                                                            false, true, false, threshold_nn_);
                                            }

                                            coarse_delta_T = Sophus::SE3(q, t); // from init guess to the map
                                            std::cout << "\nIteration " << iter_num << " - Cost: " << current_cost << ", dCost:" << std::abs(prev_cost - current_cost) << " \n\n"
                                                    << std::endl;

                                            std::cout << "coarse_delta_T:" << coarse_delta_T.log().transpose() << std::endl;

    #ifdef debug_clouds
                                            feats_undistort->clear();
                                            for (const auto &raw_point : init_georeferenced_segment->points) // for each point in the segment
                                            {
                                                V3D p_src(raw_point.x, raw_point.y, raw_point.z);
                                                V3D p_transformed = coarse_delta_T * p_src;
                                                PointType p;
                                                p.x = p_transformed.x();
                                                p.y = p_transformed.y();
                                                p.z = p_transformed.z();
                                                // p.intensity = l;
                                                p.time = segment_id;
                                                feats_undistort->push_back(p);
                                            }
                                            publish_frame_debug(pubLaserCloudDebug, feats_undistort);
                                            ros::spinOnce();
                                            rate.sleep();
    #endif
                                            if (std::abs(prev_cost - current_cost) < cost_threshold || current_cost < cost_threshold) // Check if the cost function change is small enough to stop
                                            {
                                                std::cout << "Stopping optimization: Cost change below threshold.\n";
                                                break;
                                            }

                                            prev_cost = current_cost;
                                        }

                                        feats_undistort->clear();

                                        for (int l = 0; l < lines_buffer.size(); l++) // for each line
                                        {
                                            refined_line_poses_buffer[l] = coarse_delta_T * line_poses_buffer[l]; // refined pose

                                            const auto &initial_guess = refined_line_poses_buffer[l];

                                            for (int i = 0; i < lines_buffer[l]->size(); i++) // for each point in the line
                                            {
                                                V3D p_src(lines_buffer[l]->points[i].x, lines_buffer[l]->points[i].y, lines_buffer[l]->points[i].z);
                                                //  // for debug only
                                                V3D p_transformed = initial_guess * p_src;

                                                PointType p;
                                                p.x = p_transformed.x();
                                                p.y = p_transformed.y();
                                                p.z = p_transformed.z();
                                                p.intensity = l;
                                                p.time = segment_id;

                                                feats_undistort->push_back(p);
                                            }
                                        }

                                        publish_frame_debug(pubLaserCloudDebug, feats_undistort);

                                        ros::spinOnce();
                                        rate.sleep();

                                        last_refined_pose = refined_line_poses_buffer.back();
                                        std::cout << "Segment:" << segment_id << " distance:" << (prev_mls.translation() - curr_mls.translation()).norm() << " m" << std::endl;
                                        std::cout << "Segment  " << segment_id << " completed. Press Enter to continue..." << std::endl;
                                        std::cin.get();
                                    }
                                    
                                    bool BA_refine = true;// false;
                                    if (BA_refine) // pose graph here
                                    {
                                        std::cout << "Start BA refinement..." << std::endl;
                                        // only 1 iteration for now
                                        prev_cost = std::numeric_limits<double>::max(); // Initialize with a large value
                                        current_cost = prev_cost;

                                        //threshold_nn_ = .5 * .5; // make is smaller for refinement

                                        //for (int ba_iter = 0; ba_iter < 2; ba_iter++)
                                        {
                                            //if (flg_exit || !ros::ok())
                                            //    break;

                                            // std::cout << "BA " << std::endl;
                                            // current_cost = BA_refinement(
                                            //     lines_buffer,
                                            //     refined_line_poses_buffer,
                                            //     refference_kdtree,
                                            //     reference_localMap_cloud,
                                            //     prev_segment_init,
                                            //     prev_segment,
                                            //     kdtree_prev_segment,
                                            //     cloud_pub, normals_pub,
                                            //     total_scans - mid_scan, 
                                            //     threshold_nn_, true, true);
                                            
                                            std::cout << "BA_refinement_merge_graph" << std::endl;
                                            current_cost = BA_refinement_merge_graph(
                                                lines_buffer,
                                                refined_line_poses_buffer,
                                                refference_kdtree,
                                                reference_localMap_cloud,
                                                prev_segment_init,
                                                prev_segment,
                                                kdtree_prev_segment,
                                                cloud_pub, normals_pub,
                                                2,//run_iterations
                                                total_scans - mid_scan, 
                                                threshold_nn_, false, true);


                                            feats_undistort->clear();
                                            prev_segment->clear();
                                            
                                            for (int l = 0; l < lines_buffer.size(); l++) // for each line
                                            {
                                                const auto &initial_guess = refined_line_poses_buffer[l];

                                                for (int i = 0; i < lines_buffer[l]->size(); i++) // for each point in the line
                                                {
                                                    V3D p_src(lines_buffer[l]->points[i].x, lines_buffer[l]->points[i].y, lines_buffer[l]->points[i].z);
                                                    V3D p_transformed = initial_guess * p_src;

                                                    PointType p;
                                                    p.x = p_transformed.x();
                                                    p.y = p_transformed.y();
                                                    p.z = p_transformed.z();
                                                    p.intensity = l;
                                                    p.time = segment_id;

                                                    feats_undistort->push_back(p);

                                                    if (l >= mid_scan){
                                                        prev_segment->push_back(p);
                                                    }
                                                }
                                            }
                                            
                                            // // kdtree_prev_segment->setInputCloud(prev_segment); // the prev segment
                                            // // prev_segment_init = true;
                                            
                                            std::cout<<"Prev segment:"<<prev_segment->size()<<std::endl;

                                            publish_frame_debug(pubLaserCloudDebug, feats_undistort);

                                            ros::spinOnce();
                                            rate.sleep();

                                            std::cout << "BA done, cost:" << current_cost << std::endl;
                                            last_refined_pose = refined_line_poses_buffer.back();
                                            
                                            
                                            prev_cost = current_cost;
                                            std::cin.get();
                                        }
                                    }

                                    // delete half of the data not full - for overlapping section
                                    feats_undistort->clear();
                                    for (int j = 0; j < mid_scan; j++)
                                    {
                                        const auto &line = lines_buffer.front();
                                        const auto &T = refined_line_poses_buffer.front();

                                        for (int i = 0; i < line->size(); i++) // for each point in the line
                                        {
                                            V3D p_src(line->points[i].x, line->points[i].y, line->points[i].z);
                                            V3D p_transformed = T * p_src;

                                            PointType p;
                                            p.x = p_transformed.x();
                                            p.y = p_transformed.y();
                                            p.z = p_transformed.z();

                                            feats_undistort->push_back(p);
                                        }

                                        lines_buffer.pop_front();
                                        line_poses_buffer.pop_front();
                                        refined_line_poses_buffer.pop_front();
                                    }

                                    publish_frame_debug(pubOptimizedVUX, feats_undistort);

                                    segment_id++;
                                    first_segment_aligned = true;

                                    // lines_buffer.clear();
                                    // line_poses_buffer.clear();
                                    // refined_line_poses_buffer.clear();
                                }
                            }

                            for (size_t i = 0; i < transformed_cloud->size(); i++)
                            {
                                V3D point_scanner(transformed_cloud->points[i].x, transformed_cloud->points[i].y, transformed_cloud->points[i].z);

                                V3D point_global;
                                if (true)
                                {
                                    // with our refinement
                                    point_global = T_to_be_refined * point_scanner;

                                    point_global = coarse_delta_T * point_global;
                                }

                                transformed_cloud->points[i].x = point_global.x();
                                transformed_cloud->points[i].y = point_global.y();
                                transformed_cloud->points[i].z = point_global.z();
                            }

                            //this will pub the init guess 
                            publishPointCloud_vux(transformed_cloud, point_cloud_pub);
                        }

                        // ppk gnss = 10Hz same as hesai MLS
                        // imu also 10 Hz
                    }
                }
            }
            prev_mls = Sophus::SE3(state_point.rot, state_point.pos);
            prev_mls_time = time_of_day_sec;
        }
    }
    bag.close();
}
