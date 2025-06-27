

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






std::vector<std::pair<std::vector<Sophus::SE3>, std::vector<Sophus::SE3>>> generateInterpolatedPoses(
    const std::vector<Sophus::SE3>& posesA, const std::vector<Sophus::SE3>& posesB,
    const std::vector<double>& timesA, const std::vector<double>& timesB, std::vector<double> &all_deltas
) {

        // int N_ = mls_times.size();
        //                     std::cout << " mls_times[0]:" << mls_times[0] << ", gnss_times[0]:" << gnss_times[0] << std::endl;
        //                     std::cout << " mls_times[end]:" << mls_times[N_ - 1] << ", gnss_times[end]:" << gnss_times[N_ - 1] << std::endl;

        //                     std::cout << "delta time :" << mls_times[N_ - 1] - mls_times[0] << std::endl;

    double t_start = std::max(timesA.front(), timesB.front());
    double t_end   = std::min(timesA.back(), timesB.back());
    double total_duration = t_end - t_start;

    std::cout<<"total_duration:"<<total_duration<<" s"<<std::endl;

    std::vector<std::pair<std::vector<Sophus::SE3>, std::vector<Sophus::SE3>>> all_trajectories;

    //add relative poses
    for(int i=0;i<all_deltas.size()-1;i++)
    //for (double delta_t : all_deltas) 
    {
        double &delta_t = all_deltas[i];
        std::vector<Sophus::SE3> trajectoryA, trajectoryB;

        for (double t = t_start; t <= t_end; t += delta_t) {
            if (t > timesA.back() || t > timesB.back()) break;

            Sophus::SE3 interpA = interpolateSE3Log(posesA, timesA, t);
            Sophus::SE3 interpB = interpolateSE3Log(posesB, timesB, t);

            trajectoryA.push_back(interpA);
            trajectoryB.push_back(interpB);
        }

        all_trajectories.emplace_back(trajectoryA, trajectoryB);
    }
    //add absolute poses
    std::vector<Sophus::SE3> trajectoryA, trajectoryB;

    for(int i=0;i<timesA.size();i++)
    {
        if(timesA[i] > t_start && timesA[i] < t_end)
        {
            Sophus::SE3 interpA = interpolateSE3Log(posesA, timesA, timesA[i]);
            Sophus::SE3 interpB = interpolateSE3Log(posesB, timesB, timesA[i]);

            trajectoryA.push_back(interpA);
            trajectoryB.push_back(interpB);
        }
    }

    all_trajectories.emplace_back(trajectoryA, trajectoryB);

    return all_trajectories;
}



Sophus::SE3 averageSE3Log(const std::vector<Eigen::Matrix<double, 6, 1>> &logs)
{
    Eigen::Matrix<double, 6, 1> mean = Eigen::Matrix<double, 6, 1>::Zero();
    for (const auto &xi : logs)
        mean += xi;
    mean /= static_cast<double>(logs.size());
    std::cout << "\n averageSE3Log mean:" << mean.transpose() << std::endl;
    return Sophus::SE3::exp(mean);
}

#include "clean_registration3.hpp"

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
    ros::Publisher pubLaserCloudLocal = nh.advertise<sensor_msgs::PointCloud2>("/cloud_local", 100000);

    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100);

    ros::Publisher pubLaserCloudDebug = nh.advertise<sensor_msgs::PointCloud2>("/cloud_debug", 10);
    ros::Publisher point_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/vux_data", 10000);
    ros::Publisher pubOptimizedVUX = nh.advertise<sensor_msgs::PointCloud2>("/vux_optimized", 10);

    ros::Publisher pubOptimizedVUX2 = nh.advertise<sensor_msgs::PointCloud2>("/vux_optimized2", 10);

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

    // get this as param
    std::string folder_path = "/media/eugeniu/T7/roamer/03_RIEGL_RAW/02_RXP/VUX-1HA-22-2022-B/";
    // folder_path = "/media/eugeniu/T7/roamer/03_RIEGL_RAW/02_RXP/VUX-1HA-22-2022-A/";

    vux::VuxAdaptor readVUX(std::cout, 75.);
    if (!readVUX.setUpReader(folder_path)) // get all the rxp files
    {
        std::cerr << "Cannot set up the VUX reader" << std::endl;
        return;
    }

    std::string post_processed_gnss_imu_vux_file = "/media/eugeniu/T7/roamer/evo_20240725_reCalculated2_poqmodouEugent.txt";
    std::vector<vux_gnss_post> gnss_vux_data = readMeasurements(post_processed_gnss_imu_vux_file);
    std::cout << "gnss_vux_data:" << gnss_vux_data.size() << std::endl;
    auto first_m = gnss_vux_data[0];
    V3D first_t(first_m.easting, first_m.northing, first_m.h_ell), first_t_ned;
    std::cout << "\n gps_tod:" << first_m.gps_tod << ", easting:" << first_m.easting << ", northing:" << first_m.northing << ", h_ell:" << first_m.h_ell << "\n"
              << std::endl;
    std::cout << "gps_tow:" << first_m.gps_tow << ", omega:" << first_m.omega << ", phi:" << first_m.phi << ", kappa:" << first_m.kappa << std::endl;
    int tmp_index = 0, init_guess_index = 0;

    ros::Rate rate(500);

    bool time_aligned = false;
    int some_index = 0;

    pcl::PointCloud<VUX_PointType>::Ptr next_line(new pcl::PointCloud<VUX_PointType>);
    pcl::PointCloud<PointType>::Ptr downsampled_als_cloud(new pcl::PointCloud<PointType>);

    bool vux_mls_time_aligned = false;
    pcl::VoxelGrid<VUX_PointType> downSizeFilter_vux;
    // downSizeFilter_vux.setLeafSize(filter_size_surf_min / 2, filter_size_surf_min / 2, filter_size_surf_min / 2);
    downSizeFilter_vux.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

    int scan_id = 0, vux_scan_id = 0;
    Sophus::SE3 first_vux_pose;

    Sophus::SE3 prev_mls, curr_mls;
    double prev_mls_time, curr_mls_time;

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
    bool raw_vux_imu_time_aligned = false;
    bool perform_mls_registration = true;
    bool als_integrated = false;
    Sophus::SE3 als2mls;
    Sophus::SE3 mls2als;

    //----------------------------------------------------------------------------
    M3D Rz;
    double angle = -M_PI / 2.0; // -90 degrees in radians
    Rz << cos(angle), -sin(angle), 0,
        sin(angle), cos(angle), 0,
        0, 0, 1;

    M3D R_vux2mls; // from vux scanner to mls point cloud
    R_vux2mls << 0.0064031121, -0.8606533346, -0.5091510953,
        -0.2586398121, 0.4904106092, -0.8322276624,
        0.9659526116, 0.1370155907, -0.2194590626;
    V3D t_vux2mls(-0.2238580597, -3.0124498678, -0.8051626709);
    Sophus::SE3 vux2mls_extrinsics = Sophus::SE3(R_vux2mls, t_vux2mls); // refined - vux to mls cloud

    Eigen::Matrix4d T_lidar2gnss;
    T_lidar2gnss << 
        0.0131683606, -0.9998577263, 0.0105414145,  0.0154123047,
        0.9672090675, 0.0100627670, -0.2537821120, -2.6359450601,
        0.2536399297, 0.0135376461, 0.9672039693,  -0.5896374492,
        0.0, 0.0, 0.0, 1.0;

    M3D R_lidar2gnss = T_lidar2gnss.block<3, 3>(0, 0); // Rotation
    V3D t_lidar2gnss = T_lidar2gnss.block<3, 1>(0, 3); // Translation

    // gnss should be rtansformed to mls frame
    Sophus::SE3 lidar2gnss(R_lidar2gnss, t_lidar2gnss); // FROM LIDAR 2 GNSS   T_lidar = T_gnss * lidar2gnss.inverse()
    Sophus::SE3 gnss2lidar = lidar2gnss.inverse();
    //----------------------------------------------------------------------------
    ros::Publisher pose_pub2 = nh.advertise<nav_msgs::Odometry>("/se3_pose2", 100);

    bool use_als = true;

    bool get_closest_vux_once = false;

    std::vector<pcl::PointCloud<VUX_PointType>::Ptr> vux_scans;
    std::vector<double> vux_scans_time;

    std::vector<Sophus::SE3> mls_poses, gnss_poses, gnss_poses_original;
    std::vector<double> mls_times, gnss_times;
    std::vector<Eigen::Vector6d> log_gnss_lidar_relative;

    std::vector<pcl::PointCloud<PointType>::Ptr> mls_clouds;
    int estimated_total_points = 0;
//#define integrate_vux
//#define perform_extrinsic

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
            // if (scan_id > 500) // 1050 used for data before
            // {
            //     std::cout << "Stop here... enough data" << std::endl;
            //     //break;
            // }

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
                    continue;
                }

                // undistort and provide initial guess
                imu_obj->Process(Measures, estimator_, feats_undistort);

                double t_IMU_process = omp_get_wtime();

                //publish_frame_debug(pubLaserCloudDebug, feats_undistort);

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
                            *featsFromMap = *laserCloudSurfMap;
                            als_obj->init(gnss_obj->origin_enu, gnss_obj->R_GNSS_to_MLS, featsFromMap);
                            gnss_obj->updateExtrinsic(als_obj->R_to_mls); // use the scan-based refined rotation for GNSS
                            als_obj->Update(Sophus::SE3(state_point.rot, state_point.pos));
                        }
                    }
                    else // als was set up
                    {
                        als_obj->Update(Sophus::SE3(state_point.rot, state_point.pos));
                        // update tighly fusion from MLS and ALS
                        if (!estimator_.update_tighly_fused(LASER_POINT_COV, feats_down_body, laserCloudSurfMap, als_obj->als_cloud, als_obj->localKdTree_map_als, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en))
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
                // double t_LiDAR_update = omp_get_wtime();
                // std::cout << "\nIMU_process time(ms):  " << (t_IMU_process - t00) * 1000 <<
                //", cloud_voxelization (ms): " << (t_cloud_voxelization - t_IMU_process) * 1000 <<
                //", LiDAR_update (ms): " << (t_LiDAR_update - t_cloud_voxelization) * 1000 << std::endl;

                // Crop the local map------
                state_point = estimator_.get_x();
                // get and publish the GNSS pose--------
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

                // Update the local map--------------------------------------------------
                feats_down_world->resize(feats_down_size);

                local_map_update(); // this will update local map with curr measurements and crop the map

#ifdef integrate_vux

                curr_mls = Sophus::SE3(state_point.rot, state_point.pos);
                curr_mls_time = Measures.lidar_end_time;

                double time_of_day_sec = gnss_obj->tod;
                if (gnss_obj->gps_init_origin && time_of_day_sec > .0001)
                {
                    std::cout << "MLS time_of_day_sec:" << time_of_day_sec << std::endl;

                    if (!vux_mls_time_aligned)
                    {
                        if (!get_closest_vux_once)
                        {
                            get_closest_vux_once = true;
                            if (!readVUX.timeAlign(time_of_day_sec)) // get the closest vux files to mls time
                            {
                                throw std::runtime_error("There is an issue with time aligning of raw vux and hesai mls");
                            }
                        }

                        // synch raw vux with mls------------------------------------
                        while (readVUX.next(next_line))
                        {
                            ros::spinOnce();
                            if (flg_exit || !ros::ok())
                                break;

                            if (!next_line->empty())
                            {
                                const auto &cloud_time = next_line->points[0].time;
                                V3D lla;
                                uint32_t raw_gnss_tod;
                                if (readVUX.nextGNSS(lla, raw_gnss_tod)) // get the raw GNSS into lla - if there is raw gnss
                                {
                                    std::cout << " raw time:" << raw_gnss_tod << std::endl;
                                }

                                double diff = fabs(cloud_time - time_of_day_sec);
                                std::cout << "\n VUX time:" << cloud_time << ", MLS time:" << time_of_day_sec << ", diff:" << diff << std::endl;

                                if (diff < .1)
                                {
                                    std::cout << "\nsynchronised\n, press enter..." << std::endl;
                                    vux_mls_time_aligned = true;
                                    // std::cin.get();
                                    break;
                                }

                                if (cloud_time > time_of_day_sec) // vux is ahead on time - we do not have vux for this hesai data
                                {
                                    // drop hesai frames
                                    std::cout << "Do nothing - wating till we get the VUX data based on time" << std::endl;
                                    break;
                                }

                                std::cout << "\n ====================Drop vux frames===================\n " << std::endl;
                            }
                            else
                            {
                                break;
                            }
                        }
                    }
                    else
                    {
                        vux_scans.clear();
                        vux_scans_time.clear();
                        estimated_total_points = 0;
                        while (readVUX.next(next_line))
                        {
                            ros::spinOnce();
                            if (flg_exit || !ros::ok())
                                break;

                            rate.sleep();

                            if (!next_line->empty())
                            {
                                const auto &cloud_time = next_line->points[0].time;

                                // accumulate vux scans

                                pcl::PointCloud<VUX_PointType>::Ptr downsampled_line(new pcl::PointCloud<VUX_PointType>);
                                downSizeFilter_vux.setInputCloud(next_line);
                                downSizeFilter_vux.filter(*downsampled_line);
                                TransformPoints(vux2mls_extrinsics, downsampled_line); // transform the vux cloud first to mls

                                vux_scans.push_back(downsampled_line);
                                vux_scans_time.push_back(cloud_time);
                                estimated_total_points += downsampled_line->size();

                                if (cloud_time > time_of_day_sec)
                                {
                                    break;
                                }
                            }
                            else
                            {
                                break;
                            }
                        }
                    }

                    // is is based on IMU
                    // imu_obj->Propagate2D(vux_scans, vux_scans_time, Measures.lidar_beg_time, Measures.lidar_end_time, time_of_day_sec, prev_mls, prev_mls_time);

                    // if (point_cloud_pub.getNumSubscribers() != 0)
                    {
                        pcl::PointCloud<VUX_PointType>::Ptr all_lines(new pcl::PointCloud<VUX_PointType>);

                        auto delta_predicted = (prev_mls.inverse() * curr_mls).log();
                        double scan_duration = curr_mls_time - prev_mls_time; // e.g., 0.1s
                        double tod_beg_scan = time_of_day_sec - scan_duration;
                        auto dt_tod = time_of_day_sec - tod_beg_scan;
                        
                        pcl::PointCloud<PointType>::Ptr all_lines_added_for_mapping(new pcl::PointCloud<PointType>);
                        all_lines_added_for_mapping->points.reserve(estimated_total_points);  // Optional: pre-allocate if you can estimate

                        for (size_t j = 0; j < vux_scans.size(); ++j)
                        {
                            const double t = vux_scans_time[j];
                            double alpha = (t - tod_beg_scan) / dt_tod;

                            Sophus::SE3 interpolated_pose_mls = prev_mls * Sophus::SE3::exp(alpha * delta_predicted);

                            TransformPoints(interpolated_pose_mls, vux_scans[j]);

                            *all_lines += *vux_scans[j];

                            auto& pts = vux_scans[j]->points;
                            all_lines_added_for_mapping->points.reserve(all_lines_added_for_mapping->points.size() + pts.size());
                            
                            for (const auto& pt : pts)
                            {
                                all_lines_added_for_mapping->points.emplace_back(PointType{pt.x, pt.y, pt.z});
                            }
                        }


                        *laserCloudSurfMap += *all_lines_added_for_mapping;
                        //laserCloudSurfMap->width = laserCloudSurfMap->points.size();
                        //laserCloudSurfMap->height = 1;
                        //laserCloudSurfMap->is_dense = true;

                        if (point_cloud_pub.getNumSubscribers() != 0)
                            publishPointCloud_vux(all_lines, point_cloud_pub);
                    }
                }
#endif

                prev_mls = Sophus::SE3(state_point.rot, state_point.pos);
                prev_mls_time = Measures.lidar_end_time;

#ifdef perform_extrinsic
                double time_of_day_sec = gnss_obj->tod;

                while (tmp_index < gnss_vux_data.size() - 1 && als_integrated) //
                {
                    tmp_index++;

                    double time_diff_curr = fabs(time_of_day_sec - gnss_vux_data[tmp_index].gps_tod);
                    double time_diff_next = fabs(time_of_day_sec - gnss_vux_data[tmp_index + 1].gps_tod);

                    //std::cout << "time_diff_curr:" << time_diff_curr << ", time_diff_next:" << time_diff_next << std::endl;
                    if (time_diff_curr > time_diff_next) // get to the closest message on time
                    {
                        continue; // continue to go to the next message
                    }

                    const auto &msg_time = gnss_vux_data[tmp_index].gps_tod;
                    //std::cout << "Closest GNSS time diff = " << time_diff_curr << std::endl;

                    auto ppk_gnss_imu = als2mls * gnss_vux_data[tmp_index].se3; // in mls frame

                    // a test move the gnss to hesai-imu frame with gnss2lidar
                    // ppk_gnss_imu = ppk_gnss_imu * gnss2lidar;

                    publish_ppk_gnss(ppk_gnss_imu, msg_time);

                    // std::vector<Sophus::SE3> mls_poses, gnss_poses, gnss_poses_original;

                    if (true) // perform extrinsic
                    {
                        mls_poses.push_back(prev_mls);
                        gnss_poses.push_back(ppk_gnss_imu);
                        gnss_poses_original.push_back(gnss_vux_data[tmp_index].se3);

                        mls_times.push_back(time_of_day_sec);
                        gnss_times.push_back(msg_time);

                        //feats_down_body is points in lidar frame

                        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
                        *cloud = *feats_down_body;
                        mls_clouds.push_back(cloud);//save the raw mls clouds

                        std::cout << "prev_mls    :" << prev_mls.translation().transpose() << std::endl;
                        std::cout << "ppk_gnss_imu:" << ppk_gnss_imu.translation().transpose() << std::endl;
                        std::cout << "log:" << (prev_mls.inverse() * ppk_gnss_imu).log().transpose() << std::endl;
                        // from lidar to mls
                        // auto lidar2gnss_extrinsic = prev_mls.inverse() * ppk_gnss_imu;
                        log_gnss_lidar_relative.push_back((prev_mls.inverse() * ppk_gnss_imu).log());

                        int N_data = 100;


                        if (mls_poses.size() > N_data) // got enough poses
                        {
                            std::cout << "Start extrinsic estimation press enter..." << std::endl;
                            std::cin.get();

                            Sophus::SE3 lidar2gnss_extrinsic = averageSE3Log(log_gnss_lidar_relative);
                            std::cout << "\nAveraged lidar2gnss_extrinsic SE3:\n"
                                      << lidar2gnss_extrinsic.matrix() << std::endl;

                            // Sophus::SE3 gnss2lidar = lidar2gnss.inverse();
                            // ppk_gnss_imu = ppk_gnss_imu * lidar2gnss_extrinsic.inverse();
                            // ppk_gnss_imu = ppk_gnss_imu * gnss2lidar;

                            std::vector<Sophus::SE3> A_rel, B_rel, B_rel_artificial;
                            //std::vector<double> all_deltas = {.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.}; //s
                            std::vector<double> all_deltas = {.1, .5,  1., 2., 3., 4., 5., -1}; //s 
                            std::vector<Eigen::Vector6d> relative_logs;
                            //add absolute poses
                            double delta_t = .01;
                            double t_start = std::max(mls_times.front(), gnss_times.front());
                            double t_end   = std::min(mls_times.back(), gnss_times.back());
                            double total_duration = t_end - t_start;
                            for (double t = t_start; t <= t_end; t += delta_t) {
                                if (t > mls_times.back() || t > gnss_times.back()) break;

                                Sophus::SE3 interpA = interpolateSE3Log(mls_poses, mls_times, t);
                                Sophus::SE3 interpB = interpolateSE3Log(gnss_poses, gnss_times, t);

                                relative_logs.push_back((interpA.inverse() * interpB).log());
                            }
                            
                            auto all_trajectories = generateInterpolatedPoses(mls_poses, gnss_poses, mls_times, gnss_times, all_deltas);
                            
                            for (size_t i = 0; i < all_trajectories.size(); ++i) {
                                double delta = all_deltas[i];
                                std::cout << "Δt = " << delta << "s: " << all_trajectories[i].first.size() << " samples\n";

                                int N = all_trajectories[i].first.size();
                                for(int j = 1; j<N; j++)
                                {
                                    auto Ai = all_trajectories[i].first[j - 1].inverse() * all_trajectories[i].first[j];
                                    auto Bi = all_trajectories[i].second[j - 1].inverse() * all_trajectories[i].second[j];

                                    A_rel.push_back(Ai);
                                    B_rel.push_back(Bi);

                                }
                            }
                            
                            std::cout<<"A_rel:"<<A_rel.size()<<", B_rel:"<<B_rel.size()<<std::endl;

                            // lidar2gnss_extrinsic = averageSE3Log(relative_logs);
                            // std::cout << "\nAveraged relative_logs SE3:\n"
                            //           << lidar2gnss_extrinsic.matrix() << std::endl;

                            {
                                // test with some artificially created data
                                // auto X_gt = Sophus::SE3(
                                //     Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()) *
                                //         Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY()),
                                //     Eigen::Vector3d(1.5, 2.1, -1.2));

                                // take the time interval from the begin and end
                                // and for some delta t interpolate

                                // for (int i = 1; i < mls_poses.size(); i++)
                                // {
                                //     auto Ai = mls_poses[i - 1].inverse() * mls_poses[i];

                                //     // auto Bi = gnss_poses[i-1].inverse() * gnss_poses[i];
                                //     auto Bi = gnss_poses_original[i - 1].inverse() * gnss_poses_original[i];

                                //     auto Bi_Artificial = X_gt.inverse() * Ai * X_gt;

                                //     A_rel.push_back(Ai);
                                //     B_rel.push_back(Bi);
                                //     B_rel_artificial.push_back(Bi_Artificial);
                                // }
                            }

                            auto X_est = ConventionalAXXBSVDSolver(A_rel, B_rel);
                            std::cout << "X_est ConventionalAXXBSVDSolver:\n"
                                      << X_est.matrix() << std::endl;

                            X_est = handEyeGaussNewton(A_rel, B_rel);
                            std::cout << "X_est handEyeGaussNewton       :\n"
                                      << X_est.matrix() << std::endl;

                            
                            // X_est = handEyeGaussNewton(mls_poses, gnss_poses_original);
                            // std::cout << "X_est handEyeGaussNewton(mls_poses, gnss_poses_original)       :\n"
                            //           << X_est.matrix() << std::endl;

                            
                            // X_est = handEyeGaussNewton(mls_poses, gnss_poses);
                            // std::cout << "X_est handEyeGaussNewton(mls_poses, gnss_poses)       :\n"
                            //           << X_est.matrix() << std::endl;
                            
                            ppk_gnss_imu = ppk_gnss_imu * X_est.inverse();

                            //ppk_gnss_imu = ppk_gnss_imu * lidar2gnss_extrinsic.inverse();

                            // auto X_est_artificial = handEyeGaussNewton(A_rel, B_rel_artificial);
                            // auto X_est_artificial2 = ConventionalAXXBSVDSolver(A_rel, B_rel_artificial);

                            // std::cout << "\nX_est_artificial:\n"
                            //           << X_est_artificial.matrix() << std::endl;
                            // std::cout << "Error X_est_artificial norm (Lie algebra):" << (X_gt.inverse() * X_est_artificial).log().norm() << "\n";

                            // std::cout << "\n\nX_est_artificial2:\n"
                            //           << X_est_artificial2.matrix() << std::endl;
                            // std::cout << "Error X_est_artificial2 norm (Lie algebra):" << (X_gt.inverse() * X_est_artificial2).log().norm() << "\n";

                            // throw std::runtime_error("Finished the extrinsic estimation");
                        }


                        {
                                /*
                                    try to recreate some existing map
                                    this can be used also for flying helicopters
                                        given gnss and 2d raw data
                                        given a known reference map
                                        
                                        gnss can be used with ALS or MLS, MLS can be transformed to global
                                        georeferencing the 2d vux with gnss should result in same map 

                                        scan_registered = GNSS * X * scan
                                        error = scan_registered(X) - ref_scan_registered_

                                        use this as a constraint to find the transformation between the GNSS and 2D vux 

                                    a paper for both 
                                */        
                            }



                        nav_msgs::Odometry pose_msg;
                        pose_msg.header.frame_id = "world";
                        pose_msg.header.stamp = ros::Time().fromSec(msg_time);
                        V3D trans = ppk_gnss_imu.translation();
                        pose_msg.pose.pose.position.x = trans.x();
                        pose_msg.pose.pose.position.y = trans.y();
                        pose_msg.pose.pose.position.z = trans.z();
                        Eigen::Quaterniond q(ppk_gnss_imu.so3().matrix());
                        pose_msg.pose.pose.orientation.x = q.x();
                        pose_msg.pose.pose.orientation.y = q.y();
                        pose_msg.pose.pose.orientation.z = q.z();
                        pose_msg.pose.pose.orientation.w = q.w();

                        pose_pub2.publish(pose_msg);
                    }

                     
                    

                    break;
                    
                }
#endif

                // Publish odometry and point clouds------------------------------------
                publish_odometry(pubOdomAftMapped);
                if (scan_pub_en)
                {
                    if (pubLaserCloudFull.getNumSubscribers() != 0)
                        publish_frame_world(pubLaserCloudFull);

                }

                if (pubLaserCloudMap.getNumSubscribers() != 0)
                {
                    *featsFromMap = *laserCloudSurfMap;
                    publish_map(pubLaserCloudMap);
                }

                double t11 = omp_get_wtime();
                std::cout << "Mapping time(ms):  " << (t11 - t00) * 1000 << ", feats_down_size: " << feats_down_size << ", lidar_end_time:" << lidar_end_time << "\n"
                          << std::endl;

                als2mls = als_obj->als_to_mls;
            }
        }
    }
    bag.close();
}
