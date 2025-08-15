

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
        V.col(2) *= -1;               // Reflect V to correct for reflection
        R = V * U.transpose();       // Recompute proper rotation
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

std::vector<vux_gnss_post> readMeasurements2(const std::string &filename)
{
    M3D T;
                    T <<    -1,  0,  0,
                            0,  0,  -1,
                            0,  -1,  0;
                    

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

    double stdev = 0.;
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

        // Read fields: GPSTime, Easting, Northing, H-Ell, then Phi, Omega, Kappa
        if (!(iss >> gpstime_double >> m.easting >> m.northing >> m.h_ell >> m.phi >> m.omega >> m.kappa >> stdev))
        {
            std::cerr << "Warning: Could not parse line: " << line << std::endl;
            continue;
        }
        if (measurements.empty())
        {
            std::cout << "first measurement" << std::endl;

            std::cout << "m.easting:" << m.easting << std::endl;
            std::cout << "m.northing:" << m.northing << std::endl;
            std::cout << "m.h_ell:" << m.h_ell << std::endl;
            std::cout << "m.phi:" << m.phi << std::endl;
            std::cout << "m.omega:" << m.omega << std::endl;
            std::cout << "m.kappa :" << m.kappa << std::endl;

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

        gpstime_double -= 18.; // convert to UTC since the lidar is in utc time from ros

        m.gps_tow = gpstime_double;
        m.gps_tod = std::fmod(gpstime_double, 86400.0); // Get the time of the day from time of the week;

        // these are degrees - convert to radians
        double omega = m.omega * (M_PI / 180.0);
        double phi = m.phi * (M_PI / 180.0);
        double kappa = m.kappa * (M_PI / 180.0);

        V3D translation(m.easting, m.northing, m.h_ell);

        M3D R_heikki;
        R_heikki << cos(phi) * cos(kappa), -cos(phi) * sin(kappa), sin(phi),
            cos(omega) * sin(kappa) + cos(kappa) * sin(omega) * sin(phi), cos(omega) * cos(kappa) - sin(omega) * sin(phi) * sin(kappa), -cos(phi) * sin(omega),
            sin(omega) * sin(kappa) - cos(omega) * cos(kappa) * sin(phi), cos(kappa) * sin(omega) + cos(omega) * sin(phi) * sin(kappa), cos(omega) * cos(phi);

        M3D R_enu = R_heikki * T;

        m.se3 = Sophus::SE3(Eigen::Quaterniond(R_enu), translation);

        m.stdev = std::max(stdev, .00001);

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

Sophus::SE3 averageSE3Log(const std::vector<Eigen::Matrix<double, 6, 1>> &logs)
{
    Eigen::Matrix<double, 6, 1> mean = Eigen::Matrix<double, 6, 1>::Zero();
    for (const auto &xi : logs)
        mean += xi;
    mean /= static_cast<double>(logs.size());
    std::cout << "\n averageSE3Log mean:" << mean.transpose() << std::endl;
    return Sophus::SE3::exp(mean);
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

//#include "clean_registration3.hpp"
#include "../src3/clean_registration3.hpp"
// using namespace gnss_MLS_fusion;
// #include "rangeProjection.hpp"

void DataHandler::Subscribe()
{
    std::cout << "Run test" << std::endl;
    std::cout << std::fixed << std::setprecision(12);
    std::cerr << std::fixed << std::setprecision(12);

    std::mt19937 rng(42); // Fixed seed for reproducibility

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

    Sophus::SE3 Lidar_wrt_IMU = Sophus::SE3(Lidar_R_wrt_IMU, Lidar_T_wrt_IMU);

    imu_obj->set_param(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, 10. * acc_cov),
                       V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov), V3D(b_acc_cov, b_acc_cov, 10. * b_acc_cov));

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

    ros::Publisher marker_pub = nh.advertise<visualization_msgs::MarkerArray>("tgt_covariance_markers", 1);
    ros::Publisher marker_pub2 = nh.advertise<visualization_msgs::MarkerArray>("src_covariance_markers", 1);

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

    // for the car used so far
    std::string post_processed_gnss_imu_vux_file = "/media/eugeniu/T7/roamer/evo_20240725_reCalculated2_poqmodouEugent.txt";
    std::vector<vux_gnss_post> gnss_vux_data = readMeasurements(post_processed_gnss_imu_vux_file);

    // test for drone data
    //std::string post_processed_gnss_imu_vux_file = "/home/eugeniu/Desktop/Evo_HesaiALS_20250709/Hesai_ALS_20250709_Evo_1014.txt";
    //std::vector<vux_gnss_post> gnss_vux_data = readMeasurements2(post_processed_gnss_imu_vux_file);

    
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
    downSizeFilter_vux.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

    // just a test for now
    // downSizeFilter_vux.setLeafSize(1.0, 1.0, 1.0);

    int scan_id = 0, vux_scan_id = 0;
    Sophus::SE3 first_vux_pose;

    Sophus::SE3 prev_mls, curr_mls;
    double prev_mls_time, curr_mls_time;
    cov prev_P, curr_P;

    V3D prev_enu, curr_enu;
    double prev_raw_gnss_diff_time, curr_raw_gnss_diff_time;

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
    bool raw_vux_imu_time_aligned = false;
    bool perform_mls_registration = true;
    bool als_integrated = false;
    Sophus::SE3 als2mls = Sophus::SE3();

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
    T_lidar2gnss << 0.0131683606, -0.9998577263, 0.0105414145, 0.0154123047,
                    0.9672090675, 0.0100627670, -0.2537821120, -2.6359450601,
                    0.2536399297, 0.0135376461, 0.9672039693, -0.5896374492,
                    0.0, 0.0, 0.0, 1.0;

    M3D R_lidar2gnss = T_lidar2gnss.block<3, 3>(0, 0); // Rotation
    V3D t_lidar2gnss = T_lidar2gnss.block<3, 1>(0, 3); // Translation

    // gnss should be rtansformed to mls frame
    Sophus::SE3 lidar2gnss(R_lidar2gnss, t_lidar2gnss); // FROM LIDAR 2 GNSS   T_lidar = T_gnss * lidar2gnss.inverse()
    Sophus::SE3 gnss2lidar = lidar2gnss.inverse();
    //----------------------------------------------------------------------------
    ros::Publisher pose_pub2 = nh.advertise<nav_msgs::Odometry>("/se3_pose2", 100);

    bool use_als = true;

    std::vector<pcl::PointCloud<VUX_PointType>::Ptr> vux_scans;
    std::vector<double> vux_scans_time;

    std::vector<double> mls_times, gnss_times;

    std::vector<pcl::PointCloud<PointType>::Ptr> mls_clouds;
    int estimated_total_points = 0;

    Sophus::SE3 first_ppk_gnss_pose_inverse = Sophus::SE3();

#define integrate_vux
#define integrate_ppk_gnss

    using namespace std::chrono;

    bool ppk_gnss_synced = false;
    Sophus::SE3 se3 = Sophus::SE3();
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


            if (scan_id > 1000) 
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
                    continue;
                }

                // undistort and provide initial guess
                imu_obj->Process(Measures, estimator_, feats_undistort);

                double t_IMU_process = omp_get_wtime();

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

                gnss_obj->Process(gps_buffer, lidar_end_time, state_point.pos);
                double time_of_day_sec = gnss_obj->tod;

                if(!ppk_gnss_synced)
                {
                    std::cout<<"Start time sync..."<<std::endl;
                    double time_start = time_of_day_sec - .1;
                    double time_end   = time_of_day_sec;
                    while (tmp_index < gnss_vux_data.size() - 1)
                    {
                        tmp_index++;
                        double time_diff_curr = fabs(time_start - gnss_vux_data[tmp_index].gps_tod);
                        double time_diff_next = fabs(time_start - gnss_vux_data[tmp_index + 1].gps_tod);
                        // std::cout << "time_diff_curr:" << time_diff_curr << ", time_diff_next:" << time_diff_next << std::endl;
                        std::cout << "Closest GNSS time diff = " << time_diff_curr << std::endl;
                        if (time_diff_curr > time_diff_next) // get to the closest message on time
                        {
                            continue; // continue to go to the next message
                        }
                        
                        ppk_gnss_synced = true;
                        // auto ppk_gnss_imu = als2mls * gnss_vux_data[tmp_index].se3; // in mls frame
                        auto interpolated_pose = interpolateSE3(
                                                        gnss_vux_data[tmp_index].se3, gnss_vux_data[tmp_index].gps_tod,
                                                        gnss_vux_data[tmp_index + 1].se3, gnss_vux_data[tmp_index + 1].gps_tod,
                                                        time_start) * gnss2lidar;
                        
                        first_ppk_gnss_pose_inverse = interpolated_pose.inverse();
                        std::cout << "\nsynchronised\n, press enter..." << std::endl;
                        std::cin.get();
                        break;
                    }
                }   
                else
                {
                    double time_start = time_of_day_sec - .1;
                    double time_end   = time_of_day_sec;
                    
                    while (tmp_index < gnss_vux_data.size() - 1)
                    {
                        const auto &msg_time = gnss_vux_data[tmp_index].gps_tod;

                        std::cout<<"\ntime_start:"<<time_start<<std::endl;
                        std::cout<<"time_end  :"<<time_end<<std::endl;
                        std::cout<<"msg_time  :"<<msg_time<<std::endl;

                        if(msg_time <= time_end)
                        {
                            // auto ppk_gnss_imu = als2mls * gnss_vux_data[tmp_index].se3; // in mls frame
                            auto interpolated_pose = interpolateSE3(
                                                                gnss_vux_data[tmp_index].se3, gnss_vux_data[tmp_index].gps_tod,
                                                                gnss_vux_data[tmp_index + 1].se3, gnss_vux_data[tmp_index + 1].gps_tod,
                                                                time_of_day_sec) * gnss2lidar;
        
                            


                            se3 = first_ppk_gnss_pose_inverse * interpolated_pose; //in first IMU frame 
                            publish_ppk_gnss(se3, msg_time);
                            tmp_index++;
                        }
                        else
                        {
                            break;
                        }  
                    }

                    //todo
                    /*
                    take all the raw gnss measurements from start-end scan time
                    convert them to tod 
                    get tod start and tod end of the scan 
                    use that to get all the ppk poses
                    undistort the point cloud 
                    */


                    the issue before was that gnss should have been transformed to IMU 
                        but now it works 


                    //auto T = se3 * Lidar_wrt_IMU;
                    TransformPoints(Lidar_wrt_IMU, feats_undistort); //lidar to IMU frame 

                    TransformPoints(se3, feats_undistort); //georeference with se3 in IMU frame

                    publish_frame_debug(pubLaserCloudDebug, feats_undistort);


                }

                 


                double t11 = omp_get_wtime();
                std::cout << "Mapping time(ms):  " << (t11 - t00) * 1000 << ", feats_down_size: " << feats_down_size << ", lidar_end_time:" << lidar_end_time << "\n"
                          << std::endl;
            }
        }
    }
    bag.close();
    // cv::destroyAllWindows();
}
