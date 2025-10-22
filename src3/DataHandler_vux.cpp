

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

std::vector<std::pair<std::vector<Sophus::SE3>, std::vector<Sophus::SE3>>> generateInterpolatedPoses(
    const std::vector<Sophus::SE3> &posesA, const std::vector<Sophus::SE3> &posesB,
    const std::vector<double> &timesA, const std::vector<double> &timesB, std::vector<double> &all_deltas)
{

    // int N_ = mls_times.size();
    //                     std::cout << " mls_times[0]:" << mls_times[0] << ", gnss_times[0]:" << gnss_times[0] << std::endl;
    //                     std::cout << " mls_times[end]:" << mls_times[N_ - 1] << ", gnss_times[end]:" << gnss_times[N_ - 1] << std::endl;

    //                     std::cout << "delta time :" << mls_times[N_ - 1] - mls_times[0] << std::endl;

    double t_start = std::max(timesA.front(), timesB.front());
    double t_end = std::min(timesA.back(), timesB.back());
    double total_duration = t_end - t_start;

    std::cout << "total_duration:" << total_duration << " s" << std::endl;

    std::vector<std::pair<std::vector<Sophus::SE3>, std::vector<Sophus::SE3>>> all_trajectories;

    // add relative poses
    for (int i = 0; i < all_deltas.size() - 1; i++)
    // for (double delta_t : all_deltas)
    {
        double &delta_t = all_deltas[i];
        std::vector<Sophus::SE3> trajectoryA, trajectoryB;

        for (double t = t_start; t <= t_end; t += delta_t)
        {
            if (t > timesA.back() || t > timesB.back())
                break;

            Sophus::SE3 interpA = interpolateSE3Log(posesA, timesA, t);
            Sophus::SE3 interpB = interpolateSE3Log(posesB, timesB, t);

            trajectoryA.push_back(interpA);
            trajectoryB.push_back(interpB);
        }

        all_trajectories.emplace_back(trajectoryA, trajectoryB);
    }
    // add absolute poses
    std::vector<Sophus::SE3> trajectoryA, trajectoryB;

    for (int i = 0; i < timesA.size(); i++)
    {
        if (timesA[i] > t_start && timesA[i] < t_end)
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

#include "clean_registration3.hpp"

// using namespace gnss_MLS_fusion;
// #include "rangeProjection.hpp"

// #include <novatel_oem7_msgs/INSPVA.h>

// void handleInspvaMessage(const rosbag::MessageInstance &m)
// {
//     // Try to instantiate the message as INSPVA
//     novatel_oem7_msgs::INSPVA::ConstPtr msg = m.instantiate<novatel_oem7_msgs::INSPVA>();
//     if (!msg)
//         return; // skip if wrong type

//     // Convert ROS stamp to double seconds
//     double ros_time_sec = msg->header.stamp.sec + msg->header.stamp.nsec * 1e-9;

//     // Print results
//     std::cout << std::fixed;
//     std::cout << " novatel_oem7_msgs::INSPVA "<<std::endl;
//     std::cout << "ROS Time (s): " << ros_time_sec << std::endl;
//     std::cout << "GPS Week Number: " << msg->nov_header.gps_week_number << std::endl;
//     std::cout << "GPS Week Milliseconds: " << msg->nov_header.gps_week_milliseconds << std::endl;
//     std::cout << "---------------------------------------------" << std::endl;
// }

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

    {
        //just a test
        //std::shared_ptr<Graph> imu_obj_2(new Graph());
    }


    std::shared_ptr<GNSS> gnss_obj(new GNSS());

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

    // std::ifstream file(bag_file);
    // if (!file)
    // {
    //     std::cerr << "Error: Bag file does not exist or is not accessible: " << bag_file << std::endl;
    //     return;
    // }
    // rosbag::Bag bag;
    // bag.open(bag_file, rosbag::bagmode::Read);
    // std::vector<std::string> topics;
    // topics.push_back(lid_topic);
    // topics.push_back(imu_topic);
    // topics.push_back(gnss_topic);
    // topics.push_back("/novatel/oem7/inspva");
    // rosbag::View view(bag, rosbag::TopicQuery(topics));

    std::vector<std::string> topics{lid_topic, imu_topic, gnss_topic};

    std::vector<std::string> bag_files = expandBagPattern(bag_file);
    std::cout << "bag_files:" << bag_files.size() << std::endl;
    if (bag_files.size() == 0)
    {
        std::cerr << "Error: Bag file does not exist or is not accessible: " << bag_file << std::endl;
        return;
    }
    for (auto &f : bag_files)
        std::cout << "Matched: " << f << std::endl;

    // Open all bags
    std::vector<std::shared_ptr<rosbag::Bag>> bags;
    for (const auto &file : bag_files)
    {
        auto bag = std::make_shared<rosbag::Bag>();
        bag->open(file, rosbag::bagmode::Read);
        bags.push_back(bag);
        ROS_INFO_STREAM("Opened bag: " << file);
    }

    // Build a single view from all bags
    rosbag::View view;
    for (auto &b : bags)
    {
        view.addQuery(*b, rosbag::TopicQuery(topics));
    }



    signal(SIGINT, SigHandle); // Handle Ctrl+C (SIGINT)
    flg_exit = false;

    std::cout << "Start reading the data..." << std::endl;

    // get this as param
    std::string folder_path = "/media/eugeniu/T7/roamer/03_RIEGL_RAW/02_RXP/VUX-1HA-22-2022-B/";
    // folder_path = "/media/eugeniu/T7/roamer/03_RIEGL_RAW/02_RXP/VUX-1HA-22-2022-A/";
    
    
#define integrate_vux
//#define integrate_ppk_gnss
    
    #ifdef integrate_vux
    vux::VuxAdaptor readVUX(std::cout, 75.);
    if (!readVUX.setUpReader(folder_path)) // get all the rxp files
    {
        std::cerr << "Cannot set up the VUX reader" << std::endl;
        return;
    }
    #endif 

    // for the car used so far
    // std::string post_processed_gnss_imu_vux_file = "/media/eugeniu/T7/roamer/evo_20240725_reCalculated2_poqmodouEugent.txt";
    // std::vector<vux_gnss_post> gnss_vux_data = readMeasurements(post_processed_gnss_imu_vux_file);

    // test for drone data
    std::string post_processed_gnss_imu_vux_file = "/home/eugeniu/Desktop/Evo_HesaiALS_20250709/Hesai_ALS_20250709_Evo_1014.txt";
    std::vector<vux_gnss_post> gnss_vux_data = readMeasurements2(post_processed_gnss_imu_vux_file);

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
    // R_vux2mls << 0.0064031121, -0.8606533346, -0.5091510953,
    //     -0.2586398121, 0.4904106092, -0.8322276624,
    //     0.9659526116, 0.1370155907, -0.2194590626;
    // V3D t_vux2mls(-0.2238580597, -3.0124498678, -0.8051626709);
    //THE ONE ESTIMATED USING THE ALS POINT CLOUD - this are better based on the point clouds 
    R_vux2mls << 0.0143844669, -0.8542734617, -0.5196248067,
                -0.2613330313,  0.4984032661, -0.8266191572,
                0.9651415098,  0.1476856018, -0.2160806080;
    V3D t_vux2mls(-0.2772152452, -3.1178620759, -0.8987165442);
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

    // the next 2 will be used for extrinsics estimation
    std::vector<Sophus::SE3> known_T;
    std::vector<pcl::PointCloud<VUX_PointType>::Ptr> _2d_measurements;
    Sophus::SE3 T_extrinsic = Sophus::SE3();

    bool use_als = true;

    bool get_closest_vux_once = false;

    std::vector<pcl::PointCloud<VUX_PointType>::Ptr> vux_scans;
    std::vector<double> vux_scans_time;

    std::vector<Sophus::SE3> mls_poses, gnss_poses, gnss_poses_original;
    std::vector<double> mls_times, gnss_times;
    std::vector<Eigen::Vector6d> log_gnss_lidar_relative;

    std::vector<pcl::PointCloud<PointType>::Ptr> mls_clouds;
    int estimated_total_points = 0;

    double avg_time_update = 0;
    int count = 0;
    bool ppk_gnss_init = false;
    Sophus::SE3 first_ppk_gnss_pose_inverse = Sophus::SE3();



    using namespace std::chrono;

    int tmp_count = 0;

    bool do_this_once = false, dropped = false;


    // shift_measurements_to_zero_time = true; //required for time sync
    // gnss_obj->shift_measurements_to_zero_time = shift_measurements_to_zero_time;

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
        // else if(topic == "/novatel/oem7/inspva")
        // {
        //     handleInspvaMessage(m);
        // }

        if (sync_packages(Measures))
        {
            scan_id++;
            std::cout << "scan_id:" << scan_id << std::endl;
            // if (scan_id > 8000) // 500 1050 used for data before
            // {
            //     std::cout << "Stop here... enough data 8000 scans" << std::endl;
            //     break;
            // }

            // if (scan_id > 1000) 
            // {
            //     std::cout << "Stop here... enough data 8000 scans" << std::endl;
            //     break;
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

                use_als = false;

                if (use_als)
                {
                    if (!als_obj->refine_als) // als was not setup
                    {
                        estimator_.update(LASER_POINT_COV, feats_down_body, laserCloudSurfMap, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en);

                        // Sophus::SE3 gnss_pose = (gnss_obj->use_postprocessed_gnss) ? gnss_obj->postprocessed_gps_pose : gnss_obj->gps_pose;
                        // const V3D &gnss_in_mls = gnss_pose.translation();
                        // //just test for now
                        // if (!estimator_.update_tighly_fused_test(LASER_POINT_COV, feats_down_body,
                        //     laserCloudSurfMap, als_obj->als_cloud, als_obj->localKdTree_map_als, Nearest_Points,
                        //     gnss_in_mls, 0.0001, //GNSS_VAR * GNSS_VAR,
                        //     NUM_MAX_ITERATIONS, extrinsic_est_en))
                        // {
                        //     std::cout << "\n------------------FUSION ALS-MLS update failed--------------------------------" << std::endl;
                        // }

                        // Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> Neighbours;
                        // Neighbours.resize(N_SCAN, Horizon_SCAN);
                        // projectToRangeImage(feats_undistort, Neighbours);
                        //  if (!estimator_.update_tighly_fused_test2(LASER_POINT_COV, feats_down_body, feats_undistort, Neighbours,
                        //      laserCloudSurfMap, als_obj->als_cloud, als_obj->localKdTree_map_als, Nearest_Points,
                        //      gnss_in_mls, 0.0001, //GNSS_VAR * GNSS_VAR,
                        //      NUM_MAX_ITERATIONS, extrinsic_est_en))
                        //  {
                        //      std::cout << "\n------------------FUSION ALS-MLS update failed--------------------------------" << std::endl;
                        //  }

                        // publishCovarianceEllipsoids(corr_laserCloudTgt, corr_tgt_covs,
                        //          marker_pub, "world", 2, 5, 1., .2);

                        // publishCovarianceEllipsoids(laserCloudSrc, src_covs,
                        //          marker_pub2, "world", 2, 5, .2, 1.);

                        if (gnss_obj->GNSS_extrinsic_init)
                        {
                            *featsFromMap = *laserCloudSurfMap;

                            // only for now - remove this later
                            std::string als2mls_filename = "/home/eugeniu/x_vux-georeferenced-final/_Hesai/als2mls.txt";
                            Sophus::SE3 known_als2mls;
                            readSE3FromFile(als2mls_filename, known_als2mls);
                            std::cout << "Read the known transformation" << std::endl;

                            //als_obj->init(known_als2mls);

                            if (!this->downsample) // if it is sparse ALS data from NLS
                            {
                                V3D t = known_als2mls.translation();
                                t.z() += 20.;
                                Sophus::SE3 als2mls_for_sparse_ALS = Sophus::SE3(known_als2mls.so3().matrix(), t);
                                std::cout << "Init ALS from known T map refinement" << std::endl;
                                als_obj->init(als2mls_for_sparse_ALS, laserCloudSurfMap); // with refinement
                            }
                            else
                            {
                                std::cout << "Init ALS from known T" << std::endl;
                                als_obj->init(known_als2mls);
                            }

                            //als_obj->init(gnss_obj->origin_enu, gnss_obj->R_GNSS_to_MLS, featsFromMap);

                            gnss_obj->updateExtrinsic(als_obj->R_to_mls); // use the scan-based refined rotation for GNSS
                            als_obj->Update(Sophus::SE3(state_point.rot, state_point.pos));
                            gnss_obj->als2mls_T = als_obj->als_to_mls;

                            //reset local map
                            //laserCloudSurfMap.reset(new PointCloudXYZI());
                        }
                    }
                    else // als was set up
                    {
                        als_obj->Update(Sophus::SE3(state_point.rot, state_point.pos));

                        // // update only MLS
                        // if (!estimator_.update(LASER_POINT_COV, feats_down_body, laserCloudSurfMap, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en))
                        // {
                        //     std::cout << "\n------------------MLS update failed--------------------------------" << std::endl;
                        // }

                        // // update only ALS
                        // if (!estimator_.update(LASER_POINT_COV / 4, feats_down_body, als_obj->als_cloud, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en))
                        // {
                        //     std::cout << "\n------------------ ALS update failed --------------------------------" << std::endl;
                        //     // TODO check here why -  there is not enough als data
                        // }

                        // update tighly fusion from MLS and ALS
                        if (!estimator_.update_tighly_fused(LASER_POINT_COV, feats_down_body, laserCloudSurfMap, als_obj->als_cloud, als_obj->localKdTree_map_als, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en))
                        {
                            std::cout << "\n------------------FUSION ALS-MLS update failed--------------------------------" << std::endl;
                        }    

                        if(false){ //this was used so far
                            //update tighly fusion from MLS and ALS
                            double R_gps_cov = .0001; // GNSS_VAR * GNSS_VAR;
                            Sophus::SE3 gnss_pose = (gnss_obj->use_postprocessed_gnss) ? gnss_obj->postprocessed_gps_pose : gnss_obj->gps_pose;
                            const V3D &gnss_in_mls = gnss_pose.translation();

                            bool tightly_coupled = true;
                            bool use_gnss = false; 
                            bool use_als = true;

                            if (!estimator_.update_final(
                                LASER_POINT_COV, R_gps_cov, feats_down_body, gnss_in_mls, laserCloudSurfMap, als_obj->als_cloud, als_obj->localKdTree_map_als,
                                Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en, use_gnss, use_als, tightly_coupled))
                            {
                                std::cout << "\n------------------FUSION ALS-MLS update failed--------------------------------" << std::endl;
                            }
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

                //use_als = false;
                als_integrated = true; // remove this later - just a test now

                double t_LiDAR_update = omp_get_wtime();
                std::cout << "\nIMU_process time(ms):  " << (t_IMU_process - t00) * 1000 <<
                ", cloud_voxelization (ms): " << (t_cloud_voxelization - t_IMU_process) * 1000 <<
                ", LiDAR_update (ms): " << (t_LiDAR_update - t_cloud_voxelization) * 1000 << std::endl;

                // Crop the local map------
                state_point = estimator_.get_x();
                // get and publish the GNSS pose--------
                gnss_obj->Process(gps_buffer, lidar_end_time, state_point.pos);

                Sophus::SE3 gnss_pose = (gnss_obj->use_postprocessed_gnss) ? gnss_obj->postprocessed_gps_pose : gnss_obj->gps_pose;

                // gnss_pose.so3() = state_point.rot; // use the MLS orientation
                // if (als_integrated) // if (use_gnss)
                publish_gnss_odometry(gnss_pose);

                if (gnss_obj->GNSS_extrinsic_init && use_gnss && als_integrated) // if gnss aligned
                {
                    const bool global_error = false; // set this true for global error of gps
                    // auto gps_cov_ = V3D(GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR);
                    // auto gps_cov_ = gnss_obj->gps_cov;
                    auto gps_cov_ = Eigen::Vector3d(GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR);
                    if (global_error)
                    {
                        const V3D &gnss_in_enu = gnss_obj->carthesian; // this will be relative w.r.t. zero origin
                        estimator_.update(gnss_in_enu, gps_cov_, NUM_MAX_ITERATIONS, global_error, gnss_obj->R_GNSS_to_MLS.transpose());
                    }
                    else
                    {
                        const V3D &gnss_in_mls = gnss_pose.translation();
                        estimator_.update(gnss_in_mls, gps_cov_, NUM_MAX_ITERATIONS, global_error);
                    }
                }

                curr_mls = Sophus::SE3(state_point.rot, state_point.pos);
                curr_mls_time = Measures.lidar_end_time;
                curr_P = estimator_.get_P();
                curr_enu = gnss_obj->curr_enu;
                curr_raw_gnss_diff_time = gnss_obj->diff_curr_gnss2mls;

                double time_of_day_sec = gnss_obj->tod;
#ifdef integrate_vux

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
                                double vux_cloud_dt = next_line->points.back().time - next_line->points.front().time;

                                std::cout << "vux_cloud_dt:" << vux_cloud_dt << " s" << std::endl;
                                V3D lla;
                                uint32_t raw_gnss_tod;
                                if (readVUX.nextGNSS(lla, raw_gnss_tod)) // get the raw GNSS into lla - if there is raw gnss
                                {
                                    std::cout << " raw time:" << raw_gnss_tod << std::endl;
                                }

                                double diff = fabs(cloud_time - time_of_day_sec);
                                std::cout << "\n VUX time:" << cloud_time << ", MLS time:" << time_of_day_sec << ", diff:" << diff << std::endl;

                                double diff_next = fabs(cloud_time + vux_cloud_dt - time_of_day_sec);
                                std::cout << "diff_next:" << diff_next << std::endl;

                                // if (diff < .1) //used before
                                if (diff < .1 && diff < diff_next)
                                {
                                    std::cout << "\nsynchronised\n, press enter..." << std::endl;
                                    vux_mls_time_aligned = true;
                                    std::cin.get();
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
                                double vux_cloud_dt = next_line->points.back().time - next_line->points.front().time;

                                // accumulate vux scans

                                // pcl::PointCloud<VUX_PointType>::Ptr downsampled_line(new pcl::PointCloud<VUX_PointType>);
                                // downSizeFilter_vux.setInputCloud(next_line);
                                // downSizeFilter_vux.filter(*downsampled_line);
                                // TransformPoints(vux2mls_extrinsics, downsampled_line); // transform the vux cloud first to mls

                                // vux_scans.push_back(downsampled_line);
                                // estimated_total_points += downsampled_line->size();
                                vux_scans.push_back(next_line);

                                vux_scans_time.push_back(cloud_time);
                                estimated_total_points += next_line->size();

                                // if (cloud_time > time_of_day_sec)
                                // {
                                //     break;
                                // }

                                // use this instead
                                if ((cloud_time + vux_cloud_dt) >= time_of_day_sec) // reached the end of the scan
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

                    if (true) // integrate vux into MLS mapping
                    {
                        pcl::PointCloud<VUX_PointType>::Ptr all_lines(new pcl::PointCloud<VUX_PointType>);
                        bool subscribers = point_cloud_pub.getNumSubscribers() != 0;
                        auto delta_predicted = (prev_mls.inverse() * curr_mls).log();
                        double scan_duration = curr_mls_time - prev_mls_time; // e.g., 0.1s
                        double tod_beg_scan = time_of_day_sec - scan_duration;
                        auto dt_tod = time_of_day_sec - tod_beg_scan;

                        pcl::PointCloud<PointType>::Ptr all_lines_added_for_mapping(new pcl::PointCloud<PointType>);
                        all_lines_added_for_mapping->points.reserve(estimated_total_points); // Optional: pre-allocate if you can estimate

                        for (size_t j = 0; j < vux_scans.size(); ++j)
                        {
                            pcl::PointCloud<VUX_PointType>::Ptr downsampled_line(new pcl::PointCloud<VUX_PointType>);
                            downSizeFilter_vux.setInputCloud(vux_scans[j]);
                            downSizeFilter_vux.filter(*downsampled_line);

                            const double t = vux_scans_time[j];
                            double alpha = (t - tod_beg_scan) / dt_tod;

                            Sophus::SE3 interpolated_pose_mls = prev_mls * Sophus::SE3::exp(alpha * delta_predicted);
                            Sophus::SE3 vux_pose = interpolated_pose_mls * vux2mls_extrinsics;

                            // show were is VUX compared to MLS
                            // publish_refined_ppk_gnss(vux_pose, curr_mls_time);

                            TransformPoints(vux_pose, downsampled_line);
                            if (subscribers)
                            {
                                // if(dense_pub_en)
                                // {
                                //     TransformPoints(vux_pose, vux_scans[j]);
                                //     for(int k=0;k<vux_scans[j]->size();k++)
                                //     {
                                //         vux_scans[j]->points[k].reflectance = j;
                                //     }
                                //     *all_lines += *vux_scans[j];
                                // }
                                // else
                                // {
                                for (int k = 0; k < downsampled_line->size(); k++)
                                {
                                    downsampled_line->points[k].reflectance = j;
                                }
                                *all_lines += *downsampled_line;
                                //}
                            }

                            auto &pts = downsampled_line->points;
                            all_lines_added_for_mapping->points.reserve(all_lines_added_for_mapping->points.size() + pts.size());

                            for (const auto &pt : pts)
                            {
                                all_lines_added_for_mapping->points.emplace_back(PointType{pt.x, pt.y, pt.z});
                            }
                        }

                        *laserCloudSurfMap += *all_lines_added_for_mapping;

                        if (subscribers)
                            publishPointCloud_vux(all_lines, point_cloud_pub);
                    }

                    if (false) // map based extrinsic estimation
                    {
                        // save some data
                        if (curr_mls.translation().norm() < 1)
                            continue;

                        auto delta_predicted = (prev_mls.inverse() * curr_mls).log();
                        double scan_duration = curr_mls_time - prev_mls_time; // e.g., 0.1s
                        double tod_beg_scan = time_of_day_sec - scan_duration;
                        auto dt_tod = time_of_day_sec - tod_beg_scan;

                        for (size_t j = 0; j < vux_scans.size(); ++j)
                        {
                            pcl::PointCloud<VUX_PointType>::Ptr downsampled_line(new pcl::PointCloud<VUX_PointType>);
                            downSizeFilter_vux.setInputCloud(vux_scans[j]);
                            downSizeFilter_vux.filter(*downsampled_line);

                            const double t = vux_scans_time[j];
                            double alpha = (t - tod_beg_scan) / dt_tod;

                            Sophus::SE3 interpolated_pose_mls = prev_mls * Sophus::SE3::exp(alpha * delta_predicted);

                            _2d_measurements.push_back(downsampled_line);
                            known_T.push_back(interpolated_pose_mls); // to find extrinsics for vux 2 hesai

                            std::cout << "_2d_measurements:" << _2d_measurements.size() << std::endl;
                        }

                        if (curr_mls.translation().norm() > 65) //65 so fAr
                        {
                            std::cout << "\n\nStart extrinsic estimation...\n\n"
                                      << std::endl;

                            // set the current map to kdtree----------------------------------------
                            std::cout << "kdtree set input MLS points: " << laserCloudSurfMap->size() << std::endl;
                            const auto &reference_localMap_cloud = laserCloudSurfMap; //MLS map

                            // std::cout << "kdtree set input ALS points: " << als_obj->als_cloud->size() << std::endl;
                            // const auto &reference_localMap_cloud = als_obj->als_cloud;

                            //------------------------------------------------------------------------
                            estimator_.localKdTree_map->setInputCloud(reference_localMap_cloud);
                            const auto &refference_kdtree = estimator_.localKdTree_map;

                            // M3D R_rough = Eye3d; // from vux scanner to mls point cloud
                            V3D t_rough(0, 0, 0);
                            M3D R_rough; // from vux scanner to mls point cloud
                            R_rough << 0.0064031121, -0.8606533346, -0.5091510953,
                                -0.2586398121, 0.4904106092, -0.8322276624,
                                0.9659526116, 0.1370155907, -0.2194590626;
                            
                        
                            //t_rough = V3D(-0.2238580597, -3.0124498678, -0.8051626709);
                            
                            Sophus::SE3 vux2mls_extrinsics = Sophus::SE3(R_vux2mls, t_vux2mls); // refined - vux to mls cloud

                            std::vector<double> noise_levels_deg = {1, 5, 10, 20, 30, 40, 50, 60};
                            double noise_deg = 1; // noise_levels_deg[0];

                            //noise_deg = 5.;
                            noise_deg = 10.;
                            //noise_deg = 15.; // modified from here
                            //noise_deg = 20.;
                            //noise_deg = 25.; // added p2p from here
                            //noise_deg = 30.; // added bigger kernel from here
                            
                            //noise_deg = 35.;
                            //noise_deg = 40.;
                            //noise_deg = 45.;

                            std::cout << "noise_deg:" << noise_deg << std::endl;
                            
                            M3D noise = generate_euler_noise_rotation(noise_deg);
                            /////M3D R_noisy = R_rough * noise; 
                            M3D R_noisy = noise * R_rough; //this one is better
                            R_rough = R_noisy;

                            Sophus::SE3 vux2other_extrinsic = Sophus::SE3(R_rough, t_rough); // this will be refined

                            Eigen::Quaterniond q_extrinsic(vux2other_extrinsic.so3().matrix());
                            V3D t_extrinsic = vux2other_extrinsic.translation();

                            double prev_cost = std::numeric_limits<double>::max(); // Initialize with a large value
                            double current_cost = prev_cost;
                            double cost_threshold = .001; // Threshold for stopping criterion
                            auto prev_vux2other_extrinsic = vux2other_extrinsic;

                            double radius_Sq = 2.*2.;// 3. * 3;
                            bool use_radius = false;// true;
                            double init_radius_ = 2.; //2m

                            for (int iter_num = 0; iter_num < 500; iter_num++)
                            {
                                if (flg_exit || !ros::ok())
                                    break;

                                ceres::Problem problem;
                                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.3);

                                // Ensure the quaternion stays valid during optimization
                                ceres::LocalParameterization *q_parameterization =
                                    new ceres::EigenQuaternionParameterization();
                                int points_used_for_registration = 1;

                                double q_param[4] = {q_extrinsic.x(), q_extrinsic.y(), q_extrinsic.z(), q_extrinsic.w()};
                                double t_param[3] = {t_extrinsic.x(), t_extrinsic.y(), t_extrinsic.z()};

                                // Add the quaternion parameter block with the local parameterization
                                // to Ensure the quaternion stays valid during optimization
                                problem.AddParameterBlock(q_param, 4, q_parameterization);
                                problem.AddParameterBlock(t_param, 3); // Add the translation parameter block

                                bool subs = pubLaserCloudDebug.getNumSubscribers() != 0;

                                if (subs)
                                    feats_undistort->clear();
                                
            
                                // iterate the points, georeference with init guess and search for NN
                                for (int l = 0; l < _2d_measurements.size(); l++) // for each line
                                {
                                    const auto &fixed_pose = known_T[l];                  // copy of the init guess
                                    for (int i = 0; i < _2d_measurements[l]->size(); i++) // for each point in the line
                                    {
                                        V3D p_src(_2d_measurements[l]->points[i].x, _2d_measurements[l]->points[i].y, _2d_measurements[l]->points[i].z);
                                        V3D p_transformed = fixed_pose * vux2other_extrinsic * p_src;

                                        PointType search_point;
                                        search_point.x = p_transformed.x();
                                        search_point.y = p_transformed.y();
                                        search_point.z = p_transformed.z();

                                        if (subs)
                                            feats_undistort->push_back(search_point);

                                        if (use_radius)
                                        {
                                            std::vector<int> point_idx;
                                            std::vector<float> point_dist;
                                            if (refference_kdtree->radiusSearch(search_point, init_radius_, point_idx, point_dist) > 5)
                                            {
                                                int n_points = point_idx.size();

                                                V3D mean = V3D::Zero();
                                                for (int idx : point_idx) // add the sum of points from kdtree
                                                {
                                                    mean += V3D(reference_localMap_cloud->points[idx].x, reference_localMap_cloud->points[idx].y, reference_localMap_cloud->points[idx].z);
                                                }
                                                mean /= n_points;
                                                M3D cov = M3D::Zero();
                                                for (int idx : point_idx)
                                                {
                                                    V3D diff = V3D(reference_localMap_cloud->points[idx].x, reference_localMap_cloud->points[idx].y, reference_localMap_cloud->points[idx].z) - mean;
                                                    cov += diff * diff.transpose();
                                                }

                                                cov /= (n_points - 1);

                                                // Compute Eigenvalues and Eigenvectors
                                                Eigen::SelfAdjointEigenSolver<M3D> solver(cov);

                                                if (solver.info() != Eigen::Success)
                                                {
                                                    std::cerr << "Eigen solver failed!" << std::endl;
                                                    std::cout << "n_points:" << n_points << std::endl;
                                                    continue;
                                                    // throw std::runtime_error("Error: Eigen solver failed!");
                                                }

                                                V3D norm = solver.eigenvectors().col(0); // Smallest eigenvector
                                                norm.normalize();

                                                // Compute plane offset: d = - (n * mean)
                                                double d = -norm.dot(mean);

                                                // Compute eigenvalue ratios to assess planarity
                                                const auto &eigenvalues = solver.eigenvalues();
                                                double lambda0 = eigenvalues(0); // smallest
                                                double lambda1 = eigenvalues(1);
                                                double lambda2 = eigenvalues(2);

                                                double curvature = lambda0 / (lambda0 + lambda1 + lambda2);

                                                if (curvature > .0001 && curvature <= .03)
                                                {
                                                    ceres::CostFunction *cost_function = registration::LidarPlaneNormFactor_extrinsics::Create(p_src, norm, d, fixed_pose);
                                                    problem.AddResidualBlock(cost_function, loss_function, q_param, t_param);
                                                    points_used_for_registration++;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            std::vector<int> point_idx(5);
                                            std::vector<float> point_dist(5);
                                            if (refference_kdtree->nearestKSearch(search_point, 5, point_idx, point_dist) > 0) // there are neighbours
                                            {
                                                if (point_dist[4] < radius_Sq) // 1. not too far
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
                                                                 norm(2) * reference_localMap_cloud->points[point_idx[j]].z + negative_OA_dot_norm) > .1)
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
                                                    // else{
                                                    //     V3D closest = V3D(reference_localMap_cloud->points[point_idx[0]].x,
                                                    //                       reference_localMap_cloud->points[point_idx[0]].y,
                                                    //                       reference_localMap_cloud->points[point_idx[0]].z);

                                                    //     ceres::CostFunction *cost_function = registration::LidarPointFactor_extrinsics::Create(p_src, closest, fixed_pose);
                                                    //     problem.AddResidualBlock(cost_function, loss_function, q_param, t_param);
                                                    // }
                                                }
                                            }
                                        }
                                    }
                                }

                                if (subs)
                                {
                                    publish_frame_debug(pubLaserCloudDebug, feats_undistort);
                                    ros::spinOnce();
                                    rate.sleep();
                                }

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
                                auto cost_normalized = current_cost / points_used_for_registration;
                                std::cout << "Iteration " << iter_num << " - Cost: " << current_cost << ", normalized:" << cost_normalized << std::endl;

                                // Output the refined extrinsic transformation
                                q_extrinsic = Eigen::Quaterniond(q_param[3], q_param[0], q_param[1], q_param[2]);
                                t_extrinsic = V3D(t_param[0], t_param[1], t_param[2]);
                                vux2other_extrinsic = Sophus::SE3(q_extrinsic, t_extrinsic);
                                std::cout << "t_extrinsic:" << t_extrinsic.transpose() << std::endl;

                                double delta_state = (prev_vux2other_extrinsic.inverse() * vux2other_extrinsic).log().norm();
                                std::cout << "delta_state:" << delta_state << std::endl;

                                auto error_gt = (vux2other_extrinsic.inverse() * vux2mls_extrinsics).log().norm();
                                auto error_gt_rot = (vux2other_extrinsic.inverse() * vux2mls_extrinsics).so3().log().norm();
                                auto error_gt_tran = (vux2mls_extrinsics.translation() - t_extrinsic).norm();
                                std::cout << "error_gt:" << error_gt << ",  error_gt_rot:" << error_gt_rot<<", error_gt_tran:"<<error_gt_tran << std::endl;
                                
                                std::cout << "\n\nInit vux2mls_extrinsics rotation to Euler Angles (degrees): \n"
                                    << vux2mls_extrinsics.so3().matrix().eulerAngles(0, 1, 2).transpose() * 180.0 / M_PI << std::endl;
                                std::cout << "Found vux2other_extrinsic rotation to Euler Angles (degrees): \n"
                                    << vux2other_extrinsic.so3().matrix().eulerAngles(0, 1, 2).transpose() * 180.0 / M_PI << std::endl;
                                


                                if(error_gt < 1)
                                {
                                    init_radius_ = 1.;
                                    //radius_Sq = 1;
                                    // if(error_gt < 1)
                                    // {
                                    //     //init_radius_ = .5; //m
                                    //     //radius_Sq = .5*.5;
                                    // }
                                }
                                std::cout<<"init_radius_:"<<init_radius_<<std::endl;
                                std::cout<<"radius_Sq:"<<radius_Sq<<std::endl;


                                if (false) // save the data
                                {
                                    std::ofstream foutE("/home/eugeniu/z_z_e/extrinsic_test_" + std::to_string(noise_deg) + ".txt", std::ios::app);
                                    foutE.setf(std::ios::fixed, std::ios::floatfield);
                                    foutE.precision(20);
                                    // iter_num current_cost points_used_for_registration error_gt error_gt_rot
                                    foutE << iter_num << " " << current_cost << " " << points_used_for_registration << " " << error_gt << " " << error_gt_rot<<" "<<error_gt_tran << std::endl;
                                    foutE.close();

                                    const V3D t_model = vux2other_extrinsic.translation();
                                    Eigen::Quaterniond q_model(vux2other_extrinsic.so3().matrix());
                                    q_model.normalize();

                                    std::ofstream foutP("/home/eugeniu/z_z_e/pose_extrinsic_test_" + std::to_string(noise_deg) + ".txt", std::ios::app);
                                    foutP.setf(std::ios::fixed, std::ios::floatfield);
                                    foutP.precision(20);
                                    // ' tx ty tz qx qy qz qw' - tum format(scan id, scan timestamp seconds, translation and rotation quaternion)
                                    foutP << iter_num << " " << t_model(0) << " " << t_model(1) << " " << t_model(2) << " "
                                          << q_model.x() << " " << q_model.y() << " " << q_model.z() << " " << q_model.w() << std::endl;
                                    foutP.close();
                                }

                                // Check if the cost function change is small enough to stop
                                if (std::abs(prev_cost - current_cost) < cost_threshold)
                                {
                                    std::cout << "Stopping optimization: Cost change below threshold.\n";
                                    std::cout << "delta cost:" << std::abs(prev_cost - current_cost) << std::endl;
                                    break;
                                }

                                if (delta_state < .0001)
                                {
                                    std::cout << "Stopping optimization: delta_state too small.\n";
                                    std::cout << "delta_state:" << delta_state << std::endl;
                                    break;
                                }

                                prev_cost = current_cost;
                                prev_vux2other_extrinsic = vux2other_extrinsic;
                            }

                            std::cout << "Final transform is vux2other_extrinsic log:" << vux2other_extrinsic.log().transpose() << std::endl;
                            std::cout << "Final t_extrinsic :" << t_extrinsic.transpose() << std::endl;
                            std::cout << "Final vux2other_extrinsic translation  :" << vux2other_extrinsic.translation().transpose() << std::endl;
                            std::cout << "Final rotation:\n"
                                      << vux2other_extrinsic.so3().matrix() << std::endl;

                            throw std::runtime_error("Finished the extrinsic calibration");
                            break;
                        }
                    }
                }
#endif

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

                    //auto ppk_gnss_imu = als2mls * gnss_vux_data[tmp_index].se3;
                    //publish_ppk_gnss(ppk_gnss_imu, msg_time);

                    // auto ppk_gnss_imu = als2mls * gnss_vux_data[tmp_index].se3; // in mls frame
                    auto interpolated_pose = interpolateSE3(
                                                      gnss_vux_data[tmp_index].se3, gnss_vux_data[tmp_index].gps_tod,
                                                      gnss_vux_data[tmp_index + 1].se3, gnss_vux_data[tmp_index + 1].gps_tod,
                                                      time_of_day_sec);
                    auto ppk_gnss_imu = als2mls * interpolated_pose;
                    
                    //put it in the IMU position
                    V3D ext(0.042, 0.193, 0.326);  // X, Y, Z in meters

                    //auto t_ = ppk_gnss_imu.translation() - ext;
                    //ppk_gnss_imu = Sophus::SE3(ppk_gnss_imu.so3().matrix(), t_);
                    {
                        if(!ppk_gnss_init)
                        {
                            ppk_gnss_init = true;
                            first_ppk_gnss_pose_inverse = interpolated_pose;
                        }

                        std::cout<<"first_ppk_gnss_pose_inverse:\n"<<first_ppk_gnss_pose_inverse.matrix()<<std::endl;

                        auto first_t = first_ppk_gnss_pose_inverse.translation();
                        auto curr_t = interpolated_pose.translation();

                        ppk_gnss_imu = Sophus::SE3(interpolated_pose.so3().matrix(), curr_t-first_t);

                    }
                    publish_ppk_gnss(ppk_gnss_imu, msg_time);

                    // if(!als_obj->refine_als){
                    //     //als not refined and ppk_gnss_init not set
                    //     mls_poses.push_back(curr_mls);
                    //     gnss_poses.push_back(ppk_gnss_imu); //still in the global frame 
                    // }else if(!ppk_gnss_init){
                    //     //refine the als2mls 
                    //     std::cout << "refine the als2mls  press enter..." << std::endl;
                    //     std::cout<<"mls_poses:\n"<<mls_poses.size()<<std::endl;
                    //     als2mls = als_obj->als_to_mls;
                    //     std::cout<<"als2mls:\n"<<als2mls.matrix()<<std::endl;
                    //     for(auto &pose : gnss_poses)
                    //     {
                    //         pose = als2mls * pose;
                    //         publish_ppk_gnss(pose, 0);
                    //     }
                    //     std::cin.get(); 
                        
                    //     std::vector<V3D> lidar_positions, gnss_positions;
                    //         for (size_t i = 0; i < mls_poses.size(); ++i)
                    //         {
                    //             lidar_positions.push_back(mls_poses[i].translation());
                    //             gnss_positions.push_back(gnss_poses[i].translation()); 
                    //         }

                    //     auto T_extrinsic2 = estimateRigidTransformFromPoints(lidar_positions, gnss_positions);
                    //     als2mls = T_extrinsic2 * als2mls;
                    //     std::cout<<"T_extrinsic2:\n"<<T_extrinsic2.matrix()<<std::endl;

                    //     std::cout<<"als2mls after:\n"<<als2mls.matrix()<<std::endl;
                    //     als_obj->als_to_mls = als2mls;
                        
                    //     std::cout<<"FInished..."<<std::endl;
                    //     std::cin.get();
                    //     ppk_gnss_init = true;
                    // }
                    
                    // const bool global_error = false; // set this true for global error of gps
                    // //auto gps_cov_ = V3D(.01 * .01, .01 * .01, .01 * .01);
                    
                    // double stdev = gnss_vux_data[tmp_index].stdev;
                    // double stdev_2 = stdev * stdev;
                    // std::cout<<"ppk GNSS stdev:"<<stdev<<" (m), cov:"<<stdev_2<<std::endl;

                    // auto gps_cov_ = V3D(stdev_2, stdev_2, stdev_2);
                    
                    // // auto gps_cov_ = gnss_obj->gps_cov;
                    // //auto gps_cov_ = Eigen::Vector3d(GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR);
                    // V3D gnss_in_mls = ppk_gnss_imu.translation();

                    // estimator_.update(gnss_in_mls, gps_cov_, NUM_MAX_ITERATIONS, global_error);
                    // estimator_.update(LASER_POINT_COV, feats_down_body, laserCloudSurfMap, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en);
                    // estimator_.update(gnss_in_mls, gps_cov_, NUM_MAX_ITERATIONS, global_error);
                    

                    if (false) // ppk gnss as prior pose graph
                    {
                        std::cout << "ppk gnss as prior..." << std::endl;
                        // absolute covariance values
                        V3D prev_cov_pos = prev_P.block<3, 3>(P_ID, P_ID).diagonal();
                        V3D prev_cov_rot = prev_P.block<3, 3>(R_ID, R_ID).diagonal();

                        V3D curr_cov_pos = curr_P.block<3, 3>(P_ID, P_ID).diagonal();
                        V3D curr_cov_rot = curr_P.block<3, 3>(R_ID, R_ID).diagonal();

                        Eigen::Matrix<double, 6, 6> Sigma_prev;                   // Covariance of T_w_a
                        Sigma_prev.block<3, 3>(0, 0) = prev_cov_pos.asDiagonal(); // translation first
                        Sigma_prev.block<3, 3>(3, 3) = prev_cov_rot.asDiagonal(); // rotation second

                        Eigen::Matrix<double, 6, 6> Sigma_curr;                   // Covariance of T_w_b
                        Sigma_curr.block<3, 3>(0, 0) = curr_cov_pos.asDiagonal(); // translation first
                        Sigma_curr.block<3, 3>(3, 3) = curr_cov_rot.asDiagonal(); // rotation second

                        Sophus::SE3 T_MLS_relative = prev_mls.inverse() * curr_mls;
                        // Adjoint of T_MLS_relative
                        Eigen::Matrix<double, 6, 6> Ad_Tab = T_MLS_relative.Adj();
                        /*
                            T_ab​=T_a^{−1}​⋅T_b
                            Σ_ab​=J_a​Σ_a​J_a^{⊤}​ + J_b​Σ_b​J_b^{⊤}

                            J_a​=−Adj(T_ab​)
                            J_b=I
                        */
                        // Jacobians
                        Eigen::Matrix<double, 6, 6> J_a = -Ad_Tab;
                        Eigen::Matrix<double, 6, 6> J_b = Eigen::Matrix<double, 6, 6>::Identity();

                        Eigen::Matrix<double, 6, 6> Sigma_relative =
                            J_a * Sigma_prev * J_a.transpose() +
                            J_b * Sigma_curr * J_b.transpose();

                        // std::cout << "\n\nSigma_prev:\n"
                        //           << Sigma_prev.diagonal().transpose() << std::endl;
                        // std::cout << "Sigma_curr:\n"
                        //           << Sigma_curr.diagonal().transpose() << std::endl;
                        // std::cout << "\nSigma_relative:\n"
                        //           << Sigma_relative.diagonal().transpose() << std::endl;

                        // Eigen::Matrix<double, 6, 6> Sigma_curr_check = Sigma_relative - Ad_Tab * Sigma_prev * Ad_Tab.transpose();
                        // std::cout << "\nSigma_curr_check:\n"<< Sigma_curr_check.diagonal().transpose() << std::endl;
                        // std::cout << "Error:\n"<< (Sigma_curr_check - Sigma_curr).norm() << std::endl;

                        V3D rel_pos_cov = Sigma_relative.block<3, 3>(0, 0).diagonal();
                        V3D rel_rot_cov = Sigma_relative.block<3, 3>(3, 3).diagonal();

                        // take the absolute covariance now from the MLS - TODO - get it from the GNSS
                        // V3D abs_pos_cov = V3D(1, 1, 1); // scale * Sigma_curr.block<3, 3>(0, 0).diagonal();
                        // V3D abs_rot_cov = V3D(1, 1, 1); // scale * Sigma_curr.block<3, 3>(3, 3).diagonal();

                        double scale = 1000; // 10;

                        V3D abs_pos_cov = scale * Sigma_curr.block<3, 3>(0, 0).diagonal();
                        V3D abs_rot_cov = scale * Sigma_curr.block<3, 3>(3, 3).diagonal();

                        // current GNSS absolute pose:  ppk_gnss_imu
                        // current relative transform of MLS: T_MLS_relative

                        auto fused_result = gnss_MLS_fusion::updateISam(
                            T_MLS_relative, rel_pos_cov, rel_rot_cov,
                            ppk_gnss_imu, abs_pos_cov, abs_rot_cov);

                        const Sophus::SE3 &fused_T = fused_result.first;
                        const Eigen::Vector6d &fused_cov = fused_result.second;

                        publish_refined_ppk_gnss(fused_T, msg_time);

                        // V3D rot_var = fused_cov.head<3>();   // [σ²_roll, σ²_pitch, σ²_yaw]
                        V3D trans_var = fused_cov.tail<3>(); // [σ²_x, σ²_y, σ²_z]

                        std::cout << "before update:" << Sigma_curr.block<3, 3>(0, 0).diagonal().transpose() << std::endl;
                        std::cout << "after  update:" << trans_var.transpose() << std::endl;

                        V3D gnss_in_mls = fused_T.translation();

                        // update the local estimator
                        estimator_.update(gnss_in_mls, trans_var, NUM_MAX_ITERATIONS, false);
                    }

                    if (false) // raw gnss as prior pose graph
                    {
                        // we need gps_local_i = R_ENU_to_local * (gps_i - gps_0)

                        std::cout << "raw gnss as prior..." << std::endl;
                        // absolute covariance values
                        V3D prev_cov_pos = prev_P.block<3, 3>(P_ID, P_ID).diagonal();
                        V3D prev_cov_rot = prev_P.block<3, 3>(R_ID, R_ID).diagonal();

                        V3D curr_cov_pos = curr_P.block<3, 3>(P_ID, P_ID).diagonal();
                        V3D curr_cov_rot = curr_P.block<3, 3>(R_ID, R_ID).diagonal();

                        Eigen::Matrix<double, 6, 6> Sigma_prev;                   // Covariance of T_w_a
                        Sigma_prev.block<3, 3>(0, 0) = prev_cov_pos.asDiagonal(); // translation first
                        Sigma_prev.block<3, 3>(3, 3) = prev_cov_rot.asDiagonal(); // rotation second

                        Eigen::Matrix<double, 6, 6> Sigma_curr;                   // Covariance of T_w_b
                        Sigma_curr.block<3, 3>(0, 0) = curr_cov_pos.asDiagonal(); // translation first
                        Sigma_curr.block<3, 3>(3, 3) = curr_cov_rot.asDiagonal(); // rotation second

                        Sophus::SE3 T_MLS_relative = prev_mls.inverse() * curr_mls;

                        Eigen::Matrix<double, 6, 6> Ad_Tab = T_MLS_relative.Adj(); // Adjoint of T_MLS_relative
                        // Jacobians
                        Eigen::Matrix<double, 6, 6> J_a = -Ad_Tab;
                        Eigen::Matrix<double, 6, 6> J_b = Eigen::Matrix<double, 6, 6>::Identity();

                        Eigen::Matrix<double, 6, 6> Sigma_relative =
                            J_a * Sigma_prev * J_a.transpose() +
                            J_b * Sigma_curr * J_b.transpose();

                        V3D rel_pos_cov = Sigma_relative.block<3, 3>(0, 0).diagonal();
                        V3D rel_rot_cov = Sigma_relative.block<3, 3>(3, 3).diagonal();

                        //---------------------------------------------------------------------------

                        // diff_curr_gnss2mls = gps_time - lidar_end_time; //gps_time = diff_curr_gnss2mls + lidar_end_time
                        double raw_gnss_time_prev = prev_mls_time + prev_raw_gnss_diff_time;
                        double raw_gnss_time_curr = curr_mls_time + curr_raw_gnss_diff_time;

                        // auto raw_gnss = gnss_obj->gps_pose;

                        auto raw_gnss = Sophus::SE3(als2mls.so3(), V3D(0, 0, 0)) *           // rotate to mls
                                        Sophus::SE3(Eye3d, gnss_obj->origin_enu).inverse() * // move to frame of first position
                                        interpolateSE3(                                      // interpolate to get the pose
                                            Sophus::SE3(Eye3d, prev_enu), raw_gnss_time_prev,
                                            Sophus::SE3(Eye3d, curr_enu), raw_gnss_time_curr,
                                            curr_mls_time);

                        // apply the approximate extrinsic to transform to mls FRAME
                        raw_gnss = raw_gnss * Sophus::SE3(Eye3d, V3D(-1.7, 0.07, 0.08));

                        // Next is not good directly, since raw GNSS is not in the frame of ALS cloud
                        // auto raw_gnss = Sophus::SE3(als2mls.so3(), als2mls.translation()) * // transform to mls
                        //                 interpolateSE3(                                     // interpolate to get the pose
                        //                     Sophus::SE3(Eye3d, prev_enu), raw_gnss_time_prev,
                        //                     Sophus::SE3(Eye3d, curr_enu), raw_gnss_time_curr,
                        //                     curr_mls_time);

                        // // apply the approximate extrinsic to transform to mls FRAME - not same as ALS raw GNSS
                        // raw_gnss = raw_gnss * Sophus::SE3(Eye3d, V3D(0, .208, 18.4817014440));
                        // log_gnss_lidar_relative.push_back((curr_mls.inverse() * raw_gnss).log());

                        // Sophus::SE3 lidar2gnss_extrinsic = averageSE3Log(log_gnss_lidar_relative);
                        // std::cout << "\nAveraged gnss 2 lidar SE3:\n"
                        //               << lidar2gnss_extrinsic.inverse().matrix() << std::endl;

                        // auto lidar2gnss_extrinsic = curr_mls.inverse() * raw_gnss;
                        // std::cout<<"lidar2gnss_extrinsic:\n"<<lidar2gnss_extrinsic.matrix()<<std::endl;

                        raw_gnss = gnss_pose;             // thiw will take the ppk gnss
                        raw_gnss.so3() = state_point.rot; // use the MLS orientation

                        publish_gnss_odometry(raw_gnss);

                        // V3D gps_variances = gnss_obj->gps_cov;
                        //  double scale = 100;
                        //  V3D abs_pos_cov = gnss_obj->gps_cov;// scale * Sigma_curr.block<3, 3>(0, 0).diagonal();
                        //  V3D abs_rot_cov = scale * Sigma_curr.block<3, 3>(3, 3).diagonal();

                        double scale = 100; // all done with scale 100
                        V3D abs_pos_cov = scale * Sigma_curr.block<3, 3>(0, 0).diagonal();
                        V3D abs_rot_cov = scale * Sigma_curr.block<3, 3>(3, 3).diagonal();

                        // current raw GNSS absolute pose:  raw_gnss
                        // current relative transform of MLS: T_MLS_relative
                        auto fused_result = gnss_MLS_fusion::updateISam(
                            T_MLS_relative, rel_pos_cov, rel_rot_cov,
                            raw_gnss, abs_pos_cov, abs_rot_cov);

                        const Sophus::SE3 &fused_T = fused_result.first;
                        const Eigen::Vector6d &fused_cov = fused_result.second;

                        publish_refined_ppk_gnss(fused_T, msg_time);

                        // V3D rot_var = fused_cov.head<3>();  // [σ²_roll, σ²_pitch, σ²_yaw]
                        V3D trans_var = fused_cov.tail<3>(); // [σ²_x, σ²_y, σ²_z]

                        std::cout << "before update:" << Sigma_curr.block<3, 3>(0, 0).diagonal().transpose() << ", norm:" << Sigma_curr.block<3, 3>(0, 0).diagonal().norm() << std::endl;
                        std::cout << "after  update:" << trans_var.transpose() << ", norm:" << trans_var.norm() << std::endl;

                        V3D gnss_in_mls = fused_T.translation();

                        // update the local estimator
                        estimator_.update(gnss_in_mls, trans_var, NUM_MAX_ITERATIONS, false);
                    }

                    if (false) // perform motion-based extrinsic
                    {
                        mls_poses.push_back(curr_mls);
                        gnss_poses.push_back(ppk_gnss_imu);
                        gnss_poses_original.push_back(gnss_vux_data[tmp_index].se3);

                        mls_times.push_back(time_of_day_sec);
                        gnss_times.push_back(msg_time);

                        // feats_down_body is points in lidar frame

                        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
                        *cloud = *feats_down_body;
                        mls_clouds.push_back(cloud); // save the raw mls clouds

                        std::cout << "curr_mls    :" << curr_mls.translation().transpose() << std::endl;
                        std::cout << "ppk_gnss_imu:" << ppk_gnss_imu.translation().transpose() << std::endl;
                        std::cout << "log:" << (curr_mls.inverse() * ppk_gnss_imu).log().transpose() << std::endl;
                        // from lidar to mls
                        // auto lidar2gnss_extrinsic = curr_mls.inverse() * ppk_gnss_imu;
                        log_gnss_lidar_relative.push_back((curr_mls.inverse() * ppk_gnss_imu).log());

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
                            // std::vector<double> all_deltas = {.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.}; //s
                            std::vector<double> all_deltas = {.1, .5, 1., 2., 3., 4., 5., -1}; // s
                            std::vector<Eigen::Vector6d> relative_logs;
                            // add absolute poses
                            double delta_t = .01;
                            double t_start = std::max(mls_times.front(), gnss_times.front());
                            double t_end = std::min(mls_times.back(), gnss_times.back());
                            double total_duration = t_end - t_start;
                            for (double t = t_start; t <= t_end; t += delta_t)
                            {
                                if (t > mls_times.back() || t > gnss_times.back())
                                    break;

                                Sophus::SE3 interpA = interpolateSE3Log(mls_poses, mls_times, t);
                                Sophus::SE3 interpB = interpolateSE3Log(gnss_poses, gnss_times, t);

                                relative_logs.push_back((interpA.inverse() * interpB).log());
                            }

                            auto all_trajectories = generateInterpolatedPoses(mls_poses, gnss_poses, mls_times, gnss_times, all_deltas);

                            for (size_t i = 0; i < all_trajectories.size(); ++i)
                            {
                                double delta = all_deltas[i];
                                std::cout << "Δt = " << delta << "s: " << all_trajectories[i].first.size() << " samples\n";

                                int N = all_trajectories[i].first.size();
                                for (int j = 1; j < N; j++)
                                {
                                    auto Ai = all_trajectories[i].first[j - 1].inverse() * all_trajectories[i].first[j];
                                    auto Bi = all_trajectories[i].second[j - 1].inverse() * all_trajectories[i].second[j];

                                    A_rel.push_back(Ai);
                                    B_rel.push_back(Bi);
                                }
                            }

                            std::cout << "A_rel:" << A_rel.size() << ", B_rel:" << B_rel.size() << std::endl;

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

                            // ppk_gnss_imu = ppk_gnss_imu * lidar2gnss_extrinsic.inverse();

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

                    if (false) // perform motion-based extrinsic
                    {
                        mls_poses.push_back(curr_mls);
                        gnss_poses.push_back(ppk_gnss_imu);
                        gnss_poses_original.push_back(gnss_vux_data[tmp_index].se3);

                        mls_times.push_back(time_of_day_sec);
                        gnss_times.push_back(msg_time);
                        std::cout << "mls_poses:" << mls_poses.size() << std::endl;
                        std::cout << "curr_mls    :" << curr_mls.translation().transpose() << std::endl;
                        std::cout << "ppk_gnss_imu:" << ppk_gnss_imu.translation().transpose() << std::endl;

                        auto T_extrinsic_i = curr_mls * ppk_gnss_imu.inverse();
                        // auto T_extrinsic_i = curr_mls.inverse() * ppk_gnss_imu;

                        log_gnss_lidar_relative.push_back(T_extrinsic_i.log());
                        std::cout << "log:" << T_extrinsic_i.log().transpose() << std::endl;

                        int N_data = 400;

                        auto gnss_pose = Sophus::SE3(curr_mls.so3().matrix(), T_extrinsic * (gnss_vux_data[tmp_index].se3.translation()));

                        publish_gnss_odometry(gnss_pose);

                        if (mls_poses.size() == N_data) // got enough poses
                        {
                            // Extract positions from poses
                            std::vector<Eigen::Vector3d> lidar_positions, gnss_positions;
                            for (size_t i = 0; i < mls_poses.size(); ++i)
                            {
                                lidar_positions.push_back(mls_poses[i].translation());
                                gnss_positions.push_back(gnss_poses_original[i].translation()); // or your GNSS positions
                            }

                            auto T_extrinsic2 = estimateRigidTransformFromPoints(lidar_positions, gnss_positions);
                            std::cout << "T_extrinsic2:\n"
                                      << T_extrinsic2.matrix() << std::endl;

                            // Now transform GNSS-IMU positions into LiDAR frame:
                            std::cout << "test, gnss_positions[N_data-1]:" << gnss_positions[N_data - 1].transpose() << std::endl;
                            std::cout << "test, lidar_positions[N_data-1]:" << lidar_positions[N_data - 1].transpose() << std::endl;

                            V3D lidar_estimated_position = T_extrinsic2 * gnss_positions[N_data - 1];
                            std::cout << "test, lidar_estimated_position:" << lidar_estimated_position.transpose() << std::endl;

                            T_extrinsic = T_extrinsic2;

                            // todo

                            // compute the error between the trajectories first and show it
                            // estimate the transformation
                            // transform the gnss into mls
                            // compute the error again
                            // add the known extrinsics
                            // compute the error again

                            //std::cout << "Start extrinsic estimation press enter..." << std::endl;
                            //std::cin.get();

                            // Sophus::SE3 T_extrinsic_i = averageSE3Log(log_gnss_lidar_relative);
                            // std::cout << "\nAveraged T_extrinsic_i SE3:\n"
                            //           << T_extrinsic_i.matrix() << std::endl;

                            //T_extrinsic = T_extrinsic_i;
                            

                            // gnss_obj->calibrateGnssExtrinsic(mls_poses, gnss_poses);
                            // T_extrinsic = Sophus::SE3(gnss_obj->R_GNSS_to_MLS, Zero3d);

                            std::cout << "Finished extrinsic estimation press enter..." << std::endl;
                            std::cin.get();
                        }
                    }
                    break;
                }

#endif

                // Update the local map--------------------------------------------------
                feats_down_world->resize(feats_down_size);
                local_map_update(); // this will update local map with curr measurements and crop the map

#ifdef SAVE_DATA
std::cout<<"save_poses:"<<save_poses<<", save_clouds_path:"<<save_clouds_path<<std::endl;

                if (als_integrated)
                {
                    if (save_clouds)
                    {
                        switch (lidar_type)
                        {
                        case Hesai:
                            std::cout << "Hesai save the cloud" << std::endl;
                            {
                                const pcl::PointCloud<hesai_ros::Point> &pl_orig = imu_obj->DeSkewOriginalCloud<hesai_ros::Point>(Measures.lidar_msg, state_point, save_clouds_local);
                                std::cout << "save " << pl_orig.size() << " points" << std::endl;

                                std::string filename = save_clouds_path + "Hesai/" + std::to_string(pcd_index) + "_cloud_" + std::to_string(lidar_end_time) + ".pcd";
                                pcl::io::savePCDFile(filename, pl_orig, true); // Binary format
                            }
                            break;

                        default:
                            std::cout << "Unknown LIDAR type:" << lidar_type << std::endl;
                            break;
                        }

#ifdef integrate_vux
                        auto delta_predicted = (prev_mls.inverse() * curr_mls).log();
                        double scan_duration = curr_mls_time - prev_mls_time; // e.g., 0.1s
                        double tod_beg_scan = time_of_day_sec - scan_duration;
                        auto dt_tod = time_of_day_sec - tod_beg_scan;

                        for (size_t j = 0; j < vux_scans.size(); ++j)
                        {
                            const double t = vux_scans_time[j];
                            double alpha = (t - tod_beg_scan) / dt_tod;

                            Sophus::SE3 interpolated_pose_mls = prev_mls * Sophus::SE3::exp(alpha * delta_predicted);
                            Sophus::SE3 vux_pose = interpolated_pose_mls * vux2mls_extrinsics;

                            TransformPoints(vux_pose, vux_scans[j]);

                            std::string filename = save_clouds_path + "VUX/" + std::to_string(pcd_index) + "_cloud_" + std::to_string(pcd_vux_index) + "_time_" + std::to_string(t) + ".pcd";
                            pcl::io::savePCDFile(filename, *vux_scans[j], true); // Binary format
                            pcd_vux_index++;
                        }

#endif
                    }
                    
                    if (save_poses) // this will save the MLS estimated SE3 poses
                    {
                        const V3D &t_model = state_point.pos;
                        Eigen::Quaterniond q_model(state_point.rot.matrix());
                        
                        q_model.normalize();
                        std::ofstream foutMLS(poseSavePath + "MLS.txt", std::ios::app);
                        //std::ofstream foutMLS(save_clouds_path + "MLS.txt", std::ios::app);
                        // foutMLS.setf(std::ios::scientific, std::ios::floatfield);
                        foutMLS.setf(std::ios::fixed, std::ios::floatfield);
                        foutMLS.precision(20);
                        // # ' id time tx ty tz qx qy qz qw' - tum format(scan id, scan timestamp seconds, translation and rotation quaternion)
                        foutMLS << pcd_index << " " << std::to_string(lidar_end_time) << " " << t_model(0) << " " << t_model(1) << " " << t_model(2) << " "
                                << q_model.x() << " " << q_model.y() << " " << q_model.z() << " " << q_model.w() << std::endl;
                        foutMLS.close();
                    }

                    pcd_index++;
                }
#endif

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

                std::cout<<"extrinsic R:\n"<<state_point.offset_R_L_I.matrix()<<std::endl;
                std::cout<<"extrinsic t:"<<state_point.offset_T_L_I.transpose()<<std::endl;

                als2mls = als_obj->als_to_mls;
                // std::cout<<"Debug als2mls:\n"<<als2mls.matrix()<<std::endl;
            }
        }else{
            //std::cout<<"Sync failed..."<<std::endl;
        }
    }
    // bag.close();
    for (auto &b : bags)
        b->close();
    // cv::destroyAllWindows();
}
