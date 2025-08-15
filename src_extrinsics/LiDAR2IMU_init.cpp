
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
#include <GeographicLib/UTMUPS.hpp>
#include <visualization_msgs/Marker.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/expressions.h>
#include <gtsam/navigation/ImuFactor.h>

#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

volatile bool flg_exit = false;
void SigHandle(int sig)
{
    ROS_WARN("Caught signal %d, stopping...", sig);
    flg_exit = true; // Set the flag to stop the loop
}

void publishPose(const Sophus::SE3 &_pose, const double &msg_time)
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

    // tf::Transform transform_inv = transformGPS.inverse();
    // static tf::TransformBroadcaster br2;
    // br2.sendTransform(tf::StampedTransform(transform_inv, ros::Time().fromSec(msg_time), "PPK_GNSS", "world"));
}

using gtsam::symbol_shorthand::R; // Rotation key shorthand
using namespace gtsam;

/// GTSAM Factor
class HECFactor : public gtsam::NoiseModelFactor1<gtsam::Rot3>
{
private:
    gtsam::Point3 m_axis_I_;
    gtsam::Point3 m_axis_L_;

public:
    HECFactor(gtsam::Key i, gtsam::Point3 axis_I, gtsam::Point3 axis_L, const gtsam::SharedNoiseModel &model)
        : gtsam::NoiseModelFactor1<gtsam::Rot3>(model, i), m_axis_I_(axis_I), m_axis_L_(axis_L) {}

    // gtsam::Vector evaluateError(const gtsam::Rot3 &I_R_L, boost::optional<gtsam::Matrix &> H = boost::none) const
    gtsam::Vector evaluateError(const gtsam::Rot3 &I_R_L, OptionalMatrixType H) const override
    {
        gtsam::Matrix H_Rp_R, H_Rp_p;
        gtsam::Point3 error = m_axis_I_ - I_R_L.rotate(m_axis_L_, H_Rp_R, H_Rp_p);
        if (H)
            (*H) = (-H_Rp_R).eval();

        // if (H)
        //(*H) = (gtsam::Matrix(3, 3) << -H_Rp_R).finished();
        return error; // return (gtsam::Vector(3) << error.x(), error.y(), error.z()).finished();
    }
};

void DataHandler::Subscribe()
{
    std::cout << "Run test" << std::endl;
    std::cout << std::fixed << std::setprecision(12);
    std::cerr << std::fixed << std::setprecision(12);

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
    ros::Rate rate(500);

    int scan_id = 0, vux_scan_id = 0;

    Sophus::SE3 prev_mls, curr_mls;
    double prev_mls_time, curr_mls_time;
    //----------------------------------------------------------------------------

    using namespace std::chrono;
    bool has_prev_cloud_ = false;
    sensor_msgs::ImuConstPtr last_imu_;
    PointCloudXYZI::Ptr prev_cloud_;
    prev_cloud_.reset(new PointCloudXYZI());

    Sophus::SE3 global_pose_ = Sophus::SE3();          // Global pose
    Sophus::SE3 last_motion_estimate_ = Sophus::SE3(); // Constant velocity model

    float imuAccNoise = 0.1; // 0.01;
    float imuGyrNoise = 0.1; // 0.001;
    float imuAccBiasN = 0.0002;
    float imuGyrBiasN = 0.00003;

    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt;
    int no_of_frames = 0;
    int max_frames = 250;
    std::string calibration_result_filename;

    /// GTSAM
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial_values;
    gtsam::noiseModel::Diagonal::shared_ptr rotationNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector3(1, 1, 1)));

    double imuGravity = 9.81;
    // boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
    std::shared_ptr<gtsam::PreintegrationParams> p;
    p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);

    p->accelerometerCovariance = gtsam::Matrix33::Identity() * pow(imuAccNoise, 2);
    p->gyroscopeCovariance = gtsam::Matrix33::Identity() * pow(imuGyrNoise, 2);
    p->integrationCovariance = gtsam::Matrix33::Identity() * pow(1e-4, 2);

    gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());
    imuIntegratorOpt = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);

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

            // scan registration here
            {
                *feats_undistort = *Measures.lidar;
                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(*feats_down_body);

                if (!has_prev_cloud_)
                {
                    *prev_cloud_ = *feats_down_body;
                    has_prev_cloud_ = true;
                    last_imu_ = Measures.imu.back();
                    continue;
                }

                // --- ICP Setup ---
                pcl::IterativeClosestPoint<PointType, PointType> icp;
                icp.setMaximumIterations(50);
                icp.setMaxCorrespondenceDistance(1.0);
                icp.setInputSource(feats_down_body);
                icp.setInputTarget(prev_cloud_);

                // --- Constant velocity model as initial guess ---
                Eigen::Matrix4f init_guess = last_motion_estimate_.matrix().cast<float>();

                PointCloudXYZI::Ptr aligned(new PointCloudXYZI());
                icp.align(*aligned, init_guess);

                if (!icp.hasConverged())
                {
                    ROS_WARN("ICP did not converge!");
                    return;
                }

                // --- Get relative transform from ICP ---
                Eigen::Matrix4f relative_transform_f = icp.getFinalTransformation();
                Eigen::Matrix4d relative_transform = relative_transform_f.cast<double>();

                Sophus::SE3 relative_motion(relative_transform.block<3, 3>(0, 0),
                                            relative_transform.block<3, 1>(0, 3));

                // --- Update global pose ---
                global_pose_ = global_pose_ * relative_motion;

                // --- Update constant velocity model ---
                last_motion_estimate_ = relative_motion;

                publish_frame_debug(pubLaserCloudDebug, feats_undistort);

                publishPose(global_pose_, lidar_end_time);

                *prev_cloud_ = *feats_down_body;
            }

            // imu preintregration
            {
                auto v_imu = Measures.imu;
                v_imu.push_front(last_imu_);
                int stamp_size = v_imu.size();
                std::cout << "IMU measurements " << stamp_size << std::endl;

                const double &imu_end_time = v_imu.back()->header.stamp.toSec();
                const double &imu_beg_time = v_imu.front()->header.stamp.toSec();

                const double &pcl_beg_time = Measures.lidar_beg_time;
                const double &pcl_end_time = Measures.lidar_end_time;

                double dt_lidar = pcl_end_time - pcl_beg_time;
                double dt_imu = imu_end_time - imu_beg_time;

                std::cout << "dt_lidar:" << dt_lidar << ", dt_imu:" << dt_imu << std::endl;

                V3D pos_imu, vel_imu, angvel_avr, acc_avr, acc_imu;
                M3D R_imu;

                for (int i = 1; i < stamp_size; i++)
                {
                    const auto &head = v_imu[i];
                    const auto &tail = v_imu[i - 1];

                    auto dt = head->header.stamp.toSec() - tail->header.stamp.toSec();

                    angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                        0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                        0.5 * (head->angular_velocity.z + tail->angular_velocity.z);

                    acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                        0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                        0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

                    gtsam::Vector3 acc = gtsam::Vector3(acc_avr[0], acc_avr[1], acc_avr[2]);
                    gtsam::Vector3 omega = gtsam::Vector3(angvel_avr[0], angvel_avr[1], angvel_avr[2]);

                    imuIntegratorOpt->integrateMeasurement(acc, omega, dt);
                }

                last_imu_ = Measures.imu.back();

                // Get integrated rotation from IMU
                M3D deltaR_I = imuIntegratorOpt->deltaRij().matrix();
                imuIntegratorOpt->resetIntegration();

                // delta lidar
                M3D deltaR_L = last_motion_estimate_.so3().matrix();

                //todo -> take into account the time in between for each delta R 
                
                V3D axisAngle_lidar;
                V3D axisAngle_imu;
                ceres::RotationMatrixToAngleAxis(deltaR_L.data(), axisAngle_lidar.data());
                ceres::RotationMatrixToAngleAxis(deltaR_I.data(), axisAngle_imu.data());

                /// GTSAM stuff
                graph.add(std::make_shared<HECFactor>(R(0), gtsam::Point3(axisAngle_imu.x(), axisAngle_imu.y(), axisAngle_imu.z()),
                                                        gtsam::Point3(axisAngle_lidar.x(), axisAngle_lidar.y(), axisAngle_lidar.z()), rotationNoise));
                ROS_INFO_STREAM("Frame: " << no_of_frames << " / " << max_frames);
                std::cout << "frame : " << no_of_frames << "/" << max_frames << std::endl;

                if (no_of_frames == max_frames)
                {
                    gtsam::Rot3 priorRot = gtsam::Rot3();
                    initial_values.insert(R(0), priorRot);
                    gtsam::Values result = gtsam::LevenbergMarquardtOptimizer(graph, initial_values).optimize();
                    gtsam::Rot3 finalResult = result.at<gtsam::Rot3>(R(0));
                    gtsam::Marginals marginals(graph, result);

                    std::cout << "Rotation matrix:\n" << finalResult.matrix() << std::endl;
                    std::cout << "Euler angles (deg): " << finalResult.matrix().eulerAngles(0, 1, 2).transpose() * 180.0 / M_PI << std::endl;
                    std::cout << "Marginal Covariance:\n" << marginals.marginalCovariance(R(0)) << std::endl;

                    // std::ofstream result_file(calibration_result_filename);
                    // Eigen::Matrix4d I_T_L = Eigen::Matrix4d::Identity();
                    // I_T_L.block<3,3>(0,0) = finalResult.matrix();
                    // result_file << I_T_L;
                    // result_file.close();
                    break;
                    ros::shutdown();
                }
                no_of_frames++;
            }
        }
    }
    bag.close();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mls");
    ros::NodeHandle nh;
    std::shared_ptr<DataHandler> dh = std::make_shared<DataHandler>(nh);

    dh->Subscribe();

    ros::spin();
    ros::shutdown();
    return 0;
}