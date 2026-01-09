

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
    arrow.scale.x = 0.3;
    arrow.scale.y = 0.3;
    arrow.scale.z = 0.3;

    arrow.color.r = 1.0; // Full red
    arrow.color.g = 0.5; // Medium green
    arrow.color.b = 0.0; // No blue
    arrow.color.a = 1.0; // Fully opaque

    marker_pub.publish(arrow);
}

#include "TrajectoryReader.hpp"
#include "Batch.hpp"

//-------------------------------------------
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/expressions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace gtsam;

using gtsam::symbol_shorthand::R; // Rotation key shorthand
using gtsam::symbol_shorthand::X; // extrinsic key X(0)

using Matrix6 = Eigen::Matrix<double, 6, 6>;
// Solves AX = XB problem
namespace motion_based_extrinsic_estimation
{
    // Factor constraining extrinsic X so that:
    // delta_I_meas ≈ X * delta_L_meas * X^{-1}
    class ExtrinsicSE3Factor : public NoiseModelFactor1<Pose3>
    {
    private:
        Pose3 delta_I_;
        Pose3 delta_L_;

    public:
        ExtrinsicSE3Factor(Key key, const Pose3 &delta_I, const Pose3 &delta_L, const SharedNoiseModel &model)
            : NoiseModelFactor1<Pose3>(model, key), delta_I_(delta_I), delta_L_(delta_L) {}

        Vector evaluateError(const Pose3 &T_I_L, OptionalMatrixType H) const override
        {
            // predicted delta_I from lidar delta and extrinsic
            Pose3 predicted = T_I_L * delta_L_ * T_I_L.inverse();

            // error = Logmap( delta_I^{-1} * predicted ) = Pose3::Logmap(delta_I.between(predicted))
            Pose3 errPose = delta_I_.between(predicted);
            Vector error = Pose3::Logmap(errPose); // 6x1 (rot(3), trans(3))

            if (H)
            {
                // numerical Jacobian for simplicity/robustness
                auto fun = [this](const Pose3 &X) -> Vector
                {
                    Pose3 pred = X * delta_L_ * X.inverse();
                    Pose3 e = delta_I_.between(pred);
                    return Pose3::Logmap(e);
                };
                *H = numericalDerivative11<Vector, Pose3>(fun, T_I_L, 1e-6);
            }
            return error;
        }
    };

    // define a custom point to point factor

    // save the scans in lidar frame,

    // pass the scans, imu-lidar extrinsic, and the topic publisher

    // transform the scans and publish the init velues

    // then do it iterativelly

    Sophus::SE3 GtsamToSophus(const Pose3 &pose)
    {
        Eigen::Matrix4d T = pose.matrix(); // Get 4x4 matrix
        return Sophus::SE3(T.topLeftCorner<3, 3>(), T.topRightCorner<3, 1>());
    }

    Sophus::SE3 find_extrinsic_motion_based(Sophus::SE3 &init_guess_,
                                            const std::vector<Sophus::SE3> &lidar_poses_, const std::vector<Sophus::SE3> &gnss_imu_poses_)
    {
        std::vector<Sophus::SE3> A_, B_;

        NonlinearFactorGraph graph;
        Values initial;

        double meas_sigma = .5; // rad/m synthetic measurement sigma
        auto base_noise = noiseModel::Isotropic::Sigma(6, meas_sigma);

        int N = gnss_imu_poses_.size();
        int added = 0;
        for (size_t i = 0; i + 1 < N; ++i)
        {
            auto delta_L = lidar_poses_[i].inverse() * lidar_poses_[i + 1];
            auto delta_I = gnss_imu_poses_[i].inverse() * gnss_imu_poses_[i + 1];

            double trans_motion = delta_I.translation().norm();
            double rot_motion = Sophus::SO3::log(delta_I.so3()).norm(); // radians

            if (trans_motion < 0.01 && rot_motion < (1.0 * M_PI / 180.0)) // 1 cm and 1 degree
            {
                // skip static or nearly static pairs
                // continue;
            }

            Pose3 delta_L_true = Pose3(delta_L.matrix());
            Pose3 delta_I_true = Pose3(delta_I.matrix());

            // auto huber = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(1.345), base_noise);
            // graph.add(std::make_shared<ExtrinsicSE3Factor>(X(0), delta_I_true, delta_L_true, huber));         //robust

            graph.add(std::make_shared<ExtrinsicSE3Factor>(X(0), delta_I_true, delta_L_true, base_noise)); // not robust

            A_.push_back(delta_L);
            B_.push_back(delta_I);
            added++;
        }
        if (true)
        {
            for (int step = 2; step < 4; step++)
            {
                for (size_t i = 0; i < N - 1; i += step)
                {
                    auto delta_L = lidar_poses_[i].inverse() * lidar_poses_[i + 1];
                    auto delta_I = gnss_imu_poses_[i].inverse() * gnss_imu_poses_[i + 1];

                    double trans_motion = delta_I.translation().norm();
                    double rot_motion = Sophus::SO3::log(delta_I.so3()).norm(); // radians

                    if (trans_motion < 0.01 && rot_motion < (1.0 * M_PI / 180.0)) // 1 cm and 1 degree
                    {
                        // skip static or nearly static pairs
                        // continue;
                    }

                    Pose3 delta_L_true = Pose3(delta_L.matrix());
                    Pose3 delta_I_true = Pose3(delta_I.matrix());

                    // auto huber = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(1.345), base_noise);
                    // graph.add(std::make_shared<ExtrinsicSE3Factor>(X(0), delta_I_true, delta_L_true, huber));         //robust

                    graph.add(std::make_shared<ExtrinsicSE3Factor>(X(0), delta_I_true, delta_L_true, base_noise)); // not robust

                    A_.push_back(delta_L);
                    B_.push_back(delta_I);
                    added++;
                }
            }
        }

        std::cout << "Added " << added << "/" << N << " good measurements for estimation" << std::endl;

        Pose3 init_guess = Pose3(init_guess_.matrix());
        // // Weak prior (very large sigmas = weak)
        auto priorNoise = noiseModel::Diagonal::Sigmas((Vector(6) << 999., 999., 999., 999., 999., 999.).finished());
        graph.add(PriorFactor<Pose3>(X(0), init_guess, priorNoise));
        initial.insert(X(0), init_guess);
        std::cout << "Initial guess for extrinsic:\n"
                  << init_guess << std::endl;

        // Optimize
        // LevenbergMarquardtParams params;
        // params.setVerbosity("ERROR");
        LevenbergMarquardtOptimizer optimizer(graph, initial); //, params
        Values result = optimizer.optimize();

        Pose3 T_I_L_est = result.at<Pose3>(X(0));
        std::cout << "\nEstimated extrinsic (T_I_L_est):\n"
                  << T_I_L_est << std::endl;
        return GtsamToSophus(T_I_L_est);
    }

    Sophus::SE3 find_extrinsic_(Sophus::SE3 &init_guess_,
                                const std::vector<Sophus::SE3> &lidar_poses_, const std::vector<Sophus::SE3> &gnss_imu_poses_,
                                const std::vector<PointCloudXYZI::Ptr> &scans_in_base_frame, ros::Publisher &pub_, const Sophus::SE3 &Lidar_wrt_IMU)
    {
        NonlinearFactorGraph graph;
        Values initial;

        double meas_sigma = .5; // rad/m synthetic measurement sigma
        auto base_noise = noiseModel::Isotropic::Sigma(6, meas_sigma);

        int N = gnss_imu_poses_.size();
        int added = 0;

        for (size_t i = 0; i + 1 < N; ++i)
        {
            auto delta_L = lidar_poses_[i].inverse() * lidar_poses_[i + 1];
            auto delta_I = gnss_imu_poses_[i].inverse() * gnss_imu_poses_[i + 1];

            double trans_motion = delta_I.translation().norm();
            double rot_motion = Sophus::SO3::log(delta_I.so3()).norm(); // radians

            if (trans_motion < 0.01 && rot_motion < (1.0 * M_PI / 180.0)) // 1 cm and 1 degree
            {
                // skip static or nearly static pairs
                // continue;
            }

            Pose3 delta_L_true = Pose3(delta_L.matrix());
            Pose3 delta_I_true = Pose3(delta_I.matrix());

            // auto huber = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(1.345), base_noise);
            // graph.add(std::make_shared<ExtrinsicSE3Factor>(X(0), delta_I_true, delta_L_true, huber));         //robust

            graph.add(std::make_shared<ExtrinsicSE3Factor>(X(0), delta_I_true, delta_L_true, base_noise)); // not robust

            added++;
        }

        std::cout << "Added " << added << "/" << N << " good measurements for estimation" << std::endl;

        Pose3 init_guess = Pose3(init_guess_.matrix());
        // // Weak prior (very large sigmas = weak)
        auto priorNoise = noiseModel::Diagonal::Sigmas((Vector(6) << 999., 999., 999., 999., 999., 999.).finished());
        graph.add(PriorFactor<Pose3>(X(0), init_guess, priorNoise));
        initial.insert(X(0), init_guess);
        std::cout << "Initial guess for extrinsic:\n"
                  << init_guess << std::endl;

        // Optimize
        // LevenbergMarquardtParams params;
        // params.setVerbosity("ERROR");
        LevenbergMarquardtOptimizer optimizer(graph, initial); //, params
        Values result = optimizer.optimize();

        Pose3 T_I_L_est = result.at<Pose3>(X(0));
        std::cout << "\nEstimated extrinsic (T_I_L_est):\n"
                  << T_I_L_est << std::endl;

        Sophus::SE3 E = GtsamToSophus(T_I_L_est);

        PointCloudXYZI::Ptr lc_map;
        lc_map.reset(new PointCloudXYZI());

        for (int i = 0; i < N; i++)
        {
            PointCloudXYZI::Ptr cloud_in_sensor_frame(new PointCloudXYZI);
            *cloud_in_sensor_frame = *scans_in_base_frame[i];

            // TransformPoints(Lidar_wrt_IMU, cloud_in_sensor_frame); // lidar to IMU frame - front IMU

            Sophus::SE3 gnss = E.inverse() * gnss_imu_poses_[i] * E;

            gnss = lidar_poses_[i];

            // TransformPoints(gnss, cloud_in_sensor_frame); // georeference with se3 in IMU frame

            // *lc_map += *cloud_in_sensor_frame;

            int size = cloud_in_sensor_frame->points.size();
            PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

            tbb::parallel_for(tbb::blocked_range<int>(0, size), [&](tbb::blocked_range<int> r)
                              {
                    for (int i = r.begin(); i < r.end(); i++)
                    //for (int i = 0; i < size; i++)
                    {
                        
                        auto &pi = cloud_in_sensor_frame->points[i];
                        auto &po = laserCloudWorld->points[i];
                        V3D p_body(pi.x, pi.y, pi.z);

                        //V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + state_point.pos);

                        V3D p_global = p_body;
                        //V3D p_global = Lidar_wrt_IMU * p_body;
                        //V3D p_global = gnss * Lidar_wrt_IMU * p_body;
                        //V3D p_global = gnss * p_body;

                        po.x = p_global(0);
                        po.y = p_global(1);
                        po.z = p_global(2);

                    } });

            *lc_map += *laserCloudWorld;

            break;
        }

        std::cout << "lc_map:" << lc_map->size() << std::endl;

        sensor_msgs::PointCloud2 cloud_msg;
        cloud_msg.header.stamp = ros::Time::now();

        cloud_msg.header.frame_id = "world";
        lc_map->header.frame_id = "world";
        pcl::toROSMsg(*lc_map, cloud_msg);
        pub_.publish(cloud_msg);

        return GtsamToSophus(T_I_L_est);
    }
};

void DataHandler::Subscribe()
{
    std::cout << "Run test" << std::endl;
    std::cout << std::fixed << std::setprecision(12);
    std::cerr << std::fixed << std::setprecision(12);

    std::cout << "\n=============================== Georeference the data ===============================" << std::endl;

    std::shared_ptr<IMU_Class> imu_obj(new IMU_Class());
    std::shared_ptr<GNSS> gnss_obj(new GNSS());

    Sophus::SE3 Lidar_wrt_IMU = Sophus::SE3(Lidar_R_wrt_IMU, Lidar_T_wrt_IMU);
    Sophus::SE3 Lidar_wrt_IMU_inverse = Sophus::SE3(Lidar_R_wrt_IMU, Lidar_T_wrt_IMU).inverse();
    imu_obj->set_param(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, 10. * acc_cov),
                       V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov), V3D(b_acc_cov, b_acc_cov, 10. * b_acc_cov));
    gnss_obj->set_param(GNSS_T_wrt_IMU, GNSS_IMU_calibration_distance, postprocessed_gnss_path);

    std::shared_ptr<Batch> batch_obj(new Batch());
    bool test_batch = false;//true;

    if (test_batch)
    {
        batch_obj->set_param(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, 10. * acc_cov),
                             V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov), V3D(b_acc_cov, b_acc_cov, 10. * b_acc_cov));
    }

#define USE_ALS

#ifdef USE_ALS
    std::shared_ptr<ALS_Handler> als_obj = std::make_shared<ALS_Handler>(folder_root, downsample, closest_N_files, filter_size_surf_min);
    ros::Publisher pubLaserALSMap = nh.advertise<sensor_msgs::PointCloud2>("/ALS_map", 1000);
#endif

    ros::Publisher pub_loops = nh.advertise<visualization_msgs::Marker>("loop_closure_edges", 1);
    ros::Publisher pub_points = nh.advertise<visualization_msgs::Marker>("trajectory_points", 1);
    ros::Publisher lc_map_pub = nh.advertise<sensor_msgs::PointCloud2>("loop_closure_map", 1);

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100);

    ros::Publisher pubLaserCloudDebug = nh.advertise<sensor_msgs::PointCloud2>("/cloud_debug", 10);
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("acceleration_marker", 10);

    ros::Publisher normals_pub = nh.advertise<visualization_msgs::Marker>("normals", 1);

    Sophus::SE3 curr_mls, prev_mls;
    bool perform_mls_registration = true;

    Eigen::Matrix4d T_lidar2gnss; // this is the extrinsic between the mls and the back antena GNSS-IMU solution
    T_lidar2gnss << 0.0131683606, -0.9998577263, 0.0105414145, 0.0154123047,
        0.9672090675, 0.0100627670, -0.2537821120, -2.6359450601,
        0.2536399297, 0.0135376461, 0.9672039693, -0.5896374492,
        0.0, 0.0, 0.0, 1.0;
    M3D R_lidar2gnss = T_lidar2gnss.block<3, 3>(0, 0); // Rotation
    V3D t_lidar2gnss = T_lidar2gnss.block<3, 1>(0, 3); // Translation
    // gnss should be rtansformed to mls frame
    Sophus::SE3 lidar2gnss(R_lidar2gnss, t_lidar2gnss); // FROM LIDAR 2 GNSS   T_lidar = T_gnss * lidar2gnss.inverse()
    Sophus::SE3 gnss2lidar = lidar2gnss.inverse();      // THIS FOR THE BACK ANTENA - TO TRANSFORM TO FRONT imu frame

    // the extrinsic is wrong
    gnss2lidar = Sophus::SE3();

    // if(false)
    //{
    // Pose3 predicted = T_I_L * delta_L_ * T_I_L.inverse();

    M3D R_I_L; // Euler angles (deg):   0.378909998817  -1.733002910552 -15.061522731925
    R_I_L << 0.999520748959, 0.001472395834, -0.030920938727,
        0.006610154017, 0.965678210266, 0.259657274270,
        0.030241995059, -0.259737225418, 0.965205675215;

    // R_I_L = M3D::Identity();
    V3D t_I_L = V3D(-0.078173361476, 2.491416912247, 1.124627820134);
    auto extrinsic_ = Sophus::SE3(R_I_L, t_I_L);

    // extrinsic_ = gnss2lidar;
    //}

    //----------------------------------------------------------------------------

    std::cout << "\n\nStart reading the data..." << std::endl;
    //------------------------------------------------------------------------------
    TrajectoryReader reader;
    bool use_lc = false;//true;
    // an extrinsic transformation is passed here to transform the ppk gnss-imu orientaiton into mls frame
    reader.read(postprocessed_gnss_path, gnss2lidar, use_lc, filter_size_surf_min);
    reader.Lidar_wrt_IMU = Lidar_wrt_IMU;

    // --- Access measurements ---
    const auto &measurements = reader.measurements();
    std::cout << "Parsed " << measurements.size() << " measurements." << std::endl;
    int total_m = measurements.size();

    if (!measurements.empty())
    {
        const auto &m = measurements[0]; // first measurement
        std::cout << "First measurement:" << std::endl;
        std::cout << "  GPSTime = " << m.GPSTime << " sec" << std::endl;
        std::cout << "  Position (E,N,H) = ("
                  << m.Easting << ", "
                  << m.Northing << ", "
                  << m.H_Ell << ")" << std::endl;
        std::cout << "  Orientation (Phi, Omega, Kappa) = ("
                  << m.Phi << ", "
                  << m.Omega << ", "
                  << m.Kappa << ")" << std::endl;
        std::cout << "  AccBias (X,Y,Z) = ("
                  << m.AccBiasX << ", "
                  << m.AccBiasY << ", "
                  << m.AccBiasZ << ")" << std::endl;

        std::cout << "  AngRate (X,Y,Z) = ("
                  << m.AngRateX << ", "
                  << m.AngRateY << ", "
                  << m.AngRateZ << ")" << std::endl;

        std::cout << "  VelBdy (X,Y,Z) = ("
                  << m.VelBdyX << ", "
                  << m.VelBdyY << ", "
                  << m.VelBdyZ << ")" << std::endl;

        std::cout << "First measurement m.utc_usec :" << m.utc_usec << std::endl;
        std::cout << "First measurement m.utc_usec2:" << m.utc_usec2 << std::endl;

        // First measurement m.utc_usec :   1721898390000000.000000000000
        // First measurement m.utc_usec2:   1721898390.000000000000

        // pcl_cbk msg->  .stamp.toSec():    1721900923.978538036346
    }

    auto m0 = measurements[0];
    V3D raw_gyro;
    V3D raw_acc, gravity_free_acc = V3D(m0.AccBdyX, m0.AccBdyY, m0.AccBdyZ);
    reader.addEarthGravity(measurements[reader.curr_index], raw_gyro, raw_acc, G_m_s2); // this will add the world gravity
    // reader.addGravity(measurements[reader.curr_index], se3, raw_gyro, raw_acc, G_m_s2); //gravity in curr body frame

    std::cout << "gravity_free_acc:" << gravity_free_acc.transpose() << std::endl;
    std::cout << "raw_acc:" << raw_acc.transpose() << std::endl;

    Sophus::SE3 first_ppk_gnss_pose_inverse = Sophus::SE3();
    reader.toSE3(m0, first_ppk_gnss_pose_inverse);
    first_ppk_gnss_pose_inverse = first_ppk_gnss_pose_inverse.inverse();

    int tmp_index = 0;
    ros::Rate rate(500);

    int scan_id = 0;

    bool use_als = true, als_integrated = false;
    using namespace std::chrono;

    bool ppk_gnss_synced = false;
    Sophus::SE3 se3 = Sophus::SE3();
    // Alignment transform: GNSS -> LiDAR
    Sophus::SE3 T_LG = Sophus::SE3();
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

    std::vector<std::shared_ptr<rosbag::Bag>> bags;
    for (const auto &file : bag_files)
    {
        auto bag = std::make_shared<rosbag::Bag>();
        bag->open(file, rosbag::bagmode::Read);
        bags.push_back(bag);
        ROS_INFO_STREAM("Opened bag: " << file);
    }

    rosbag::View view;
    for (auto &b : bags)
    {
        view.addQuery(*b, rosbag::TopicQuery(topics));
    }

    signal(SIGINT, SigHandle); // Handle Ctrl+C (SIGINT)
    flg_exit = false;
    perform_mls_registration = true;
    bool ppk_gnss_oriented = false;

    Sophus::SE3 tmp_pose = Sophus::SE3();
    Sophus::SE3 prev_se3 = Sophus::SE3();
    Sophus::SE3 last_trigger = Sophus::SE3();

    // for extrinsics mls - gnss
    std::vector<Sophus::SE3> lidar_poses_;
    std::vector<Sophus::SE3> gnss_imu_poses_;
    std::vector<PointCloudXYZI::Ptr> scans_in_base_frame;
    int iters = 0;
    bool first_time = true;

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

            // if (scan_id > 3000) // just for some checkings
            // {
            //     std::cout << "Stop here... enough data 2500 scans" << std::endl;
            //     break;
            // }

            {
                double t00 = omp_get_wtime();

                gnss_obj->Process(gps_buffer, lidar_end_time, state_point.pos);
                Sophus::SE3 gnss_pose = (gnss_obj->use_postprocessed_gnss) ? gnss_obj->postprocessed_gps_pose : gnss_obj->gps_pose;
                // gnss_pose.so3() = state_point.rot; // use the MLS orientation
                // if (als_integrated) // if (use_gnss)
                publish_gnss_odometry(gnss_pose);

                double time_of_day_sec = gnss_obj->tod;

                if (!reader.initted)
                {
                    if (!reader.init(time_of_day_sec))
                    {
                        std::cerr << "Cannot initialize the GNSS-IMU reader..." << std::endl;
                        throw std::runtime_error("Cannot initialize the GNSS-IMU reader...: time_of_day_sec " + std::to_string(time_of_day_sec));
                    }
                    else
                    {
                        reader.initted = true;
                        tmp_index = reader.curr_index;
                        std::cout << "init Initialization succeeded.." << std::endl;
                        std::cout << "tmp_index:" << tmp_index << std::endl;
                    }
                    continue;

                    // reader.initted = false;

                    // if (!reader.init_unix(lidar_end_time))
                    // {
                    //     std::cerr << "Cannot init_unix initialize the GNSS-IMU reader..." << std::endl;
                    //     return;
                    // }

                    // tmp_index = reader.curr_index;
                    // std::cout << "init_unix Initialization succeeded.." << std::endl;
                    // std::cout << "tmp_index:" << tmp_index << std::endl;
                }

                if (reader.initted)
                {
                    tmp_index = reader.curr_index;
                    if (!ppk_gnss_synced)
                    {
                        const auto &m = measurements[tmp_index];
                        Sophus::SE3 interpolated_pose;
                        reader.toSE3(m, interpolated_pose);

                        // take only the position of the first pose - keeps the orientation as it it, so gravity = earth gravity
                        first_ppk_gnss_pose_inverse = Sophus::SE3(M3D::Identity(), interpolated_pose.translation()).inverse();

                        // doing this we put everything in the frame of the first pose - gravity here is not the earth gravity
                        // first_ppk_gnss_pose_inverse = interpolated_pose.inverse(); //this will rotate the world - so that gravity
                        // the earth gravity can be added using the current system rotation in the world frame

                        tmp_pose = first_ppk_gnss_pose_inverse * interpolated_pose;
                        tmp_pose = Sophus::SE3(tmp_pose.so3().matrix(), V3D::Zero());
                        // tmp_pose = extrinsic_.inverse()  * tmp_pose * extrinsic_;

                        // Convert to Euler (ZYX: yaw-pitch-roll)
                        V3D euler = interpolated_pose.so3().matrix().eulerAngles(2, 1, 0);
                        // euler[0] = yaw (around Z), euler[1] = pitch (around Y), euler[2] = roll (around X)
                        std::cout << "Euler angles (rad): " << euler.transpose() << std::endl;
                        std::cout << "Euler angles (deg): " << euler.transpose() * 180.0 / M_PI << std::endl;

                        ppk_gnss_synced = true;
                        std::cout << "\nsynchronised\n, press enter to continue ..." << std::endl;
                        std::cin.get();
                        continue;
                    }

                    double time_start = time_of_day_sec - .1;
                    double time_end = time_of_day_sec;

                    auto interpolated_pose = reader.closestPose(time_end);

                    tmp_index = reader.curr_index;
                    const auto &msg_time = measurements[tmp_index].tod;

                    // interpolated gnss-ins pose is transformed to frame of the first pose and then rotated to MLS system
                    se3 = T_LG * first_ppk_gnss_pose_inverse * interpolated_pose;

                    publish_ppk_gnss(se3, msg_time);

                    if (false)
                    {
                        // reader.addEarthGravity(measurements[reader.curr_index], raw_gyro, raw_acc, G_m_s2); //this will add the world gravity
                        reader.addGravity(measurements[reader.curr_index], se3, raw_gyro, raw_acc, G_m_s2); // gravity in curr body frame

                        // todo - we can do it the other way around and add the gravity in IMU body frame
                        publishAccelerationArrow(marker_pub, -raw_acc, msg_time);

                        *feats_undistort = *Measures.lidar;              // lidar frame
                        TransformPoints(Lidar_wrt_IMU, feats_undistort); // lidar to IMU frame - front IMU

                        reader.undistort_const_vel(time_start, feats_undistort); // const vel model
                        // reader.undistort_imu(time_start, feats_undistort); //imu measurements

                        downSizeFilterSurf.setInputCloud(feats_undistort);
                        downSizeFilterSurf.filter(*feats_down_body);
                        feats_down_size = feats_down_body->points.size();

                        TransformPoints(se3, feats_down_body); // georeference with se3 in IMU frame

                        publish_frame_debug(pubLaserCloudDebug, feats_down_body);
                    }
                }
                else
                {
                    std::cout << "GNSS reader not initted..." << std::endl;
                    throw std::runtime_error("GNSS reader not initted...");
                }

                //------------------------------------------------------------
                // perform_mls_registration = false;
                if (perform_mls_registration)
                {
                    if (flg_first_scan)
                    {
                        first_lidar_time = Measures.lidar_beg_time;
                        flg_first_scan = false;
                        curr_mls = Sophus::SE3(state_point.rot, state_point.pos);
                        prev_mls = curr_mls;
                        continue;
                    }

                    if (!test_batch)
                    {
                        //  undistort and provide initial guess
                        imu_obj->Process(Measures, estimator_, feats_undistort);
                        if (imu_obj->imu_need_init_)
                        {
                            std::cout << "IMU was not initialised " << std::endl;
                            continue;
                        }
                        
                        if (!ppk_gnss_oriented)
                        {
                            state_point = estimator_.get_x();
                            curr_mls = Sophus::SE3(state_point.rot, state_point.pos);
                            prev_mls = curr_mls;

                            // lidar predicted pose
                            // Sophus::SE3 T_L0 = Sophus::SE3(state_point.rot, state_point.pos); // first LiDAR–inertial pose
                            Sophus::SE3 T_L0 = Sophus::SE3(state_point.rot, V3D(0, 0, 0)); // first LiDAR–inertial pose

                            
                            // Alignment transform: GNSS -> LiDAR
                            // T_LG = T_L0 * se3.inverse();

                            T_LG = T_L0 * tmp_pose.inverse();

                            // use only the yaw angle
                            double yaw = T_LG.so3().matrix().eulerAngles(2, 1, 0)[0]; // rotation around Z // yaw, pitch, roll (Z,Y,X order)
                            Eigen::AngleAxisd yawRot(yaw, V3D::UnitZ());
                            T_LG = Sophus::SE3(yawRot.toRotationMatrix(), V3D::Zero());

                            // Transform any GNSS pose into LiDAR-imu frame
                            // Sophus::SE3 T_Lk = T_LG * se3;

                            // T_LG = Sophus::SE3(); //this shoud be identity when using for extrinsics

                            ppk_gnss_oriented = true;

                            Sophus::SE3 put_trajectory_in_this_Frame = T_LG * first_ppk_gnss_pose_inverse;
                            reader.visualize_trajectory(put_trajectory_in_this_Frame, 8000, pub_points, pub_loops);

                                                
                            prev_se3 = T_LG * tmp_pose;

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
                                                  {
                                                      pointBodyToWorld(&(feats_undistort->points[i]), &(feats_down_world->points[i])); // transform to world coordinates
                                                  }
                                              });

                            *laserCloudSurfMap += *feats_down_world;
                            map_init = true;
                            continue;
                        }

                        Nearest_Points.resize(feats_down_size);

                        bool use_se3_update = false;// true;
                        V3D std_pos_m = V3D(.01, .01, .01);
                        V3D std_rot_deg = V3D(.1, .1, .1);

                        // with manual cov - its closer to gnss-ins solution - better
                        // with given ones - also works but slightly worse

                        if (true) //
                        {
                            const auto &mi = measurements[reader.curr_index];

                            double scale = 3; // 1; // 1-sigma  //100; // 9;
                            // scale = 6;
                            //scale = 12;
                            //M3D R = (T_LG * first_ppk_gnss_pose_inverse).so3().matrix();
                            //std::cout << "\n\nOriginal translation std (m): " << V3D(mi.SDEast, mi.SDNorth, mi.SDHeight).transpose() << "\n";
                            //std::cout << "Original rotation std (deg): " << V3D(mi.RollSD, mi.PitchSD, mi.HdngSD).transpose() << "\n";
                            // std_pos_m = scale * (R * V3D(mi.SDEast, mi.SDNorth, mi.SDHeight)).cwiseAbs();
                            // std_rot_deg = scale * ((R * V3D(mi.RollSD, mi.PitchSD, mi.HdngSD)).cwiseAbs()); 

                            scale = 9; //used when the gnss uses as absolute prior
                            
                            std_pos_m = scale * V3D::Ones()*V3D(mi.SDEast, mi.SDNorth, mi.SDHeight).norm();
                            std_rot_deg = scale * V3D::Ones()*V3D(mi.RollSD, mi.PitchSD, mi.HdngSD).norm();
                            std::cout << "\nTransformed translation std (m): " << std_pos_m.transpose() << "\n";
                            std::cout << "Transformed rotation std (deg): " << std_rot_deg.transpose() << "\n";
                        }

                        bool use_als_update = false; // true;

                        use_als = false;// true;
                        if (use_als)
                        {
                            if (!als_obj->refine_als) // als was not setup
                            {
                                use_als_update = false; // ALS not set yet

                                if (gnss_obj->GNSS_extrinsic_init)
                                {
                                    *featsFromMap = *laserCloudSurfMap;

                                    // only for now - remove this later
                                    // std::string als2mls_filename = "/home/eugeniu/x_vux-georeferenced-final/_Hesai/als2mls.txt";
                                    std::string als2mls_filename = "/home/eugeniu/zz_zx_final/ref/als_to_mls.txt";

                                    Sophus::SE3 known_als2mls;
                                    readSE3FromFile(als2mls_filename, known_als2mls);
                                    std::cout << "Read the known transformation" << std::endl;
                                    std::cout << "known_als2mls:\n"
                                              << known_als2mls.matrix() << std::endl;

                                    // als_obj->init(known_als2mls);

                                    if (!this->downsample) // if it is sparse ALS data from NLS
                                    {
                                        V3D t = known_als2mls.translation();
                                        t.z() += 20.;
                                        Sophus::SE3 als2mls_for_sparse_ALS = Sophus::SE3(known_als2mls.so3().matrix(), t);
                                        std::cout << "Init ALS from known T map refinement" << std::endl;
                                        // als_obj->init(als2mls_for_sparse_ALS, featsFromMap); // with refinement used before

                                        als_obj->init(gnss_obj->origin_enu, gnss_obj->R_GNSS_to_MLS, featsFromMap); //test now
                                    }
                                    else
                                    {
                                        // std::cout << "Init ALS from known T" << std::endl;
                                        // als_obj->init(known_als2mls);

                                        als_obj->init(gnss_obj->origin_enu, gnss_obj->R_GNSS_to_MLS, featsFromMap);
                                    }
                                    // TO BE DONE: take the initialization from GNSS-INS as int guess
                                    // als_obj->init(known_als2mls, laserCloudSurfMap); // with refinement

                                    gnss_obj->updateExtrinsic(als_obj->R_to_mls); // use the scan-based refined rotation for GNSS
                                    als_obj->Update(Sophus::SE3(state_point.rot, state_point.pos));
                                    gnss_obj->als2mls_T = als_obj->als_to_mls;

                                    std::cout << "T_LG:\n"
                                              << T_LG.matrix() << std::endl;
                                    std::cout << "als_obj->als_to_mls:\n"
                                              << als_obj->als_to_mls.matrix() << std::endl;

                                    std::cout << "T_LG       Euler angles (deg): " << T_LG.so3().matrix().eulerAngles(2, 1, 0).transpose() * 180.0 / M_PI << std::endl;
                                    std::cout << "als_to_mls Euler angles (deg): " << als_obj->als_to_mls.so3().matrix().eulerAngles(2, 1, 0).transpose() * 180.0 / M_PI << std::endl;

                                    // add the orientation correction for GNSS extrinsic here
                                    T_LG = Sophus::SE3(als_obj->als_to_mls.so3(), T_LG.translation());
                                    // double yaw = als_obj->als_to_mls.so3().matrix().eulerAngles(2, 1, 0)[0]; // rotation around Z // yaw, pitch, roll (Z,Y,X order)
                                    // Eigen::AngleAxisd yawRot(yaw, V3D::UnitZ());
                                    // T_LG = Sophus::SE3(yawRot.toRotationMatrix(), T_LG.translation()); //worse


                                    Sophus::SE3 put_trajectory_in_this_Frame = T_LG * first_ppk_gnss_pose_inverse;
                                    reader.visualize_trajectory(put_trajectory_in_this_Frame, 8000, pub_points, pub_loops);

                                    std::cout << "\nupdated the ALS2MLS orientation, press enter..." << std::endl;
                                    std::cin.get();

                                    // reset local map - this should be reset, since it contains accumulated drift
                                    laserCloudSurfMap.reset(new PointCloudXYZI());
                                }
                            }
                            else // als was set up
                            {
                                als_obj->Update(Sophus::SE3(state_point.rot, state_point.pos));
                                als_integrated = true;

                                use_als_update = true; // use ALS now

                                {
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
                                    // if (!estimator_.update_tighly_fused(LASER_POINT_COV, feats_down_body, laserCloudSurfMap, als_obj->als_cloud, als_obj->localKdTree_map_als, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en))
                                    // {
                                    //     std::cout << "\n------------------FUSION ALS-MLS update failed--------------------------------" << std::endl;
                                    // }

                                    // if (false)
                                    // { // this was used so far
                                    //     // update tighly fusion from MLS and ALS
                                    //     double R_gps_cov = .0001; // GNSS_VAR * GNSS_VAR;
                                    //     Sophus::SE3 gnss_pose = (gnss_obj->use_postprocessed_gnss) ? gnss_obj->postprocessed_gps_pose : gnss_obj->gps_pose;
                                    //     const V3D &gnss_in_mls = gnss_pose.translation();

                                    //     bool tightly_coupled = true;
                                    //     bool use_gnss = false;
                                    //     bool use_als = true;

                                    //     if (!estimator_.update_final(
                                    //             LASER_POINT_COV, R_gps_cov, feats_down_body, gnss_in_mls, laserCloudSurfMap, als_obj->als_cloud, als_obj->localKdTree_map_als,
                                    //             Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en, use_gnss, use_als, tightly_coupled))
                                    //     {
                                    //         std::cout << "\n------------------FUSION ALS-MLS update failed--------------------------------" << std::endl;
                                    //     }
                                    // }
                                }
                            }

                            if (pubLaserALSMap.getNumSubscribers() != 0)
                            {
                                als_obj->getCloud(featsFromMap);
                                publish_map(pubLaserALSMap);
                            }
                        }

                        //use_als_update = false;
                        bool have_lc_ref = false;
                        if (use_lc)
                        {
                            // predicted pose
                            curr_mls = Sophus::SE3(state_point.rot, state_point.pos);
                            double dist = (curr_mls.inverse() * last_trigger).log().norm();
                            double threshold_ = 1; // m
                            if (dist >= threshold_)
                            {
                                std::cout << "[INFO] System traveled more than " << threshold_ << " meters: " << dist << " m" << std::endl;
                                if (reader.try_LC(10)) // this should not be called too often
                                {
                                    have_lc_ref = true;
                                    last_trigger = curr_mls;
                                }
                            }

                            std::cout << "reader.lc_map:" << reader.lc_map->size() << std::endl;
                            if (reader.lc_map->size() > 0)
                            {
                                if (lc_map_pub.getNumSubscribers() != 0)
                                {
                                    sensor_msgs::PointCloud2 cloud_msg;
                                    cloud_msg.header.stamp = ros::Time::now();
                                    cloud_msg.header.frame_id = "MLS";
                                    pcl::toROSMsg(*reader.lc_map, cloud_msg);
                                    lc_map_pub.publish(cloud_msg);
                                }
                            }
                        }


                        // auto global_se3 = prev_mls * prev_se3.inverse() * se3;
                        // prev_se3 = se3;
                        iters = estimator_.update_MLS(LASER_POINT_COV, feats_down_body, laserCloudSurfMap, NUM_MAX_ITERATIONS, extrinsic_est_en,
                                                      use_als_update, als_obj->als_cloud, als_obj->localKdTree_map_als,
                                                      use_se3_update, se3, std_pos_m, std_rot_deg,
                                                      have_lc_ref, reader.lc_map, reader.lc_tree);

                        curr_mls = Sophus::SE3(state_point.rot, state_point.pos); // updated pose
                        if (use_lc)
                        {
                            state_point = estimator_.get_x();
                            auto cov_ = estimator_.get_P().block<6, 6>(0, 0);
                            reader.cloud_and_odom_callback(pubLaserCloudDebug, feats_down_body, curr_mls, cov_, have_lc_ref); // build the graph and save the scans
                        }

                        als_integrated = true; // to save the poses

                        // this is good
                        //  if(imu_obj->backwardPass(Measures, estimator_, *feats_undistort))
                        //  {
                        //      std::cout<<"enough smoothing has been done, downsample and update..."<<std::endl;
                        //      downSizeFilterSurf.setInputCloud(feats_undistort);
                        //      downSizeFilterSurf.filter(*feats_down_body);
                        //      feats_down_size = feats_down_body->points.size();

                        //     //update NOT USED
                        //     // estimator_.update_MLS(LASER_POINT_COV, feats_down_body, laserCloudSurfMap, 1, extrinsic_est_en,
                        //     //                   use_als_update, als_obj->als_cloud, als_obj->localKdTree_map_als,
                        //     //                   use_se3_update, se3, std_pos_m, std_rot_deg,
                        //     //                   have_lc_ref, reader.lc_map, reader.lc_tree);
                        // }

                        if (false)
                        {
                            // take this from std or separation data
                            auto std_pos_m = V3D(.5, .5, .5); // take this from the measurement itself - 10cm
                            auto std_rot_deg = V3D(1, 1, 1);  //- 5 degrees

                            std_pos_m = V3D(.01, .01, .01);
                            std_rot_deg = V3D(.1, .1, .1);

                            if (false)
                            {
                                const auto &mi = measurements[reader.curr_index];
                                V3D tran_std = V3D(mi.SDEast, mi.SDNorth, mi.SDHeight); // in meters
                                V3D rot_std = V3D(mi.RollSD, mi.PitchSD, mi.HdngSD);    // in degrees
                                // std::cout<<"GNSS tran:"<<tran_std.transpose()<<" m"<<std::endl;
                                // std::cout<<"GNSS rot :"<<rot_std.transpose()<<" deg"<<std::endl;

                                {
                                    double scale = 1; // 10;
                                    M3D R = (T_LG * first_ppk_gnss_pose_inverse).so3().matrix();
                                    std_pos_m = scale * (R * tran_std).cwiseAbs();
                                    std_rot_deg = scale * ((R * rot_std).cwiseAbs() * 180.0 / M_PI);

                                    std::cout << "Transformed translation std (m): " << std_pos_m.transpose() << "\n";
                                    std::cout << "Transformed rotation std (deg): " << std_rot_deg.transpose() << "\n";
                                }
                            }

                            estimator_.update_se3(se3, NUM_MAX_ITERATIONS, std_pos_m, std_rot_deg);
                            // std_pos_m = V3D(GNSS_VAR, GNSS_VAR, GNSS_VAR);
                            // std_rot_deg = V3D(100, 100, 100);
                            // estimator_.update_se3(se3, NUM_MAX_ITERATIONS, std_pos_m, std_rot_deg);

                            // THIS ONE WORKS
                            //  const bool global_error = false; // set this true for global error of gps
                            //  // auto gps_cov_ = gnss_obj->gps_cov;
                            // auto gps_cov_ = V3D(GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR);
                            //  // const V3D &gnss_in_mls = gnss_pose.translation();
                            //  V3D gnss_in_mls = se3.translation();
                            //  estimator_.update(gnss_in_mls, gps_cov_, NUM_MAX_ITERATIONS, global_error);
                        }

                        // Crop the local map------
                        state_point = estimator_.get_x();
                        curr_mls = Sophus::SE3(state_point.rot, state_point.pos);
                    }
                    else
                    {
                        batch_obj->Process(Measures, estimator_, feats_undistort); // feats_undistort is in IMU frame 
                        if (batch_obj->imu_need_init_)
                        {
                            std::cout << "IMU was not initialised " << std::endl;
                            continue;
                        }

                        std::cout << "feats_undistort now :" << feats_undistort->size() << std::endl;

                        downSizeFilterSurf.setInputCloud(feats_undistort);
                        downSizeFilterSurf.filter(*feats_down_body);
                        feats_down_size = feats_down_body->points.size();

                        if (!ppk_gnss_oriented)
                        {
                            state_point = estimator_.get_x();
                            // lidar predicted pose
                            // Sophus::SE3 T_L0 = Sophus::SE3(state_point.rot, state_point.pos); // first LiDAR–inertial pose
                            Sophus::SE3 T_L0 = Sophus::SE3(state_point.rot, V3D(0, 0, 0)); // first LiDAR–inertial pose

                            // Alignment transform: GNSS -> LiDAR
                            // T_LG = T_L0 * se3.inverse();

                            T_LG = T_L0 * tmp_pose.inverse();

                            // use only the yaw angle
                            double yaw = T_LG.so3().matrix().eulerAngles(2, 1, 0)[0]; // rotation around Z // yaw, pitch, roll (Z,Y,X order)
                            Eigen::AngleAxisd yawRot(yaw, V3D::UnitZ());
                            T_LG = Sophus::SE3(yawRot.toRotationMatrix(), V3D::Zero());

                            // Transform any GNSS pose into LiDAR-imu frame
                            // Sophus::SE3 T_Lk = T_LG * se3;

                            // T_LG = Sophus::SE3(); //this shoud be identity when using for extrinsics

                            ppk_gnss_oriented = true;

                            Sophus::SE3 put_trajectory_in_this_Frame = T_LG * first_ppk_gnss_pose_inverse;
                            reader.visualize_trajectory(put_trajectory_in_this_Frame, 8000, pub_points, pub_loops);

                            continue;
                        }


                        if (feats_down_size < 5)
                        {
                            ROS_WARN("No feats_down_body point, skip this scan!\n");
                            std::cout << "feats_undistort:" << feats_undistort->size() << std::endl;
                            std::cout << "feats_down_body:" << feats_down_size << std::endl;
                            throw std::runtime_error("NO feats_down_body points -> ERROR");
                        }
                        state_point = estimator_.get_x();

                        if (!map_init)
                        {
                            feats_down_size = feats_undistort->size();
                            feats_down_world->resize(feats_down_size);
                            TransformPoints(Lidar_wrt_IMU_inverse, feats_undistort); // IMU back to lidar
                            tbb::parallel_for(tbb::blocked_range<int>(0, feats_down_size),
                                              [&](tbb::blocked_range<int> r)
                                              {
                                                  for (int i = r.begin(); i < r.end(); i++)
                                                  {
                                                      pointBodyToWorld(&(feats_undistort->points[i]), &(feats_down_world->points[i])); // transform to world coordinates
                                                  }
                                              });

                            *laserCloudSurfMap += *feats_down_world;
                            map_init = true;
                            continue;
                        }

                        //TransformPoints(Lidar_wrt_IMU, feats_down_body); // lidar to IMU frame - front IMU

                        gtsam::Matrix6 out_cov_pose;

                        // std::cout << "kdtree set input ALS points: " << als_obj->als_cloud->size() << std::endl;
                        // const auto &reference_localMap_cloud = als_obj->als_cloud;
                        //------------------------------------------------------------------------
                        estimator_.localKdTree_map->setInputCloud(laserCloudSurfMap);
                        const auto &refference_kdtree = estimator_.localKdTree_map;

                        bool debug = true;
                        batch_obj->test(state_point, Measures.lidar_beg_time, Measures.lidar_end_time, feats_down_body,
                                        laserCloudSurfMap, refference_kdtree, true,  // MLS map
                                        laserCloudSurfMap, refference_kdtree, false, // ALS map
                                        out_cov_pose, normals_pub, debug);

                        estimator_.set_x(state_point);
                                                als_integrated = true; // to save the poses

                        TransformPoints(Lidar_wrt_IMU_inverse, feats_down_body); // IMU back to lidar

                        if (false) // this will save the MLS estimated SE3 poses
                        {
                            auto s_ekf = estimator_.get_x();
                            auto s_graph = state_point; // was taken from the graph

                            std::ofstream foutS1("/home/eugeniu/S1.txt", std::ios::app);
                            std::ofstream foutS2("/home/eugeniu/S2.txt", std::ios::app);

                            V3D t_s1 = s_ekf.pos;
                            V3D v1 = s_ekf.vel;
                            V3D bg1 = s_ekf.bg;
                            V3D ba1 = s_ekf.ba;
                            Eigen::Quaterniond q_s1(s_ekf.rot.matrix());
                            q_s1.normalize();

                            foutS1.setf(std::ios::fixed, std::ios::floatfield);
                            foutS1.precision(20);
                            // # ' id t q v bg ba'
                            foutS1 << scan_id << " " << t_s1(0) << " " << t_s1(1) << " " << t_s1(2) << " "
                                   << q_s1.x() << " " << q_s1.y() << " " << q_s1.z() << " " << q_s1.w() << " "
                                   << v1.x() << " " << v1.y() << " " << v1.z() << " " << bg1.x() << " " << bg1.y() << " " << bg1.z() << " "
                                   << ba1.x() << " " << ba1.y() << " " << ba1.z() << std::endl;
                            foutS1.close();

                            t_s1 = s_graph.pos;
                            v1 = s_graph.vel;
                            bg1 = s_graph.bg;
                            ba1 = s_graph.ba;
                            Eigen::Quaterniond q_s2(s_graph.rot.matrix());
                            q_s2.normalize();

                            foutS2.setf(std::ios::fixed, std::ios::floatfield);
                            foutS2.precision(20);
                            // # ' id t q v bg ba'
                            foutS2 << scan_id << " " << t_s1(0) << " " << t_s1(1) << " " << t_s1(2) << " "
                                   << q_s2.x() << " " << q_s2.y() << " " << q_s2.z() << " " << q_s2.w() << " "
                                   << v1.x() << " " << v1.y() << " " << v1.z() << " " << bg1.x() << " " << bg1.y() << " " << bg1.z() << " "
                                   << ba1.x() << " " << ba1.y() << " " << ba1.z() << std::endl;
                            foutS2.close();
                        }
                    }

                    // Update the local map--------------------------------------------------
                    feats_down_world->resize(feats_down_size);
                    local_map_update(); // this will update local map with curr measurements and crop the map

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

                    // EXTRINSIC ESTIMATION - Motion based
                    if (false && als_integrated)
                    {
                        curr_mls = Sophus::SE3(state_point.rot, state_point.pos); // updated pose
                        lidar_poses_.push_back(curr_mls);                         // mls frame
                        gnss_imu_poses_.push_back(se3);

                        //*feats_down_body = *Measures.lidar;

                        scans_in_base_frame.push_back(feats_down_body);

                        auto dist = se3.translation().norm();

                        std::cout << "dist:" << dist << ", als_integrated:" << als_integrated << std::endl;
                        if (dist >= 500) // 200
                        {
                            Sophus::SE3 extrins = Sophus::SE3();

                            std::cout << "\nStart calibration, press enter..." << std::endl;
                            std::cin.get();

                            extrins = motion_based_extrinsic_estimation::find_extrinsic_motion_based(extrins, lidar_poses_, gnss_imu_poses_);

                            V3D euler = extrins.so3().matrix().eulerAngles(2, 1, 0);
                            // euler[0] = yaw (around Z), euler[1] = pitch (around Y), euler[2] = roll (around X)
                            std::cout << "Euler angles (rad): " << euler.transpose() << std::endl;
                            std::cout << "Euler angles (deg): " << euler.transpose() * 180.0 / M_PI << std::endl;

                            std::cout << "extrins:\n"
                                      << extrins.matrix() << std::endl;

                            // extrins = Sophus::SE3();

                            // extrins = motion_based_extrinsic_estimation::find_extrinsic_(extrins, lidar_poses_, gnss_imu_poses_,
                            //                           scans_in_base_frame, lc_map_pub, Lidar_wrt_IMU);

                            std::cout << "\nEnd calibration, press enter..." << std::endl;
                            std::cin.get();

                            // throw std::runtime_error("Finished extrinsic estimation");
                        }
                    }

                    double t11 = omp_get_wtime();
                    double duration_time = (t11 - t00) * 1000;
                    std::cout << "Mapping time(ms):  " << duration_time << ", feats_down_size: " << feats_down_size << ", lidar_end_time:" << lidar_end_time << "\n"
                              << std::endl;


                    double velocity = (curr_mls.translation() - prev_mls.translation()).norm() / 0.1;
                    std::cout<<"velocity:"<<velocity<<" (m/s)"<<std::endl;
                    if(velocity > 50) //max about 11 m/s
                    {
                        throw std::runtime_error("Crash detected, stop...");
                    }

                    prev_mls = curr_mls;
                    imu_obj->done_update_ = true;
#ifdef SAVE_DATA
                    std::cout << "save_poses:" << save_poses << ", save_clouds_path:" << save_clouds_path << std::endl;

                    if (als_integrated)
                    {

                        if (!als2mls_saved)
                        {
                            // save als_obj->als_to_mls refine transformation
                            std::ofstream foutG(poseSavePath + "als_to_mls.txt", std::ios::app);
                            // foutG.setf(std::ios::scientific, std::ios::floatfield);
                            foutG.setf(std::ios::fixed, std::ios::floatfield);
                            foutG.precision(20);

                            foutG << als_obj->als_to_mls.translation().transpose() << "\n"; //"Position: " <<
                            foutG << als_obj->als_to_mls.so3().matrix() << "\n";            //"Rotation (SO3):\n"
                            foutG.close();
                            als2mls_saved = true;
                        }

                        if (save_poses) // this will save the MLS estimated SE3 poses
                        {
                            const V3D &t_model = state_point.pos;
                            Eigen::Quaterniond q_model(state_point.rot.matrix());

                            q_model.normalize();
                            std::ofstream foutMLS(poseSavePath + "MLS.txt", std::ios::app);
                            // std::ofstream foutMLS(save_clouds_path + "MLS.txt", std::ios::app);
                            //  foutMLS.setf(std::ios::scientific, std::ios::floatfield);
                            foutMLS.setf(std::ios::fixed, std::ios::floatfield);
                            foutMLS.precision(20);
                            // # ' id time tx ty tz qx qy qz qw' - tum format(scan id, scan timestamp seconds, translation and rotation quaternion)
                            foutMLS << pcd_index << " " << std::to_string(lidar_end_time) << " " << t_model(0) << " " << t_model(1) << " " << t_model(2) << " "
                                    << q_model.x() << " " << q_model.y() << " " << q_model.z() << " " << q_model.w() << std::endl;
                            foutMLS.close();

                            const V3D &t_gnss = se3.translation();
                            Eigen::Quaterniond q_gnss(se3.so3().matrix());
                            q_gnss.normalize();
                            std::ofstream foutGNSS(poseSavePath + "MLS_gnss.txt", std::ios::app);
                            foutGNSS.setf(std::ios::fixed, std::ios::floatfield);
                            foutGNSS.precision(20);
                            // # ' id time tx ty tz qx qy qz qw' - tum format(scan id, scan timestamp seconds, translation and rotation quaternion)
                            foutGNSS << pcd_index << " " << std::to_string(lidar_end_time) << " " << t_gnss(0) << " " << t_gnss(1) << " " << t_gnss(2) << " "
                                     << q_gnss.x() << " " << q_gnss.y() << " " << q_gnss.z() << " " << q_gnss.w() << std::endl;
                            foutGNSS.close();

                            std::ofstream foutMLS_time(poseSavePath + "MLS_time.txt", std::ios::app);
                            foutMLS_time.setf(std::ios::fixed, std::ios::floatfield);
                            foutMLS_time.precision(20);
                            // # ' id time duration_time' - tum format(scan id, scan timestamp seconds, duration_time)
                            foutMLS_time << pcd_index << " " << std::to_string(lidar_end_time) << " " << duration_time << std::endl;
                            foutMLS_time.close();

                            std::ofstream foutMLS_iter(poseSavePath + "MLS_iter.txt", std::ios::app);
                            foutMLS_iter.setf(std::ios::fixed, std::ios::floatfield);
                            foutMLS_iter.precision(20);
                            // # ' id time duration_time' - tum format(scan id, scan timestamp seconds, duration_time)
                            foutMLS_iter << pcd_index << " " << std::to_string(lidar_end_time) << " " << iters << std::endl;
                            foutMLS_iter.close();
                        }
                        pcd_index++;
                    }
#endif
                }

                // std::this_thread::sleep_for(std::chrono::milliseconds(50)); // to simulate lidar measurements
            }
        }
    }
    // bag.close();
    for (auto &b : bags)
        b->close();

    // cv::destroyAllWindows(); */
    plt::close();
}
