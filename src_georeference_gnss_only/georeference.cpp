

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

// for drone we had this 
//     M3D T;
//     T << -1, 0, 0,
//         0, 0, -1,
//         0, -1, 0;

//     M3D R_enu = R_heikki * T;
//     m.se3 = Sophus::SE3(Eigen::Quaterniond(R_enu), translation);

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

// #include "../src3/clean_registration3.hpp"

#include "TrajectoryReader.hpp"

void DataHandler::Subscribe()
{
    std::cout << "Run test" << std::endl;
    std::cout << std::fixed << std::setprecision(12);
    std::cerr << std::fixed << std::setprecision(12);

    std::cout << "\n=============================== Georeference the data ===============================" << std::endl;

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

    // #ifdef USE_ALS
    //     std::shared_ptr<ALS_Handler> als_obj = std::make_shared<ALS_Handler>(folder_root, downsample, closest_N_files, filter_size_surf_min);
    //     ros::Publisher pubLaserALSMap = nh.advertise<sensor_msgs::PointCloud2>("/ALS_map", 1000);
    // #endif

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudLocal = nh.advertise<sensor_msgs::PointCloud2>("/cloud_local", 100000);

    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100);

    ros::Publisher pubLaserCloudDebug = nh.advertise<sensor_msgs::PointCloud2>("/cloud_debug", 10);
    ros::Publisher point_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/vux_data", 10000);
    ros::Publisher pubOptimizedVUX = nh.advertise<sensor_msgs::PointCloud2>("/vux_optimized", 10);

    ros::Publisher pubOptimizedVUX2 = nh.advertise<sensor_msgs::PointCloud2>("/vux_optimized2", 10);
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("acceleration_marker", 10);

    std::cout << "\n\nStart reading the data..." << std::endl;

    // for the car used so far - from the back antena
    std::string ppk_gnss_imu_file = "/media/eugeniu/T7/roamer/evo_20240725_reCalculated2_poqmodouEugent.txt";

    // from the front antena - bad no proper orientation
    // ppk_gnss_imu_file = "/media/eugeniu/T7/evo-bags/GT-MLS-25-07.txt";
    // ppk_gnss_imu_file = "/media/eugeniu/T71/Lowcost_trajs/Lieksa/Kiimasuo/Lieksa_20250519_1.txt";


    //------------------------------------------------------------------------------
    TrajectoryReader reader;
    reader.read(ppk_gnss_imu_file);

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

    /*
    --setup the proper GNSS reader for the better trajectory
    --option to add the raw acc with gravity
    --publish the data
    --publish the gravity vector
    --option to transform it into first IMU frame
    --read the bag files at it is but do not do anything with the data yet

    synch the data

    transform everything into frame of the first pose

    */


    int tmp_index = 0;
    ros::Rate rate(500);

    int scan_id = 0;

    Sophus::SE3 curr_mls;
    bool perform_mls_registration = true;

    Eigen::Matrix4d T_lidar2gnss;
    T_lidar2gnss << 0.0131683606, -0.9998577263, 0.0105414145, 0.0154123047,
        0.9672090675, 0.0100627670, -0.2537821120, -2.6359450601,
        0.2536399297, 0.0135376461, 0.9672039693, -0.5896374492,
        0.0, 0.0, 0.0, 1.0;
    M3D R_lidar2gnss = T_lidar2gnss.block<3, 3>(0, 0); // Rotation
    V3D t_lidar2gnss = T_lidar2gnss.block<3, 1>(0, 3); // Translation
    // gnss should be rtansformed to mls frame
    Sophus::SE3 lidar2gnss(R_lidar2gnss, t_lidar2gnss); // FROM LIDAR 2 GNSS   T_lidar = T_gnss * lidar2gnss.inverse()
    Sophus::SE3 gnss2lidar = lidar2gnss.inverse(); //THIS FOR THE BACK ANTENA - TO TRANSFORM TO FRONT imu frame 
    //----------------------------------------------------------------------------

    bool use_als = true;
    using namespace std::chrono;

    bool ppk_gnss_synced = false;
    Sophus::SE3 se3 = Sophus::SE3();

    std::vector<std::string> topics{lid_topic, imu_topic, gnss_topic};

    // std::ifstream file(bag_file);
    // if (!file)
    // {
    //     std::cerr << "Error: Bag file does not exist or is not accessible: " << bag_file << std::endl;
    //     return;
    // }
    // rosbag::Bag bag;
    // bag.open(bag_file, rosbag::bagmode::Read);
    // rosbag::View view(bag, rosbag::TopicQuery(topics));

    //{
    std::string bag_pattern = "/media/eugeniu/T71/Lowcost_trajs/Lieksa/Kiimasuo/Kiimasuo_hesai_*.bag";
    bag_pattern = bag_file;

    std::vector<std::string> bag_files = expandBagPattern(bag_pattern);
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
    //}

    signal(SIGINT, SigHandle); // Handle Ctrl+C (SIGINT)
    flg_exit = false;

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
                    continue;
                }

                if(false){
                    // undistort and provide initial guess
                    imu_obj->Process(Measures, estimator_, feats_undistort);
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
                    
                }
                
                gnss_obj->Process(gps_buffer, lidar_end_time, state_point.pos);
                double time_of_day_sec = gnss_obj->tod;
                
                if (!reader.initted)
                {
                    if (!reader.init(time_of_day_sec))
                    {
                        std::cerr << "Cannot initialize the GNSS-IMU reader..." << std::endl;
                        return;
                    }
                    else
                    {
                        tmp_index = reader.curr_index;
                        std::cout << "init Initialization succeeded.." << std::endl;
                        std::cout << "tmp_index:" << tmp_index << std::endl;
                    }
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
                else
                {
                    std::cout << "Cannot find the right ros unix association" << std::endl;
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

                        // Convert to Euler (ZYX: yaw-pitch-roll)
                        Eigen::Vector3d euler = interpolated_pose.so3().matrix().eulerAngles(2, 1, 0);
                        // euler[0] = yaw (around Z), euler[1] = pitch (around Y), euler[2] = roll (around X)
                        std::cout << "Euler angles (rad): " << euler.transpose() << std::endl;
                        std::cout << "Euler angles (deg): " << euler.transpose() * 180.0 / M_PI << std::endl;

                        ppk_gnss_synced = true;
                        std::cout << "\nsynchronised\n, press enter..." << std::endl;
                        std::cin.get();
                        continue;
                    }

                    double time_start = time_of_day_sec - .1;
                    double time_end = time_of_day_sec;
                    auto interpolated_pose = reader.closestPose(time_end); // * gnss2lidar;
                    
                    //auto interpolated_pose = reader.closestPoseUnix(lidar_end_time);
                    //reader.toSE3(measurements[tmp_index+27], interpolated_pose);

                    tmp_index = reader.curr_index;
                    const auto &msg_time = measurements[tmp_index].tod;

                    std::cout<<"utc_usec :"<<measurements[tmp_index].utc_usec<<std::endl;
                    std::cout<<"utc_usec2:"<<measurements[tmp_index].utc_usec2<<std::endl;


                    se3 = first_ppk_gnss_pose_inverse * interpolated_pose; // in first frame
                    publish_ppk_gnss(se3, msg_time);

                    // reader.addEarthGravity(measurements[reader.curr_index], raw_gyro, raw_acc, G_m_s2); //this will add the world gravity
                    reader.addGravity(measurements[reader.curr_index], se3, raw_gyro, raw_acc, G_m_s2); // gravity in curr body frame

                    // todo - we can do it the other way around and add the gravity in IMU body frame
                    publishAccelerationArrow(marker_pub, -raw_acc, msg_time);

                    {
                        // add the reader.Rz.transpose()  before the gnss2lidar, since gnss2lidar contains it
                        se3 = first_ppk_gnss_pose_inverse * interpolated_pose * Sophus::SE3(reader.Rz.transpose(), V3D(0, 0, 0)) * gnss2lidar;
                        

                        undistort here ...


                        *feats_undistort = *Measures.lidar; //lidar frame
                        TransformPoints(Lidar_wrt_IMU, feats_undistort); // lidar to IMU frame
                        TransformPoints(se3, feats_undistort); // georeference with se3 in IMU frame



                        downSizeFilterSurf.setInputCloud(feats_undistort);
                        downSizeFilterSurf.filter(*feats_down_body);
                        feats_down_size = feats_down_body->points.size();



                        // auto T = se3 * Lidar_wrt_IMU;
                        //TransformPoints(Lidar_wrt_IMU, feats_down_body); // lidar to IMU frame
                        //TransformPoints(se3, feats_down_body); // georeference with se3 in IMU frame

                        publish_frame_debug(pubLaserCloudDebug, feats_down_body);
                    }
                }
                else
                {
                    std::cout << "GNSS reader not initted..." << std::endl;
                }
                continue; //------------------------------------------------------------

                double t11 = omp_get_wtime();
                std::cout << "Mapping time(ms):  " << (t11 - t00) * 1000 << ", feats_down_size: " << feats_down_size << ", lidar_end_time:" << lidar_end_time << "\n"
                          << std::endl;

                
                 std::this_thread::sleep_for(std::chrono::milliseconds(100)); //to simulate lidar measurements 
            }
        }
    }
    // bag.close();
    for (auto &b : bags)
        b->close();

    // cv::destroyAllWindows(); */
}
