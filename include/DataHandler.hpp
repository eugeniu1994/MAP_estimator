#ifndef USE_DATAHANDLER_VUX_H1
#define USE_DATAHANDLER_VUX_H1

#define PCL_NO_PRECOMPILE

#include <omp.h>
#include <iostream>
#include <thread>
#include <math.h>
#include <csignal>
#include <unistd.h>
#include <ros/ros.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <rosbag/message_instance.h>
#include <mutex>

#include <utils.h>
#include "Estimator.hpp"

class DataHandler
{
public:
    ros::NodeHandle nh;

    enum LiDAR_Type
    {
        Hesai = 0,
        VLS128 = 1,
        Ouster = 2,
        Unknown = -1
    };

    std::string lid_topic, imu_topic, gnss_topic;
    std::mutex mutex_lock;

    bool time_sync_en = false, extrinsic_est_en = false;

    double time_diff_lidar_to_imu = 0.0, GNSS_IMU_calibration_distance = 10., lidar_tod = 0.0;
    double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
    double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
    double filter_size_surf_min = 0, filter_size_map_min = 0;
    double lidar_end_time = 0, first_lidar_time = 0.0;
    int feats_down_size = 0, NUM_MAX_ITERATIONS = 0;

    bool lidar_pushed, flg_first_scan = true, flg_EKF_inited, als2mls_saved = false;
    bool scan_pub_en = false, dense_pub_en = false, map_init = false;

    double timediff_lidar_wrt_imu = 0.0;
    bool timediff_set_flg = false;

    double _first_lidar_time = 0., _first_imu_time = 0.;
    bool _lidar_init = false, _imu_init = false;

    double lidar_mean_scantime = 0.0;
    int scan_id = 0;
    double min_dist = 2, max_dist = 100;
    double max_dist_sq, min_dist_sq;
    bool gnss_init = false;
    int point_step = 1;

    // ALS
    bool downsample = false;
    int closest_N_files = 4;
    std::string folder_root = "";

    bool use_gnss = false;

    std::vector<double> extrinT_LiDAR2IMU, extrinR_LiDAR2IMU, extrinsic_T_GNSS2IMU;

    V3D Lidar_T_wrt_IMU, GNSS_T_wrt_IMU;
    M3D Lidar_R_wrt_IMU;

    double first_point_time_;

    std::deque<double> time_buffer;
    std::deque<PointCloudXYZI::Ptr> lidar_buffer;
    std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
    std::deque<gps_common::GPSFix::ConstPtr> gps_buffer;

    PointCloudXYZI::Ptr laserCloudSurfMap, featsFromMap, feats_undistort, feats_down_body, feats_down_world;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::CropBox<PointType> cropBoxFilter;

    MeasureGroup Measures;
    MAP_ estimator_;

    state state_point;
    V3D pos_lid;

    nav_msgs::Odometry odomAftMapped, odomGPS;

    LiDAR_Type lidar_type = DataHandler::Hesai; // Member variable to hold the LIDAR type
    std::string bag_file = "";

    template <typename T>
    void set_posestamp(T &out)
    {
        out.pose.position.x = state_point.pos(0);
        out.pose.position.y = state_point.pos(1);
        out.pose.position.z = state_point.pos(2);

        auto q_ = Eigen::Quaterniond(state_point.rot.matrix());
        out.pose.orientation.x = q_.coeffs()[0];
        out.pose.orientation.y = q_.coeffs()[1];
        out.pose.orientation.z = q_.coeffs()[2];
        out.pose.orientation.w = q_.coeffs()[3];
    }

    DataHandler(ros::NodeHandle &nh_)
    {
        nh = nh_;
        laserCloudSurfMap.reset(new PointCloudXYZI());
        featsFromMap.reset(new PointCloudXYZI());
        feats_undistort.reset(new PointCloudXYZI());
        feats_down_body.reset(new PointCloudXYZI());
        feats_down_world.reset(new PointCloudXYZI());

        nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
        nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
        nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        nh.param<std::string>("common/lid_topic", lid_topic, "/hesai/pandar");
        nh.param<std::string>("common/imu_topic", imu_topic, "/imu/data_raw");
        nh.param<std::string>("common/gnss_topic", gnss_topic, "/gps/gps");
        nh.param<bool>("common/time_sync_en", time_sync_en, false);
        nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        nh.param<double>("common/lidar_tod", lidar_tod, 0.0);

        nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
        nh.param<double>("filter_size_map", filter_size_map_min, 0.5);

        nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
        nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
        nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
        nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);

        nh.param<int>("preprocess/point_step", point_step, 1);

        int tmp_lidar_type = 0;
        nh.param<int>("preprocess/lidar_type", tmp_lidar_type, 0);
        if (tmp_lidar_type >= Hesai && tmp_lidar_type <= Ouster)
        {
            lidar_type = static_cast<LiDAR_Type>(tmp_lidar_type);
        }
        else
        {
            lidar_type = Unknown; // Handle invalid index
            std::cerr << "Invalid LIDAR type index: " << tmp_lidar_type << std::endl;
        }

        std::cout << "lidar_type:" << lidar_type << std::endl;
        nh.param<double>("preprocess/min_dist", min_dist, 1.0);
        nh.param<double>("preprocess/max_dist", max_dist, 100);
        max_dist_sq = max_dist * max_dist, min_dist_sq = min_dist * min_dist;
        nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
        nh.param<std::vector<double>>("mapping/extrinsic_T", extrinT_LiDAR2IMU, std::vector<double>());
        nh.param<std::vector<double>>("mapping/extrinsic_R", extrinR_LiDAR2IMU, std::vector<double>());
        nh.param<double>("mapping/GNSS_IMU_calibration_distance", GNSS_IMU_calibration_distance, 10.0);

        nh.param<std::string>("Bag", bag_file, "");
        std::cout << "bag_file:" << bag_file << std::endl;

        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

        Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT_LiDAR2IMU);
        Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR_LiDAR2IMU);

        // ALS
        nh.param<bool>("als/downsample", downsample, false);
        nh.param<bool>("als/use_gnss", use_gnss, false);
        nh.param<int>("als/closest_N_files", closest_N_files, 4);
        nh.param<std::string>("als/als_path", folder_root, "");
    };

    ~DataHandler() {};

    void msg2cloud(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
    void pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);
    void gps_cbk(const gps_common::GPSFix::ConstPtr &msg);

    bool sync_packages(MeasureGroup &meas);
    void pointBodyToWorld(PointType const *const pi, PointType *const po);
    void pointBodyLidarToIMU(PointType const *const pi, PointType *const po);
    void local_map_update();
    void publish_frame_world(const ros::Publisher &pubLaserCloudFull_);
    void publish_frame_body(const ros::Publisher &pubLaserCloudFull_);
    void publish_frame_debug(const ros::Publisher &pubLaserCloudFrame_, const PointCloudXYZI::Ptr &frame_);
    void publish_map(const ros::Publisher &pubLaserCloudMap);
    void publish_odometry(const ros::Publisher &pubOdomAftMapped);
    void publish_gnss_odometry(const Sophus::SE3 &gnss_pose);

    void Subscribe();
};

#endif