#ifndef USE_DATAHANDLER_H1
#define USE_DATAHANDLER_H1

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

#include "Estimator.hpp"

#include "RIEKF.hpp"

//#ifndef USE_EKF
#include "Estimator_ICP.hpp"
// #endif

#define INIT_TIME (0.1)

//#define LASER_POINT_COV (0.01)
// #define LASER_POINT_COV (0.05) //  5 cm used with gps
#define GNSS_VAR (0.05)
#define very_good_gnss_var (0.05) //this is for measurements from the postprocessed gnss-imu file

#define LASER_POINT_COV     (0.001)

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

    int add_point_size = 0;
    bool time_sync_en = false, extrinsic_est_en = false;

    float DET_RANGE = 300.0f;
    const float MOV_THRESHOLD = 1.0f; // 1.5f;
    double time_diff_lidar_to_imu = 0.0, GNSS_IMU_calibration_distance = 10., lidar_tod = 0.0;
    double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
    double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
    double filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
    double cube_len = 0, lidar_end_time = 0, first_lidar_time = 0.0;
    int feats_down_size = 0, NUM_MAX_ITERATIONS = 0, pcd_index = 0;

    bool lidar_pushed, flg_first_scan = true, flg_EKF_inited, als2mls_saved = false;
    bool scan_pub_en = false, dense_pub_en = false, map_init = false;

    double timediff_lidar_wrt_imu = 0.0;
    bool timediff_set_flg = false;

    double _first_lidar_time = 0., _first_imu_time = 0.;
    bool _lidar_init = false, _imu_init = false;

    bool shift_measurements_to_zero_time = false;
    
    double lidar_mean_scantime = 0.0;
    int scan_id = 0;
    bool Localmap_Initialized = false;
    double min_dist = 2, max_dist = 100;
    double max_dist_sq, min_dist_sq;
    bool gnss_init = false, save_clouds = false, save_clouds_local = false, save_poses=false;
    std::string save_clouds_path = "", poseSavePath = "";
    int point_step = 1;

    //ALS
    bool downsample = false;
    int closest_N_files = 4;
    std::string folder_root = "/media/eugeniu/T7/NLS_las/"; // this is the ALS from NlS
    // std::string folder_root = "/media/eugeniu/T7/EVO_las/"; // dense ALS

    bool use_gnss = false, saveALS_NN_2_MLS = false, use_ransac_alignment = false;
    std::string postprocessed_gnss_path = "";
    std::vector<PointVector> Nearest_Points;

    std::vector<double> extrinT_LiDAR2IMU, extrinR_LiDAR2IMU, extrinsic_T_GNSS2IMU;

    V3D Lidar_T_wrt_IMU, GNSS_T_wrt_IMU;
    M3D Lidar_R_wrt_IMU;
    
    double first_point_time_;
    
    std::deque<double> time_buffer;
    std::deque<PointCloudXYZI::Ptr> lidar_buffer;
    std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
    std::deque<gps_common::GPSFix::ConstPtr> gps_buffer;
#ifdef SAVE_DATA
    std::deque<sensor_msgs::PointCloud2::ConstPtr> lidar_msg_buffer;
#endif

    PointCloudXYZI::Ptr featsFromMap, feats_undistort, feats_down_body, feats_down_world;

    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    MeasureGroup Measures;

#if USE_STATIC_KDTREE == 0
    std::vector<BoxPointType> cub_needrm;
    KD_TREE<PointType> ikdtree;
    BoxPointType A;
    V3D prev_pos_LiD;
#else
    PointCloudXYZI::Ptr laserCloudSurfMap;
    pcl::CropBox<PointType> cropBoxFilter;
#endif

#ifdef USE_EKF
    RIEKF estimator_;
#else
    ICP estimator_;
#endif  

    ICP estimator_icp;
    
    state state_point, updated_state, before_als_state;
    V3D pos_lid;

    nav_msgs::Odometry odomAftMapped, odomGPS;
    geometry_msgs::PoseStamped msg_body_pose;

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
        featsFromMap.reset(new PointCloudXYZI());
        feats_undistort.reset(new PointCloudXYZI());
        feats_down_body.reset(new PointCloudXYZI());
        feats_down_world.reset(new PointCloudXYZI());

#if USE_STATIC_KDTREE == 1
        laserCloudSurfMap.reset(new PointCloudXYZI());
        std::cout << "Using static kdtree" << std::endl;
#else
        std::cout << "Using ikdtree" << std::endl;
#endif
        nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
        nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
        nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        nh.param<std::string>("common/lid_topic", lid_topic, "/livox/lidar");
        nh.param<std::string>("common/imu_topic", imu_topic, "/livox/imu");
        nh.param<std::string>("common/gnss_topic", gnss_topic, "/gps/gps");
        nh.param<bool>("common/time_sync_en", time_sync_en, false);
        nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);

        nh.param<double>("common/lidar_tod", lidar_tod, 0.0);
        
        nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
        nh.param<double>("filter_size_map", filter_size_map_min, 0.5);

        nh.param<double>("cube_side_length", cube_len, 200);
        nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
        nh.param<double>("mapping/fov_degree", fov_deg, 180);
        nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
        nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
        nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
        nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);

        nh.param<int>("preprocess/point_step", point_step, 1);
        std::cout<<"point_step:"<<point_step<<std::endl;

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
        nh.param<std::vector<double>>("mapping/extrinsic_T_GNSS2IMU", extrinsic_T_GNSS2IMU, std::vector<double>());
        nh.param<double>("mapping/GNSS_IMU_calibration_distance", GNSS_IMU_calibration_distance, 10.0);

        nh.param<std::string>("Bag", bag_file, "");
        std::cout << "bag_file:" << bag_file << std::endl;

        std::cout<<"filter_size_surf_min:"<<filter_size_surf_min<<std::endl;
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT_LiDAR2IMU);
        Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR_LiDAR2IMU);
        GNSS_T_wrt_IMU << VEC_FROM_ARRAY(extrinsic_T_GNSS2IMU);

        //ALS
        nh.param<bool>("als/downsample", downsample, false);
        nh.param<bool>("als/use_gnss", use_gnss, false);
        nh.param<bool>("als/use_ransac_alignment", use_ransac_alignment, false);
        
        nh.param<bool>("als/saveALS_NN_2_MLS", saveALS_NN_2_MLS, false);
        nh.param<int>("als/closest_N_files", closest_N_files, 4);
        nh.param<std::string>("als/als_path", folder_root, "");
        nh.param<std::string>("als/postprocessed_gnss_path", postprocessed_gnss_path, "");

#ifdef SAVE_DATA
        std::cout<<"Built with save mode on..."<<std::endl;
        nh.param<bool>("publish/save_clouds", save_clouds, false);
        nh.param<bool>("publish/save_clouds_local", save_clouds_local, false);
        if (save_clouds)
        {
            nh.param<std::string>("publish/save_clouds_path", save_clouds_path, "");
            if (save_clouds_path.empty())
            {
                save_clouds = false;
                std::cout << "\033[31msave_clouds_path is empty, set a proper path\033[0m" << std::endl;
            }
            else
            {
                std::cout << "\033[32mSave the clouds:\033[0m" << save_clouds_path << std::endl;
            }
        }
        
        nh.param<bool>("publish/save_poses", save_poses, false);
        if(save_poses)
        {
            nh.param<std::string>("publish/poseSavePath", poseSavePath, "");
            if (poseSavePath.empty())
            {
                save_poses = false;
                std::cout << "\033[31mposeSavePath is empty, set a proper path\033[0m" << std::endl;
            }
            else
            {
                // Check if directory exists
                struct stat info;
                if (stat(poseSavePath.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR))
                {
                    save_poses = false;
                    std::cout << "\033[31mDirectory does not exist: " << poseSavePath << "\033[0m" << std::endl;
                }
                else
                {
                    std::cout << "\033[32mSave the poses to: " << poseSavePath << "\033[0m" << std::endl;
                }
            }
            namespace fs = boost::filesystem;

                fs::path dir(poseSavePath);

                if (!fs::exists(dir)) {
                    throw std::runtime_error("ERROR: Directory does not exist: " + poseSavePath);
                }
                if (!fs::is_directory(dir)) {
                    throw std::runtime_error("ERROR: Path exists but is not a directory: " + poseSavePath);
                }
                std::cout << "[INFO] Directory exists: " << poseSavePath << std::endl;
        }
#endif

    };

    ~DataHandler() {};

    void msg2cloud(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);

    void pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);
    void gps_cbk(const gps_common::GPSFix::ConstPtr &msg);

    bool sync_packages(MeasureGroup &meas);
    bool sync_packages_no_IMU(MeasureGroup &meas);

    void pointBodyToWorld(PointType const *const pi, PointType *const po);
    void pointBodyLidarToIMU(PointType const *const pi, PointType *const po);

    void local_map_update();
    void local_map_update_from_ALS(const std::vector<PointVector> &Nearest_Points);
    void RemovePointsFarFromLocation();

    void publish_frame_world(const ros::Publisher &pubLaserCloudFull_);
    void publish_frame_body(const ros::Publisher &pubLaserCloudFull_);
    void publish_frame_debug(const ros::Publisher &pubLaserCloudFrame_, const PointCloudXYZI::Ptr &frame_);
    void publish_map(const ros::Publisher &pubLaserCloudMap);
    void publish_odometry(const ros::Publisher &pubOdomAftMapped);
    void publish_gnss_odometry(const Sophus::SE3 &gnss_pose);

    void BagHandler();
    void BagHandler_Arvo();
};

#endif