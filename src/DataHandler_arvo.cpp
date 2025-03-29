
#include "DataHandler.hpp"

#include "IMU.hpp"
#include "GNSS.hpp"

#ifndef USE_EKF
#include "PoseGraph.hpp"
#endif

#ifdef USE_ALS
#include "ALS.hpp"
#endif

#include "arvo_reader.hpp"

volatile bool flg_exit2 = false;
void SigHandle2(int sig)
{
    ROS_WARN("Caught signal %d, stopping...", sig);
    flg_exit2 = true; // Set the flag to stop the loop
}

//--------------------------------------------------------------------------------------------
struct DataRow
{
    double toh;
    double utcTime;
    double gpsTime;
    double easting;
    double northing;
    double hEll;
    double omega;
    double phi;
    double kappa;
    int q;
    double sdHori;
    double sdHeig;
    double velBdyX;
    double velBdyY;
    double velBdyZ;
    double vEast;
    double vNorth;
    double vUp;
    double angRateX;
    double angRateY;
    double angRateZ;
    double accBdyX;
    double accBdyY;
    double accBdyZ;
};

inline sensor_msgs::Imu::ConstPtr createImuMessage(const std::vector<DataRow> &gnss_data, size_t gnss_index, Sophus::SE3 &_gt_gnss)
{
    sensor_msgs::Imu::Ptr msg_in(new sensor_msgs::Imu());

    // Convert angular rates from degrees/s to radians/s
    double AngRateX_rad = gnss_data[gnss_index].angRateX * (M_PI / 180.0); // Convert degrees to radians
    double AngRateY_rad = gnss_data[gnss_index].angRateY * (M_PI / 180.0);
    double AngRateZ_rad = gnss_data[gnss_index].angRateZ * (M_PI / 180.0);

    // Assign angular rates
    msg_in->angular_velocity.x = AngRateX_rad;
    msg_in->angular_velocity.y = AngRateY_rad;
    msg_in->angular_velocity.z = AngRateZ_rad;

    double x = gnss_data[gnss_index].easting;
    double y = gnss_data[gnss_index].northing;
    double z = gnss_data[gnss_index].hEll;

    // these are degrees - convert to radians
    double omega = gnss_data[gnss_index].omega * (M_PI / 180.0);
    double phi = gnss_data[gnss_index].phi * (M_PI / 180.0);
    double kappa = gnss_data[gnss_index].kappa * (M_PI / 180.0);

    auto se3 = xyzypr2tf_(x, y, z, kappa, phi, omega); // tf

    V3D acceleration_in_body_no_gravity(gnss_data[gnss_index].accBdyX, gnss_data[gnss_index].accBdyY, gnss_data[gnss_index].accBdyZ);

    tf::Quaternion tfQuaternion = se3.getRotation();
    Eigen::Quaterniond eigenQuaternion(tfQuaternion.w(), tfQuaternion.x(), tfQuaternion.y(), tfQuaternion.z());
    M3D R_imu = eigenQuaternion.toRotationMatrix();

    V3D translation;
    translation << se3.getOrigin().getX(), se3.getOrigin().getY(), se3.getOrigin().getZ();

    Sophus::SE3 gt_pose(R_imu, translation);
    _gt_gnss = gt_pose;
    // defined somewhere G_m_s2 = 9.81;
    V3D accel_body_with_g = gt_pose.so3().inverse() * ((gt_pose.so3() * acceleration_in_body_no_gravity) + V3D(0, 0, G_m_s2));

    // std::cout << "acceleration_in_body_no_gravity:" << acceleration_in_body_no_gravity.transpose() << std::endl;
    // std::cout << "accel_body_with_g              :" << accel_body_with_g.transpose() << std::endl;

    // Assign linear accelerations
    msg_in->linear_acceleration.x = accel_body_with_g[0];
    msg_in->linear_acceleration.y = accel_body_with_g[1];
    msg_in->linear_acceleration.z = accel_body_with_g[2];

    msg_in->header.stamp = ros::Time(gnss_data[gnss_index].toh);
    msg_in->header.frame_id = "IMU_frame";

    return msg_in;
}

std::vector<DataRow> get_gnss(std::string &txt_file)
{
    std::vector<DataRow> data;
    std::uint64_t fullWeekSecs = 0;
    std::string line;
    std::ifstream file(txt_file);

    std::vector<unsigned int> yearMonthDay = {2021, 3, 31}; // 31 march 2021
    fullWeekSecs = dateTime2UnixTime_(yearMonthDay[0], yearMonthDay[1], yearMonthDay[2]);
    fullWeekSecs -= 86400. * getDayOfWeekIndex_(fullWeekSecs);

    while (std::getline(file, line))
    {
        if (line.find("UTCTime") != std::string::npos)
        {
            std::getline(file, line); // the units row
            break;
        }
    }

    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        DataRow row;

        iss >> row.utcTime >> row.gpsTime >> row.easting >> row.northing >> row.hEll;
        iss >> row.omega >> row.phi >> row.kappa >> row.q >> row.sdHori >> row.sdHeig;
        iss >> row.velBdyX >> row.velBdyY >> row.velBdyZ >> row.vEast >> row.vNorth >> row.vUp;
        iss >> row.angRateX >> row.angRateY >> row.angRateZ >> row.accBdyX >> row.accBdyY >> row.accBdyZ;

        //row.toh = std::fmod(row.gpsTime, 3600.);
        row.toh = std::fmod(row.utcTime, 3600.);

        // row.utcTime += fullWeekSecs;
        // row.gpsTime += fullWeekSecs;

        data.push_back(row);
    }
    return data;
}

void DataHandler::BagHandler_Arvo()
{
    std::cout << std::fixed << std::setprecision(12);

    std::cout << "\n===============================BagHandler_Arvo===============================" << std::endl;
#ifdef MP_EN
    std::cout << "Open_MP is available" << std::endl;
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

    signal(SIGINT, SigHandle2); // Handle Ctrl+C (SIGINT)
    flg_exit2 = false;

    //----------------------------------------
    std::string bag_directory = "/media/eugeniu/T7 Shield/Masalantie_data";

    // Get all bag files in the directory
    std::vector<std::string> bag_files;
    for (const auto &entry : boost::filesystem::directory_iterator(bag_directory))
    {
        if (boost::filesystem::is_regular_file(entry) &&
            entry.path().filename().string().find("cloud_Masalantie_itaan_lidar_VLS128_Center") == 0 &&
            entry.path().extension() == ".bag")
        {
            bag_files.push_back(entry.path().string());
        }
    }
    std::cout << "there are " << bag_files.size() << " bags" << std::endl;
    std::sort(bag_files.begin(), bag_files.end());

    std::vector<std::string> topics;
    topics.push_back(lid_topic);

    // read the gnss-imu data

    std::string trajectory_file = "/media/eugeniu/T7 Shield/Masalantie_data/20210331_masalantie_itaan_vrs.txt";
    std::vector<DataRow> gnss_data = get_gnss(trajectory_file);
    std::cout << "gnss_data:" << gnss_data.size() << std::endl;
    std::cout << "weekTimeSec:" << gnss_data[0].gpsTime << ", toh:" << gnss_data[0].toh << std::endl;
    int gps_index = 0, total_gnss = gnss_data.size();

    {
        double E = 363116., N = 6671521.;
        double dist = 0;
        while (gps_index < total_gnss)
        {
            dist = std::pow(gnss_data[gps_index].easting - E, 2) + std::pow(gnss_data[gps_index].northing - N, 2);
            dist = sqrt(dist);
            std::cout << "dist:" << dist << " gps_index:" << gps_index << "/" << total_gnss << std::endl;
            if (dist < 5)
            {
                break;
            }
            gps_index++;
        }
    }

    Sophus::SE3 first_gnss_pose;

    bool gps_lidar_time_aligned = false;
    bool do_once = true;
    Sophus::SE3 mls2als; // als_to_mls
    M3D r1;
    V3D t1;
    for (const auto &bag_file : bag_files)
    {
        if (flg_exit2)
            break;
        ROS_INFO("\nOpening bag: %s", bag_file.c_str());
        rosbag::Bag bag;
        try
        {
            bag.open(bag_file, rosbag::bagmode::Read);

            rosbag::View view(bag, rosbag::TopicQuery(topics));
            for (const rosbag::MessageInstance &m : view)
            {
                if (flg_exit2)
                    break;

                std::string topic = m.getTopic();
                if (topic == lid_topic)
                {
                    sensor_msgs::PointCloud2::ConstPtr pcl_msg = m.instantiate<sensor_msgs::PointCloud2>();
                    if (pcl_msg)
                    {
                        PointCloudXYZI::Ptr ptr_tmp(new PointCloudXYZI());
                        msg2cloud(pcl_msg, ptr_tmp);

                        if (gps_index >= total_gnss)
                        {
                            std::cout << "Reatched the end of gnss file" << std::endl;
                            break;
                        }

                        double diff = 0;
                        if (gnss_data[gps_index].toh > first_point_time_)
                        {
                            diff = fabs(first_point_time_ - gnss_data[gps_index].toh);
                            std::cout << "Drop scan first_point_time_:" << first_point_time_ << ", GNSS toh:" << gnss_data[gps_index].toh << std::endl;
                            continue;
                        }

                        sensor_msgs::PointCloud2 modified_pcl_msg = *pcl_msg;
                        modified_pcl_msg.header.stamp = ros::Time(first_point_time_);

                        sensor_msgs::PointCloud2::Ptr modified_pcl_msg_ptr = boost::make_shared<sensor_msgs::PointCloud2>(modified_pcl_msg);

                        // pcl_cbk(pcl_msg);
                        pcl_cbk(modified_pcl_msg_ptr);

                        if (!gps_lidar_time_aligned)
                        {
                            std::cout << "first_point_time_:" << first_point_time_ << ", GNSS toh:" << gnss_data[gps_index].toh << std::endl;
                            diff = fabs(first_point_time_ - gnss_data[gps_index].toh);
                            std::cout << "diff:" << diff << std::endl;
                            if (gnss_data[gps_index].toh > first_point_time_) // gps is in the future - drop scans
                            {
                                // It will drop scans until the lidar will be in the future
                                std::cout << "gps is in the future - drop scans" << std::endl;
                                lidar_buffer.pop_front();
                                time_buffer.pop_front();
                            }
                            else // lidar is in the future drop gps
                            {
                                std::cout << "lidar is in the future drop gps" << std::endl;
                                double prev_diff = diff;
                                while (abs(diff) > 0.1 && gps_index < total_gnss)
                                {
                                    gps_index++;
                                    diff = first_point_time_ - gnss_data[gps_index].toh;
                                    std::cout << "diff:" << diff << ", GPS time:" << gnss_data[gps_index].toh << " first_point_time_:" << first_point_time_ << " gps_index:" << gps_index << "/" << total_gnss << std::endl;
                                    if (abs(prev_diff) < abs(diff))
                                    {
                                        std::cout << "prev_diff:" << prev_diff << " stop here" << std::endl;
                                        break;
                                    }
                                    prev_diff = diff;
                                }
                                std::cout << "\n Data is aligned \n"
                                          << std::endl;
                                gps_lidar_time_aligned = true;

                                sensor_msgs::Imu::ConstPtr imu_msg = createImuMessage(gnss_data, gps_index, first_gnss_pose);
                                first_gnss_pose = first_gnss_pose.inverse();
                            }

                            continue;
                        }

                        // std::cout << "\nlidar_buffer:" << lidar_buffer.size() << ", time_buffer:" << time_buffer.size() << std::endl;

                        double start_time = first_point_time_;
                        double end_time = start_time + lidar_buffer.front()->points[0].time;
                        std::cout<<"gps_index:"<<gps_index<<"/"<<total_gnss<<std::endl;

                        Sophus::SE3 gnss_pose_from_postprocessed_file;
                        while (gps_index < total_gnss && gnss_data[gps_index].toh <= end_time)
                        {
                            sensor_msgs::Imu::ConstPtr imu_msg = createImuMessage(gnss_data, gps_index, gnss_pose_from_postprocessed_file);
                            imu_cbk(imu_msg);
                            gps_index++;
                        }
                        // std::cout << "imu_buffer:" << imu_buffer.size() << std::endl;

#define compile_this
#ifdef compile_this
                        if (sync_packages(Measures))
                        {
                            std::cout << "\nIMU:" << imu_buffer.size() << ", GPS:" << gps_buffer.size() << ", LiDAR:" << lidar_buffer.size() << std::endl;

                            double t00 = omp_get_wtime();
                            if (flg_first_scan)
                            {
                                first_lidar_time = Measures.lidar_beg_time;
                                flg_first_scan = false;
                                continue;
                            }

                            Sophus::SE3 gnss_pose = first_gnss_pose * gnss_pose_from_postprocessed_file;
                            std::cout << "gnss_pose:" << gnss_pose.translation().transpose() << std::endl;

                            imu_obj->Process(Measures, estimator_, feats_undistort);

                            if (!imu_obj->init_from_GT)
                            {
                                imu_obj->IMU_init_from_GT(Measures, estimator_, gnss_pose);
                            }

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
                            if (do_once)
                            {
                                // discard the gyro bias since no static position
                                state_point.bg = Zero3d;
                                estimator_.set_x(state_point);
                                do_once = false;
                            }
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
                            std::cout << "feats_down_body:" << feats_down_body->size() << std::endl;

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
                            // Crop the local map----------------------------------------------------
                            state_point = estimator_.get_x();
                            before_als_state = estimator_.get_x();
                            // get the pos and rot from the GNSS
                            /// state_point.pos = gnss_pose.translation();
                            // state_point.rot = gnss_pose.so3();
                            pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

                            std::cout << "offset_T_L_I:" << state_point.offset_T_L_I.transpose() << std::endl;
                            std::cout << "offset_R_L_I\n:" << state_point.offset_R_L_I << std::endl;

                            RemovePointsFarFromLocation();

                            // get and publish the GNSS pose-----------------------------------------
                            // gnss_obj->Process(gps_buffer, lidar_end_time, state_point.pos);
                            gnss_obj->Process(gnss_pose, gnss_pose_from_postprocessed_file, lidar_end_time, state_point.pos);
                            gnss_pose = (gnss_obj->use_postprocessed_gnss) ? gnss_obj->postprocessed_gps_pose : gnss_obj->gps_pose;

                            // gnss_pose.so3() = state_point.rot; // use the MLS orientation
                            publish_gnss_odometry(gnss_pose);

                            {
                                if (gnss_obj->GNSS_extrinsic_init && use_gnss) // if gnss aligned
                                {
                                    const bool global_error = false; // set this true for global error of gps
                                    // auto gps_cov_ = V3D(GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR);
                                    // auto gps_cov_ = gnss_obj->gps_cov;
                                    // auto gps_cov_ = Eigen::Vector3d(GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR, GNSS_VAR * GNSS_VAR);
                                    auto gps_cov_ = Eigen::Vector3d(very_good_gnss_var * very_good_gnss_var, very_good_gnss_var * very_good_gnss_var, very_good_gnss_var * very_good_gnss_var);

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
                                        if (als_obj->init(gnss_obj->origin_enu, gnss_obj->R_GNSS_to_MLS, featsFromMap))
                                        {
                                            gnss_obj->updateExtrinsic(als_obj->R_to_mls); // use the scan-based refined rotation for GNSS
                                        };
                                    }
                                }
                                else
                                {
                                    // std::cout << "ALS refined - use it for registration " << std::endl;
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
                            }

                            // Update the local map--------------------------------------------------
                            feats_down_world->resize(feats_down_size);
                            local_map_update();

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

#else
// the icp variant was here
#endif
                        }

#endif

#ifdef SAVE_DATA
                        if (als_obj->refine_als)
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

                                mls2als = als_obj->als_to_mls.inverse();
                                r1 = als_obj->als_to_mls.so3().matrix().transpose();
                                t1 = -r1 * als_obj->als_to_mls.translation();

                                als2mls_saved = true;
                            }
                            if (save_clouds)
                            {
                                switch (lidar_type)
                                {
                                case Hesai:
                                    std::cout << "Hesai save the cloud" << std::endl;
                                    {
                                        const pcl::PointCloud<hesai_ros::Point> &pl_orig = imu_obj->DeSkewOriginalCloud<hesai_ros::Point>(Measures.lidar_msg, state_point, save_clouds_local);
                                        std::cout << "save " << pl_orig.size() << " points" << std::endl;

                                        std::string filename = save_clouds_path + std::to_string(pcd_index) + "_cloud_" + std::to_string(lidar_end_time) + ".pcd";
                                        pcl::io::savePCDFile(filename, pl_orig, true); // Binary format
                                    }
                                    break;

                                case VLS128:
                                    std::cout << "Velodyne save the cloud" << std::endl;
                                    {
                                        //state_point = before_als_state;
                                        const pcl::PointCloud<velodyne_ros::Point> &pl_orig = imu_obj->DeSkewOriginalCloud<velodyne_ros::Point>(Measures.lidar_msg, state_point, save_clouds_local);
                                        std::cout << "save " << pl_orig.size() << " points" <<"feats_undistort:"<<feats_undistort->size()<< std::endl;

                                        std::string filename = save_clouds_path + "mls_" + std::to_string(pcd_index) + "_cloud_" + std::to_string(lidar_end_time) + ".pcd";
                                        pcl::io::savePCDFile(filename, pl_orig, true); // Binary format

                                        if (true)
                                        {
                                            // add option to transform it to global ALS frame use the inverse of the ALS2MLS
                                            pcl::PointCloud<velodyne_ros::Point> pl_transformed;
                                            pl_transformed.reserve(pl_orig.size());

                                            Eigen::Matrix3d R = r1; //mls2als.so3().matrix();
                                            Eigen::Vector3d t = t1; //  mls2als.translation();

                                            for (const auto &pt : pl_orig)
                                            {
                                                velodyne_ros::Point transformed_pt = pt;

                                                V3D p_orig(pt.x, pt.y, pt.z);
                                                V3D p_transformed = R * p_orig + t;

                                                transformed_pt.x = p_transformed.x();
                                                transformed_pt.y = p_transformed.y();
                                                transformed_pt.z = p_transformed.z();

                                                pl_transformed.push_back(transformed_pt);
                                            }
                                            std::cout << "save the transformed cloud to ALS frame " << pl_transformed.size() << std::endl;

                                            filename = save_clouds_path + "als_" + std::to_string(pcd_index) + "_cloud_" + std::to_string(lidar_end_time) + ".pcd";
                                            pcl::io::savePCDFile(filename, pl_transformed, true); // Binary format
                                        }
                                    }
                                    break;

                                case Ouster:
                                    std::cout << "Ouster not implemented" << std::endl;
                                    break;

                                default:
                                    std::cout << "Unknown LIDAR type:" << lidar_type << std::endl;
                                    break;
                                }
                            }

                            if (save_poses)
                            {
                                const V3D &t_model = state_point.pos;
                                Eigen::Quaterniond q_model(state_point.rot.matrix());
                                q_model.normalize();

                                std::ofstream foutMLS(poseSavePath + "MLS.txt", std::ios::app);
                                // foutMLS.setf(std::ios::scientific, std::ios::floatfield);
                                foutMLS.setf(std::ios::fixed, std::ios::floatfield);
                                foutMLS.precision(20);
                                // # ' id tx ty tz qx qy qz qw' - tum format(scan id, scan timestamp seconds, translation and rotation quaternion)
                                foutMLS << pcd_index << " " << std::to_string(lidar_end_time) << " " << t_model(0) << " " << t_model(1) << " " << t_model(2) << " "
                                        << q_model.x() << " " << q_model.y() << " " << q_model.z() << " " << q_model.w() << std::endl;
                                foutMLS.close();
                            }

                            pcd_index++;
                        }
#endif
                    }
                }
            }
            // break; // remove later

            bag.close();
        }
        catch (const std::exception &e)
        {
            ROS_ERROR("Error reading bag %s: %s", bag_file.c_str(), e.what());
        }
    }
    std::cout << "Stop function" << std::endl;
}
