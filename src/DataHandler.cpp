

#include "DataHandler.hpp"

#include "IMU.hpp"
#include "GNSS.hpp"

#ifdef USE_ALS
#include "ALS.hpp"
#endif

#include <GeographicLib/UTMUPS.hpp>

#include <liblas/liblas.hpp>
#include <chrono>

void DataHandler::publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "world";
    odomAftMapped.child_frame_id = "MLS"; //"MLS" moves relative to "world"
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);

    auto P = estimator_.get_P();

    // float64[36] covariance,  its a 6x6 mat
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "world", "MLS"));

    // tf::Transform transform_inv = transform.inverse();
    // static tf::TransformBroadcaster br2;
    // br2.sendTransform(tf::StampedTransform(transform_inv, odomAftMapped.header.stamp, "MLS", "world"));
}

void DataHandler::publish_gnss_odometry(const Sophus::SE3 &gnss_pose)
{
    tf::Transform transformGPS;
    tf::Quaternion q;
    static tf::TransformBroadcaster br;

    auto t = gnss_pose.translation();
    auto R_yaw = gnss_pose.so3().matrix();

    transformGPS.setOrigin(tf::Vector3(t[0], t[1], t[2]));
    Eigen::Quaterniond quat_(R_yaw);
    q = tf::Quaternion(quat_.x(), quat_.y(), quat_.z(), quat_.w());
    transformGPS.setRotation(q);
    br.sendTransform(tf::StampedTransform(transformGPS, ros::Time().fromSec(lidar_end_time), "world", "GPSFix"));

    // tf::Transform transform_inv = transformGPS.inverse();
    // static tf::TransformBroadcaster br2;
    // br2.sendTransform(tf::StampedTransform(transform_inv, ros::Time().fromSec(lidar_end_time), "GPSFix", "world"));
}

void DataHandler::publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "world";
    pubLaserCloudMap.publish(laserCloudMap);
}

void DataHandler::publish_frame_world(const ros::Publisher &pubLaserCloudFull_)
{
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    tbb::parallel_for(tbb::blocked_range<int>(0, size), [&](tbb::blocked_range<int> r)
                      {
                    for (int i = r.begin(); i < r.end(); i++)
                    //for (int i = 0; i < size; i++)
                    {
                        pointBodyToWorld(&laserCloudFullRes->points[i],
                                        &laserCloudWorld->points[i]);
                    } });

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "world";
    pubLaserCloudFull_.publish(laserCloudmsg);
}

void DataHandler::publish_frame_body(const ros::Publisher &pubLaserCloudFull_body)
{
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    tbb::parallel_for(tbb::blocked_range<int>(0, size), [&](tbb::blocked_range<int> r)
                      {
                    for (int i = r.begin(); i < r.end(); i++)
                    {
                         pointBodyLidarToIMU(&laserCloudFullRes->points[i],
                                         &laserCloudWorld->points[i]);

                         //laserCloudWorld->points[i] = laserCloudFullRes->points[i];
                    } });

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    // pcl::toROSMsg(*laserCloudFullRes, laserCloudmsg);

    // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.stamp = ros::Time(0); // <- "no transform needed"

    laserCloudmsg.header.frame_id = "MLS";
    pubLaserCloudFull_body.publish(laserCloudmsg);
}

void DataHandler::publish_frame_debug(const ros::Publisher &pubLaserCloudFrame_, const PointCloudXYZI::Ptr &frame_)
{
    if (pubLaserCloudFrame_.getNumSubscribers() == 0)
        return;

    std::cout << "publish_frame_debug frame_:" << frame_->size() << std::endl;
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*frame_, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "world";
    pubLaserCloudFrame_.publish(laserCloudmsg);
}

void DataHandler::local_map_update()
{
    tbb::parallel_for(tbb::blocked_range<int>(0, feats_down_size),
                      [&](tbb::blocked_range<int> r)
                      {
                          for (int i = r.begin(); i < r.end(); i++)
                          // for (int i = 0; i < feats_down_size; i++)
                          {
                              pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                          }
                      });

    *laserCloudSurfMap += *feats_down_world;

    pcl::PointCloud<PointType>::Ptr tmpSurf(new pcl::PointCloud<PointType>());
    double local_map_radius = 75; // 100;

    double x_min = state_point.pos.x() - local_map_radius;
    double y_min = state_point.pos.y() - local_map_radius;
    double z_min = state_point.pos.z() - local_map_radius;
    double x_max = state_point.pos.x() + local_map_radius;
    double y_max = state_point.pos.y() + local_map_radius;
    double z_max = state_point.pos.z() + local_map_radius;

    // ROS_INFO("size : %f,%f,%f,%f,%f,%f", x_min, y_min, z_min,x_max, y_max, z_max);
    cropBoxFilter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1));
    cropBoxFilter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1));
    cropBoxFilter.setNegative(false);

    cropBoxFilter.setInputCloud(laserCloudSurfMap);
    cropBoxFilter.filter(*tmpSurf);

    downSizeFilterSurf.setInputCloud(tmpSurf);
    downSizeFilterSurf.filter(*laserCloudSurfMap);
}

void DataHandler::pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);

    V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
    po->time = pi->time;
    //po->ring = pi->ring;
}

void DataHandler::pointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
    po->time = pi->time;
    //po->ring = pi->ring;
}

void DataHandler::gps_cbk(const gps_common::GPSFix::ConstPtr &msg)
{
    auto status = msg->status.status;
    if (status != 0)
    {
        std::cout << "status:" << status << std::endl;
        std::cout << "Unable to get a fix on the location." << std::endl;
        return;
    }

    if (std::isnan(msg->latitude + msg->longitude + msg->altitude))
    {
        std::cout << "is nan GPS" << std::endl;
        return;
    }

    gps_buffer.push_back(msg);
}

void DataHandler::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);

    if (!_imu_init)
    {
        _imu_init = true;
        _first_imu_time = msg_in->header.stamp.toSec();
    }

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        // std::cout << "change IMU time with timediff_lidar_wrt_imu:" << timediff_lidar_wrt_imu << std::endl;
        msg->header.stamp =
            ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    if (timestamp < last_timestamp_imu)
    {
        std::cout << "imu loop back, clear buffer" << std::endl;
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;
    imu_buffer.push_back(msg);
}

void DataHandler::pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg_in)
{
    // std::cout<<"\npcl_cbk msg_in->header.stamp.toSec()->"<<msg_in->header.stamp.toSec()<<", lidar_buffer:"<<lidar_buffer.size()<<std::endl;

    sensor_msgs::PointCloud2::Ptr msg(new sensor_msgs::PointCloud2(*msg_in));

    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    if (!_lidar_init)
    {
        _lidar_init = true;
        _first_lidar_time = msg->header.stamp.toSec();
    }

    last_timestamp_lidar = msg->header.stamp.toSec();
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
        std::cout << "dt:" << abs(last_timestamp_imu - last_timestamp_lidar) << std::endl;
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        std::cout << "Self sync IMU and LiDAR, time diff is " << timediff_lidar_wrt_imu << std::endl;
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    msg2cloud(msg, ptr);

    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
}

void DataHandler::msg2cloud(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
    double first_point_time, range;
    size_t index;
    switch (lidar_type)
    {
    case Hesai:
        // std::cout << "Hesai" << std::endl;
        {
            pcl::PointCloud<hesai_ros::Point> pl_orig;
            pcl::fromROSMsg(*msg, pl_orig);

            int n = pl_orig.points.size();
            pcl_out->resize(n / point_step);
            first_point_time = pl_orig.points[0].timestamp;

            // std::cout<<"first_point_time:"<<first_point_time<<", last point time:"<<pl_orig.points[n-1].timestamp<<std::endl;

            index = 0;
            for (int i = 0; i < n; i += point_step)
            {
                const auto &point = pl_orig.points[i];
                range = point.x * point.x + point.y * point.y + point.z * point.z;

                if (range < min_dist_sq || range > max_dist_sq)
                    continue;

                // Assign to the preallocated index
                pcl_out->points[index].x = point.x;
                pcl_out->points[index].y = point.y;
                pcl_out->points[index].z = point.z;
                pcl_out->points[index].intensity = sqrt(sqrt(range));             // Save the sqrt range in the intensity field
                pcl_out->points[index].time = point.timestamp - first_point_time; // Time relative to first point
                // pcl_out->points[index].ring = point.ring;

                index++;
            }
            pcl_out->resize(index);
        }
        break;

    case VLS128:
        std::cout << "VLS128 not implemented" << std::endl;
        break;

    case Ouster:
        std::cout << "Ouster not implemented" << std::endl;
        break;

        // TODO: add support for more lidars

    default:
        std::cout << "Unknown LIDAR type:" << lidar_type << std::endl;
        break;
    }
}

bool DataHandler::sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        std::cout << "lidar_buffer:" << lidar_buffer.size() << ", imu_buffer:" << imu_buffer.size() << std::endl;
        return false;
    }

    if (!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front() + meas.lidar->points.front().time;
        lidar_end_time = time_buffer.front() + meas.lidar->points.back().time;
        meas.lidar_end_time = lidar_end_time;
        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();

    while ((!imu_buffer.empty())) //&& (imu_time < lidar_end_time)
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
        {
            // std::cout << "IMU time from future, imu_time:" << imu_time << ", lidar_end_time:" << lidar_end_time << std::endl;
            break;
        }

        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    if (!lidar_pushed || meas.imu.empty())
    {
        std::cout << "\n\nIssue in sync_packages - the data in not synched\n\n"
                  << std::endl;
        // throw std::runtime_error("\n\nIssue in sync_packages - the data in not synched\n\n");

        return false;
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

volatile bool flg_exit = false;
void SigHandle(int sig)
{
    ROS_WARN("Caught signal %d, stopping...", sig);
    flg_exit = true; // Set the flag to stop the loop
}

void DataHandler::Subscribe()
{
    std::cout << "--------------------Subscribe------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(12);
    std::cerr << std::fixed << std::setprecision(12);

    std::shared_ptr<IMU_Class> imu_obj(new IMU_Class());
    std::shared_ptr<GNSS> gnss_obj(new GNSS());

    imu_obj->set_param(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, 10. * acc_cov),
                       V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov), V3D(b_acc_cov, b_acc_cov, 10. * b_acc_cov));
    gnss_obj->set_param(GNSS_IMU_calibration_distance);

#ifdef USE_ALS
    std::shared_ptr<ALS_Handler> als_obj = std::make_shared<ALS_Handler>(folder_root, downsample, closest_N_files, filter_size_surf_min);
    ros::Publisher pubLaserALSMap = nh.advertise<sensor_msgs::PointCloud2>("/ALS_map", 1000);
#endif

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100);

    Sophus::SE3 curr_mls, prev_mls;
    ros::Rate rate(500);

    int scan_id = 0;
    bool als_integrated = false;

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

    int iters = 0,pcd_index = 0;
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

            gnss_obj->Process(gps_buffer, lidar_end_time, state_point.pos);
            Sophus::SE3 gnss_pose = (gnss_obj->use_postprocessed_gnss) ? gnss_obj->postprocessed_gps_pose : gnss_obj->gps_pose;
            // gnss_pose.so3() = state_point.rot; // use the MLS orientation
            // if (als_integrated) // if (use_gnss)
            publish_gnss_odometry(gnss_pose);

            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                flg_first_scan = false;
                curr_mls = Sophus::SE3(state_point.rot, state_point.pos);
                prev_mls = curr_mls;
                continue;
            }

            //  undistort and provide initial guess
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

#ifdef USE_ALS
            bool use_als_update = false;
            bool use_se3_update = false, use_se3_rel = false;
            Sophus::SE3 absolute_se3, rel_se3;
            V3D abs_std_pos_m, abs_std_rot_deg, rel_std_pos_m, rel_std_rot_deg;

            if (!als_obj->refine_als) // als was not setup
            {
                use_als_update = false; // ALS not set yet
                *featsFromMap = *laserCloudSurfMap;
                if (gnss_obj->GNSS_extrinsic_init)
                {
                    als_obj->init(gnss_obj->origin_enu, gnss_obj->R_GNSS_to_MLS, featsFromMap);

                    gnss_obj->updateExtrinsic(als_obj->R_to_mls); // use the scan-based refined rotation for GNSS
                    als_obj->Update(Sophus::SE3(state_point.rot, state_point.pos));
                    gnss_obj->als2mls_T = als_obj->als_to_mls;
                }
            }
            else // als was set up
            {
                als_obj->Update(Sophus::SE3(state_point.rot, state_point.pos));
                als_integrated = true;

                use_als_update = true; // use ALS now
            }

            if (pubLaserALSMap.getNumSubscribers() != 0)
            {
                als_obj->getCloud(featsFromMap);
                publish_map(pubLaserALSMap);
            }

            iters = estimator_.update(NUM_MAX_ITERATIONS, extrinsic_est_en, feats_down_body, laserCloudSurfMap,
                                      use_als_update, als_obj->als_cloud, als_obj->localKdTree_map_als,
                                      use_se3_update, absolute_se3, abs_std_pos_m, abs_std_rot_deg,
                                      use_se3_rel, rel_se3, rel_std_pos_m, rel_std_rot_deg, prev_mls);

#else
            iters = estimator_.update(NUM_MAX_ITERATIONS, extrinsic_est_en, feats_down_body, laserCloudSurfMap);
#endif

            state_point = estimator_.get_x();
            curr_mls = Sophus::SE3(state_point.rot, state_point.pos);

            // Update the local map--------------------------------------------------
            feats_down_world->resize(feats_down_size);
            local_map_update();

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

            prev_mls = curr_mls;
        }
    }

    for (auto &b : bags)
        b->close();
}
