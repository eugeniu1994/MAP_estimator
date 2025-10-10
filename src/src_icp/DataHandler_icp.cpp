
#include "DataHandler.hpp"

#include "IMU.hpp"
#include "GNSS.hpp"

#ifndef USE_EKF
#include "PoseGraph.hpp"
#endif

//#ifdef USE_ALS
#include "ALS.hpp"
//#endif

void DataHandler::publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "world";
    odomAftMapped.child_frame_id = "MLS";
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

void DataHandler::publish_frame_debug(const ros::Publisher &pubLaserCloudFrame_, const PointCloudXYZI::Ptr &frame_)
{
    std::cout << "publish_frame_debug frame_:" << frame_->size() << std::endl;
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*frame_, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "world";
    pubLaserCloudFrame_.publish(laserCloudmsg);
}

void DataHandler::local_map_update()
{
#if USE_STATIC_KDTREE == 0
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));

        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType mid_point;
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int j = 0; j < NUM_MATCH_POINTS; j++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(points_near[j], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();

#else
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
    
    //double threshold = 75;
    double threshold = 150;

    double x_min = state_point.pos.x() - threshold;
    double y_min = state_point.pos.y() - threshold;
    double z_min = state_point.pos.z() - threshold;
    double x_max = state_point.pos.x() + threshold;
    double y_max = state_point.pos.y() + threshold;
    double z_max = state_point.pos.z() + threshold;

    // ROS_INFO("size : %f,%f,%f,%f,%f,%f", x_min, y_min, z_min,x_max, y_max, z_max);
    cropBoxFilter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1));
    cropBoxFilter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1));
    cropBoxFilter.setNegative(false);

    cropBoxFilter.setInputCloud(laserCloudSurfMap);
    cropBoxFilter.filter(*tmpSurf);

    downSizeFilterSurf.setInputCloud(tmpSurf);
    downSizeFilterSurf.filter(*laserCloudSurfMap);
#endif
}

void DataHandler::pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
#ifdef USE_EKF
    V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I) + state_point.pos);
#else
    V3D p_global(state_point.rot.matrix() * p_body + state_point.pos); // for icp the cloud already is in IMU frame
#endif
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
    po->time = pi->time;
}

bool DataHandler::sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }

    if (!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front() + meas.lidar->points.front().time;
        lidar_end_time = time_buffer.front() + meas.lidar->points.back().time;
        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
#ifdef SAVE_DATA
        meas.lidar_msg = lidar_msg_buffer.front();
#endif
    }

    if (last_timestamp_imu < lidar_end_time) // If lst imu timestamp is less than the lidar final time, it means that not enough imu data has been collected.
    {
        // std::cout<<"last_timestamp_imu is smaller than lidar_end_time,  return False in the sync"<<std::endl;
        // std::cout<<"last_timestamp_imu:"<<last_timestamp_imu<<", lidar_end_time:"<<lidar_end_time<<std::endl;
        return false;
    }

    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
            break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    if(!lidar_pushed || meas.imu.empty())
    {
        std::cout<<"lidar_pushed:"<<lidar_pushed<<std::endl;
        std::cout<<"meas.imu:"<<meas.imu.size()<<std::endl;

        std::cout<<"meas.lidar_beg_time:"<<meas.lidar_beg_time<<std::endl;
        std::cout<<"meas.lidar_end_time:"<<meas.lidar_end_time<<std::endl;
        if(!imu_buffer.empty())
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();
            std::cout<<"imu_time:"<<imu_time<<std::endl;
        }
        throw std::runtime_error("Issue in sync_packages - the data in not synched");

        return false;
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
#ifdef SAVE_DATA
    lidar_msg_buffer.pop_front();
#endif
    return true;
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
    if (!_imu_init)
    {
        _imu_init = true;
        _first_imu_time = msg_in->header.stamp.toSec();
    }

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        std::cout << "change IMU time with timediff_lidar_wrt_imu:" << timediff_lidar_wrt_imu << std::endl;
        msg->header.stamp = ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }
    // msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    double timestamp = msg->header.stamp.toSec();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
}

void DataHandler::pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    //std::cout << "\nstandard_pcl_cbk msg->header.stamp.toSec()->" << msg->header.stamp.toSec() << std::endl;

    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
#ifdef SAVE_DATA
        lidar_msg_buffer.clear();
#endif
    }

    last_timestamp_lidar = msg->header.stamp.toSec();
    if (!_lidar_init)
    {
        _lidar_init = true;
        _first_lidar_time = msg->header.stamp.toSec();
    }

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    // time_sync_en - self alignment, estimate shift in time
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
#ifdef SAVE_DATA
    lidar_msg_buffer.push_back(msg);
#endif
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
            //std::cout<<"first_point_time:"<<first_point_time<<std::endl;
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
                pcl_out->points[index].intensity = sqrt(sqrt(range));             // Save the range in the intensity field
                pcl_out->points[index].time = point.timestamp - first_point_time; // Time relative to first point

                index++;
            }
            pcl_out->resize(index); // Resize to the actual number of points added
        }
        break;

    case VLS128:
    {
        // std::string fields = pcl::getFieldsList(*msg);
        // std::cout << "Available fields in PointCloud2: " << fields << std::endl;
        // std::cout << "PointCloud2 Fields:\n";
        // for (const auto &field : msg->fields)
        // {
        //     std::cout << "  Name: " << field.name
        //               << ", Offset: " << field.offset
        //               << ", Datatype: " << static_cast<int>(field.datatype)
        //               << ", Count: " << field.count << std::endl;
        // }
        
        pcl::PointCloud<velodyne_ros::Point> pl_orig;
        pcl::fromROSMsg(*msg, pl_orig);

        int n = pl_orig.points.size();
        pcl_out->resize(n / point_step);
        first_point_time_ = pl_orig.points[0].time;
        index = 0;
        first_point_time = first_point_time_;
        // In some occasions point time can actually be > 3600, this is caused by the Velodyne driver,
        // therefore have to take modulus here
        first_point_time_ = std::fmod(first_point_time_, 3600);
        std::cout << "first_point_time:" << first_point_time_ << std::endl;
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
            pcl_out->points[index].intensity = sqrt(sqrt(range));        // Save the range in the intensity field
            pcl_out->points[index].time = point.time - first_point_time; // Time relative to first point

            index++;
        }
        pcl_out->resize(index); // Resize to the actual number of points added
    }
    break;

    case Ouster:
        throw std::runtime_error("msg2cloud Ouster not implemented");
        break;

    default:
        throw std::runtime_error("msg2cloud Unknown LIDAR type:");
        std::cout << "" << lidar_type << std::endl;
        break;
    }
}

volatile bool flg_exit = false;
void SigHandle(int sig)
{
    ROS_WARN("Caught signal %d, stopping...", sig);
    flg_exit = true; // Set the flag to stop the loop
}

void DataHandler::RemovePointsFarFromLocation()
{
#if USE_STATIC_KDTREE == 0
    V3D pos_LiD = pos_lid;
    // std::cout<<"DET_RANGE:"<<DET_RANGE<<", cube_len:"<<cube_len<<std::endl;
    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            A.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            A.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
            prev_pos_LiD = pos_LiD;
        }
        Localmap_Initialized = true;
        return;
    }

    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - A.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - A.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            need_move = true;
            // std::cout<<"Need to move MOV_THRESHOLD:"<<MOV_THRESHOLD<<", DET_RANGE:"<<DET_RANGE<<", x dist :"<<dist_to_map_edge[i][0]<<", y dist:"<<dist_to_map_edge[i][1]<<std::endl;
        }
    }
    if (!need_move)
    {
        return;
    }

    // std::cout << " need_move lasermap_fov_segment,  DET_RANGE:" << DET_RANGE << ", MOV_THRESHOLD * DET_RANGE:" << MOV_THRESHOLD * DET_RANGE << std::endl;

    Eigen::Vector3d moving = pos_LiD - prev_pos_LiD;
    BoxPointType B = A, tmp_boxpoints;
    for (int i = 0; i < 3; i++)
    {
        // update the the map cube
        // B.vertex_max[i] += moving[i];B.vertex_min[i] += moving[i];
        B.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
        B.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        tmp_boxpoints = A;
        if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) // dist to max
        {
            // std::cout<<"enough positive motion"<<std::endl;
            tmp_boxpoints.vertex_max[i] = A.vertex_min[i] + moving[i];
            cub_needrm.push_back(tmp_boxpoints);
        }
        else if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            // dist to min
            // std::cout<<"enough negative motion"<<std::endl;
            tmp_boxpoints.vertex_min[i] = A.vertex_max[i] + moving[i];
            cub_needrm.push_back(tmp_boxpoints);
        }
    }

    prev_pos_LiD = pos_LiD;
    A = B;

    // PointVector points_history; //this should be used to create the global map - if required
    // ikdtree.acquire_removed_points(points_history);
    // std::cout<<"points_history:"<<points_history.size()<<std::endl;

    if (cub_needrm.size() > 0)
    {
        // std::cout << "Delete_Point_Boxes:" << cub_needrm.size() << std::endl;
        ikdtree.Delete_Point_Boxes(cub_needrm);
    }
    cub_needrm.clear();
#endif
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

void DataHandler::BagHandler()
{
    std::cout << "\n===============================BagHandler===============================" << std::endl;
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
    
    gnss_obj->use_ransac_alignment = use_ransac_alignment;

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
    int scan_id = 0;
    std::cout << "Start reading the data..." << std::endl;
    V3D prev_t = V3D::Zero();
    double travelled_distance = 0.0;
    for (const rosbag::MessageInstance &m : view)
    {
        // scan_id++;
        // if (scan_id < 45100) // this is only for the 0 bag
        //      continue;

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

        if (flg_exit)
            break;

        if (sync_packages(Measures))
        {
            scan_id++;
            //if (scan_id < 45100) // this is only for the 0 bag
            //continue;

            std::cout << "scan_id:" << scan_id << std::endl;
            // if (scan_id > 1310) // 1400
            // {
            //     std::cout << "Stop here... enough data" << std::endl;
            //     break;
            // }

            std::cout<<"scan_id:"<<scan_id<<", travelled_distance:"<<travelled_distance<<std::endl;

            std::cout << "\nIMU:" << imu_buffer.size() << ", GPS:" << gps_buffer.size() << ", LiDAR:" << lidar_buffer.size() << std::endl;

            double t00 = omp_get_wtime();

            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                flg_first_scan = false;
                continue;
            }

            // imu_obj->Process(Measures, estimator_, feats_undistort);
            //todo - populate the feats_undistort here 
            double t_IMU_process = omp_get_wtime();

            // publish_frame_debug(pubLaserCloudDebug, feats_undistort);

            state_point = estimator_.get_x();
            pos_lid = state_point.pos + state_point.offset_R_L_I.matrix() * state_point.offset_T_L_I;
            flg_EKF_inited = true;// (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;




            const auto &[source, frame_downsample] = estimator_.Voxelize(feats_undistort);
            // MLS registration
            if (!estimator_.update(source, estimator_.local_map_, true))
            {
                std::cout << "\n------------------MLS update failed--------------------------------" << std::endl;
            }

            // double t_LiDAR_update = omp_get_wtime();
            // std::cout << "IMU_process time(ms):  " << (t_IMU_process - t00) * 1000 << ", LiDAR_update (ms): " << (t_LiDAR_update - t_IMU_process) * 1000 << std::endl;

            // get and publish the GNSS pose-----------------------------------------
            //gnss_obj->Process(gps_buffer, lidar_end_time, state_point.pos);
            //Sophus::SE3 gnss_pose = (gnss_obj->use_postprocessed_gnss) ? gnss_obj->postprocessed_gps_pose : gnss_obj->gps_pose;

            feats_down_size = frame_downsample.size();
            state_point = estimator_.get_x(); // state after registration
/*
#ifdef USE_ALS
            if (!als_obj->refine_als)
            {                                      // not initialized
                if (gnss_obj->GNSS_extrinsic_init) // if gnss aligned
                {
                    estimator_.LocalMap(featsFromMap);
                    als_obj->init(gnss_obj->origin_enu, gnss_obj->R_GNSS_to_MLS, featsFromMap);
                    gnss_obj->updateExtrinsic(als_obj->R_to_mls); // use the scan-based refined rotation for GNSS
                }
                // imu_obj->update(state_point, Measures.lidar_beg_time, Measures.lidar_end_time); // state after graph update
                updated_state = estimator_.get_x(); // state after ALS registration
                imu_obj->update(state_point, Measures.lidar_beg_time, Measures.lidar_end_time,
                                false, Sophus::SE3(updated_state.rot, updated_state.pos),
                                use_gnss, updated_state.pos);
                estimator_.set_x(state_point);
            }
            else
            {
                als_obj->Update(Sophus::SE3(state_point.rot, state_point.pos));
                // ALS refinement
                bool als_success = estimator_.update(frame_downsample, als_obj->local_map_, false, saveALS_NN_2_MLS);
                if (!als_success)
                {
                    std::cout << "\n------------------ALS update failed--------------------------------" << std::endl;
                }
                /* this works more or less but not good enough ----
                   // global_graph_obj->addGPSFactor(gnss_pose.translation());

                   updated_state = estimator_.get_x(); // state after ALS registration
                   // global_graph_obj->update(state_point, updated_state); //without gps
                   global_graph_obj->update(state_point, updated_state, true, gnss_pose.translation());

                   estimator_.set_x(updated_state);
                   imu_obj->prevPose_ = global_graph_obj->prevPose_;

                   // imu_obj->update(state_point, Measures.lidar_beg_time, Measures.lidar_end_time, true, Sophus::SE3(updated_state.rot, updated_state.pos)); //state after graph update
                   // estimator_.set_x(state_point); // * /

                updated_state = estimator_.get_x(); // state after ALS registration
                imu_obj->update(state_point, Measures.lidar_beg_time, Measures.lidar_end_time,
                                als_success, Sophus::SE3(updated_state.rot, updated_state.pos),
                                use_gnss, gnss_pose.translation());

                estimator_.set_x(state_point);
            }

            if (pubLaserALSMap.getNumSubscribers() != 0)
            {
                als_obj->getCloud(featsFromMap);
                publish_map(pubLaserALSMap);
            }

#else
            imu_obj->update(state_point, Measures.lidar_beg_time, Measures.lidar_end_time,
                            false, Sophus::SE3(updated_state.rot, updated_state.pos),
                            use_gnss, gnss_pose.translation());
            estimator_.set_x(state_point);
#endif

*/
           // do the imu_obj->update

            state_point = estimator_.get_x();
            estimator_.local_map_.Update(frame_downsample, Sophus::SE3(state_point.rot, state_point.pos));

            // gnss_pose.so3() = state_point.rot; // use the MLS orientation
            //publish_gnss_odometry(gnss_pose);

            // Publish odometry and point clouds------------------------------------
            publish_odometry(pubOdomAftMapped);
            if (scan_pub_en)
            {
                if (pubLaserCloudFull.getNumSubscribers() != 0)
                {
                    if (!dense_pub_en)
                    {
                        Eigen2PCL(feats_down_body, frame_downsample);
                    }
                    publish_frame_world(pubLaserCloudFull);
                }
            }

            if (pubLaserCloudMap.getNumSubscribers() != 0)
            {
                estimator_.LocalMap(featsFromMap);
                publish_map(pubLaserCloudMap);
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
                            const pcl::PointCloud<velodyne_ros::Point> &pl_orig = imu_obj->DeSkewOriginalCloud<velodyne_ros::Point>(Measures.lidar_msg, state_point, save_clouds_local);
                            std::cout << "save " << pl_orig.size() << " points" << std::endl;
                            std::string filename = save_clouds_path + std::to_string(pcd_index) + "_cloud_" + std::to_string(lidar_end_time) + ".pcd";
                            pcl::io::savePCDFile(filename, pl_orig, true); // Binary format
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
                    // # ' id time tx ty tz qx qy qz qw' - tum format(scan id, scan timestamp seconds, translation and rotation quaternion)
                    foutMLS << pcd_index << " " << std::to_string(lidar_end_time) << " " << t_model(0) << " " << t_model(1) << " " << t_model(2) << " "
                    << q_model.x() << " " << q_model.y() << " " << q_model.z() << " " << q_model.w() << std::endl;
                    foutMLS.close();
                }

                pcd_index++;
            }
#endif

            //state_point = estimator_.get_x();
            //std::cout<<"extrinsic_est_en:"<<extrinsic_est_en<<std::endl;
            //std::cout<<"\n Extrinsics R:\n"<<state_point.offset_R_L_I.matrix()<<"\nt:"<<state_point.offset_T_L_I.transpose()<<std::endl;

            double t11 = omp_get_wtime();
            std::cout << "Mapping time(ms):  " << (t11 - t00) * 1000 << ", feats_down_size: " << feats_down_size << ", lidar_end_time:" << lidar_end_time << std::endl;
        }
    }
    bag.close();
}
