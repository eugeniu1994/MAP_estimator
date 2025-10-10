

#include "DataHandler_vux.hpp"

#include "IMU.hpp"
#include "GNSS.hpp"

#ifndef USE_EKF
#include "PoseGraph.hpp"
#endif

#ifdef USE_ALS
#include "ALS.hpp"
#endif

#include "Vux_reader.hpp"
#include <GeographicLib/UTMUPS.hpp>

#include <liblas/liblas.hpp>

void DataHandler::publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "world";
    odomAftMapped.child_frame_id = "MLS";    //"MLS" moves relative to "world"
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

    //tf::Transform transform_inv = transform.inverse();
    //static tf::TransformBroadcaster br2;
    //br2.sendTransform(tf::StampedTransform(transform_inv, odomAftMapped.header.stamp, "MLS", "world"));
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
    //pcl::toROSMsg(*laserCloudFullRes, laserCloudmsg);

    //laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.stamp = ros::Time(0);  // <- "no transform needed"

    laserCloudmsg.header.frame_id =  "MLS";
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
    
    double local_map_radius = 150;// 100;// 75;// ;
    //local_map_radius = 400;
    
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
#endif
}


void DataHandler::local_map_update_from_ALS(const std::vector<PointVector> &Nearest_Points)
{
#ifdef USE_EKF
#if USE_STATIC_KDTREE == 0
    PointVector PointToAdd;
    PointToAdd.reserve(Nearest_Points.size());
    for (int i = 0; i < Nearest_Points.size(); i++)
    {
        // Append all points from the current PointVector to all_points
        if (point_selected_surf[i])
        {
            const PointVector &point_vector = Nearest_Points[i];
            PointToAdd.push_back(point_vector[0]); // keep only the closest point
        }
    }
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    // std::cout<<"local_map_update_from_ALS add_point_size:"<<add_point_size<<std::endl;
#else

    pcl::PointCloud<PointType>::Ptr tmpSurf(new pcl::PointCloud<PointType>());
    tmpSurf->points.resize(Nearest_Points.size());
    for (int i = 0; i < Nearest_Points.size(); i++)
    {
        if (point_selected_surf[i])
        {
            const PointVector &point_vector = Nearest_Points[i];
            PointType &point = tmpSurf->points[i];

            point.x = point_vector[0].x;
            point.y = point_vector[0].y;
            point.z = point_vector[0].z;

            point.intensity = -100.0f; // Set to a default value 0
            // point.time = 0.0f;      // Set to a default value
        }
    }
    // std::cout<<"Added "<<tmpSurf->size()<<" points from ALS to MLS"<<std::endl;

    *laserCloudSurfMap += *tmpSurf;

#endif
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

void DataHandler::pointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.offset_R_L_I.matrix() * p_body + state_point.offset_T_L_I);

    //V3D p_global(state_point.offset_R_L_I.matrix().transpose() * p_body - state_point.offset_R_L_I.matrix().transpose() * state_point.offset_T_L_I);

    //p_global = p_body;

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
    po->time = pi->time;
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

    //std::cout<<"we are here imu_cbk \n"<<std::endl;
    
    // { //remove this later 
    //     // Scale acceleration values
    //     msg->linear_acceleration.x /= 2.0;
    //     msg->linear_acceleration.y /= 2.0;
    //     msg->linear_acceleration.z /= -2.0;

    //     // Scale gyroscope values
    //     msg->angular_velocity.x /= 2.0;
    //     msg->angular_velocity.y /= 2.0;
    //     msg->angular_velocity.z /= 2.0;

    //     auto acc = V3D(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    //     std::cout<<"acc:"<<acc.transpose()<<" at time:"<< msg_in->header.stamp.toSec() <<std::endl;

    // }

    

    if (!_imu_init)
    {
        _imu_init = true;
        _first_imu_time = msg_in->header.stamp.toSec();
    }
    
    if(shift_measurements_to_zero_time && _imu_init)
    {
        msg->header.stamp =
            ros::Time().fromSec(msg_in->header.stamp.toSec() - _first_imu_time);
    }

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        //std::cout << "change IMU time with timediff_lidar_wrt_imu:" << timediff_lidar_wrt_imu << std::endl;
        msg->header.stamp =
            ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    // msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    double timestamp = msg->header.stamp.toSec();

    if (timestamp < last_timestamp_imu)
    {
        std::cout<<"imu loop back, clear buffer"<<std::endl;
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    //std::cout<<"IMU_CBK: imu_buffer-"<<imu_buffer.size()<<std::endl;
}

void DataHandler::pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg_in)
{
    std::cout<<"\npcl_cbk msg_in->header.stamp.toSec()->"<<msg_in->header.stamp.toSec()<<", lidar_buffer:"<<lidar_buffer.size()<<std::endl;
    
    sensor_msgs::PointCloud2::Ptr msg(new sensor_msgs::PointCloud2(*msg_in));

    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
#ifdef SAVE_DATA
        lidar_msg_buffer.clear();
#endif
    }

    
    if (!_lidar_init)
    {
        _lidar_init = true;
        _first_lidar_time = msg->header.stamp.toSec();
    }

    if(shift_measurements_to_zero_time && _lidar_init)
    {
        std::cout<<"LiDAR time shift from "<<msg->header.stamp.toSec()<<" to "<<(msg->header.stamp.toSec() - _first_lidar_time)<<std::endl;
        msg->header.stamp =
            ros::Time().fromSec(msg->header.stamp.toSec() - _first_lidar_time);
    }

    last_timestamp_lidar = msg->header.stamp.toSec();
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
        std::cout<<"dt:"<<abs(last_timestamp_imu - last_timestamp_lidar)<<std::endl;
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

            std::cout<<"first_point_time:"<<first_point_time<<", last point time:"<<pl_orig.points[n-1].timestamp<<std::endl;

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

                pcl_out->points[index].ring = point.ring;
                
                index++;
            }
            pcl_out->resize(index); // Resize to the actual number of points added
        }
        break;

    case VLS128:
        std::cout << "VLS128 not implemented" << std::endl;
        break;

    case Ouster:
        std::cout << "Ouster not implemented" << std::endl;
        break;

    default:
        std::cout << "Unknown LIDAR type:" << lidar_type << std::endl;
        break;
    }
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

void DataHandler::vux_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mutex_lock.lock();
    // std::cout<<"vux_cbk"<<std::endl;
    vux_buffer.push_back(msg);
    mutex_lock.unlock();

    // get the vux message and convert it to point cloud

    // std::cout<<"standard_pcl_cbk msg->header.stamp.toSec()->"<<msg->header.stamp.toSec()<<std::endl;
    /*if (msg->header.stamp.toSec() < last_timestamp_lidar)
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
#endif*/
}

bool DataHandler::sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        std::cout<<"lidar_buffer:"<<lidar_buffer.size()<<", imu_buffer:"<<imu_buffer.size()<<std::endl;
        return false;
    }

    if (!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front() + meas.lidar->points.front().time;
        lidar_end_time = time_buffer.front() + meas.lidar->points.back().time;
        meas.lidar_end_time = lidar_end_time;

        //std::cout<<"meas.lidar:"<<meas.lidar->size()<<std::endl;

        lidar_pushed = true;
#ifdef SAVE_DATA
        meas.lidar_msg = lidar_msg_buffer.front();
#endif
    }

    if (last_timestamp_imu < lidar_end_time) // If lst imu timestamp is less than the lidar final time, it means that not enough imu data has been collected.
    {
        //std::cout<<"last_timestamp_imu is smaller than lidar_end_time,  return False in the sync"<<std::endl;
        //std::cout<<"last_timestamp_imu:"<<last_timestamp_imu<<", lidar_end_time:"<<lidar_end_time<<std::endl;
        //std::cout<<"time difference:"<<fabs(last_timestamp_imu - lidar_end_time)<<std::endl;
        return false;
    }

    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();

    while ((!imu_buffer.empty()) ) //&& (imu_time < lidar_end_time)
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
        {
            std::cout<<"IMU time from future, imu_time:"<<imu_time<<", lidar_end_time:"<<lidar_end_time<<std::endl;
            break;
        }
            
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    if(!lidar_pushed || meas.imu.empty())
    {
        std::cout<<"lidar_pushed:"<<lidar_pushed<<std::endl;
        std::cout<<"meas.imu:"<<meas.imu.size()<<std::endl;

        std::cout<<"meas.lidar_beg_time:"<<meas.lidar_beg_time<<std::endl;
        std::cout<<"meas.lidar_end_time:"<<meas.lidar_end_time<<std::endl;
        std::cout<<"imu_buffer:"<<imu_buffer.size()<<std::endl;
        
        std::cout<<"imu_time:"<<imu_time<<std::endl;
        
        std::cout<<"\n\nIssue in sync_packages - the data in not synched\n\n"<<std::endl;
        //throw std::runtime_error("\n\nIssue in sync_packages - the data in not synched\n\n");

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
