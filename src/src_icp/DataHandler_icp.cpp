
#include "DataHandler.hpp"

#include "IMU.hpp"
#include "GNSS.hpp"

// #ifdef USE_ALS
#include "ALS.hpp"
// #endif

#include "TrajectoryReader.hpp"

void DataHandler::publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "world";
    odomAftMapped.child_frame_id = "MLS";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);

    auto P = estimator_icp.get_P();

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

    if (!lidar_pushed || meas.imu.empty())
    {
        std::cout << "lidar_pushed:" << lidar_pushed << std::endl;
        std::cout << "meas.imu:" << meas.imu.size() << std::endl;

        std::cout << "meas.lidar_beg_time:" << meas.lidar_beg_time << std::endl;
        std::cout << "meas.lidar_end_time:" << meas.lidar_end_time << std::endl;
        if (!imu_buffer.empty())
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();
            std::cout << "imu_time:" << imu_time << std::endl;
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

bool DataHandler::sync_packages_no_IMU(MeasureGroup &meas)
{
    if (lidar_buffer.empty())
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
    // std::cout << "\nstandard_pcl_cbk msg->header.stamp.toSec()->" << msg->header.stamp.toSec() << std::endl;

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
            // std::cout<<"first_point_time:"<<first_point_time<<std::endl;
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

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

std::vector<double> gaussianSmooth(const std::vector<double>& signal, double sigma, int kernel_radius = 3)
{
    int n = signal.size();
    std::vector<double> smoothed(n, 0.0);
    std::vector<double> kernel(2 * kernel_radius + 1);

    // Build Gaussian kernel
    double sum = 0.0;
    for (int i = -kernel_radius; i <= kernel_radius; ++i) {
        double val = std::exp(-0.5 * (i * i) / (sigma * sigma));
        kernel[i + kernel_radius] = val;
        sum += val;
    }
    for (double &v : kernel) v /= sum; // normalize

    // Convolution
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        double wsum = 0.0;
        for (int k = -kernel_radius; k <= kernel_radius; ++k) {
            int idx = i + k;
            if (idx < 0 || idx >= n) continue;
            acc += signal[idx] * kernel[k + kernel_radius];
            wsum += kernel[k + kernel_radius];
        }
        smoothed[i] = (wsum > 0) ? acc / wsum : signal[i];
    }

    return smoothed;
}

int findMaxIndex(const std::vector<double>& v)
{
    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

class SignalProcessor
{
public:
    // Basic cross-correlation without interpolation
    static double computeCrossCorrelationBasic(const std::vector<double> &sig1,
                                               const std::vector<double> &sig2,
                                               double offset, double dt)
    {
        int sample_offset = static_cast<int>(std::round(offset / dt));
        return computeCrossCorrelationDiscrete(sig1, sig2, sample_offset);
    }

    // Cross-correlation with linear interpolation for sub-sample accuracy
    static double computeCrossCorrelationInterpolated(const std::vector<double> &sig1,
                                                      const std::vector<double> &sig2,
                                                      double offset, double dt)
    {
        if (sig1.empty() || sig2.empty() || dt <= 0)
        {
            return 0.0;
        }

        double correlation = 0.0;
        int count = 0;

        // Convert time offset to fractional sample offset
        double sample_offset = offset / dt;
        int integer_offset = static_cast<int>(std::floor(sample_offset));
        double fractional = sample_offset - integer_offset;

        if (integer_offset >= 0)
        {
            for (size_t i = 0; i < sig1.size(); ++i)
            {
                double target_index = i + sample_offset;
                int lower_index = static_cast<int>(std::floor(target_index));
                int upper_index = lower_index + 1;

                if (lower_index >= 0 && upper_index < static_cast<int>(sig2.size()))
                {
                    // Linear interpolation
                    double lower_val = sig2[lower_index];
                    double upper_val = sig2[upper_index];
                    double interpolated_val = lower_val + (upper_val - lower_val) *
                                                              (target_index - lower_index);

                    correlation += sig1[i] * interpolated_val;
                    count++;
                }
            }
        }
        else
        {
            for (size_t j = 0; j < sig2.size(); ++j)
            {
                double target_index = j - sample_offset; // sample_offset is negative
                int lower_index = static_cast<int>(std::floor(target_index));
                int upper_index = lower_index + 1;

                if (lower_index >= 0 && upper_index < static_cast<int>(sig1.size()))
                {
                    // Linear interpolation
                    double lower_val = sig1[lower_index];
                    double upper_val = sig1[upper_index];
                    double interpolated_val = lower_val + (upper_val - lower_val) *
                                                              (target_index - lower_index);

                    correlation += sig2[j] * interpolated_val;
                    count++;
                }
            }
        }

        return count > 0 ? correlation / count : 0.0;
    }

private:
    static double computeCrossCorrelationDiscrete(const std::vector<double> &sig1,
                                                  const std::vector<double> &sig2,
                                                  int sample_offset)
    {
        double correlation = 0.0;
        int count = 0;

        if (sample_offset >= 0)
        {
            // sig2 is delayed relative to sig1
            for (size_t i = 0; i < sig1.size(); ++i)
            {
                int j = i + sample_offset;
                if (j >= 0 && j < static_cast<int>(sig2.size()))
                {
                    correlation += sig1[i] * sig2[j];
                    count++;
                }
            }
        }
        else
        {
            // sig1 is delayed relative to sig2
            for (size_t j = 0; j < sig2.size(); ++j)
            {
                int i = j - sample_offset;
                if (i >= 0 && i < static_cast<int>(sig1.size()))
                {
                    correlation += sig1[i] * sig2[j];
                    count++;
                }
            }
        }

        return count > 0 ? correlation / count : 0.0;
    }

    static double computeMean(const std::vector<double> &signal)
    {
        if (signal.empty())
            return 0.0;
        double sum = 0.0;
        for (double val : signal)
        {
            sum += val;
        }
        return sum / signal.size();
    }

    static double computeStandardDeviation(const std::vector<double> &signal, double mean)
    {
        if (signal.size() <= 1)
            return 0.0;
        double sum_sq_diff = 0.0;
        for (double val : signal)
        {
            double diff = val - mean;
            sum_sq_diff += diff * diff;
        }
        return std::sqrt(sum_sq_diff / (signal.size() - 1));
    }
};

double estimateTimeOffset(
    const std::vector<Sophus::SE3> &sensor1_poses, // LiDAR poses
    const std::vector<Sophus::SE3> &sensor2_poses, // GNSS-IMU poses
    const std::vector<double> &sensor1_time,
    const std::vector<double> &sensor2_time, double &best_offset, double min_motion = 10)
{
    if (sensor1_poses.size() < 2 || sensor2_poses.size() < 2)
    {
        std::cerr << "Not enough poses in one of the sensors!\n";
        return false;
    }

    // 1. Check motion magnitude ---
    double dist1 = (sensor1_poses.front().translation() - sensor1_poses.back().translation()).norm();
    double dist2 = (sensor2_poses.front().translation() - sensor2_poses.back().translation()).norm();
    if (dist1 < min_motion || dist2 < min_motion)
    {
        std::cerr << "Not enough motion in trajectories (" << dist1 << " m, " << dist2 << " m)\n";
        return false;
    }

    //  2. Compute cumulative translation distances ---
    auto computeCumulativeDistance = [](const std::vector<Sophus::SE3> &poses)
    {
        std::vector<double> dist(poses.size(), 0.0);
        for (size_t i = 1; i < poses.size(); ++i)
            dist[i] = dist[i - 1] + (poses[i].translation() - poses[i - 1].translation()).norm();
        return dist;
    };

    std::vector<double> dist1_traj = computeCumulativeDistance(sensor1_poses);
    std::vector<double> dist2_traj = computeCumulativeDistance(sensor2_poses);

    // --- Plot original trajectories ---
    plt::figure();
    plt::named_plot("LiDAR", sensor1_time, dist1_traj, "b-");
    plt::named_plot("GNSS-IMU", sensor2_time, dist2_traj, "r--");
    plt::xlabel("Time [s]");
    plt::ylabel("Cumulative distance [m]");
    plt::title("Trajectory cumulative distance alignment");
    plt::legend();
    plt::grid(true);
    plt::draw();

    // --- 3. Linear interpolation helper ---
    auto interpolate = [](const std::vector<double> &time, const std::vector<double> &dist, double t)
    {
        if (t <= time.front())
            return dist.front();
        if (t >= time.back())
            return dist.back();
        auto it = std::lower_bound(time.begin(), time.end(), t);
        size_t idx = std::distance(time.begin(), it);
        if (idx == 0)
            return dist.front();
        double t1 = time[idx - 1], t2 = time[idx];
        double d1 = dist[idx - 1], d2 = dist[idx];
        double alpha = (t - t1) / (t2 - t1);
        return d1 + alpha * (d2 - d1);
    };

    // Determine which sensor started first
    double offset_start = 0.0;
    double offset_end = 0.0;
    double offset_step = 0.05; // 50 ms step

    if (dist1_traj.back() > dist2_traj.back())
    {
        std::cout << "LiDAR traveled more, may have started first\n";
        // LiDAR started first - search positive offsets
        offset_start = 0.0;
        offset_end = 500.0;
    }
    else
    {
        std::cout << "GNSS-IMU traveled more, may have started first\n";
        // GNSS-IMU started first -  search negative offsets
        offset_start = -500.0;
        offset_end = 0.0;
    }

    // --- Search for best time offset ---
    best_offset = 0.0;
    double best_error = std::numeric_limits<double>::max();

    for (double offset = offset_start; offset <= offset_end; offset += offset_step)
    {
        double total_error = 0.0;
        int count = 0;
        for (size_t i = 0; i < sensor1_time.size(); ++i)
        {
            double t_shifted = sensor1_time[i] + offset;
            if (t_shifted < sensor2_time.front() || t_shifted > sensor2_time.back())
                continue; // skip if out of range
            double interp_dist2 = interpolate(sensor2_time, dist2_traj, t_shifted);
            double err = dist1_traj[i] - interp_dist2;
            total_error += err * err;
            count++;
        }
        if (count < 10)
            continue; // not enough overlap
        double mean_error = total_error / count;
        if (mean_error < best_error)
        {
            best_error = mean_error;
            best_offset = offset;
        }
    }

    std::cout << "Estimated time offset (LiDAR -> GNSS-IMU): " << best_offset
              << " s, mean alignment error = " << best_error << std::endl;

    plt::figure();
    plt::named_plot("LiDAR", sensor1_time, dist1_traj, "b-");
    plt::named_plot("GNSS-IMU", sensor2_time, dist2_traj, "r--");
    // Shift GNSS-IMU time by estimated best_offset
    std::vector<double> sensor2_time_shifted(sensor2_time.size());
    for (size_t i = 0; i < sensor2_time.size(); ++i)
        sensor2_time_shifted[i] = sensor2_time[i] - best_offset; // align to LiDAR time

    plt::named_plot("GNSS-IMU (shifted)", sensor2_time_shifted, dist2_traj, "g--");

    plt::xlabel("Time [s]");
    plt::ylabel("Cumulative distance [m]");
    plt::title("Trajectory cumulative distance alignment");
    plt::legend();
    plt::grid(true);
    plt::draw();

    //-----------------------------------------------------------------------
    auto computeAngularVelocities = [](const std::vector<Sophus::SE3> &poses)
    {
        std::vector<double> angular_vel(poses.size(), 0.0);
        for (size_t i = 1; i < poses.size(); ++i)
        {
            // double dt = trajectory[i].first - trajectory[i-1].first;
            Sophus::SE3 relative = poses[i - 1].inverse() * poses[i];
            angular_vel[i] = (relative.so3().log() * 180. / M_PI).norm(); // / dt
        }
        return angular_vel;
    };

    std::vector<double> sensor1_angvel = computeAngularVelocities(sensor1_poses);
    std::vector<double> sensor2_angvel = computeAngularVelocities(sensor2_poses);

    double sigma = 1.0;
    // Smooth angular velocities
    auto smoothed1 = gaussianSmooth(sensor1_angvel, sigma);
    auto smoothed2 = gaussianSmooth(sensor2_angvel, sigma);

    // Find peaks
    int idx1 = findMaxIndex(smoothed1);
    int idx2 = findMaxIndex(smoothed2);

    double t1 = sensor1_time[idx1];
    double t2 = sensor2_time[idx2];

    // Offset (positive → sensor1 lags behind sensor2)
    double time_diff = t1 - t2;

    std::cout << "Peak angular velocity index: sensor1=" << idx1
              << " (t=" << t1 << "), sensor2=" << idx2
              << " (t=" << t2 << ")\n";
    std::cout << "Estimated time offset: " << time_diff << " seconds\n";


    plt::figure();
    
    plt::named_plot("smoothed LiDAR", sensor1_time, smoothed1, "b-");
    plt::named_plot("smoothed GNSS-IMU", sensor2_time, smoothed2, "r--");

    plt::named_plot("LiDAR", sensor1_time, sensor1_angvel, "g-");
    plt::named_plot("GNSS-IMU", sensor2_time, sensor2_angvel, "y--");

    plt::xlabel("Time [s]");
    plt::ylabel("Angular Velocities [deg]");
    plt::title("Trajectory velocities alignment");
    plt::legend();
    plt::grid(true);
    plt::draw();

    {
        std::vector<double> signal1 = sensor1_angvel;
        std::vector<double> signal2 = sensor2_angvel;

        double dt = 0.1; // 10Hz sampling

        std::cout << "Testing cross-correlation:" << std::endl;

        // Find best offset (like in the original code)
        double search_range = 100.0; // seconds
        double search_step = 0.05; // seconds

        double best_offset = 0.0;
        double max_correlation = -std::numeric_limits<double>::max();

        for (double offset = -search_range; offset <= search_range; offset += search_step)
        {
            double correlation = SignalProcessor::computeCrossCorrelationInterpolated(
                signal1, signal2, offset, dt);

            if (correlation > max_correlation)
            {
                max_correlation = correlation;
                best_offset = offset;
            }
        }

        std::cout << "\nBest offset: " << best_offset << " seconds with correlation: "
                  << max_correlation << std::endl;
    }


    plt::show();
    return true;
}

void DataHandler::BagHandler()
{
    std::cout << "\n===============================Subscribe===============================" << std::endl;
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
    Sophus::SE3 Lidar_wrt_IMU = Sophus::SE3(Lidar_R_wrt_IMU, Lidar_T_wrt_IMU);

#ifdef USE_ALS
    // std::shared_ptr<ALS_Handler> als_obj = std::make_shared<ALS_Handler>(folder_root, downsample, closest_N_files, filter_size_surf_min);

    ros::Publisher pubLaserALSMap = nh.advertise<sensor_msgs::PointCloud2>("/ALS_map", 1000);
#endif

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100);

    ros::Publisher pubLaserCloudDebug = nh.advertise<sensor_msgs::PointCloud2>("/cloud_debug", 10);

    std::vector<std::string> topics{lid_topic}; //, imu_topic, gnss_topic

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
    int scan_id = 0;
    std::cout << "Start reading the data..." << std::endl;
    V3D prev_t = V3D::Zero();
    double travelled_distance = 0.0;

    TrajectoryReader reader;
    // an extrinsic transformation is passed here to transform the ppk gnss-imu orientaiton into mls frame
    Sophus::SE3 extr = Sophus::SE3();
    Sophus::SE3 se3 = Sophus::SE3();
    reader.read(postprocessed_gnss_path, extr);

    // Alignment transform: GNSS -> LiDAR
    Sophus::SE3 T_LG = Sophus::SE3();

    int tmp_index = 0;
    //{
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

    std::cout << "time_diff_lidar_to_imu:" << time_diff_lidar_to_imu << std::endl;

    //}
    Sophus::SE3 lidar_pose = Sophus::SE3();

    std::vector<Sophus::SE3> lidar, gnss_imu;
    std::vector<double> lidar_time, gnss_imu_time;

    double time_of_day_sec = time_diff_lidar_to_imu + m0.tod + 0.1;

    bool ppk_gnss_synced = false;
    bool shift_time_sinc = false;
    for (const rosbag::MessageInstance &m : view)
    {
        std::string topic = m.getTopic();
        if (topic == lid_topic)
        {
            sensor_msgs::PointCloud2::ConstPtr pcl_msg = m.instantiate<sensor_msgs::PointCloud2>();
            if (pcl_msg)
            {
                pcl_cbk(pcl_msg);
            }
        }

        if (flg_exit)
            break;

        if (sync_packages_no_IMU(Measures))
        {
            scan_id++;
            // if (scan_id < 45100) // this is only for the 0 bag
            // continue;

            std::cout << "scan_id:" << scan_id << std::endl;
            std::cout << "scan_id:" << scan_id << ", travelled_distance:" << travelled_distance << std::endl;

            std::cout << "\nIMU:" << imu_buffer.size() << ", GPS:" << gps_buffer.size() << ", LiDAR:" << lidar_buffer.size() << std::endl;

            double t00 = omp_get_wtime();

            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                flg_first_scan = false;
                continue;
            }

            // imu_obj->Process(Measures, estimator_icp, feats_undistort);
            *feats_undistort = *Measures.lidar; // populate the feats_undistort here

            // publish_frame_debug(pubLaserCloudDebug, feats_undistort);

            state_point = estimator_icp.get_x();
            pos_lid = state_point.pos + state_point.offset_R_L_I.matrix() * state_point.offset_T_L_I;
            flg_EKF_inited = true; // (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

            bool deskew = true;
            const auto &[source, frame_downsample] = estimator_icp.Voxelize(feats_undistort, deskew);

            std::cout << "source:" << source.size() << ", frame_downsample:" << frame_downsample.size() << std::endl;
            // MLS registration
            if (!estimator_icp.update(source, estimator_icp.local_map_, true, false))
            {
                std::cout << "\n------------------MLS update failed--------------------------------" << std::endl;
            }

            feats_down_size = frame_downsample.size();
            state_point = estimator_icp.get_x(); // state after registration

            lidar_pose = Sophus::SE3(state_point.rot, state_point.pos);
            // update MLS map
            estimator_icp.local_map_.Update(frame_downsample, lidar_pose);

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
                estimator_icp.LocalMap(featsFromMap);
                publish_map(pubLaserCloudMap);
            }

            double t11 = omp_get_wtime();
            std::cout << "Mapping time(ms):  " << (t11 - t00) * 1000 << ", feats_down_size: " << feats_down_size << ", lidar_end_time:" << lidar_end_time << std::endl;

            //-------------------------------------------------------------------

            time_of_day_sec += (Measures.lidar_end_time - Measures.lidar_beg_time);
            std::cout << "time_of_day_sec:" << time_of_day_sec << ", m0.tod:" << m0.tod << std::endl;
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
                // continue;
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

                    // tmp_pose = first_ppk_gnss_pose_inverse * interpolated_pose;

                    // Convert to Euler (ZYX: yaw-pitch-roll)
                    V3D euler = interpolated_pose.so3().matrix().eulerAngles(2, 1, 0);
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

                auto interpolated_pose = reader.closestPose(time_start);
                // auto interpolated_pose = reader.closestPoseUnix(lidar_end_time);
                // reader.toSE3(measurements[tmp_index+27], interpolated_pose);

                tmp_index = reader.curr_index;
                const auto &msg_time = measurements[tmp_index].tod;

                se3 = T_LG * first_ppk_gnss_pose_inverse * interpolated_pose; // in first frame

                if (true)
                {
                    publish_ppk_gnss(se3, msg_time);

                    // reader.addEarthGravity(measurements[reader.curr_index], raw_gyro, raw_acc, G_m_s2); //this will add the world gravity
                    reader.addGravity(measurements[reader.curr_index], se3, raw_gyro, raw_acc, G_m_s2); // gravity in curr body frame

                    // todo - we can do it the other way around and add the gravity in IMU body frame
                    // publishAccelerationArrow(marker_pub, -raw_acc, msg_time);

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

                if (!shift_time_sinc)
                {
                    lidar.push_back(lidar_pose);
                    gnss_imu.push_back(se3);
                    lidar_time.push_back(time_of_day_sec);
                    gnss_imu_time.push_back(msg_time);

                    double time_diff = 0;

                    shift_time_sinc = estimateTimeOffset(lidar, gnss_imu, lidar_time, gnss_imu_time, time_diff, GNSS_IMU_calibration_distance);
                    std::cout << "time_diff:" << time_diff << std::endl;
                    time_of_day_sec += time_diff;
                    if (shift_time_sinc)
                    {
                        std::cout << "\nsynchronised\n, press enter..., time_of_day_sec:" << time_of_day_sec << std::endl;
                        std::cin.get();

                        auto first_lidar_time = lidar_time[0] + time_diff;
                        auto first_lidar_pose = lidar[0];

                        reader.init(first_lidar_time);

                        Sophus::SE3 gnss_at_first_lidar_time = reader.closestPose(first_lidar_time); // gnss at first lidar time
                        tmp_index = reader.curr_index;

                        // THERE IS A BUG HERE

                        // FIGURE OUT HOW TO SOLVE IT

                        // take only the position of the first pose - keeps the orientation as it it, so gravity = earth gravity
                        first_ppk_gnss_pose_inverse = Sophus::SE3(M3D::Identity(), gnss_at_first_lidar_time.translation()).inverse();

                        auto tmp_pose = first_ppk_gnss_pose_inverse * interpolated_pose;

                        Sophus::SE3 T_L0 = first_lidar_pose; // first LiDAR–inertial pose
                        // Alignment transform: GNSS -> LiDAR
                        // T_LG = T_L0 * se3.inverse();
                        T_LG = T_L0 * tmp_pose.inverse();

                        // use only the yaw angle
                        double yaw = T_LG.so3().matrix().eulerAngles(2, 1, 0)[0]; // rotation around Z // yaw, pitch, roll (Z,Y,X order)
                        Eigen::AngleAxisd yawRot(yaw, V3D::UnitZ());
                        T_LG = Sophus::SE3(yawRot.toRotationMatrix(), V3D::Zero());

                        // Transform any GNSS pose into LiDAR–Inertial frame
                        // Sophus::SE3 T_Lk = T_LG * se3;
                    }

                    // find the motion change in first and motion change in second

                    // take the lidar odometry pose + time from init + curr scan delta
                    // take the gnss-imu time  pose + time tod

                    // given

                    // std::vector<Sophus::SE3> sensor1_poses; (lidar)
                    // std::vector<Sophus::SE3> sensor2_poses; (gnss-imu)

                    // std::vector<double> sensor1_time;
                    // std::vector<double> sensor2_time;

                    // check if delta pose from the first-last sensor1_poses > 50
                    // check if delta pose from the first-last sensor2_poses > 50
                    // there is enough motion in both trajectorries

                    // aling the trajectory and find the time_diff

                    // i know that gnss-imu starts earlier
                }
            }
            else
            {
                std::cout << "GNSS reader not initted..." << std::endl;
                throw std::runtime_error("GNSS reader not initted...");
            }
        }
    }
    for (auto &b : bags)
        b->close();
}
