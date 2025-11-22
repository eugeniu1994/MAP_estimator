#include "IMU.hpp"

IMU_Class::IMU_Class() : imu_need_init_(true), start_timestamp_(-1)
{
    init_iter_num = 1;
    Q = process_noise_cov();
    cov_acc = V3D(0.1, 0.1, 0.1);
    cov_gyr = V3D(0.1, 0.1, 0.1);
    cov_bias_gyr = V3D(0.0001, 0.0001, 0.0001);
    cov_bias_acc = V3D(0.0001, 0.0001, 0.0001);
    mean_acc = V3D(0, 0, -1.0);
    mean_gyr = V3D(0, 0, 0);
    angvel_last = Zero3d;
    Lidar_T_wrt_IMU = Zero3d;
    Lidar_R_wrt_IMU = Eye3d;
    last_imu_.reset(new sensor_msgs::Imu());
}

IMU_Class::~IMU_Class() {}

void IMU_Class::reset()
{
    ROS_WARN("IMU reset");
    mean_acc = V3D(0, 0, -1.0);
    mean_gyr = V3D(0, 0, 0);
    angvel_last = Zero3d;
    imu_need_init_ = true;
    start_timestamp_ = -1;
    init_iter_num = 1;
    IMU_Buffer.clear();
    last_imu_.reset(new sensor_msgs::Imu());
}

void IMU_Class::set_param(const V3D &tran, const M3D &rot, const V3D &gyr, const V3D &acc, const V3D &gyr_bias, const V3D &acc_bias)
{
    Lidar_T_wrt_IMU = tran;
    Lidar_R_wrt_IMU = rot;
    cov_gyr_scale = gyr;
    cov_acc_scale = acc;
    cov_bias_gyr = gyr_bias;
    cov_bias_acc = acc_bias;

    std::cout << "set_param:\n"
              << std::endl;
    std::cout << "cov_acc_scale:" << cov_acc_scale.transpose() << "\ncov_gyr_scale:" << cov_gyr_scale.transpose() << std::endl;
}

/**
 * inline Sophus::SO3d align_accel_to_z_world(const V3D &accel) {
    //  unobservable in the gravity direction, and the z in R.log() will always be 0
    const V3D z_world = {0.0, 0.0, 1.0};
    const Eigen::Quaterniond quat_accel = Eigen::Quaterniond::FromTwoVectors(accel, z_world);
    return Sophus::SO3d(quat_accel);
}
void IMU_Class::Set_init(Eigen::Vector3d &tmp_gravity, Eigen::Matrix3d &rot)
{
   1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity
  // V3D tmp_gravity = - mean_acc / mean_acc.norm() * G_m_s2; // state_gravity;
  M3D hat_grav;
  hat_grav << 0.0, gravity_(2), -gravity_(1),
              -gravity_(2), 0.0, gravity_(0),
              gravity_(1), -gravity_(0), 0.0;
  double align_norm = (hat_grav * tmp_gravity).norm() / gravity_.norm() / tmp_gravity.norm();
  double align_cos = gravity_.transpose() * tmp_gravity;
  align_cos = align_cos / gravity_.norm() / tmp_gravity.norm();
  if (align_norm < 1e-6)
  {
    if (align_cos > 1e-6)
    {
      rot = Eye3d;
    }
    else
    {
      rot = -Eye3d;
    }
  }
  else
  {
    V3D align_angle = hat_grav * tmp_gravity / (hat_grav * tmp_gravity).norm() * acos(align_cos);
    rot = Exp(align_angle(0), align_angle(1), align_angle(2));
  }
}**/

void IMU_Class::IMU_init(const MeasureGroup &meas, Estimator &kf_state, int &N)
{
    V3D cur_acc, cur_gyr;

    if (b_first_frame_)
    {
        reset();
        N = 1;
        b_first_frame_ = false;
        const auto &imu_acc = meas.imu.front()->linear_acceleration;
        const auto &gyr_acc = meas.imu.front()->angular_velocity;
        mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    }

    for (const auto &imu : meas.imu)
    {
        const auto &imu_acc = imu->linear_acceleration;
        const auto &gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

        mean_acc += (cur_acc - mean_acc) / N;
        mean_gyr += (cur_gyr - mean_gyr) / N;

        cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) / N;
        cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) / N / N * (N - 1);

        N++;

        // accelNoiseEstimator.update(cur_acc);
    }

    state init_state = kf_state.get_x();

    std::cout << "Initialization will be done with:" << meas.imu.size() << " measurements " << std::endl;

    bool init_orientation = true;

    if (init_orientation)
    {
        // assuming the IMU is mostly stationary.
        V3D sum_acc(0., 0., 0.);
        std::vector<V3D> linear_accelerations; // used later
        for (const auto &imu : meas.imu)
        {
            const auto &imu_acc = imu->linear_acceleration;
            cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
            sum_acc += cur_acc;

            linear_accelerations.emplace_back(
                imu->linear_acceleration.x,
                imu->linear_acceleration.y,
                imu->linear_acceleration.z);
        }

        V3D average_gravity = sum_acc / (double)meas.imu.size(), median_gravity;
        std::cout << "average gravity:" << average_gravity.transpose() << std::endl;
        // Calculate median---------------------------------------------------------
        std::sort(linear_accelerations.begin(), linear_accelerations.end(), [](const Eigen::Vector3d &a, const Eigen::Vector3d &b)
                  {
                      return a.norm() < b.norm(); // Sort ascending by the norm of the vector
                  });
        size_t size_linear_accelerations = linear_accelerations.size();
        if (size_linear_accelerations % 2 == 0)
        {
            median_gravity = (linear_accelerations[size_linear_accelerations / 2 - 1] + linear_accelerations[size_linear_accelerations / 2]) / 2.0; // Average of middle two
        }
        else
        {
            median_gravity = linear_accelerations[size_linear_accelerations / 2]; // Middle element
        }
        std::cout << "median gravity:" << median_gravity.transpose() << std::endl;
        V3D stdDev = calculateStdDev(linear_accelerations, average_gravity);
        std::cout << "Standard Deviation of gravity: " << stdDev.transpose() << std::endl;
        // TODO - check if stdDev is too big - initialization is not good

        const V3D &z_axis = average_gravity.normalized();
        // const Eigen::Vector3d &z_axis = median_gravity.normalized(); //test this
        V3D x_axis = V3D::UnitX() - z_axis * z_axis.transpose() * V3D::UnitX();
        x_axis.normalize();
        V3D y_axis = z_axis.cross(x_axis); // y_axis = skew_x(z_axis)*x_axis;
        y_axis.normalize();

        Rbw.block<3, 1>(0, 0) = x_axis;
        Rbw.block<3, 1>(0, 1) = y_axis;
        Rbw.block<3, 1>(0, 2) = z_axis;

        std::cout << "Init rotation:\n"
                  << Rbw.transpose() << std::endl;
        V3D eulerAngles = Rbw.eulerAngles(0, 1, 2); // Roll, Pitch, Yaw
        std::cout << "Init rotation to Euler Angles (degrees): \n"
                  << eulerAngles.transpose() * 180.0 / M_PI << std::endl;

        // with given init rotation
        init_state.grav = Eigen::Vector3d(0, 0, -G_m_s2);
        init_state.rot = Sophus::SO3(Rbw.transpose());
    }
    else
    {
        Rbw = Eye3d;
        init_state.grav = -mean_acc / mean_acc.norm() * G_m_s2;
    }

    // to be done - set the acceleration bias
    // V3D ba = mean_acc - Rbw * _gravity;

    init_state.bg = mean_gyr;

    init_state.offset_T_L_I = Lidar_T_wrt_IMU;
    init_state.offset_R_L_I = Sophus::SO3(Lidar_R_wrt_IMU);
    kf_state.set_x(init_state);
    std::cout << "Init state gravity:" << init_state.grav.transpose() << std::endl;

    cov init_P = kf_state.get_P();
    init_P.block(Re_ID, Re_ID, 3, 3) = Eye3d * 0.00001;
    init_P.block(Te_ID, Te_ID, 3, 3) = Eye3d * 0.00001;
    init_P.block(BG_ID, BG_ID, 3, 3) = Eye3d * 0.0001;
    init_P.block(BA_ID, BA_ID, 3, 3) = Eye3d * 0.001;
    init_P.block(G_ID, G_ID, 3, 3) = Eye3d * 0.00001;

    kf_state.set_P(init_P);
    last_imu_ = meas.imu.back();

    std::cout << "IMU_init\n"
              << std::endl;
    std::cout << "cov_gyr:" << cov_gyr.transpose() << ", cov_acc:" << cov_acc.transpose() << std::endl;
    std::cout << "cov_bias_gyr:" << cov_bias_gyr.transpose() << ", cov_bias_acc:" << cov_bias_acc.transpose() << std::endl;

    Q.block<3, 3>(G_VAR_ID, G_VAR_ID).diagonal() = cov_gyr;
    Q.block<3, 3>(A_VAR_ID, A_VAR_ID).diagonal() = cov_acc;
    Q.block<3, 3>(BG_VAR_ID, BG_VAR_ID).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(BA_VAR_ID, BA_VAR_ID).diagonal() = cov_bias_acc;

    std::cout << "Q G_VAR imu:\n"
              << Q.block<3, 3>(G_VAR_ID, G_VAR_ID) << std::endl;
    std::cout << "Q A_VAR_ID imu:\n"
              << Q.block<3, 3>(A_VAR_ID, A_VAR_ID) << std::endl;
}

void IMU_Class::IMU_init_from_GT(const MeasureGroup &meas, Estimator &kf_state, const Sophus::SE3 &gt)
{
    std::cout << "IMU_init_from_GT" << std::endl;
    V3D cur_acc, cur_gyr;

    int N = 1;
    if (b_first_frame_)
    {
        reset();
        N = 1;
        b_first_frame_ = false;
        const auto &imu_acc = meas.imu.front()->linear_acceleration;
        const auto &gyr_acc = meas.imu.front()->angular_velocity;
        mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    }

    for (const auto &imu : meas.imu)
    {
        const auto &imu_acc = imu->linear_acceleration;
        const auto &gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

        mean_acc += (cur_acc - mean_acc) / N;
        mean_gyr += (cur_gyr - mean_gyr) / N;

        cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) / N;
        cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) / N / N * (N - 1);

        N++;
    }

    state init_state = kf_state.get_x();

    std::cout << "Initialization will be done with:" << meas.imu.size() << " measurements " << std::endl;

    Rbw = gt.so3().matrix();
    std::cout << "gt rotation:\n"
              << Rbw << std::endl;
    auto eulerAngles = Rbw.eulerAngles(0, 1, 2); // Roll, Pitch, Yaw
    std::cout << "gt rotation to Euler Angles (degrees): \n"
              << eulerAngles.transpose() * 180.0 / M_PI << std::endl;

    // with given init rotation
    init_state.grav = Eigen::Vector3d(0, 0, -G_m_s2);
    init_state.rot = Sophus::SO3(Rbw); //

    // TODO
    /*
    add option to pass directly the values of the gravity
    */

    kf_state.set_x(init_state);
    std::cout << "Init state gravity:" << init_state.grav.transpose() << std::endl;
    init_from_GT = true;

    // to be done - set the acceleration bias
    // V3D ba = mean_acc - Rbw * _gravity;

    init_state.bg = mean_gyr;

    init_state.offset_T_L_I = Lidar_T_wrt_IMU;
    init_state.offset_R_L_I = Sophus::SO3(Lidar_R_wrt_IMU);
    kf_state.set_x(init_state);
    std::cout << "Init state gravity:" << init_state.grav.transpose() << std::endl;

    cov init_P = kf_state.get_P();
    init_P.block(Re_ID, Re_ID, 3, 3) = Eye3d * 0.00001;
    init_P.block(Te_ID, Te_ID, 3, 3) = Eye3d * 0.00001;
    init_P.block(BG_ID, BG_ID, 3, 3) = Eye3d * 0.0001;
    init_P.block(BA_ID, BA_ID, 3, 3) = Eye3d * 0.001;
    init_P.block(G_ID, G_ID, 3, 3) = Eye3d * 0.00001;

    kf_state.set_P(init_P);
    last_imu_ = meas.imu.back();

    std::cout << "IMU_init\n"
              << std::endl;
    std::cout << "cov_gyr:" << cov_gyr.transpose() << ", cov_acc:" << cov_acc.transpose() << std::endl;
    std::cout << "cov_bias_gyr:" << cov_bias_gyr.transpose() << ", cov_bias_acc:" << cov_bias_acc.transpose() << std::endl;

    Q.block<3, 3>(G_VAR_ID, G_VAR_ID).diagonal() = cov_gyr;
    Q.block<3, 3>(A_VAR_ID, A_VAR_ID).diagonal() = cov_acc;
    Q.block<3, 3>(BG_VAR_ID, BG_VAR_ID).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(BA_VAR_ID, BA_VAR_ID).diagonal() = cov_bias_acc;

    std::cout << "Q G_VAR imu:\n"
              << Q.block<3, 3>(G_VAR_ID, G_VAR_ID) << std::endl;
    std::cout << "Q A_VAR_ID imu:\n"
              << Q.block<3, 3>(A_VAR_ID, A_VAR_ID) << std::endl;
}

void IMU_Class::Propagate2D(std::vector<pcl::PointCloud<VUX_PointType>::Ptr> &vux_scans,
                            const std::vector<double> &vux_scans_time,
                            const double &pcl_beg_time,
                            const double &pcl_end_time,
                            const double &tod_end_scan,
                            const Sophus::SE3 &prev_mls, const double &prev_mls_time)
{
    std::cout << "IMU Propagate2D vux_scans: " << vux_scans.size() << std::endl;
    if (vux_scans.empty() || IMU_Buffer.size() < 2)
    {
        std::cout << "No valid data..." << std::endl;
        return;
    }

    double scan_duration = pcl_end_time - pcl_beg_time; // e.g., 0.1s
    double tod_beg_scan = tod_end_scan - scan_duration;

    if (false) // propagate for each vux from the closest imu
    {
        V3D pos_imu, vel_imu, angvel_avr, acc_avr, acc_imu;
        M3D R_imu;

        int pcl_idx = static_cast<int>(vux_scans.size()) - 1;

        for (int imu_idx = static_cast<int>(IMU_Buffer.size()) - 1; imu_idx > 0 && pcl_idx >= 0; imu_idx--)
        {
            const auto &head = IMU_Buffer[imu_idx - 1];
            const auto &tail = IMU_Buffer[imu_idx];

            R_imu << MAT_FROM_ARRAY(head.rot);
            vel_imu << VEC_FROM_ARRAY(head.vel);
            pos_imu << VEC_FROM_ARRAY(head.pos);
            acc_imu << VEC_FROM_ARRAY(tail.acc);
            angvel_avr << VEC_FROM_ARRAY(tail.gyr);

            std::cout << "\nIMU_Buffer [" << imu_idx << "] time: " << head.offset_time << " -> " << tail.offset_time << std::endl;

            // Process 2D scans in this IMU interval
            while (pcl_idx >= 0)
            {
                pcl::PointCloud<VUX_PointType>::Ptr &scan = vux_scans[pcl_idx]; // get the current scan
                if (scan->points.empty())
                {
                    pcl_idx--;
                    continue;
                }

                double tod_time = vux_scans_time[pcl_idx]; // scan->points[0].time;
                double relative_time = tod_time - tod_beg_scan;

                if (relative_time <= head.offset_time)
                    break; // move to earlier IMU segment

                double dt = relative_time - head.offset_time;
                std::cout << "  VUX[" << pcl_idx << "] rel_time: " << relative_time << ", dt: " << dt << std::endl;

                M3D R_i = R_imu * Sophus::SO3::exp(angvel_avr * dt).matrix();
                V3D T_ei = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

                Sophus::SE3 interpolated_imu_pose(R_i, T_ei);

                // apply transform to each point in the scan
                for (auto &pt : scan->points)
                {
                    Eigen::Vector3d p_raw(pt.x, pt.y, pt.z);
                    Eigen::Vector3d p_new = interpolated_imu_pose * p_raw;
                    pt.x = p_new.x();
                    pt.y = p_new.y();
                    pt.z = p_new.z();
                }

                pcl_idx--; // move to earlier 2D scan
            }
        }
    }
    else // interpolate from the begin and end of the 2d scan
    {
        // also maybe apply voxelization before transformation
        // when combining with hesai - combined after voxelization
        //     each scan is voxelized individually
        //     to prevent the voxel grid problems when too close to each other
        //         if bad imu -> will result in false measurements

        std::cout << std::endl;
        // std::cout<<"prev_mls_time:"<<prev_mls_time<<", last_lidar_end_time_:"<<last_lidar_end_time_<<std::endl;
        // std::cout<<"tod_beg_scan:"<<tod_beg_scan<<", tod_end_scan:"<<tod_end_scan<<std::endl;

        // auto dt_ = last_lidar_end_time_ - prev_mls_time;
        auto dt_tod = tod_end_scan - tod_beg_scan;
        // std::cout<<"dt_:"<<dt_<<", dt_tod:"<<dt_tod<<std::endl;

        auto delta_predicted = (prev_mls.inverse() * Sophus::SE3(imu_state.rot, imu_state.pos)).log();

        for (int i = 0; i < vux_scans.size(); i++)
        {
            const double &t = vux_scans_time[i];        // vux_scans[i]->points[0].time;
            double alpha = (t - tod_beg_scan) / dt_tod; // when t will be bigger thatn time 2 it will extrapolate

            Sophus::SE3 interpolated_imu_pose = prev_mls * Sophus::SE3::exp(alpha * delta_predicted);

            int n_points = vux_scans[i]->points.size();
#pragma omp parallel for
            for (int j = 0; j < n_points; ++j)
            {
                auto &pt = vux_scans[i]->points[j];
                V3D p_raw(pt.x, pt.y, pt.z);
                V3D p_new = interpolated_imu_pose * p_raw;
                pt.x = p_new.x();
                pt.y = p_new.y();
                pt.z = p_new.z();
                pt.reflectance = i;
            }
        }

        // THE NEXT IS INTERPOLATING BETWEEN THE FIRST AND LAST IMU poses
        //  M3D R_imu1, R_imu2;
        //  V3D pos_imu1, pos_imu2;
        //  double time1 = IMU_Buffer[0].offset_time;
        //  R_imu1 << MAT_FROM_ARRAY(IMU_Buffer[0].rot);
        //  pos_imu1 << VEC_FROM_ARRAY(IMU_Buffer[0].pos);
        //  const Sophus::SE3 pose1(R_imu1, pos_imu1);

        // int n = IMU_Buffer.size() - 1;
        // double time2 = IMU_Buffer[n].offset_time;
        // R_imu2 << MAT_FROM_ARRAY(IMU_Buffer[n].rot);
        // pos_imu2 << VEC_FROM_ARRAY(IMU_Buffer[n].pos);
        // const Sophus::SE3 pose2(R_imu2, pos_imu2);

        // Sophus::SE3 delta = pose1.inverse() * pose2;

        // for (int i = 0; i < vux_scans.size(); i++)
        // {
        //     const double &tod_time = vux_scans_time[i];// vux_scans[i]->points[0].time;
        //     double t = tod_time - tod_beg_scan;           // relative_time
        //     double alpha = (t - time1) / (time2 - time1); // when t will be bigger thatn time 2 it will extrapolate

        //     Sophus::SE3 interpolated_imu_pose = pose1 * Sophus::SE3::exp(alpha * delta.log());

        //     int n_points = vux_scans[i]->points.size();
        //     #pragma omp parallel for
        //     for (int j = 0; j < n_points; ++j)
        //     {
        //         auto &pt = vux_scans[i]->points[j];
        //         V3D p_raw(pt.x, pt.y, pt.z);
        //         V3D p_new = interpolated_imu_pose * p_raw;
        //         pt.x = p_new.x();
        //         pt.y = p_new.y();
        //         pt.z = p_new.z();
        //         pt.reflectance = j;
        //     }
        // }
    }
}

void IMU_Class::ConstVelUndistort(const MeasureGroup &meas, Estimator &kf_state, PointCloudXYZI::Ptr &frame, const Sophus::SE3 &prev_, const Sophus::SE3 &curr_)
{
    *frame = *(meas.lidar);
    const size_t N = frame->size();

    double first_point_time = frame->points[0].time;
    double last_point_time = frame->points.back().time;
    std::cout << "first_point_time:" << first_point_time << ", last_point_time:" << last_point_time << std::endl;
    double mid_ = (last_point_time - first_point_time) / 2.; // this works just fine for lieksa
    mid_ = 0;                                                // worked for evo
    mid_ = (last_point_time - first_point_time);             // end scan

    auto delta_pose = (prev_.inverse() * curr_).log();
    auto velocity = delta_pose / (last_point_time - first_point_time);

    imu_state = kf_state.get_x();
    const auto &R_L2I = imu_state.offset_R_L_I.matrix();
    const auto &R_I2L = imu_state.offset_R_L_I.matrix().transpose();

    // tbb::parallel_for(size_t(0), N, [&](size_t i)
    for (int i = 0; i < N; i++)
    {
        V3D_4 P_i(frame->points[i].x, frame->points[i].y, frame->points[i].z);

        // decoupled rotation and translation: Sophus::SE3d T_j = Sophus::interpolate(T_begin, T_end, scale);

        // coupled rotation and translation
        const auto motion_imu = Sophus::SE3::exp((frame->points[i].time - first_point_time - mid_) * velocity);

        V3D P_compensate = R_I2L * ((motion_imu * (R_L2I * P_i + imu_state.offset_T_L_I)) - imu_state.offset_T_L_I);

        // P_compensate(2) += (frame->points[i].time - first_point_time) * 10;

        frame->points[i].x = P_compensate(0);
        frame->points[i].y = P_compensate(1);
        frame->points[i].z = P_compensate(2);
    }
    //);
}

void IMU_Class::Propagate(const MeasureGroup &meas, Estimator &kf_state, PointCloudXYZI &pcl_out)
{
    auto v_imu = meas.imu;
    v_imu.push_front(last_imu_);

    const double &imu_end_time = v_imu.back()->header.stamp.toSec();
    const double &pcl_beg_time = meas.lidar_beg_time;
    const double &pcl_end_time = meas.lidar_end_time;
    const double &imu_beg_time = v_imu.front()->header.stamp.toSec();

    double dt = pcl_end_time - pcl_beg_time;
    if (dt < 0)
    {
        throw std::invalid_argument("Negative pcl_dt time");
    }

    pcl_out = *(meas.lidar);
    // Sort the point cloud by the timestamp if not sorted already
    // sort(pcl_out.points.begin(), pcl_out.points.end(), time_list); // sort by timestamp

    imu_state = kf_state.get_x();

    IMU_Buffer.clear();
    IMU_Buffer.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.matrix()));

    bool use_smoothig = true;
    forward_results_.clear();
    if (use_smoothig)
    {
        ForwardResult f0; // first state is from the prev update state, pred = update
        f0.x_pred = kf_state.get_x();
        f0.P_pred = kf_state.get_P();
        f0.x_update = kf_state.get_x();
        f0.P_update = kf_state.get_P();

        f0.x_update2 = kf_state.get_x();
        f0.P_update2 = kf_state.get_P();

        forward_results_.push_back(f0);
    }

    V3D pos_imu, vel_imu, angvel_avr, acc_avr, acc_imu;
    M3D R_imu;

    input in;

    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
    {
        auto &&head = *(it_imu);
        auto &&tail = *(it_imu + 1);

        if (tail->header.stamp.toSec() < last_lidar_end_time_)
            continue;

        angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
            0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
            0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
        acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
            0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
            0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

        acc_avr = acc_avr * G_m_s2 / mean_acc.norm();

        if (head->header.stamp.toSec() < last_lidar_end_time_)
        {
            dt = tail->header.stamp.toSec() - last_lidar_end_time_;
        }
        else
        {
            dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        }

        in.acc = acc_avr;
        in.gyro = angvel_avr;

        kf_state.predict(dt, Q, in);
        imu_state = kf_state.get_x();

        // this is new stuff---------------------------------------------------
        bool perform_imu_update = false; // true;
        if (perform_imu_update)
        { // TODO: put this before
            //------------------------------------------------------------------
            // const V3D loc_grav_measured = imu_state.rot.inverse() * Eigen::Vector3d(0, 0, -G_m_s2); // gravity local
            // const V3D loc_grav_predicted = imu_state.rot.inverse() * imu_state.grav;                // estimated
            // std::cout << "\nloc_grav_1:" << loc_grav_measured.transpose() << "\nloc_grav_2:" << loc_grav_predicted.transpose() << std::endl;

            // std::cout << "r:" << r.transpose() << "--------------------" << std::endl;

            if (false)
            {
                // numeric Jacobian
                auto compute_residual = [&](const state &s) -> Eigen::Vector3d
                {
                    Eigen::Matrix3d R_ = s.rot.matrix();
                    Eigen::Vector3d g_w_ = s.grav;
                    Eigen::Vector3d Delta_g_ = Eigen::Vector3d(0, 0, -G_m_s2) - g_w_;
                    return R_.transpose() * Delta_g_;
                };

                double eps = 1e-7;
                Eigen::Matrix<double, 3, state_size> H_num;
                state x0 = imu_state;
                Eigen::Vector3d r0 = compute_residual(x0);

                for (int i = 0; i < state_size; ++i)
                {
                    Eigen::Matrix<double, state_size, 1> dv = Eigen::Matrix<double, state_size, 1>::Zero();
                    dv(i) = eps;
                    state x_p = kf_state.boxplus(x0, dv);
                    Eigen::Vector3d r1 = compute_residual(x_p);
                    H_num.col(i) = (r1 - r0) / eps;
                }
                M3D Rinv = imu_state.rot.matrix().transpose();
                V3D g_w = imu_state.grav;
                V3D g_ref_world(0.0, 0.0, -G_m_s2);

                Eigen::Matrix<double, 3, state_size> H = Eigen::Matrix<double, 3, state_size>::Zero();
                H.block<3, 3>(0, R_ID) = Sophus::SO3::hat(Rinv * (g_ref_world - g_w));
                H.block<3, 3>(0, G_ID) = -Rinv;

                // H_num with analytic H
                std::cout << "||H_num - H_analytic|| = " << (H_num - H).norm() << std::endl;

                // std::cout<<"\nH_analytic:\n"<<"H_R:\n"<<H.block<3, 3>(0, R_ID)<<"\nH_G:\n"<<H.block<3, 3>(0, G_ID)<<std::endl;
                // std::cout<<"H_num:\n"<<"H_R:\n"<<H_num.block<3, 3>(0, R_ID)<<"\nH_G:\n"<<H_num.block<3, 3>(0, G_ID)<<std::endl;
            }

            const V3D g_ref_world(0.0, 0.0, -G_m_s2); // reference gravity in world
            M3D R = imu_state.rot.matrix();           // world -> IMU rotation
            V3D g_w = imu_state.grav;                 // gravity in world (state)
            // predicted local gravity (in IMU frame)
            // not used V3D g_l = R.transpose() * g_w; // = R^{-1} * g_w

            V3D Delta_g = g_ref_world - g_w; // 3x1
            // predicted measured local gravity difference: r = R^{-1} * Delta_g
            M3D Rinv = R.transpose(); // R^{-1}
            V3D r = Rinv * Delta_g;   // residual (3x1) - we use meas - pred form   r = R.T * (g_ref - g_pred)
            // std::cout << "r:" << r.transpose() << std::endl;

            //  Jacobians - compute dh(x)/dx
            Eigen::Matrix<double, 3, state_size> H = Eigen::Matrix<double, 3, state_size>::Zero();
            H.block<3, 3>(0, R_ID) = Sophus::SO3::hat(Rinv * Delta_g);
            H.block<3, 3>(0, G_ID) = -Rinv;

            //  measurement covariance (3x3)
            // double meas_noise_std = 0.001;// 0.01;// 0.1;// 1; // take this from data itself - Q matrix
            // M3D Rm = (meas_noise_std * meas_noise_std) * M3D::Identity();
            // std::cout<<"Q:\n"<<Q.block<3, 3>(A_VAR_ID, A_VAR_ID).diagonal().transpose()<<std::endl;

            accelNoiseEstimator.update(r); // update stats
            V3D meas_std = accelNoiseEstimator.stddev();
            std::cout << "meas_std:" << meas_std.transpose() << std::endl;
            M3D Rm = accelNoiseEstimator.covariance();
            // std::cout<<"Rm:\n"<<Rm<<std::endl;

            // Kalman update
            auto P = kf_state.get_P();                                                // 24x24
            M3D S = H * P * H.transpose() + Rm;                                       // 3x3
            Eigen::Matrix<double, state_size, 3> K = P * H.transpose() * S.inverse(); // 24x3
            Eigen::Matrix<double, state_size, 1> dx = -K * r;                         // 24x1 - take negative sign due to r=z−h(x)  dr/dx = dr/dh(x) * dh(x)/dx = -1*dh(x)/dx

            imu_state = kf_state.boxplus(imu_state, dx); // dx is the delta corection that should be applied

            // double grav_err_prev = Delta_g.norm();
            // double grav_err_now = (g_ref_world - imu_state.grav).norm();

            double r_init = r.norm();
            double r_now = (imu_state.rot.matrix().transpose() * (g_ref_world - imu_state.grav)).norm();
            // std::cout<<"grav_err_prev:"<<grav_err_prev<<", grav_err_now:"<<grav_err_now<<",  diff:"<<(grav_err_prev - grav_err_now)<<std::endl;
            std::cout << "r_init:" << r_init << ", r_now:" << r_now << ",  diff:" << (r_init - r_now) << std::endl;
            std::cout << "Curr imu_state.grav:" << imu_state.grav.transpose() << std::endl;

            //  Covariance update (Joseph form)
            Eigen::Matrix<double, state_size, state_size> I = Eigen::Matrix<double, state_size, state_size>::Identity();
            // P -= K * H * P;
            kf_state.set_x(imu_state);
            // kf_state.set_P(P);
        }

        angvel_last = V3D(tail->angular_velocity.x, tail->angular_velocity.y, tail->angular_velocity.z) - imu_state.bg;
        acc_s_last = V3D(tail->linear_acceleration.x, tail->linear_acceleration.y, tail->linear_acceleration.z) * G_m_s2 / mean_acc.norm();

        if (G_m_s2 != 1)
            acc_s_last = imu_state.rot * (acc_s_last - imu_state.ba) + imu_state.grav;
        else
            acc_s_last = imu_state.rot * (acc_s_last - imu_state.ba);

        double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
        IMU_Buffer.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.matrix()));

        if (use_smoothig)
        {
            ForwardResult f_curr;
            f_curr.x_pred = kf_state.get_x();
            f_curr.P_pred = kf_state.get_P();
            f_curr.F = kf_state.get_Fx();
            forward_results_.push_back(f_curr);
        }
    }
    std::cout << "----------------------------------------------------------" << std::endl;

    dt = abs(pcl_end_time - imu_end_time);

    kf_state.predict(dt, Q, in); // Predict the IMU state at the end of the scan
    if (use_smoothig)
    {
        ForwardResult f_curr;
        f_curr.x_pred = kf_state.get_x();
        f_curr.P_pred = kf_state.get_P();
        f_curr.F = kf_state.get_Fx();
        forward_results_.push_back(f_curr);
    }

    imu_state = kf_state.get_x();

    last_imu_ = meas.imu.back();
    last_lidar_end_time_ = pcl_end_time;

    auto it_pcl = pcl_out.points.end() - 1;
    auto begin_pcl = pcl_out.points.begin();

    const auto &end_R_T = imu_state.rot.matrix().transpose();
    const auto &R_L2I = imu_state.offset_R_L_I.matrix();
    const auto &R_I2L = imu_state.offset_R_L_I.matrix().transpose();

    std::cout << "grav_1:" << V3D(0, 0, -G_m_s2).transpose() << "\ngrav_2:" << imu_state.grav.transpose() << std::endl;

    for (auto it_kp = IMU_Buffer.end() - 1; it_kp != IMU_Buffer.begin(); it_kp--)
    {
        auto head = it_kp - 1;
        auto tail = it_kp;
        R_imu << MAT_FROM_ARRAY(head->rot);
        vel_imu << VEC_FROM_ARRAY(head->vel);
        pos_imu << VEC_FROM_ARRAY(head->pos);
        acc_imu << VEC_FROM_ARRAY(tail->acc);
        angvel_avr << VEC_FROM_ARRAY(tail->gyr);

        if (true) // iterative
        {
            for (; it_pcl->time > head->offset_time; it_pcl--)
            {
                dt = it_pcl->time - head->offset_time;
                // P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei)

                M3D R_i(R_imu * Sophus::SO3::exp(angvel_avr * dt).matrix());
                V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
                V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
                V3D P_compensate = R_I2L * (end_R_T * (R_i * (R_L2I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);

                // just a test to skip the motion distortion
                // P_compensate = P_i; // use the original point

                it_pcl->x = P_compensate(0);
                it_pcl->y = P_compensate(1);
                it_pcl->z = P_compensate(2);

                if (it_pcl == begin_pcl)
                    break;
            }
        }
        else
        {
            int end_ = it_pcl - begin_pcl;
            for (; it_pcl->time > head->offset_time; it_pcl--)
            {
                if (it_pcl == begin_pcl)
                    break;
            }
            int start_ = it_pcl - begin_pcl;

            if (start_ < 0 || end_ >= pcl_out.points.size())
            {
                std::cout << "ERROR  start_: " << start_ << ", end_:" << end_ << ", pcl_out->size():" << pcl_out.points.size() << std::endl;
                throw std::invalid_argument("ERROR in undistort");
            }

            // std::cout << "test start_:" << start_ << ", end_:" << end_ << std::endl;
            try
            {
                tbb::parallel_for(tbb::blocked_range<int>(start_, end_),
                                  [&](tbb::blocked_range<int> r)
                                  {
                                      for (int i = r.begin(); i < r.end(); i++)
                                      {
                                          double dt_ = pcl_out.points[i].time - head->offset_time;

                                          M3D R_i(R_imu * Sophus::SO3::exp(angvel_avr * dt_).matrix());
                                          V3D P_i(pcl_out.points[i].x, pcl_out.points[i].y, pcl_out.points[i].z);
                                          V3D T_ei(pos_imu + vel_imu * dt_ + 0.5 * acc_imu * dt_ * dt_ - imu_state.pos);

                                          V3D P_compensate = R_I2L * (end_R_T * (R_i * (R_L2I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);

                                          pcl_out.points[i].x = P_compensate(0);
                                          pcl_out.points[i].y = P_compensate(1);
                                          pcl_out.points[i].z = P_compensate(2);
                                      }
                                  });
            }
            catch (const std::exception &e)
            {
                std::cerr << "Exception in parallel loop: " << e.what() << std::endl;
                throw;
            }
        }
    }
}


// RTS smoother after forward pass
void IMU_Class::backwardPass(Estimator &kf_state)
{
    std::cout << "IMU_Buffer:" << IMU_Buffer.size() << std::endl;
    std::cout << "forward_results_:" << forward_results_.size() << std::endl;

    int N = forward_results_.size();

    // Initialize smoother with the final updated state
    forward_results_[N - 1].x_update = kf_state.get_x();
    forward_results_[N - 1].P_update = kf_state.get_P();

    forward_results_[N - 1].P_update2 = kf_state.get_P();
    forward_results_[N - 1].x_update2 = kf_state.get_x();

    auto P_prev = forward_results_[0].P_update;
    auto P_curr = kf_state.get_P();

    for (int k = 0; k < N; k++)
    {
        double alpha = static_cast<double>(k) / N;
        forward_results_[k].P_update2 = (1.0 - alpha) * P_prev + alpha * P_curr;;
    }
        
    // Backward recursion
    for (int k = N - 2; k >= 0; k--)
    {
        // Predicted covariance at time k classic RTS - I need the updated covariance here
        cov P_k_pred = forward_results_[k].P_pred; // P_{k|k-1} - this will non-zero correction reaching the previous lidar state.
        cov P_k1_pred = forward_results_[k + 1].P_pred; // P_{k+1|k}
        cov F_k1 = forward_results_[k + 1].F;           // F_{k+1}

        // Smoother gain for prediction states
        cov C_k = P_k_pred * F_k1.transpose() * P_k1_pred.inverse();

        // Smoothed state : x_{k | N} = x_{k | k - 1} + C_k * (x_{k + 1 | N} - x_{k + 1 | k})
        vectorized_state dx_correction_ = C_k * kf_state.boxminus(forward_results_[k + 1].x_update, forward_results_[k + 1].x_pred);
        forward_results_[k].x_update = kf_state.boxplus(forward_results_[k].x_pred, dx_correction_);

        // Smoothed covariance: P_{k|N} = P_{k|k-1} + C_k * (P_{k+1|N} - P_{k+1|k}) * C_k^T
        forward_results_[k].P_update = P_k_pred +
                                       C_k * (forward_results_[k + 1].P_update - P_k1_pred) * C_k.transpose();

        auto predicted_ = forward_results_[k].x_pred;
        auto updated_ = forward_results_[k].x_update;
        auto t_diff = (predicted_.pos - updated_.pos).norm();

        std::cout << "\nk=" << k << " first  correction norm (pos): " << t_diff << std::endl;

        //-------------------------------------------------------------------------

        P_k_pred = forward_results_[k].P_update2; //taken from linear interpolation of updated covs 
        //F_k1 = forward_results_[k].F;
        C_k = P_k_pred * F_k1.transpose() * P_k1_pred.inverse();
        dx_correction_ = C_k * kf_state.boxminus(forward_results_[k + 1].x_update2, forward_results_[k + 1].x_pred);
        forward_results_[k].x_update2 = kf_state.boxplus(forward_results_[k].x_pred, dx_correction_);

        forward_results_[k].P_update2 = P_k_pred +
                                        C_k * (forward_results_[k + 1].P_update2 - P_k1_pred) * C_k.transpose();

        updated_ = forward_results_[k].x_update2;
        t_diff = (predicted_.pos - updated_.pos).norm();
        std::cout << "k=" << k << " second correction norm (pos): " << t_diff << std::endl;
    }
}

void IMU_Class::Process(const MeasureGroup &meas, Estimator &kf_state, PointCloudXYZI::Ptr &pcl_un_)
{
    if (meas.imu.empty())
    {
        std::cout << "Empty IMU list" << std::endl;
        *pcl_un_ = *meas.lidar;
        return;
    };
    ROS_ASSERT(meas.lidar != nullptr);

    // std::cout<<"Q G_VAR imu:\n"<<Q.block<3, 3>(G_VAR_ID, G_VAR_ID)<<std::endl;
    // std::cout<<"Q A_VAR_ID imu:\n"<<Q.block<3, 3>(A_VAR_ID, A_VAR_ID)<<std::endl;
    // std::cout<<"cov_gyr_scale:"<<cov_gyr_scale.transpose()<<std::endl;
    // std::cout<<"cov_acc_scale:"<<cov_acc_scale.transpose()<<std::endl;
    if (imu_need_init_)
    {
        std::cout << "IMU_init from the raw acceleration values ..." << std::endl;
        if (!init_from_GT)
        {
            IMU_init(meas, kf_state, init_iter_num);
        }
        else
        {
            std::cout << "IMU was initialized from GT data..." << std::endl;
            init_iter_num = meas.imu.size();
        }

        imu_need_init_ = true;
        last_imu_ = meas.imu.back();
        if (init_iter_num > MIN_INIT_COUNT)
        {
            imu_need_init_ = false;

            std::cout << "\n\nInit" << std::endl;
            std::cout << "cov_acc:" << cov_acc.transpose() << std::endl;
            std::cout << "cov_gyr:" << cov_gyr.transpose() << std::endl;

            cov_acc = cov_acc_scale;
            cov_gyr = cov_gyr_scale;

            Q.block<3, 3>(G_VAR_ID, G_VAR_ID).diagonal() = cov_gyr;
            Q.block<3, 3>(A_VAR_ID, A_VAR_ID).diagonal() = cov_acc;
            Q.block<3, 3>(BG_VAR_ID, BG_VAR_ID).diagonal() = cov_bias_gyr;
            Q.block<3, 3>(BA_VAR_ID, BA_VAR_ID).diagonal() = cov_bias_acc;
            // std::cout<<"Q G_VAR imu:\n"<<Q.block<3, 3>(G_VAR_ID, G_VAR_ID)<<std::endl;
            // std::cout<<"Q A_VAR_ID imu:\n"<<Q.block<3, 3>(A_VAR_ID, A_VAR_ID)<<std::endl;
            ROS_INFO("IMU Initialization Done");
        }
        else
        {
            std::cout << "\n\n Not enough IMU,  only:" << init_iter_num << std::endl;
        }

        return;
    }

    Propagate(meas, kf_state, *pcl_un_);
}

#ifdef SAVE_DATA
double saved_scan = 0;
// Explicit template instantiation - required by the linker
// otherwise I have to implement them in the header file
template pcl::PointCloud<hesai_ros::Point> IMU_Class::DeSkewOriginalCloud<hesai_ros::Point>(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg, const state &imu_state, bool save_clouds_local);
template pcl::PointCloud<velodyne_ros::Point> IMU_Class::DeSkewOriginalCloud<velodyne_ros::Point>(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg, const state &imu_state, bool save_clouds_local);

template <typename PointT>
pcl::PointCloud<PointT> IMU_Class::DeSkewOriginalCloud(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg, const state &imu_state, bool save_clouds_local)
{
    pcl::PointCloud<PointT> pcl_out;
    pcl::fromROSMsg(*cloud_msg, pcl_out);

    auto it_pcl = pcl_out.points.end() - 1;
    auto begin_pcl = pcl_out.points.begin();

    V3D pos_imu, vel_imu, angvel_avr, acc_avr, acc_imu;
    M3D R_imu;
    const auto &end_R_T = imu_state.rot.matrix().transpose();

    const M3D &R_L2I = imu_state.offset_R_L_I.matrix();
    const M3D &R_I2L = imu_state.offset_R_L_I.matrix().transpose();

    // if constexpr to conditionally compile code based on the type of PointT
    if constexpr (std::is_same<PointT, velodyne_ros::Point>::value)
    {
        // make sure points are local not global
        auto first_point_time = pcl_out.points[0].time;
        // In some occasions point time can actually be > 3600, this is caused by the Velodyne driver,
        // therefore have to take modulus here
        first_point_time = std::fmod(first_point_time, 3600);

        tbb::parallel_for(tbb::blocked_range<int>(0, pcl_out.size()),
                          [&](const tbb::blocked_range<int> &r)
                          {
                              for (int i = r.begin(); i < r.end(); ++i)
                              {
                                  pcl_out.points[i].time -= first_point_time;
                              }
                          });
        std::cout << "Deskew for Velodyne, IMU_Buffer:" << IMU_Buffer.size() << std::endl;
        for (auto it_kp = IMU_Buffer.end() - 1; it_kp != IMU_Buffer.begin(); it_kp--)
        {
            auto head = it_kp - 1;
            auto tail = it_kp;
            R_imu << MAT_FROM_ARRAY(head->rot);
            vel_imu << VEC_FROM_ARRAY(head->vel);
            pos_imu << VEC_FROM_ARRAY(head->pos);
            acc_imu << VEC_FROM_ARRAY(tail->acc);
            angvel_avr << VEC_FROM_ARRAY(tail->gyr);

            int end_ = it_pcl - begin_pcl;
            // for (; it_pcl->time * 1e-9 > head->offset_time; it_pcl--)
            for (; it_pcl->time > head->offset_time; it_pcl--)
            {
                if (it_pcl == begin_pcl)
                    break;
            }
            int start_ = it_pcl - begin_pcl;

            if (start_ < 0 || end_ >= pcl_out.points.size())
            {
                std::cout << "ERROR DeSkewOriginalCloud start_: " << start_ << ", end_:" << end_ << ", pcl_out->size():" << pcl_out.points.size() << std::endl;
                throw std::invalid_argument("ERROR in DeSkewOriginalCloud");
            }

            tbb::parallel_for(tbb::blocked_range<int>(start_, end_),
                              [&](tbb::blocked_range<int> r)
                              {
                                  for (int i = r.begin(); i < r.end(); i++)
                                  {
                                      // double dt_ = pcl_out.points[i].time * 1e-9 - head->offset_time;
                                      double dt_ = pcl_out.points[i].time - head->offset_time;

                                      M3D R_i(R_imu * Sophus::SO3::exp(angvel_avr * dt_).matrix());
                                      V3D P_i(pcl_out.points[i].x, pcl_out.points[i].y, pcl_out.points[i].z);
                                      V3D T_ei(pos_imu + vel_imu * dt_ + 0.5 * acc_imu * dt_ * dt_ - imu_state.pos);

                                      V3D P_compensate = R_I2L * (end_R_T * (R_i * (R_L2I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);

                                      // P_compensate = P_i; //remove this later

                                      pcl_out.points[i].x = P_compensate(0);
                                      pcl_out.points[i].y = P_compensate(1);
                                      pcl_out.points[i].z = P_compensate(2);

                                      pcl_out.points[i].time += first_point_time; // put the time back
                                  }
                              });
        }
    }
    else if constexpr (std::is_same<PointT, hesai_ros::Point>::value)
    {
        // make sure points are local not global
        auto first_point_time = pcl_out.points[0].timestamp;
        tbb::parallel_for(tbb::blocked_range<int>(0, pcl_out.size()),
                          [&](const tbb::blocked_range<int> &r)
                          {
                              for (int i = r.begin(); i < r.end(); ++i)
                              {
                                  pcl_out.points[i].timestamp -= first_point_time;
                                  V3D P_i(pcl_out.points[i].x, pcl_out.points[i].y, pcl_out.points[i].z);
                                  pcl_out.points[i].range = P_i.norm();
                              }
                          });

        // first_point_time = std::fmod(first_point_time, 3600); //not sure about this - added to decrease the absolute value of time
        std::cout << "Deskew for Hesai, IMU_Buffer:" << IMU_Buffer.size() << ", first_point_time:" << first_point_time << std::endl;

        for (auto it_kp = IMU_Buffer.end() - 1; it_kp != IMU_Buffer.begin(); it_kp--)
        {
            auto head = it_kp - 1;
            auto tail = it_kp;
            R_imu << MAT_FROM_ARRAY(head->rot);
            vel_imu << VEC_FROM_ARRAY(head->vel);
            pos_imu << VEC_FROM_ARRAY(head->pos);
            acc_imu << VEC_FROM_ARRAY(tail->acc);
            angvel_avr << VEC_FROM_ARRAY(tail->gyr);

            int end_ = it_pcl - begin_pcl;
            for (; it_pcl->timestamp > head->offset_time; it_pcl--)
            {
                if (it_pcl == begin_pcl)
                    break;
            }
            int start_ = it_pcl - begin_pcl;

            if (start_ < 0 || end_ >= pcl_out.points.size())
            {
                std::cout << "ERROR DeSkewOriginalCloud start_: " << start_ << ", end_:" << end_ << ", pcl_out->size():" << pcl_out.points.size() << std::endl;
                throw std::invalid_argument("ERROR in DeSkewOriginalCloud");
            }

            tbb::parallel_for(tbb::blocked_range<int>(start_, end_),
                              [&](tbb::blocked_range<int> r)
                              {
                                  for (int i = r.begin(); i < r.end(); i++)
                                  {
                                      double dt_ = pcl_out.points[i].timestamp - head->offset_time;

                                      M3D R_i(R_imu * Sophus::SO3::exp(angvel_avr * dt_).matrix());
                                      V3D P_i(pcl_out.points[i].x, pcl_out.points[i].y, pcl_out.points[i].z);
                                      V3D T_ei(pos_imu + vel_imu * dt_ + 0.5 * acc_imu * dt_ * dt_ - imu_state.pos);

                                      V3D P_compensate = R_I2L * (end_R_T * (R_i * (R_L2I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);

                                      pcl_out.points[i].x = P_compensate(0);
                                      pcl_out.points[i].y = P_compensate(1);
                                      pcl_out.points[i].z = P_compensate(2);
                                  }
                              });
        }

        saved_scan++;
        tbb::parallel_for(tbb::blocked_range<int>(0, pcl_out.size()),
                          [&](const tbb::blocked_range<int> &r)
                          {
                              for (int i = r.begin(); i < r.end(); ++i)
                              {
                                  // pcl_out.points[i].timestamp += first_point_time; //put the time back
                                  pcl_out.points[i].timestamp += (saved_scan * .1); // 10Hz
                              }
                          });
    }
    else
    {
        std::cout << "Unsupported point type" << std::endl;
    }

    if (!save_clouds_local)
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, pcl_out.size()),
                          [&](const tbb::blocked_range<int> &r)
                          {
                              for (int i = r.begin(); i < r.end(); ++i)
                              {
                                  V3D global_position = imu_state.rot.matrix() * (R_L2I * V3D(pcl_out.points[i].x, pcl_out.points[i].y, pcl_out.points[i].z) + imu_state.offset_T_L_I) + imu_state.pos;
                                  pcl_out.points[i].x = global_position.x();
                                  pcl_out.points[i].y = global_position.y();
                                  pcl_out.points[i].z = global_position.z();
                              }
                          });
    }

    return pcl_out;
}

#endif
