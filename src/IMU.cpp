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

    std::cout<<"set_param:\n"<<std::endl;
    std::cout<<"cov_acc_scale:"<<cov_acc_scale.transpose()<<"\ncov_gyr_scale:"<<cov_gyr_scale.transpose()<<std::endl;

}

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
        V3D y_axis = z_axis.cross(x_axis); //y_axis = skew_x(z_axis)*x_axis;
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

    //to be done - set the acceleration bias
    //V3D ba = mean_acc - Rbw * _gravity;

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

    std::cout<<"IMU_init\n"<<std::endl;
    std::cout<<"cov_gyr:"<<cov_gyr.transpose()<<", cov_acc:"<<cov_acc.transpose()<<std::endl;
    std::cout<<"cov_bias_gyr:"<<cov_bias_gyr.transpose()<<", cov_bias_acc:"<<cov_bias_acc.transpose()<<std::endl;

    Q.block<3, 3>(G_VAR_ID, G_VAR_ID).diagonal() = cov_gyr;
    Q.block<3, 3>(A_VAR_ID, A_VAR_ID).diagonal() = cov_acc;
    Q.block<3, 3>(BG_VAR_ID, BG_VAR_ID).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(BA_VAR_ID, BA_VAR_ID).diagonal() = cov_bias_acc;

    std::cout<<"Q G_VAR imu:\n"<<Q.block<3, 3>(G_VAR_ID, G_VAR_ID)<<std::endl;
    std::cout<<"Q A_VAR_ID imu:\n"<<Q.block<3, 3>(A_VAR_ID, A_VAR_ID)<<std::endl;

}

void IMU_Class::IMU_init_from_GT(const MeasureGroup &meas, Estimator &kf_state, const Sophus::SE3 &gt)
{
    std::cout << "IMU_init_from_GT" << std::endl;

    state init_state = kf_state.get_x();
    Rbw = gt.so3().matrix();
    std::cout << "gt rotation:\n"
              << Rbw << std::endl;
    auto eulerAngles = Rbw.eulerAngles(0, 1, 2); // Roll, Pitch, Yaw
    std::cout << "gt rotation to Euler Angles (degrees): \n"
              << eulerAngles.transpose() * 180.0 / M_PI << std::endl;

    // with given init rotation
    init_state.grav = Eigen::Vector3d(0, 0, -G_m_s2);
    init_state.rot = Sophus::SO3(Rbw); //

    kf_state.set_x(init_state);
    std::cout << "Init state gravity:" << init_state.grav.transpose() << std::endl;
    init_from_GT = true;
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

                double tod_time = vux_scans_time[pcl_idx];// scan->points[0].time;
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

        std::cout<<std::endl;
        //std::cout<<"prev_mls_time:"<<prev_mls_time<<", last_lidar_end_time_:"<<last_lidar_end_time_<<std::endl;
        //std::cout<<"tod_beg_scan:"<<tod_beg_scan<<", tod_end_scan:"<<tod_end_scan<<std::endl;

        //auto dt_ = last_lidar_end_time_ - prev_mls_time;
        auto dt_tod = tod_end_scan - tod_beg_scan;
        //std::cout<<"dt_:"<<dt_<<", dt_tod:"<<dt_tod<<std::endl;

        auto delta_predicted = (prev_mls.inverse() * Sophus::SE3(imu_state.rot, imu_state.pos)).log();
        
        for (int i = 0; i < vux_scans.size(); i++)
        {
            const double &t = vux_scans_time[i];// vux_scans[i]->points[0].time;
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


        //THE NEXT IS INTERPOLATING BETWEEN THE FIRST AND LAST IMU poses
        // M3D R_imu1, R_imu2;
        // V3D pos_imu1, pos_imu2;
        // double time1 = IMU_Buffer[0].offset_time;
        // R_imu1 << MAT_FROM_ARRAY(IMU_Buffer[0].rot);
        // pos_imu1 << VEC_FROM_ARRAY(IMU_Buffer[0].pos);
        // const Sophus::SE3 pose1(R_imu1, pos_imu1);

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

        angvel_last = V3D(tail->angular_velocity.x, tail->angular_velocity.y, tail->angular_velocity.z) - imu_state.bg;
        acc_s_last = V3D(tail->linear_acceleration.x, tail->linear_acceleration.y, tail->linear_acceleration.z) * G_m_s2 / mean_acc.norm();

        if (G_m_s2 != 1)
            acc_s_last = imu_state.rot * (acc_s_last - imu_state.ba) + imu_state.grav;
        else
            acc_s_last = imu_state.rot * (acc_s_last - imu_state.ba);

        double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
        IMU_Buffer.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.matrix()));
    }
    dt = abs(pcl_end_time - imu_end_time);
    kf_state.predict(dt, Q, in);
    imu_state = kf_state.get_x();
    last_imu_ = meas.imu.back();
    last_lidar_end_time_ = pcl_end_time;

    auto it_pcl = pcl_out.points.end() - 1;
    auto begin_pcl = pcl_out.points.begin();

    const auto &end_R_T = imu_state.rot.matrix().transpose();
    const auto &R_L2I = imu_state.offset_R_L_I.matrix();
    const auto &R_I2L = imu_state.offset_R_L_I.matrix().transpose();

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
        std::cout << "IMU_init ..." << std::endl;
        IMU_init(meas, kf_state, init_iter_num);
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
            //std::cout<<"Q G_VAR imu:\n"<<Q.block<3, 3>(G_VAR_ID, G_VAR_ID)<<std::endl;
            //std::cout<<"Q A_VAR_ID imu:\n"<<Q.block<3, 3>(A_VAR_ID, A_VAR_ID)<<std::endl;
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
