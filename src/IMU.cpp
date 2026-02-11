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

    bool world_orientation = true;

    if (world_orientation)
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

        // average_gravity = median_gravity; //try to use median instead of mean

        const V3D &z_axis = average_gravity.normalized();
        V3D x_axis = V3D::UnitX() - z_axis * z_axis.transpose() * V3D::UnitX();
        x_axis.normalize();
        V3D y_axis = z_axis.cross(x_axis);
        y_axis.normalize();

        Rbw.block<3, 1>(0, 0) = x_axis;
        Rbw.block<3, 1>(0, 1) = y_axis;
        Rbw.block<3, 1>(0, 2) = z_axis;

        std::cout << "Init rotation:\n"
                  << Rbw.transpose() << std::endl;
        V3D eulerAngles = Rbw.eulerAngles(0, 1, 2); // Roll, Pitch, Yaw
        std::cout << "Init rotation to Euler Angles (degrees): \n"
                  << eulerAngles.transpose() * 180.0 / M_PI << std::endl;

        init_state.grav = V3D(0, 0, -G_m_s2);
        init_state.rot = Sophus::SO3(Rbw.transpose());
    }
    else
    {
        Rbw = Eye3d;
        init_state.grav = -mean_acc / mean_acc.norm() * G_m_s2;
    }

    init_state.bg = mean_gyr;
    init_state.offset_T_L_I = Lidar_T_wrt_IMU;
    init_state.offset_R_L_I = Sophus::SO3(Lidar_R_wrt_IMU);
    kf_state.set_x(init_state);

    cov init_P = kf_state.get_P();
    init_P.block(Re_ID, Re_ID, 3, 3) = Eye3d * 0.00001;
    init_P.block(Te_ID, Te_ID, 3, 3) = Eye3d * 0.00001;
    init_P.block(BG_ID, BG_ID, 3, 3) = Eye3d * 0.0001;
    init_P.block(BA_ID, BA_ID, 3, 3) = Eye3d * 0.001;
    init_P.block(G_ID, G_ID, 3, 3) = Eye3d * 0.00001;

    kf_state.set_P(init_P);
    last_imu_ = meas.imu.back();

    Q.block<3, 3>(G_VAR_ID, G_VAR_ID).diagonal() = cov_gyr;
    Q.block<3, 3>(A_VAR_ID, A_VAR_ID).diagonal() = cov_acc;
    Q.block<3, 3>(BG_VAR_ID, BG_VAR_ID).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(BA_VAR_ID, BA_VAR_ID).diagonal() = cov_bias_acc;
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

        acc_s_last = imu_state.rot * (acc_s_last - imu_state.ba) + imu_state.grav;

        double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
        IMU_Buffer.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.matrix()));
    }

    dt = abs(pcl_end_time - imu_end_time);

    kf_state.predict(dt, Q, in); // Predict the IMU state at the end of the scan
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

        // TODO: rewrite the code to do this in parallel
        for (; it_pcl->time > head->offset_time; it_pcl--)
        {
            dt = it_pcl->time - head->offset_time;

            M3D R_i(R_imu * Sophus::SO3::exp(angvel_avr * dt).matrix());
            V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
            V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
            V3D P_compensate = R_I2L * (end_R_T * (R_i * (R_L2I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);

            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);

            if (it_pcl == begin_pcl)
                break;
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

    if (imu_need_init_)
    {
        std::cout << "IMU_init from the raw acceleration values ..." << std::endl;
        IMU_init(meas, kf_state, init_iter_num);

        imu_need_init_ = true;
        last_imu_ = meas.imu.back();
        if (init_iter_num > MIN_INIT_COUNT)
        {
            imu_need_init_ = false;

            cov_acc = cov_acc_scale;
            cov_gyr = cov_gyr_scale;

            Q.block<3, 3>(G_VAR_ID, G_VAR_ID).diagonal() = cov_gyr;
            Q.block<3, 3>(A_VAR_ID, A_VAR_ID).diagonal() = cov_acc;
            Q.block<3, 3>(BG_VAR_ID, BG_VAR_ID).diagonal() = cov_bias_gyr;
            Q.block<3, 3>(BA_VAR_ID, BA_VAR_ID).diagonal() = cov_bias_acc;
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

