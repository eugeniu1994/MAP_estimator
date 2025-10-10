#include "PoseGraph.hpp"

using namespace custom_factor;

template <typename T>
inline double ROS_TIME(T msg)
{
    return msg->header.stamp.toSec();
}

void Graph::resetParams()
{
    std::cout << "===================resetParams==================" << std::endl;
    lastImuT_imu = -1;
    doneFirstOpt = false;
    systemInitialized = false;
}

void Graph::resetOptimization()
{
    gtsam::ISAM2Params optParameters;
    optParameters.relinearizeThreshold = 0.1;
    optParameters.relinearizeSkip = 1;
    optimizer = gtsam::ISAM2(optParameters);

    gtsam::NonlinearFactorGraph newGraphFactors;
    graphFactors = newGraphFactors;

    gtsam::Values NewGraphValues;
    graphValues = NewGraphValues;
}

bool Graph::failureDetection(const gtsam::Vector3 &velCur, const gtsam::imuBias::ConstantBias &biasCur)
{
    Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
    if (vel.norm() > 30)
    {
        ROS_WARN("Large velocity, reset IMU-preintegration!");
        return true;
    }

    Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
    Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
    if (ba.norm() > 1.0 || bg.norm() > 1.0)
    {
        ROS_WARN("Large bias, reset IMU-preintegration!");
        return true;
    }

    return false;
}

void Graph::IMU_init(const MeasureGroup &meas, Estimator &kf_state, int &N)
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

    // boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(G_m_s2);
    p = gtsam::PreintegrationParams::MakeSharedU(G_m_s2);

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

        // with given init rotation
        init_state.grav = Eigen::Vector3d(0, 0, -G_m_s2);
        init_state.rot = Sophus::SO3(Rbw.transpose());
    }
    else
    {
        Rbw = Eye3d;
        init_state.grav = -mean_acc / mean_acc.norm() * G_m_s2;
    }

    init_state.bg = mean_gyr;
    // for icp cloud already is in IMU frame
    // init_state.offset_T_L_I = Lidar_T_wrt_IMU;
    // init_state.offset_R_L_I = Sophus::SO3(Lidar_R_wrt_IMU);
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

    Q.block<3, 3>(G_VAR_ID, G_VAR_ID).diagonal() = cov_gyr;
    Q.block<3, 3>(A_VAR_ID, A_VAR_ID).diagonal() = cov_acc;
    Q.block<3, 3>(BG_VAR_ID, BG_VAR_ID).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(BA_VAR_ID, BA_VAR_ID).diagonal() = cov_bias_acc;
    

    std::cout<<"cov_gyr:"<<cov_gyr.transpose()<<std::endl;
    std::cout<<"cov_acc:"<<cov_acc.transpose()<<std::endl;
    std::cout<<"cov_bias_gyr:"<<cov_bias_gyr.transpose()<<std::endl;
    std::cout<<"cov_bias_acc:"<<cov_bias_acc.transpose()<<std::endl;

    std::cout<<"cov_gyr_scale:"<<cov_gyr_scale.transpose()<<std::endl;
    std::cout<<"cov_acc_scale:"<<cov_acc_scale.transpose()<<std::endl;

    float imuAccNoise = 0.1; // 0.01;
    float imuGyrNoise = 0.1; // 0.001;
    float imuAccBiasN = 0.0002;
    float imuGyrBiasN = 0.00003;

    p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2);             // acc white noise in continuous
    p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);                 // gyro white noise in continuous
    noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();

    /*noiseModelBetweenBias = (gtsam::Vector(6) << cov_bias_acc[0], cov_bias_acc[1], cov_bias_acc[2], cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2]).finished();
    // Set the covariance matrices
    p->accelerometerCovariance = (Eigen::Matrix3d() << cov_acc_scale(0), 0, 0,
                                                        0, cov_acc_scale(1), 0,
                                                        0, 0, cov_acc_scale(2)).finished();
    p->gyroscopeCovariance = (Eigen::Matrix3d() << cov_gyr_scale(0), 0, 0,
                                                    0, cov_gyr_scale(1), 0,
                                                    0, 0, cov_gyr_scale(2)).finished();*/

    // Print for verification (optional)
    std::cout << "Accelerometer Covariance: \n" << p->accelerometerCovariance << std::endl;
    std::cout << "Gyroscope Covariance: \n" << p->gyroscopeCovariance << std::endl;

    p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);                      // error committed in integrating position from velocities
    gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished()); // assume zero initial bias

    priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m

    priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);   // m/s
    priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good

    // correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());   // m, m, m, rad,rad,rad,
    correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01).finished());  // m, m, m, rad,rad,rad,
    correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.05, 0.05, 0.05).finished()); // m, m, m, rad,rad,rad,
    correctionNoise3 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // m, m, m, rad,rad,rad,

    defaultGNSSnoise = (gtsam::Vector(3) << 0.5, 0.5, 0.5).finished();


    imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
    imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization
                                                                                    // initial velocity
    prevVel_ = gtsam::Vector3(0, 0, 0);
    prevBias_ = prior_imu_bias;

    gtsam::Rot3 initialRotation(init_state.rot.matrix());
    // Define initial state using the rotation matrix
    gtsam::Point3 initialPosition(0.0, 0.0, 0.0);
    gtsam::Vector3 initialVelocity(0.0, 0.0, 0.0);
    gtsam::NavState initialNavState(initialRotation, initialPosition, initialVelocity);

    prevStateOdom = initialNavState;
    prevState_ = initialNavState;
}

void Graph::Process(MeasureGroup &meas, Estimator &kf_state, PointCloudXYZI::Ptr &pcl_un_)
{
    // transform points to IMU frame
    TransformPoints(Lidar_R_wrt_IMU, Lidar_T_wrt_IMU, meas.lidar);
    if (meas.imu.empty())
    {
        std::cout << "Graph::Process IMU list is empty" << std::endl;
        return;
    };
    ROS_ASSERT(meas.lidar != nullptr);

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
            std::cout << "cov_gyr:" << cov_acc.transpose() << std::endl;
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

void Graph::Propagate(const MeasureGroup &meas, Estimator &kf_state, PointCloudXYZI &pcl_out)
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

    V3D pos_imu, vel_imu, angvel_avr, acc_avr, acc_imu;
    M3D R_imu;

    state imu_state = kf_state.get_x();
    imu_state.pos = prevState_.pose().translation();
    imu_state.rot = Sophus::SO3(prevState_.pose().rotation().matrix());
    imu_state.vel = prevState_.v();
    imu_state.bg = prevBias_.gyroscope();
    imu_state.ba = prevBias_.accelerometer();

    IMU_Buffer.clear();
    IMU_Buffer.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.matrix()));

    sensor_msgs::Imu thisImu;
    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
    {
        auto &&head = *(it_imu);
        auto &&tail = *(it_imu + 1);

        if (tail->header.stamp.toSec() < last_lidar_end_time_)
            continue;

        thisImu = *head;

        angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
            0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
            0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
        acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
            0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
            0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

        acc_avr = acc_avr * G_m_s2 / mean_acc.norm();

        thisImu.linear_acceleration.x = acc_avr.x();
        thisImu.linear_acceleration.y = acc_avr.y();
        thisImu.linear_acceleration.z = acc_avr.z();

        thisImu.angular_velocity.x = angvel_avr.x();
        thisImu.angular_velocity.y = angvel_avr.y();
        thisImu.angular_velocity.z = angvel_avr.z();

        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        if (head->header.stamp.toSec() < last_lidar_end_time_)
        {
            dt = tail->header.stamp.toSec() - last_lidar_end_time_;
        }
        else
        {
            dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        }

        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(acc_avr[0], acc_avr[1], acc_avr[2]),
                                                gtsam::Vector3(angvel_avr[0], angvel_avr[1], angvel_avr[2]), dt);

        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);
        // gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        imu_state.pos = currentState.pose().translation();
        imu_state.rot = Sophus::SO3(currentState.pose().rotation().matrix());
        imu_state.vel = currentState.v();

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
    last_imu_ = meas.imu.back();
    last_lidar_end_time_ = pcl_end_time;
    imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                            gtsam::Vector3(thisImu.angular_velocity.x, thisImu.angular_velocity.y, thisImu.angular_velocity.z), dt);
    gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);
    imu_state.pos = currentState.pose().translation();
    imu_state.rot = Sophus::SO3(currentState.pose().rotation().matrix());
    imu_state.vel = currentState.v();
    kf_state.set_x(imu_state);

    auto it_pcl = pcl_out.points.end() - 1;
    auto begin_pcl = pcl_out.points.begin();

    auto end_R = imu_state.rot.matrix().transpose();

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

        tbb::parallel_for(tbb::blocked_range<int>(start_, end_),
                          [&](tbb::blocked_range<int> r)
                          {
                              for (int i = r.begin(); i < r.end(); i++)
                              {
                                  double dt_ = pcl_out.points[i].time - head->offset_time;

                                  M3D R_i(R_imu * Sophus::SO3::exp(angvel_avr * dt_).matrix());
                                  V3D P_i(pcl_out.points[i].x, pcl_out.points[i].y, pcl_out.points[i].z);
                                  V3D T_ei(pos_imu + vel_imu * dt_ + 0.5 * acc_imu * dt_ * dt_ - imu_state.pos);
                                  V3D P_compensate = end_R * (R_i * P_i + T_ei);

                                  pcl_out.points[i].x = P_compensate(0);
                                  pcl_out.points[i].y = P_compensate(1);
                                  pcl_out.points[i].z = P_compensate(2);
                              }
                          });
    }
}

void Graph::update_se3(state &_state, const double &lidar_end_time)
{
    std::cout << "update_se3" << std::endl;
    if (imuQueOpt.empty())
    {
        std::cout << "No imuQueOpt messages" << std::endl;
        return; // no imu msgs
    }

    gtsam::Pose3 measuredPose(_state.rot.matrix(), _state.pos);

    if (!systemInitialized)
    {
        resetOptimization();

        while (!imuQueOpt.empty())
        {
            if (ROS_TIME(&imuQueOpt.front()) < lidar_end_time)
            {
                lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                imuQueOpt.pop_front();
                imuQueImu.pop_front();
            }
            else
                break;
        }

        // initial pose
        prevPose_ = measuredPose; // this should contain the init rotation that was passed to registration
        gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
        graphFactors.add(priorPose);
        // initial velocity
        prevVel_ = gtsam::Vector3(0, 0, 0);
        gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
        graphFactors.add(priorVel);
        // initial bias
        prevBias_ = gtsam::imuBias::ConstantBias();
        gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
        graphFactors.add(priorBias);
        // add values
        graphValues.insert(X(0), prevPose_);
        graphValues.insert(V(0), prevVel_);
        graphValues.insert(B(0), prevBias_);

        // optimize once
        optimizer.update(graphFactors, graphValues);
        graphFactors.resize(0);
        graphValues.clear();

        imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

        key = 1;
        systemInitialized = true;
        return;
    }

    if (key == 50) // reset graph for speed
    {
        std::cout << "============================= reset graph =============================" << std::endl;
        // get updated noise before reset
        gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
        gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
        gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));

        resetOptimization();
        // add pose
        gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
        graphFactors.add(priorPose);
        // add velocity
        gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
        graphFactors.add(priorVel);
        // add bias
        gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
        graphFactors.add(priorBias);
        // add values
        graphValues.insert(X(0), prevPose_);
        graphValues.insert(V(0), prevVel_);
        graphValues.insert(B(0), prevBias_);
        
        // optimize once
        optimizer.update(graphFactors, graphValues);
        graphFactors.resize(0);
        graphValues.clear();

        key = 1;
    }

    // 1. integrate imu data and optimize
    while (!imuQueOpt.empty())
    {
        // pop and integrate imu data that is between two optimizations
        sensor_msgs::Imu *thisImu = &imuQueOpt.front();
        double imuTime = ROS_TIME(thisImu);
        if (imuTime < lidar_end_time)
        {
            double dt = (lastImuT_opt < 0) ? (1.0 / imuRate) : (imuTime - lastImuT_opt);
            imuIntegratorOpt_->integrateMeasurement(
                gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt);

            lastImuT_opt = imuTime;
            imuQueOpt.pop_front();
        }
        else
            break;
    }

    // add imu factor to graph
    const gtsam::PreintegratedImuMeasurements &preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(*imuIntegratorOpt_);
    gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
    graphFactors.add(imu_factor);
    // add imu bias between factor
    graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                                                                        gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
    // add pose factor
    gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), measuredPose, correctionNoise);
    graphFactors.add(pose_factor);
    

    // insert predicted values
    gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
    graphValues.insert(X(key), propState_.pose());
    graphValues.insert(V(key), propState_.v());
    graphValues.insert(B(key), prevBias_);

    // optimize
    optimizer.update(graphFactors, graphValues);
    optimizer.update();
    graphFactors.resize(0);
    graphValues.clear();
    // Overwrite the beginning of the preintegration for the next step.
    gtsam::Values result = optimizer.calculateEstimate();

    prevPose_ = result.at<gtsam::Pose3>(X(key));
    prevVel_ = result.at<gtsam::Vector3>(V(key));
    prevState_ = gtsam::NavState(prevPose_, prevVel_);
    prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));
    //if (got_gnss)
    //    prevGNSS_ = result.at<gtsam::Pose3>(G(key));
    
    _state.pos = prevState_.pose().translation();
    _state.rot = Sophus::SO3(prevState_.pose().rotation().matrix());
    _state.vel = prevState_.v();
    _state.bg = prevBias_.gyroscope();
    _state.ba = prevBias_.accelerometer();

    // Calculate the marginal covariances for all variables
    // gtsam::Marginals marginals(optimizer.getFactorsUnsafe(), result);
    // system_cov = marginals.marginalCovariance(X(key));
    // std::cout<<"system_cov:\n"<<system_cov<<std::endl;

    // Reset the optimization preintegration object.
    imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
    // check optimization
    if (failureDetection(prevVel_, prevBias_))
    {
        resetParams();
        return;
    }

    // 2. after optiization, re-propagate imu odometry preintegration
    gtsam::NavState tmp_state = prevStateOdom;
    prevStateOdom = prevState_;
    prevBiasOdom = prevBias_;

    // first pop imu message older than current correction data
    double lastImuQT = -1;
    while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < lidar_beg_time)
    {
        lastImuQT = ROS_TIME(&imuQueImu.front());
        imuQueImu.pop_front();
    }

    if (!imuQueImu.empty())
    {
        // reset bias use the newly optimized bias
        imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
        // integrate imu message from the beginning of this optimization
        for (int i = 0; i < (int)imuQueImu.size(); ++i)
        {
            sensor_msgs::Imu *thisImu = &imuQueImu[i];
            double imuTime = ROS_TIME(thisImu);
            double dt = (lastImuQT < 0) ? (1.0 / imuRate) : (imuTime - lastImuQT);

            // this should always be reset before undistor starts
            // imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
            //                                         gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt);
            lastImuQT = imuTime;

            // predict odometry
            // gtsam::NavState currentState = imuIntegratorImu_->predict(tmp_state, prevBiasOdom);
            // gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());

            // Eigen::Matrix3d r = imuPose.rotation().matrix();
            // Eigen::Vector3d t = imuPose.translation();

            // TODO - use the above to perform re-deskew with better estimation
        }
    }

    // std::cout << "update key:" << key << std::endl;
    // std::cout<<"imuQueImu:"<<imuQueImu.size()<<", imuQueOpt:"<<imuQueOpt.size()<<std::endl;

    ++key;
    doneFirstOpt = true;
}

















/*
void Graph::update(state &_state, const double &lidar_beg_time, const double &lidar_end_time,
                   bool got_als, const Sophus::SE3 &als_pose,
                   bool got_gnss, const V3D &gnss_pos)
{
    // std::cout << "UPDATE GRAPH" << std::endl;
    if (imuQueOpt.empty())
    {
        std::cout << "No imuQueOpt messages" << std::endl;
        return; // no imu msgs
    }

    gtsam::Rot3 gtsam_rot(_state.rot.matrix());
    gtsam::Point3 gtsam_pos(_state.pos);
    gtsam::Pose3 curPose(gtsam_rot, gtsam_pos);

    if(got_gnss)
        std::cout<<"got_gnss"<<std::endl;

    if (!systemInitialized)
    {
        resetOptimization();

        while (!imuQueOpt.empty())
        {
            if (ROS_TIME(&imuQueOpt.front()) < lidar_end_time)
            {
                lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                imuQueOpt.pop_front();
                imuQueImu.pop_front();
            }
            else
                break;
        }

        // initial pose
        prevPose_ = curPose; // this should contain the init rotation that was passed to registration
        gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
        graphFactors.add(priorPose);
        // initial velocity
        prevVel_ = gtsam::Vector3(0, 0, 0);
        gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
        graphFactors.add(priorVel);
        // initial bias
        prevBias_ = gtsam::imuBias::ConstantBias();
        gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
        graphFactors.add(priorBias);
        // add values
        graphValues.insert(X(0), prevPose_);
        graphValues.insert(V(0), prevVel_);
        graphValues.insert(B(0), prevBias_);

        if (got_gnss)
        {
            prevGNSS_ = gtsam::Pose3(gtsam_rot, gtsam::Point3(gnss_pos));
            gtsam::PriorFactor gps_factor(G(0), prevGNSS_, correctionNoise2);
            graphValues.insert(G(0), prevGNSS_);
        }

        // optimize once
        optimizer.update(graphFactors, graphValues);
        graphFactors.resize(0);
        graphValues.clear();

        imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

        key = 1;
        systemInitialized = true;
        return;
    }

    if (key == 100) // reset graph for speed
    {
        std::cout << "============================= reset graph =============================" << std::endl;
        // get updated noise before reset
        gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
        gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
        gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));

        resetOptimization();
        // add pose
        gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
        graphFactors.add(priorPose);
        // add velocity
        gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
        graphFactors.add(priorVel);
        // add bias
        gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
        graphFactors.add(priorBias);
        // add values
        graphValues.insert(X(0), prevPose_);
        graphValues.insert(V(0), prevVel_);
        graphValues.insert(B(0), prevBias_);
        if (got_gnss)
        {
            //gtsam::noiseModel::Gaussian::shared_ptr updatedGNSSNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(G(key - 1)));
            // add gnss
            /*gtsam::GPSFactor gps_factor(G(0), prevGNSS_, updatedGNSSNoise);
            graphFactors.add(gps_factor);
            graphValues.insert(G(0), prevGNSS_);* /

            prevGNSS_ = gtsam::Pose3(gtsam_rot, gtsam::Point3(gnss_pos));
            //gtsam::PriorFactor gps_factor(G(0), prevGNSS_, updatedGNSSNoise);
            gtsam::PriorFactor gps_factor(G(0), prevGNSS_, correctionNoise2);

            graphFactors.add(gps_factor);
            graphValues.insert(G(0), prevGNSS_);
        }

        // optimize once
        optimizer.update(graphFactors, graphValues);
        graphFactors.resize(0);
        graphValues.clear();

        key = 1;
    }

    // 1. integrate imu data and optimize
    while (!imuQueOpt.empty())
    {
        // pop and integrate imu data that is between two optimizations
        sensor_msgs::Imu *thisImu = &imuQueOpt.front();
        double imuTime = ROS_TIME(thisImu);
        if (imuTime < lidar_end_time)
        {
            double dt = (lastImuT_opt < 0) ? (1.0 / imuRate) : (imuTime - lastImuT_opt);
            imuIntegratorOpt_->integrateMeasurement(
                gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt);

            lastImuT_opt = imuTime;
            imuQueOpt.pop_front();
        }
        else
            break;
    }

    // add imu factor to graph
    const gtsam::PreintegratedImuMeasurements &preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(*imuIntegratorOpt_);
    gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
    graphFactors.add(imu_factor);
    // add imu bias between factor
    graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                                                                        gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));

    if (got_als)
    {
        //  This is the refined LiDAR pose obtained from ALS
        gtsam::Pose3 ALS_pose_gtsam(gtsam::Rot3(als_pose.so3().matrix()), gtsam::Point3(als_pose.translation()));

        // add pose factor
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), ALS_pose_gtsam, correctionNoise);
        graphFactors.add(pose_factor);

        gtsam::Pose3 relative_pose = prevPose_.between(curPose); // prev to curr mls estimation
        //  Add the BetweenFactor using ALS data as a refined estimate
        //graphFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(X(key - 1), X(key), relative_pose, correctionNoise2));
    }
    else
    {
        // add pose factor
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, correctionNoise);
        graphFactors.add(pose_factor);
    }

    // insert predicted values
    gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
    graphValues.insert(X(key), propState_.pose());
    graphValues.insert(V(key), propState_.v());
    graphValues.insert(B(key), prevBias_);

    if (got_gnss)
    {
        /*gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(defaultGNSSnoise);
        prevGNSS_ = gtsam::Vector3(gnss_pos);
        gtsam::GPSFactor gps_factor(G(key), prevGNSS_, gps_noise);
        graphFactors.add(gps_factor);
        graphValues.insert(G(key), propState_.pose().translation());* /

        auto currGNSS_ = gtsam::Pose3(gtsam_rot, gtsam::Point3(gnss_pos));
        //gtsam::Pose3 relative_pose = prevGNSS_.between(currGNSS_); // prev to curr GNSS estimation
        //graphFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(G(key - 1), G(key), relative_pose, correctionNoise2));

        gtsam::PriorFactor<gtsam::Pose3> pose_factor(G(key), currGNSS_, correctionNoise2);
        graphFactors.add(pose_factor);

        prevGNSS_ = currGNSS_;
        auto relative_pose = prevPose_.between(currGNSS_);
        graphFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(X(key - 1), G(key), relative_pose, correctionNoise2));

        graphValues.insert(G(key), currGNSS_);
    }

    // optimize
    optimizer.update(graphFactors, graphValues);
    optimizer.update();
    graphFactors.resize(0);
    graphValues.clear();
    // Overwrite the beginning of the preintegration for the next step.
    gtsam::Values result = optimizer.calculateEstimate();

    prevPose_ = result.at<gtsam::Pose3>(X(key));
    prevVel_ = result.at<gtsam::Vector3>(V(key));
    prevState_ = gtsam::NavState(prevPose_, prevVel_);
    prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));
    //if (got_gnss)
    //    prevGNSS_ = result.at<gtsam::Pose3>(G(key));
    
    _state.pos = prevState_.pose().translation();
    _state.rot = Sophus::SO3(prevState_.pose().rotation().matrix());
    _state.vel = prevState_.v();
    _state.bg = prevBias_.gyroscope();
    _state.ba = prevBias_.accelerometer();

    // Calculate the marginal covariances for all variables
    // gtsam::Marginals marginals(optimizer.getFactorsUnsafe(), result);
    // system_cov = marginals.marginalCovariance(X(key));
    // std::cout<<"system_cov:\n"<<system_cov<<std::endl;

    // Reset the optimization preintegration object.
    imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
    // check optimization
    if (failureDetection(prevVel_, prevBias_))
    {
        resetParams();
        return;
    }

    // 2. after optiization, re-propagate imu odometry preintegration
    gtsam::NavState tmp_state = prevStateOdom;
    prevStateOdom = prevState_;
    prevBiasOdom = prevBias_;

    // first pop imu message older than current correction data
    double lastImuQT = -1;
    while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < lidar_beg_time)
    {
        lastImuQT = ROS_TIME(&imuQueImu.front());
        imuQueImu.pop_front();
    }

    if (!imuQueImu.empty())
    {
        // reset bias use the newly optimized bias
        imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
        // integrate imu message from the beginning of this optimization
        for (int i = 0; i < (int)imuQueImu.size(); ++i)
        {
            sensor_msgs::Imu *thisImu = &imuQueImu[i];
            double imuTime = ROS_TIME(thisImu);
            double dt = (lastImuQT < 0) ? (1.0 / imuRate) : (imuTime - lastImuQT);

            // this should always be reset before undistor starts
            // imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
            //                                         gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt);
            lastImuQT = imuTime;

            // predict odometry
            // gtsam::NavState currentState = imuIntegratorImu_->predict(tmp_state, prevBiasOdom);
            // gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());

            // Eigen::Matrix3d r = imuPose.rotation().matrix();
            // Eigen::Vector3d t = imuPose.translation();

            // TODO - use the above to perform re-deskew with better estimation
        }
    }

    // std::cout << "update key:" << key << std::endl;
    // std::cout<<"imuQueImu:"<<imuQueImu.size()<<", imuQueOpt:"<<imuQueOpt.size()<<std::endl;

    ++key;
    doneFirstOpt = true;
}



//-----------------------------------------------------------------------------------------------

GlobalGraph::GlobalGraph()
{
    priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished());
    // correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.05, 0.05, 0.05).finished());        // m, m, m, rad,rad,rad,
    correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.005, 0.005, 0.005, 0.005, 0.005, 0.005).finished()); // m, m, m, rad,rad,rad,
    betweenNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.2, 0.2, 0.2, 0.2, 0.2, 0.2).finished());                // m, m, m, rad,rad,rad,
}

void GlobalGraph::resetOptimization()
{
    gtsam::ISAM2Params optParameters;
    optParameters.relinearizeThreshold = 0.1;
    optParameters.relinearizeSkip = 1;
    optimizer = gtsam::ISAM2(optParameters);

    gtsam::NonlinearFactorGraph newGraphFactors;
    graphFactors = newGraphFactors;

    gtsam::Values NewGraphValues;
    graphValues = NewGraphValues;
}

void GlobalGraph::update_(state &mls_state, state &als_state)
{
    gtsam::Pose3 MLS_Pose(gtsam::Rot3(mls_state.rot.matrix()), gtsam::Point3(mls_state.pos));
    gtsam::Pose3 ALS_Pose(gtsam::Rot3(als_state.rot.matrix()), gtsam::Point3(als_state.pos));

    // std::cout<<"mls_state:"<<mls_state.pos.transpose()<<", als_state:"<<als_state.pos.transpose()<<std::endl;
    // gtsam::Pose3 ALS_Pose( gtsam::Rot3(mls_state.rot.matrix()), gtsam::Point3(mls_state.pos));
    // gtsam::Pose3 MLS_Pose( gtsam::Rot3(als_state.rot.matrix()), gtsam::Point3(als_state.pos));

    if (!systemInitialized)
    {
        resetOptimization();

        // initial pose
        prevPose_ = MLS_Pose;
        gtsam::PriorFactor<gtsam::Pose3> priorPose(0, prevPose_, priorPoseNoise);
        graphFactors.add(priorPose);
        // add values
        graphValues.insert(0, prevPose_);

        // optimize once
        optimizer.update(graphFactors, graphValues);
        graphFactors.resize(0);
        graphValues.clear();

        key = 1;
        systemInitialized = true;
        return;
    }

    if (key == 400) // reset graph for speed
    {
        std::cout << "============================= reset GLOBAL graph =============================" << std::endl;
        // get updated noise before reset
        gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
        resetOptimization();
        // add pose
        gtsam::PriorFactor<gtsam::Pose3> priorPose(0, prevPose_, updatedPoseNoise);
        graphFactors.add(priorPose);
        // add values
        graphValues.insert(0, prevPose_);

        // optimize once
        optimizer.update(graphFactors, graphValues);
        graphFactors.resize(0);
        graphValues.clear();

        key = 1;
    }

    // gtsam::Pose3 relative_pose = prevPose_.between(ALS_Pose); //prev to als mls estimation
    //   Add the BetweenFactor using ALS data as a refined estimate
    // graphFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(key - 1, key, relative_pose, correctionNoise));

    // add pose factor
    gtsam::PriorFactor<gtsam::Pose3> pose_factor(key, ALS_Pose, correctionNoise);
    graphFactors.add(pose_factor);

    // insert MLS_Pose as predicted values
    graphValues.insert(key, MLS_Pose);

    // optimize
    optimizer.update(graphFactors, graphValues);
    optimizer.update();
    graphFactors.resize(0);
    graphValues.clear();
    // Overwrite the beginning of the preintegration for the next step.
    gtsam::Values result = optimizer.calculateEstimate();

    prevPose_ = result.at<gtsam::Pose3>(key);

    als_state.pos = prevPose_.translation();
    als_state.rot = Sophus::SO3(prevPose_.rotation().matrix());

    ++key;
}

void GlobalGraph::update(state &mls_state, state &als_state, bool has_gps, V3D gps_pos)
{
    // gtsam::Pose3 MLS_Pose( gtsam::Rot3(mls_state.rot.matrix()), gtsam::Point3(mls_state.pos));
    // gtsam::Pose3 ALS_Pose( gtsam::Rot3(als_state.rot.matrix()), gtsam::Point3(als_state.pos));

    gtsam::Pose3 ALS_Pose(gtsam::Rot3(mls_state.rot.matrix()), gtsam::Point3(mls_state.pos));
    gtsam::Pose3 MLS_Pose(gtsam::Rot3(als_state.rot.matrix()), gtsam::Point3(als_state.pos));

    if (!systemInitialized)
    {
        resetOptimization();

        // initial pose
        prevPose_ = ALS_Pose;
        gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
        graphFactors.add(priorPose);
        // add values
        graphValues.insert(X(0), prevPose_);

        if (has_gps)
        {
            //gtsam::noiseModel::Diagonal::shared_ptr gpsNoise = gtsam::noiseModel::Diagonal::Sigmas(
            //    (gtsam::Vector(3) << 1.0, 1.0, 1.0).finished() // Adjust these sigmas to GPS noise level
            //);
            //gtsam::Point3 gps_point(gps_pos);
            //graphFactors.add(gtsam::GPSFactor(0, gps_point, gpsNoise));
        }

        // optimize once
        optimizer.update(graphFactors, graphValues);
        graphFactors.resize(0);
        graphValues.clear();

        key = 1;
        systemInitialized = true;
        return;
    }

    if (key == 300) // reset graph for speed
    {
        std::cout << "============================= reset GLOBAL graph =============================" << std::endl;
        // get updated noise before reset
        gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
        resetOptimization();
        // add pose
        gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
        graphFactors.add(priorPose);
        // add values
        graphValues.insert(X(0), prevPose_);
        if (has_gps)
        {
            //gtsam::noiseModel::Diagonal::shared_ptr gpsNoise = gtsam::noiseModel::Diagonal::Sigmas(
            //     (gtsam::Vector(3) << 1.0, 1.0, 1.0).finished() // Adjust these sigmas to GPS noise level
            // );
            // gtsam::Point3 gps_point(gps_pos);
            // graphFactors.add(gtsam::GPSFactor(0, gps_point, gpsNoise));
        }
        // optimize once
        optimizer.update(graphFactors, graphValues);
        graphFactors.resize(0);
        graphValues.clear();

        key = 1;
    }

    // Add ALS pose to initial estimate
    graphValues.insert(X(key), ALS_Pose);

    // Between factor using MLS as additional measurement
    gtsam::Pose3 relative_pose = prevPose_.between(MLS_Pose);
    graphFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(X(key - 1), X(key), relative_pose, correctionNoise));

    // gtsam::PriorFactor<gtsam::Pose3> pose_factor(key, MLS_Pose, betweenNoise);
    // graphFactors.add(pose_factor);

    // Integrate GPS measurement if available
    if (has_gps)
    {
        //gtsam::noiseModel::Diagonal::shared_ptr gpsNoise = gtsam::noiseModel::Diagonal::Sigmas(
        //    (gtsam::Vector(3) << 1.0, 1.0, 1.0).finished() // Adjust these sigmas to GPS noise level
        //);
        //gtsam::Point3 gps_point(gps_pos);
        //graphFactors.add(gtsam::GPSFactor(key, gps_point, gpsNoise));
        //maybe I have to add the pose as se3 pose 
    }

    // optimize
    optimizer.update(graphFactors, graphValues);
    optimizer.update();
    graphFactors.resize(0);
    graphValues.clear();
    // Overwrite the beginning of the preintegration for the next step.
    gtsam::Values result = optimizer.calculateEstimate();

    prevPose_ = result.at<gtsam::Pose3>(X(key));

    als_state.pos = prevPose_.translation();
    als_state.rot = Sophus::SO3(prevPose_.rotation().matrix());

    ++key;
}
*/