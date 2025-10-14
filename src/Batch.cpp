#include "Batch.hpp"

#include <visualization_msgs/Marker.h>

using namespace gtsam;

gtsam::Pose3 sophusToGtsam(const Sophus::SE3 &pose)
{
    return gtsam::Pose3(pose.matrix());
}

Sophus::SE3 GtsamToSophus(const gtsam::Pose3 &pose)
{
    Eigen::Matrix4d T = pose.matrix(); // Get 4x4 matrix
    return Sophus::SE3(T.topLeftCorner<3, 3>(), T.topRightCorner<3, 1>());
}

namespace custom_factor
{
    // Custom factor for point-to-point constraints
    class PointToPointFactor : public NoiseModelFactor1<Pose3>
    {
    private:
        Point3 measured_point_; // Point in sensor frame
        Point3 target_point_;   // Corresponding point in world frame

        double huber_delta_ = .2;
        // Huber weight calculation
        double huberWeight(double error_norm) const
        {
            return (error_norm <= huber_delta_) ? 1.0 : huber_delta_ / error_norm;
        }

    public:
        PointToPointFactor(Key pose_key,
                           const Point3 &measured_point,
                           const Point3 &target_point,
                           const SharedNoiseModel &model) : NoiseModelFactor1<Pose3>(model, pose_key),
                                                            measured_point_(measured_point),
                                                            target_point_(target_point) {}

        // Vector evaluateError(const Pose3 &pose, boost::optional<Matrix &> H = boost::none) const override
        Vector evaluateError(const gtsam::Pose3 &pose, OptionalMatrixType H) const override
        {
            // Transform measured point to world frame
            Matrix36 H_point;
            Point3 world_point = pose.transformFrom(measured_point_, H_point); // computes ∂(world_point)/∂pose

            // Calculate error vector
            Vector3 error = world_point - target_point_;

            auto robust_weight = 1.0; // huberWeight(error.norm());

            if (H)
            {
                // Jacobian: ∂error/∂pose = ∂p_world/∂pose
                *H = H_point * robust_weight;
            }

            return robust_weight * error;

            // if (H)
            // {
            //     // // Compute Jacobian if requested
            //     // Matrix36 H_point_wrt_pose;
            //     // pose.transformFrom(measured_point_, H_point_wrt_pose);
            //     // (*H) = H_point_wrt_pose * weight_;
            // }

            /*
            // Transform landmark from world frame to LiDAR frame
            gtsam::Point3 predicted = pose.transformTo(target_point_, H);

            // Compute residual: measured - predicted
            return measured_point_ - predicted;
            */

            // return error;
        }

        virtual gtsam::NonlinearFactor::shared_ptr clone() const override
        {
            return std::make_shared<PointToPointFactor>(
                this->key(),
                measured_point_,
                target_point_,
                this->noiseModel());
        }
    };

    // Custom factor for point-to-plane constraints
    class PointToPlaneFactor : public NoiseModelFactor1<Pose3>
    {
    private:
        Point3 measured_point_; // Point in sensor frame
        Point3 plane_normal_;   // Plane normal (normalized)
        Point3 target_point_;   // A point on the plane in world frame
        double d_;
        bool use_alternative_method_; // Flag to choose between calculation methods

        double huber_delta_ = .5; // .1;  maybe take this as parameter

        double huberWeight(double error_norm) const
        {
            return (error_norm <= huber_delta_) ? 1.0 : huber_delta_ / error_norm;
        }

    public:
        PointToPlaneFactor(Key pose_key,
                           const Point3 &measured_point,
                           const Point3 &plane_norm,
                           const Point3 &target_point,
                           double d,
                           bool use_alternative_method,
                           const SharedNoiseModel &model) : NoiseModelFactor1<Pose3>(model, pose_key),
                                                            measured_point_(measured_point),
                                                            plane_normal_(plane_norm),
                                                            target_point_(target_point),
                                                            d_(d),
                                                            use_alternative_method_(use_alternative_method) {}

        Vector evaluateError(const Pose3 &pose, OptionalMatrixType H) const override
        {
            Matrix36 H_point_wrt_pose;
            Point3 p_transformed = pose.transformFrom(measured_point_, H ? &H_point_wrt_pose : 0); // Transform measured point to world frame

            double error = 0.0;
            if (use_alternative_method_)
            {
                error = (p_transformed - target_point_).dot(plane_normal_);
            }
            else
            {
                error = plane_normal_.dot(p_transformed) + d_;
            }

            // double robust_weight = huberWeight(fabs(error)); //  Apply robust weighting
            double robust_weight = 1.0;

            if (H)
            {
                *H = (plane_normal_.transpose() * H_point_wrt_pose) * robust_weight;
            }

            return (Vector(1) << error * robust_weight).finished();
        }

        // Clone method for deep copy
        virtual gtsam::NonlinearFactor::shared_ptr clone() const override
        {
            return std::make_shared<PointToPlaneFactor>(
                this->key(),
                measured_point_,
                plane_normal_,
                target_point_,
                d_,
                use_alternative_method_,
                this->noiseModel());
        }
    };

    class PointToLineFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
    {
        using Base = gtsam::NoiseModelFactor1<gtsam::Pose3>;

        Point3 measured_point_; // Point in sensor frame
        Point3 line_dir_;       // direction (normalized) should be unit
        Point3 target_point_;   // A point on the plane in world frame

        double huber_delta_ = .1;
        // Huber weight calculation
        double huberWeight(double error_norm) const
        {
            return (error_norm <= huber_delta_) ? 1.0 : huber_delta_ / error_norm;
        }

    public:
        PointToLineFactor(gtsam::Key poseKey,
                          const Point3 &measured_point,
                          const Point3 &target_point,
                          const Point3 &line_dir,
                          const gtsam::SharedNoiseModel &model)
            : Base(model, poseKey),
              measured_point_(measured_point),
              target_point_(target_point),
              line_dir_(line_dir.normalized()) {}

        gtsam::Vector evaluateError(const gtsam::Pose3 &pose, OptionalMatrixType H) const override
        {
            Matrix36 H_point_wrt_pose;
            // Transform the measured point to world frame
            Point3 world_point = pose.transformFrom(measured_point_, H ? &H_point_wrt_pose : 0);

            // Vector from line center to point
            Point3 point_to_center = world_point - target_point_;

            gtsam::Vector3 cross_err = point_to_center.cross(line_dir_);
            double error_norm = cross_err.norm();

            // double robust_weight = huberWeight(error_norm);
            double robust_weight = 1.0;

            if (H)
            {
                // Compute Jacobian for scalar error
                gtsam::Matrix13 de_dworld_point;
                if (error_norm < 1e-8)
                {
                    de_dworld_point.setZero();
                }
                else
                {
                    // ∂(‖a×b‖)/∂a = (a×b)^T/‖a×b‖ * [b]×
                    de_dworld_point = cross_err.transpose() / error_norm * gtsam::skewSymmetric(line_dir_);
                }

                // Chain rule: ∂e/∂pose = ∂e/∂world_point * ∂world_point/∂pose
                *H = de_dworld_point * H_point_wrt_pose * robust_weight;
            }

            return (Vector(1) << error_norm * robust_weight).finished();
        }

        virtual gtsam::NonlinearFactor::shared_ptr clone() const override
        {
            return std::make_shared<PointToLineFactor>(
                this->key(),
                measured_point_,
                target_point_,
                line_dir_,
                this->noiseModel());
        }
    };
};

double robust_kernel = .1;

using namespace custom_factor;

template <typename T>
inline double ROS_TIME(T msg)
{
    return msg->header.stamp.toSec();
}

void Batch::resetParams()
{
    std::cout << "===================resetParams==================" << std::endl;
    lastImuT_imu = -1;
    doneFirstOpt = false;
    systemInitialized = false;
}

void Batch::resetOptimization()
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

bool Batch::failureDetection(const gtsam::Vector3 &velCur, const gtsam::imuBias::ConstantBias &biasCur)
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

void Batch::IMU_init(const MeasureGroup &meas, Estimator &kf_state, int &N)
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

    Q.block<3, 3>(G_VAR_ID, G_VAR_ID).diagonal() = cov_gyr;
    Q.block<3, 3>(A_VAR_ID, A_VAR_ID).diagonal() = cov_acc;
    Q.block<3, 3>(BG_VAR_ID, BG_VAR_ID).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(BA_VAR_ID, BA_VAR_ID).diagonal() = cov_bias_acc;

    std::cout << "cov_gyr:" << cov_gyr.transpose() << std::endl;
    std::cout << "cov_acc:" << cov_acc.transpose() << std::endl;
    std::cout << "cov_bias_gyr:" << cov_bias_gyr.transpose() << std::endl;
    std::cout << "cov_bias_acc:" << cov_bias_acc.transpose() << std::endl;

    std::cout << "cov_gyr_scale:" << cov_gyr_scale.transpose() << std::endl;
    std::cout << "cov_acc_scale:" << cov_acc_scale.transpose() << std::endl;

    float imuAccNoise = 0.1; // 0.01;
    float imuGyrNoise = 0.1; // 0.001;
    float imuAccBiasN = 0.0002;
    float imuGyrBiasN = 0.00003;

    p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2); // acc white noise in continuous
    p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);     // gyro white noise in continuous
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
    std::cout << "Accelerometer Covariance: \n"
              << p->accelerometerCovariance << std::endl;
    std::cout << "Gyroscope Covariance: \n"
              << p->gyroscopeCovariance << std::endl;

    p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);                      // error committed in integrating position from velocities
    gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished()); // assume zero initial bias

    priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m

    priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);   // m/s
    priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good

    // correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());   // m, m, m, rad,rad,rad,
    correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01).finished());  // m, m, m, rad,rad,rad,
    correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.05, 0.05, 0.05).finished()); // m, m, m, rad,rad,rad,
    correctionNoise3 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished());                   // m, m, m, rad,rad,rad,

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

void Batch::Process(MeasureGroup &meas, Estimator &kf_state, PointCloudXYZI::Ptr &pcl_un_)
{
    // transform points to IMU frame
    //TransformPoints(Lidar_R_wrt_IMU, Lidar_T_wrt_IMU, meas.lidar);
    if (meas.imu.empty())
    {
        std::cout << "Batch::Process IMU list is empty" << std::endl;
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

void Batch::Propagate(const MeasureGroup &meas, Estimator &kf_state, PointCloudXYZI &pcl_out)
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

    auto end_R_T = imu_state.rot.matrix().transpose();

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
        // std::cout << "start_:" << start_ << ", end_:" << end_ << std::endl;
        if (true)
        {
            for (; it_pcl->time > head->offset_time; it_pcl--)
            {
                dt = it_pcl->time - head->offset_time;
                // P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei)

                M3D R_i(R_imu * Sophus::SO3::exp(angvel_avr * dt).matrix());
                V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
                V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
                //V3D P_compensate = end_R_T * (R_i * P_i + T_ei);
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
            tbb::parallel_for(tbb::blocked_range<int>(start_, end_),
                              [&](tbb::blocked_range<int> r)
                              {
                                  for (int i = r.begin(); i < r.end(); i++)
                                  {
                                      double dt_ = pcl_out.points[i].time - head->offset_time;

                                      M3D R_i(R_imu * Sophus::SO3::exp(angvel_avr * dt_).matrix());
                                      V3D P_i(pcl_out.points[i].x, pcl_out.points[i].y, pcl_out.points[i].z);
                                      V3D T_ei(pos_imu + vel_imu * dt_ + 0.5 * acc_imu * dt_ * dt_ - imu_state.pos);
                                      //V3D P_compensate = end_R_T * (R_i * P_i + T_ei);
                                      V3D P_compensate = R_I2L * (end_R_T * (R_i * (R_L2I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);


                                      pcl_out.points[i].x = P_compensate(0);
                                      pcl_out.points[i].y = P_compensate(1);
                                      pcl_out.points[i].z = P_compensate(2);
                                  }
                              });
        }
    }
}

void Batch::update_se3(state &_state, const double &lidar_beg_time, const double &lidar_end_time, gtsam::Matrix6 &out_cov_pose)
{
    std::cout << "update_se3" << std::endl;
    if (imuQueOpt.empty())
    {
        std::cout << "No imuQueOpt messages" << std::endl;
        return; // no imu msgs
    }

    Sophus::SE3 curr_position = Sophus::SE3(_state.rot.matrix(), _state.pos);
    Pose3 measuredPose = sophusToGtsam(curr_position);

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

    if (key == max_key) // reset graph for speed
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

    // save the latest estimate on _state
    _state.pos = prevState_.pose().translation();
    _state.rot = Sophus::SO3(prevState_.pose().rotation().matrix());
    _state.vel = prevState_.v();
    _state.bg = prevBias_.gyroscope();
    _state.ba = prevBias_.accelerometer();

    // Calculate the marginal covariances for all variables
    gtsam::Marginals marginals(optimizer.getFactorsUnsafe(), result);

    // full 6x6 covariance of the pose (position + rotation)
    out_cov_pose = marginals.marginalCovariance(X(key)); //

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

struct landmark
{
    int map_point_index;   // index of the point from the reference map
    int cloud_point_index; // index pf the points from the cloud
    V3D norm;              // the normal of the plane in global frame (normalized)
    double d;              // d parameter of the plane
    double var;            // plane measurement variance
};

// =============================
// Global static buffers
// =============================
static std::vector<landmark> global_landmarks(100000);
static std::vector<bool> global_valid(100000, false);

void establishCorrespondences(const PointCloudXYZI::Ptr &pcl_un_, const Sophus::SE3 &T,
                              const pcl::PointCloud<PointType>::Ptr &_map, const pcl::KdTreeFLANN<PointType>::Ptr &_tree)
{

    constexpr double MAX_SQ_DIST = 1.0;

    size_t N = pcl_un_->size();
    // if (global_landmarks.size() < N)
    // {
    //     global_landmarks.resize(N);
    //     global_valid.resize(N);
    // }

#ifdef MP_EN
// omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < N; i++)
    {
        global_valid[i] = false;

        V3D p_src(pcl_un_->points[i].x, pcl_un_->points[i].y, pcl_un_->points[i].z);
        V3D p_transformed = T * p_src; // local to map frame

        // Nearest neighbor search
        PointType search_point;
        search_point.x = p_transformed.x();
        search_point.y = p_transformed.y();
        search_point.z = p_transformed.z();

        std::vector<int> pointSearchInd(NUM_MATCH_POINTS);
        std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        if (_tree->nearestKSearch(search_point, NUM_MATCH_POINTS, pointSearchInd, pointSearchSqDis) >= NUM_MATCH_POINTS)
        {
            if (pointSearchSqDis[NUM_MATCH_POINTS - 1] <= MAX_SQ_DIST) // MAX_SQ_DIST = 1.0
            {
                PointVector points_near;
                std::vector<double> point_weights;
                points_near.reserve(NUM_MATCH_POINTS);
                point_weights.reserve(NUM_MATCH_POINTS);
                for (int j = 0; j < pointSearchInd.size(); j++)
                {
                    points_near.push_back(_map->points[pointSearchInd[j]]);
                    point_weights.push_back(1.);
                }

                Eigen::Matrix<float, 4, 1> pabcd;
                double plane_var = 0;
                if (ekf::esti_plane_pca(pabcd, points_near, .03, point_weights, plane_var, true))
                {
                    landmark l;
                    l.map_point_index = pointSearchInd[0];
                    l.cloud_point_index = i;
                    l.norm = V3D(pabcd(0), pabcd(1), pabcd(2));
                    l.d = pabcd(3);
                    l.var = plane_var;

                    global_valid[i] = true;
                    global_landmarks[i] = l;
                }
            }
        }
    }
}

pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZINormal>());

void debug_CloudWithNormals(const ros::Publisher &normals_pub)
{
    std::cout << "debug_CloudWithNormals with " << cloud_with_normals->size() << " points" << std::endl;

    // --- 1. Publish Normals as Markers ---
    visualization_msgs::Marker normals_marker;
    normals_marker.header.frame_id = "world";
    normals_marker.type = visualization_msgs::Marker::LINE_LIST;
    normals_marker.action = visualization_msgs::Marker::ADD;
    normals_marker.scale.x = 0.05; // Line width
    normals_marker.color.a = 1.0;  // Full opacity
    double normal_length = 3.;     // 5.;     // Length of normal lines

    for (const auto &point : cloud_with_normals->points)
    {
        geometry_msgs::Point p1, p2;

        // Start of normal (point)
        p1.x = point.x;
        p1.y = point.y;
        p1.z = point.z;
        normals_marker.points.push_back(p1);

        // End of normal (point + normal * length)
        p2.x = point.x + normal_length * point.normal_x;
        p2.y = point.y + normal_length * point.normal_y;
        p2.z = point.z + normal_length * point.normal_z;
        normals_marker.points.push_back(p2);

        std_msgs::ColorRGBA color;
        // if (point.curvature < 2) // seen less than 10 times
        // {
        //     color.r = 1.0; // not seen enough
        //     color.g = 0.0;
        //     color.b = 0.0;
        // }
        // else
        // {
        color.r = 0.0;
        color.g = 1.0; // Green for high curvature
        color.b = 0.0;
        //}
        color.a = 1.0; // Full opacity

        normals_marker.colors.push_back(color); // Color for start point
        normals_marker.colors.push_back(color); // Color for end point
    }

    normals_pub.publish(normals_marker);
}

void Batch::update_all(state &_state, const double &lidar_beg_time, const double &lidar_end_time, const PointCloudXYZI::Ptr &pcl_un_,
                       const pcl::PointCloud<PointType>::Ptr &mls_map, const pcl::KdTreeFLANN<PointType>::Ptr &mls_tree, bool use_mls,
                       const pcl::PointCloud<PointType>::Ptr &als_map, const pcl::KdTreeFLANN<PointType>::Ptr &als_tree, bool use_als,
                       gtsam::Matrix6 &out_cov_pose, const ros::Publisher &normals_pub, bool debug)
{
    // todo
    /*
    ---proceed as with se3 update - as before
    ---a function to do parallel search for neighbours and find the landmars given a map and a tree
    ---add the landmark measurements - either planes with estimated covs, maybe integrate the robust kernel
    ---debug option: use the existing code to visualize the normals of the measurements

    option to keep the last cloud - ICP relative, curr to prev measurement

    option to integrate the position only GPS factors

    option to integrate any SE3 measurement as in between measurements (ICP, GNSS-IMU, etc)
        if GNSS-IMU - with provided cov, else set a manual one

    modify the _state input, it should be reference to the iekf_estimator,
        call the iekf update after the graph optimize

    change the correctionNoise for X(.) - take it from the system provided it
    */

    std::cout << "update_se3" << std::endl;
    if (imuQueOpt.empty())
    {
        std::cout << "No imuQueOpt messages" << std::endl;
        return; // no imu msgs
    }
    Sophus::SE3 curr_position = Sophus::SE3(_state.rot.matrix(), _state.pos);
    Pose3 measuredPose = sophusToGtsam(curr_position);

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

    if (key == max_key) // reset graph for speed
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

    // if (key > 1 && key < max_key)
    {
        size_t N = pcl_un_->size();
        if (use_mls)
        {
            std::cout << "establish MLS measurements..." << std::endl;
            establishCorrespondences(pcl_un_, curr_position, mls_map, mls_tree);
            int added_planes = 0;
            bool deb = (normals_pub.getNumSubscribers() != 0) && debug;
            if (deb)
            {
                cloud_with_normals->clear();
            }

            for (int i = 0; i < N; i++) // for each point
            {
                if (global_valid[i])
                {
                    const auto &lm = global_landmarks[i];

                    Point3 plane_norm(lm.norm.x(), lm.norm.y(), lm.norm.z());
                    Point3 measured_point(pcl_un_->points[i].x, pcl_un_->points[i].y, pcl_un_->points[i].z); // measured_landmar_in_sensor_frame
                    // Point3 target_point(mls_map->points[lm.map_point_index].x, mls_map->points[lm.map_point_index].y, mls_map->points[lm.map_point_index].z);
                    Point3 target_point(0, 0, 0);
                    // if (use_alternative_method_)
                    //     error = (p_transformed - target_point_).dot(plane_normal_);
                    // else
                    //     error = plane_normal_.dot(p_transformed) + d_;

                    bool use_alternative_method = true;
                    // use_alternative_method = false;

                    auto robust_noise = gtsam::noiseModel::Robust::Create(
                        gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel),
                        gtsam::noiseModel::Isotropic::Sigma(1, sqrt(lm.var)));

                    // double sigma = std::sqrt(lm.var);
                    // double cauchy_scale = std::max(1e-3, 3.0 * sigma); // kernel scale proportional to plane noise

                    // auto robust_noise = gtsam::noiseModel::Robust::Create(
                    //     gtsam::noiseModel::mEstimator::Cauchy::Create(cauchy_scale),
                    //     gtsam::noiseModel::Isotropic::Sigma(1, sigma));

                    // planes
                    graphFactors.emplace_shared<PointToPlaneFactor>(X(key), measured_point, plane_norm, target_point, lm.d,
                                                                    use_alternative_method, robust_noise);
                    added_planes++;
                    // p2p
                    // auto point_noise_cauchy = gtsam::noiseModel::Robust::Create(
                    // gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel), // Robust kernel for outliers
                    // gtsam::noiseModel::Isotropic::Sigma(3, sigma_point));
                    // auto point_noise = gtsam::noiseModel::Robust::Create(
                    //     gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel), // Robust kernel for outliers 10cm
                    //     gtsam::noiseModel::Isotropic::Sigma(3, .5));
                    // this_Graph.emplace_shared<PointToPointFactor>(X(pose_key), measured_point, target_point, point_noise);
                    if (deb)
                    {
                        pcl::PointXYZINormal pt;
                        pt.curvature = 1; // <0 plotted as blue

                        pt.x = mls_map->points[lm.map_point_index].x;
                        pt.y = mls_map->points[lm.map_point_index].y;
                        pt.z = mls_map->points[lm.map_point_index].z;

                        pt.normal_x = lm.norm.x();
                        pt.normal_y = lm.norm.y();
                        pt.normal_z = lm.norm.z();

                        pt.intensity = lm.var; // just for test

                        cloud_with_normals->push_back(pt);
                    }
                }
            }

            if (deb)
            {
                debug_CloudWithNormals(normals_pub);
            }
            std::cout << "added_planes:" << added_planes << std::endl;
        }
        if (use_als)
        {
            std::cout << "establish ALS measurements..." << std::endl;
            // establishCorrespondences(pcl_un_, curr_position, als_map, als_tree);
            //  todo copy the code from MLS to ALS
        }
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

    // save the latest estimate on _state - call the update function on this
    _state.pos = prevState_.pose().translation();
    _state.rot = Sophus::SO3(prevState_.pose().rotation().matrix());
    _state.vel = prevState_.v();
    _state.bg = prevBias_.gyroscope();
    _state.ba = prevBias_.accelerometer();

    // Calculate the marginal covariances for all variables
    gtsam::Marginals marginals(optimizer.getFactorsUnsafe(), result);
    // full 6x6 covariance of the pose (position + rotation)
    out_cov_pose = marginals.marginalCovariance(X(key)); //
    // std::cout<<"system_cov:\n"<<out_cov_pose<<std::endl;

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
