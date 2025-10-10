#ifndef USE_POSEGRAPH_H1
#define USE_POSEGRAPH_H1

#include "IMU.hpp"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/geometry/Unit3.h> // For plane normals
#include <gtsam/base/numericalDerivative.h>

using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::G; // GNSS position (x,y,z)

Pose3 sophusToGtsam(const Sophus::SE3 &pose)
{
    return Pose3(pose.matrix());
}

Sophus::SE3 GtsamToSophus(const Pose3 &pose)
{
    Eigen::Matrix4d T = pose.matrix(); // Get 4x4 matrix
    return Sophus::SE3(T.topLeftCorner<3, 3>(), T.topRightCorner<3, 1>());
}

// Local graph
class Graph : public IMU_Class
{
public:
    void Process(MeasureGroup &meas, Estimator &kf_state, PointCloudXYZI::Ptr &pcl_un_);
    // void update(state &_state, const double &lidar_beg_time, const double &lidar_end_time,
    //              bool got_als = false, const Sophus::SE3 &als_pose = Sophus::SE3(),
    //              bool got_gnss = false, const V3D &gnss_pos = Zero3d);

    

    void update_se3(state &_state, const double &lidar_end_time); //update the system with se3 pose only

    pass the MLS and/or ALS as ref

    void update_all(state &_state, const double &lidar_end_time, PointCloudXYZI::Ptr &pcl_un_,
        const pcl::PointCloud<PointType>::Ptr &mls_map, const pcl::KdTreeFLANN<PointType>::Ptr &mls_tree, bool use_mls,
        const pcl::PointCloud<PointType>::Ptr &als_map, const pcl::KdTreeFLANN<PointType>::Ptr &als_tree, bool use_als,       
    
    )
    {
        COPY ALL THIS CODE IN A NEW FILE AND WRITE IT CLEAN

        INTEGRATE INTO EXISTING CODE EASY
            OPTION TO UPDATE THE IEKF FROM THIS
            WE CAN JUST PASS THE SE3 to iekf 


        something like this


        get the map cloud and its kdtree
        get the src cloud
        for each point - search the neighbours in the map

        iekf should stay pose should be absolute 

        to integrate the GNSS - add option for 3d global position only 

        option to integrate the SE3 from gnss-imu or visual odome - as relative inbetween factors 

        //TODO

        each measurement will have its own std - variance

        se3 from gnss-imu, or iekf, planes var, etc


        double robust_kernel = .1;
        double landmarks_sigma = 1; // this was used for all the tests so far

        // bool use_artificial_uncertainty = false;
        bool use_artificial_uncertainty = true;

        auto plane_noise_cauchy = gtsam::noiseModel::Robust::Create(
            // gtsam::noiseModel::mEstimator::Cauchy::Create(.2), // Less aggressive than Tukey
            gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel), // 2cm
            gtsam::noiseModel::Isotropic::Sigma(1, landmarks_sigma));

        auto point_noise_cauchy = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel), // Robust kernel for outliers
            gtsam::noiseModel::Isotropic::Sigma(3, sigma_point));
            
        Point3 measured_point(raw_point.x, raw_point.y, raw_point.z); // measured_landmark_in_sensor_frame
        Point3 target_point(land.landmark_point.x(), land.landmark_point.y(), land.landmark_point.z());

                    if (land.is_plane)
                    {
                        Point3 plane_norm(land.norm.x(), land.norm.y(), land.norm.z());
                        // if (use_alternative_method_)
                        //     error = (p_transformed - target_point_).dot(plane_normal_);
                        // else
                        //     error = plane_normal_.dot(p_transformed) + d_;

                        bool use_alternative_method = true;
                        // use_alternative_method = false;

                        auto robust_noise = gtsam::noiseModel::Robust::Create(
                            gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel),
                            gtsam::noiseModel::Isotropic::Sigma(1, 3 * land.sigma));

                        if (use_artificial_uncertainty)
                        {
                            robust_noise = plane_noise_cauchy; // use the artificial one
                        }

                        //planes
                        graphFactors.emplace_shared<PointToPlaneFactor>(X(pose_key), measured_point, plane_norm, target_point, land.d,
                                                                      use_alternative_method, robust_noise);

                        // p2p
                    // auto point_noise_cauchy = gtsam::noiseModel::Robust::Create(
                    // gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel), // Robust kernel for outliers
                    // gtsam::noiseModel::Isotropic::Sigma(3, sigma_point));

                    // auto point_noise = gtsam::noiseModel::Robust::Create(
                    //     gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel), // Robust kernel for outliers 10cm
                    //     gtsam::noiseModel::Isotropic::Sigma(3, .5));

                    // this_Graph.emplace_shared<PointToPointFactor>(X(pose_key), measured_point, target_point, point_noise);
    }


    gtsam::Pose3 prevPose_;
    gtsam::Pose3 prevGNSS_;

private:
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise, correctionNoise2, correctionNoise3;
    gtsam::Vector defaultGNSSnoise;

    gtsam::Vector noiseModelBetweenBias;

    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    std::deque<sensor_msgs::Imu> imuQueOpt;
    std::deque<sensor_msgs::Imu> imuQueImu;

    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false, gnss_initialized = false;
    bool systemInitialized = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    std::shared_ptr<gtsam::PreintegrationParams> p;

    int key = 1;

    float imuRate = 100;

    void resetOptimization();
    bool failureDetection(const gtsam::Vector3 &velCur, const gtsam::imuBias::ConstantBias &biasCur);
    void resetParams();
    void IMU_init(const MeasureGroup &meas, Estimator &kf_state, int &N) override;
    void Propagate(const MeasureGroup &meas, Estimator &kf_state, PointCloudXYZI &pcl_out) override;
};

namespace custom_factor
{
    using namespace gtsam;

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

            // auto robust_weight = GM_robust_kernel(error.norm());
            auto robust_weight = huberWeight(error.norm());

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

            double robust_weight = huberWeight(fabs(error)); //  Apply robust weighting
            // double robust_weight = 1.0;

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

            double robust_weight = huberWeight(error_norm);
            // double robust_weight = 1.0;

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






















//----------------------------------------------------------------
class GlobalGraph
{
private:
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr betweenNoise;

    bool systemInitialized = false;

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    int key = 0;

    void resetOptimization();

public:
    GlobalGraph();
    ~GlobalGraph() {};

    gtsam::Pose3 prevPose_;
    void update_(state &mls_state, state &als_state);
    void update(state &mls_state, state &als_state, bool has_gps = false, V3D gps_pos = Zero3d);

    void addGPSFactor(const V3D &pos);
};



#endif