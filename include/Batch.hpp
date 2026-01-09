#ifndef USE_BATCH_H1
#define USE_BATCH_H1

#include "IMU.hpp"
#include "utils.h"

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
using gtsam::symbol_shorthand::G; // GNSS position (x,y,z)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)


// Local graph Batch
class Batch : public IMU_Class
{
public:
    void Process(MeasureGroup &meas, Estimator &kf_state, PointCloudXYZI::Ptr &pcl_un_);

    void update_se3(state &_state, const double &lidar_beg_time, const double &lidar_end_time, gtsam::Matrix6 &out_cov_pose); // update the system with se3 pose only

    void update_all(state &_state, const double &lidar_beg_time, const double &lidar_end_time, const PointCloudXYZI::Ptr &pcl_un_,
                    const pcl::PointCloud<PointType>::Ptr &mls_map, const pcl::KdTreeFLANN<PointType>::Ptr &mls_tree, bool use_mls,
                    const pcl::PointCloud<PointType>::Ptr &als_map, const pcl::KdTreeFLANN<PointType>::Ptr &als_tree, bool use_als,
                    gtsam::Matrix6 &out_cov_pose, const ros::Publisher &normals_pub, bool debug = false);

    void test(state &_state, const double &lidar_beg_time, const double &lidar_end_time, const PointCloudXYZI::Ptr &pcl_un_,
                    const pcl::PointCloud<PointType>::Ptr &mls_map, const pcl::KdTreeFLANN<PointType>::Ptr &mls_tree, bool use_mls,
                    const pcl::PointCloud<PointType>::Ptr &als_map, const pcl::KdTreeFLANN<PointType>::Ptr &als_tree, bool use_als,
                    gtsam::Matrix6 &out_cov_pose, const ros::Publisher &normals_pub, bool debug = false);
    
    gtsam::Pose3 prevPose_;
    bool doneFirstOpt = false;
    state imu_state;

    private:
        gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
        gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
        gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
        gtsam::noiseModel::Diagonal::shared_ptr correctionNoise, correctionNoise2, correctionNoise3;

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

        
        bool systemInitialized = false;
        double lastImuT_imu = -1;
        double lastImuT_opt = -1;

        gtsam::ISAM2 optimizer;
        gtsam::NonlinearFactorGraph graphFactors;
        gtsam::Values graphValues;

        std::shared_ptr<gtsam::PreintegrationParams> p;

        int key = 1, max_key = 50;

        float imuRate = 100;

        void resetOptimization();
        bool failureDetection(const gtsam::Vector3 &velCur, const gtsam::imuBias::ConstantBias &biasCur);
        void resetParams();
        void IMU_init(const MeasureGroup &meas, Estimator &kf_state, int &N) override;
        void Propagate(const MeasureGroup &meas, Estimator &kf_state, PointCloudXYZI &pcl_out) override;
    };
#endif