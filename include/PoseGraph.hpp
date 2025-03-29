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

using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::G; // GNSS position (x,y,z)

// Local graph
class Graph : public IMU_Class
{
public:
    void Process(MeasureGroup &meas, Estimator &kf_state, PointCloudXYZI::Ptr &pcl_un_);
    void update(state &_state, const double &lidar_beg_time, const double &lidar_end_time,
                 bool got_als = false, const Sophus::SE3 &als_pose = Sophus::SE3(),
                 bool got_gnss = false, const V3D &gnss_pos = Zero3d);
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