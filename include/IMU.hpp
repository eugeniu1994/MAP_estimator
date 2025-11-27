#ifndef USE_IMU_H1
#define USE_IMU_H1

#include <cmath>
#include <math.h>
#include <deque>
#include <thread>
#include <csignal>
#include <memory> // For std::shared_p

#include <utils.h>
#include <Estimator.hpp>

#define MIN_INIT_COUNT (10)

struct AccelNoiseEstimator
{
    const double min_std_ = 0.001;

    int n = 0;
    V3D mean = V3D::Zero();
    V3D M2 = V3D::Zero(); // sum of squared diffs

    void update(const V3D &sample)
    {
        ++n;
        V3D delta = sample - mean;
        mean += delta / n;
        V3D delta2 = sample - mean;
        M2 += delta.cwiseProduct(delta2);
    }

    V3D variance() const
    {
        return (n > 1) ? (M2 / (n - 1)).eval() : V3D::Constant(min_std_ * min_std_);
    }

    V3D stddev() const
    {
        return variance().cwiseSqrt().cwiseMax(V3D::Constant(min_std_));
    }

    M3D covariance() const
    {
        V3D stdv = stddev();
        return stdv.array().square().matrix().asDiagonal();
    }
};

struct ForwardResult
{
    state x_pred;   // x_{k|k-1}
    state x_update; // x_{k|k}
    cov P_pred;     // P_{k|k-1}
    cov P_update;   // P_{k|k}
    cov F;          // State transition Jacobian

    state x_update2;
    cov P_update2;  
};


class IMU_Class
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    IMU_Class();
    ~IMU_Class();

    AccelNoiseEstimator accelNoiseEstimator;

    void set_param(const V3D &tran, const M3D &rot, const V3D &gyr, const V3D &acc, const V3D &gyr_bias, const V3D &acc_bias);
    virtual void Process(const MeasureGroup &meas, Estimator &kf_state, PointCloudXYZI::Ptr &pcl_un_);

    bool imu_need_init_ = true, init_from_GT = false;
#ifdef SAVE_DATA
    template <typename PointT>
    pcl::PointCloud<PointT> DeSkewOriginalCloud(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg, const state &imu_state, bool save_clouds_local);
#endif

    void IMU_init_from_GT(const MeasureGroup &meas, Estimator &kf_state, const Sophus::SE3 &gt);
    void Propagate2D(std::vector<pcl::PointCloud<VUX_PointType>::Ptr> &vux_scans,
                     const std::vector<double> &vux_scans_time, const double &pcl_beg_time, const double &pcl_end_time, const double &tod,
                     const Sophus::SE3 &prev_mls, const double &prev_mls_time);

    void ConstVelUndistort(const MeasureGroup &meas, Estimator &kf_state, PointCloudXYZI::Ptr &pcl_out, const Sophus::SE3 &prev_, const Sophus::SE3 &curr_);
    
    bool backwardPass(const MeasureGroup &meas, Estimator &kf_state, PointCloudXYZI &pcl_out);
    
    void plot_values(const std::vector<ForwardResult> &forward_results_);

protected:
    state imu_state;
    Eigen::Matrix<double, noise_size, noise_size> Q;
    V3D cov_acc, cov_gyr;
    V3D cov_acc_scale, cov_gyr_scale;
    V3D cov_bias_gyr, cov_bias_acc;

    M3D Rbw;
    bool b_first_frame_ = true;
    sensor_msgs::ImuConstPtr last_imu_;
    std::vector<Pose6D> IMU_Buffer;

    std::vector<ForwardResult> forward_results_;

    // extrinsics with LiDAR
    M3D Lidar_R_wrt_IMU;
    V3D Lidar_T_wrt_IMU;

    V3D mean_acc, mean_gyr;
    V3D angvel_last, acc_s_last;

    double start_timestamp_, last_lidar_end_time_;
    int init_iter_num = 1;

    void reset();
    virtual void IMU_init(const MeasureGroup &meas, Estimator &kf_state, int &N);
    virtual void Propagate(const MeasureGroup &meas, Estimator &kf_state, PointCloudXYZI &pcl_out);
};

#endif
