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

#include <geometry_msgs/Quaternion.h>

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
        V3D stdv = 3*stddev();
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

static inline void orientationChangeFromGyro(double q0, double q1, double q2,
                                             double q3, double gx, double gy,
                                             double gz, double &qDot1,
                                             double &qDot2, double &qDot3,
                                             double &qDot4)
{
    // Rate of change of quaternion from gyroscope
    // See EQ 12
    qDot1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
    qDot2 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
    qDot3 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
    qDot4 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);
}

static float invSqrt(float x)
{
    float xhalf = 0.5f * x;
    union
    {
        float x;
        int i;
    } u;
    u.x = x;
    u.i = 0x5f3759df - (u.i >> 1);
    /* The next line can be repeated any number of times to increase accuracy */
    u.x = u.x * (1.5f - xhalf * u.x * u.x);
    return u.x;
}

template <typename T>
static inline void normalizeVector(T &vx, T &vy, T &vz)
{
    T recipNorm = invSqrt(vx * vx + vy * vy + vz * vz);
    vx *= recipNorm;
    vy *= recipNorm;
    vz *= recipNorm;
}

template <typename T>
static inline void normalizeQuaternion(T &q0, T &q1, T &q2, T &q3)
{
    T recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    q0 *= recipNorm;
    q1 *= recipNorm;
    q2 *= recipNorm;
    q3 *= recipNorm;
}

static inline void rotateAndScaleVector(float q0, float q1, float q2, float q3,
                                        float _2dx, float _2dy, float _2dz,
                                        float &rx, float &ry, float &rz)
{
    // result is half as long as input
    rx = _2dx * (0.5f - q2 * q2 - q3 * q3) + _2dy * (q0 * q3 + q1 * q2) +
         _2dz * (q1 * q3 - q0 * q2);
    ry = _2dx * (q1 * q2 - q0 * q3) + _2dy * (0.5f - q1 * q1 - q3 * q3) +
         _2dz * (q0 * q1 + q2 * q3);
    rz = _2dx * (q0 * q2 + q1 * q3) + _2dy * (q2 * q3 - q0 * q1) +
         _2dz * (0.5f - q1 * q1 - q2 * q2);
}

static inline void compensateGyroDrift(float q0, float q1, float q2, float q3,
                                       float s0, float s1, float s2, float s3,
                                       float dt, float zeta, float &w_bx,
                                       float &w_by, float &w_bz, float &gx,
                                       float &gy, float &gz)
{
    // w_err = 2 q x s
    float w_err_x =
        2.0f * q0 * s1 - 2.0f * q1 * s0 - 2.0f * q2 * s3 + 2.0f * q3 * s2;
    float w_err_y =
        2.0f * q0 * s2 + 2.0f * q1 * s3 - 2.0f * q2 * s0 - 2.0f * q3 * s1;
    float w_err_z =
        2.0f * q0 * s3 - 2.0f * q1 * s2 + 2.0f * q2 * s1 - 2.0f * q3 * s0;

    w_bx += w_err_x * dt * zeta;
    w_by += w_err_y * dt * zeta;
    w_bz += w_err_z * dt * zeta;

    gx -= w_bx;
    gy -= w_by;
    gz -= w_bz;
}

static inline void orientationChangeFromGyro(float q0, float q1, float q2,
                                             float q3, float gx, float gy,
                                             float gz, float &qDot1,
                                             float &qDot2, float &qDot3,
                                             float &qDot4)
{
    // Rate of change of quaternion from gyroscope
    // See EQ 12
    qDot1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
    qDot2 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
    qDot3 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
    qDot4 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);
}

static inline void addGradientDescentStep(float q0, float q1, float q2,
                                          float q3, float _2dx, float _2dy,
                                          float _2dz, float mx, float my,
                                          float mz, float &s0, float &s1,
                                          float &s2, float &s3)
{
    float f0, f1, f2;

    // Gradient decent algorithm corrective step
    // EQ 15, 21
    rotateAndScaleVector(q0, q1, q2, q3, _2dx, _2dy, _2dz, f0, f1, f2);

    f0 -= mx;
    f1 -= my;
    f2 -= mz;

    // EQ 22, 34
    // Jt * f
    s0 += (_2dy * q3 - _2dz * q2) * f0 + (-_2dx * q3 + _2dz * q1) * f1 +
          (_2dx * q2 - _2dy * q1) * f2;
    s1 += (_2dy * q2 + _2dz * q3) * f0 +
          (_2dx * q2 - 2.0f * _2dy * q1 + _2dz * q0) * f1 +
          (_2dx * q3 - _2dy * q0 - 2.0f * _2dz * q1) * f2;
    s2 += (-2.0f * _2dx * q2 + _2dy * q1 - _2dz * q0) * f0 +
          (_2dx * q1 + _2dz * q3) * f1 +
          (_2dx * q0 + _2dy * q3 - 2.0f * _2dz * q2) * f2;
    s3 += (-2.0f * _2dx * q3 + _2dy * q0 + _2dz * q1) * f0 +
          (-_2dx * q0 - 2.0f * _2dy * q3 + _2dz * q2) * f1 +
          (_2dx * q1 + _2dy * q2) * f2;
}
class IMU_Class
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Madgwick's IMU and AHRS------------------------------------------
    float gain_ = 0.05; // 0.1; // algorithm gain
    // **** state variables
    float q0 = 1.0, q1 = 0.0, q2 = 0.0, q3 = 0.0; // quaternion
    bool initialized_ = false;

    void getOrientation(float &q0, float &q1, float &q2, float &q3)
    {
        q0 = this->q0;
        q1 = this->q1;
        q2 = this->q2;
        q3 = this->q3;

        // perform precise normalization of the output, using 1/sqrt()
        // instead of the fast invSqrt() approximation. Without this,
        // TF2 complains that the quaternion is not normalized.
        double recipNorm = 1 / sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
        q0 *= recipNorm;
        q1 *= recipNorm;
        q2 *= recipNorm;
        q3 *= recipNorm;
    }

    void setOrientation(float q0, float q1, float q2, float q3)
    {
        this->q0 = q0;
        this->q1 = q1;
        this->q2 = q2;
        this->q3 = q3;
    }

    void madgwickAHRSupdateIMU(float gx, float gy, float gz, float ax, float ay, float az, float dt)
    {
        float s0, s1, s2, s3;
        float qDot1, qDot2, qDot3, qDot4;

        // Rate of change of quaternion from gyroscope
        orientationChangeFromGyro(q0, q1, q2, q3, gx, gy, gz, qDot1, qDot2, qDot3, qDot4);

        // Compute feedback only if accelerometer measurement valid (avoids NaN in
        // accelerometer normalisation)
        if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f)))
        {
            // Normalise accelerometer measurement
            normalizeVector(ax, ay, az);

            // Gradient decent algorithm corrective step
            s0 = 0.0;
            s1 = 0.0;
            s2 = 0.0;
            s3 = 0.0;

            // Gravity: [0, 0, 1]
            addGradientDescentStep(q0, q1, q2, q3, 0.0, 0.0, 2.0, ax, ay, az, s0, s1, s2, s3);

            // normalizeQuaternion(s0, s1, s2, s3);

            normalizeQuaternion(s0, s1, s2, s3);

            // Apply feedback step
            qDot1 -= gain_ * s0;
            qDot2 -= gain_ * s1;
            qDot3 -= gain_ * s2;
            qDot4 -= gain_ * s3;
        }

        // Integrate rate of change of quaternion to yield quaternion
        q0 += qDot1 * dt;
        q1 += qDot2 * dt;
        q2 += qDot3 * dt;
        q3 += qDot4 * dt;

        // Normalise quaternion
        normalizeQuaternion(q0, q1, q2, q3);
    }

    void getGravity(float &rx, float &ry, float &rz, float gravity)
    {
        // Gravity: [0, 0, 1]
        rotateAndScaleVector(q0, q1, q2, q3, 0.0, 0.0, 2.0 * gravity, rx, ry, rz);
    }

    void madgwickAHRSupdateIMU2(float gx, float gy, float gz,
                                float ax, float ay, float az,
                                float dt)
    {
        float s0, s1, s2, s3;
        float qDot1, qDot2, qDot3, qDot4;

        // Gyro-based quaternion derivative
        orientationChangeFromGyro(q0, q1, q2, q3,
                                  gx, gy, gz,
                                  qDot1, qDot2, qDot3, qDot4);

        bool use_gradient = false;
        float best_gain = 0.0f;
        float best_error = std::numeric_limits<float>::infinity();

        // -----------------------------
        // Gyro-only reference
        // -----------------------------
        float qa0 = q0 + qDot1 * dt;
        float qa1 = q1 + qDot2 * dt;
        float qa2 = q2 + qDot3 * dt;
        float qa3 = q3 + qDot4 * dt;
        normalizeQuaternion(qa0, qa1, qa2, qa3);

        Eigen::Vector3f g_ref(0, 0, 1);
        Eigen::Vector3f g_gyro;
        rotateAndScaleVector(qa0, qa1, qa2, qa3,
                             0.0f, 0.0f, 1.0f,
                             g_gyro[0], g_gyro[1], g_gyro[2]);
        g_gyro.normalize();

        float err_gyro = 1.0f - g_gyro.dot(g_ref);
        best_error = err_gyro;

        // -----------------------------
        // Gradient computation
        // -----------------------------
        if (!(ax == 0.0f && ay == 0.0f && az == 0.0f))
        {
            normalizeVector(ax, ay, az);

            s0 = s1 = s2 = s3 = 0.0f;
            addGradientDescentStep(q0, q1, q2, q3,
                                   0.0f, 0.0f, 2.0f,
                                   ax, ay, az,
                                   s0, s1, s2, s3);

            // Gradient norm check
            float s_norm = std::sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
            if (s_norm > 1e-6f)
            {
                normalizeQuaternion(s0, s1, s2, s3);

                // -----------------------------
                // Line search on gain
                // -----------------------------
                const float gain_min = 0.001f;
                const float gain_max = 0.5f;
                const int N = 10; // number of samples

                for (int i = 0; i < N; ++i)
                {
                    float gain = gain_min + (gain_max - gain_min) * float(i) / float(N - 1);

                    float qb0 = q0 + (qDot1 - gain * s0) * dt;
                    float qb1 = q1 + (qDot2 - gain * s1) * dt;
                    float qb2 = q2 + (qDot3 - gain * s2) * dt;
                    float qb3 = q3 + (qDot4 - gain * s3) * dt;
                    normalizeQuaternion(qb0, qb1, qb2, qb3);

                    Eigen::Vector3f g_test;
                    rotateAndScaleVector(qb0, qb1, qb2, qb3,
                                         0.0f, 0.0f, 1.0f,
                                         g_test[0], g_test[1], g_test[2]);
                    g_test.normalize();

                    float err = 1.0f - g_test.dot(g_ref);
                    
                    if (err < best_error)
                    {
                        best_error = err;
                        best_gain = gain;
                        use_gradient = true;
                    }
                }
                std::cout<<"best_error:"<<best_error<<", best_gain:"<<best_gain<<std::endl;
            }
        }

        // -----------------------------
        // Apply best correction
        // -----------------------------
        if (use_gradient)
        {
            qDot1 -= best_gain * s0;
            qDot2 -= best_gain * s1;
            qDot3 -= best_gain * s2;
            qDot4 -= best_gain * s3;
        }

        // Integrate quaternion
        q0 += qDot1 * dt;
        q1 += qDot2 * dt;
        q2 += qDot3 * dt;
        q3 += qDot4 * dt;

        normalizeQuaternion(q0, q1, q2, q3);
    }

    IMU_Class();
    ~IMU_Class();

    AccelNoiseEstimator accelNoiseEstimator;

    bool done_update_ = false;

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
