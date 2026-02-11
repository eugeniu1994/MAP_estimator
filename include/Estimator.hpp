
#ifndef USE_ESTIMATOR_H1
#define USE_ESTIMATOR_H1

#pragma once

#include <cstdlib>
#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <utils.h>

// pos, rot, extrinsic_R, extrinsic_t, vel, bg, ba, grav
enum StateID
{
    P_ID = 0, 
    R_ID = 3,
    Re_ID = 6,
    Te_ID = 9,
    V_ID = 12,
    BG_ID = 15,
    BA_ID = 18,
    G_ID = 21
};
// gyroscope variance, acceleration variance, bias gyro var, bias acc var
enum NoiseID
{
    G_VAR_ID = 0,
    A_VAR_ID = 3,
    BG_VAR_ID = 6,
    BA_VAR_ID = 9
};

constexpr int state_size = 24;
constexpr int noise_size = 12;
typedef Eigen::Matrix<double, state_size, state_size> cov;     // 24X24 covariance matrix
typedef Eigen::Matrix<double, state_size, 1> vectorized_state; // 24X1 vector
using Jacobian_plane = Eigen::Matrix<double, 1, state_size>;
using Vector4d = Eigen::Matrix<double, 4, 1>;

typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// 24-dimensional state x
struct state
{
    V3D pos = Zero3d;
    Sophus::SO3 rot = Sophus::SO3(Eigen::Matrix3d::Identity());
    Sophus::SO3 offset_R_L_I = Sophus::SO3(Eigen::Matrix3d::Identity());
    V3D offset_T_L_I = Zero3d;
    V3D vel = Zero3d;
    V3D bg = Zero3d;
    V3D ba = Zero3d;
    V3D grav = Eigen::Vector3d(0, 0, -G_m_s2);
};

// Input u
struct input
{
    V3D acc = Zero3d;
    V3D gyro = Zero3d;
};

Eigen::Matrix<double, noise_size, noise_size> process_noise_cov();
Eigen::Matrix<double, state_size, 1> f(state s, input in);
Eigen::Matrix<double, state_size, state_size> df_dx(state s, input in);
Eigen::Matrix<double, state_size, noise_size> df_dw(state s, input in);

using namespace ekf;

struct obj_struct
{
    bool valid;                                               
    bool converge;                                             
};

class Estimator
{
public:

    Estimator() {};
    ~Estimator() {};

    state get_x();
    cov get_P();
    cov get_Fx();

    void set_x(state &input_state);
    void set_P(cov &input_cov);

    void predict(double &dt, Eigen::Matrix<double, noise_size, noise_size> &Q, const input &i_in);
    
    state boxplus(state x, vectorized_state f_);
    vectorized_state boxminus(state x1, state x2);

protected:
    state x_;
    cov P_ = cov::Identity(); // 24X24
    cov Fx = cov::Identity(); // 24X24
};

class MAP_ : public Estimator
{
public:
    

    MAP_()
    {
        localKdTree_map.reset(new pcl::KdTreeFLANN<PointType>());
    };

    ~MAP_() {};

    int update(int maximum_iter, bool extrinsic_est, PointCloudXYZI::Ptr &feats_down_body, const PointCloudXYZI::Ptr &map, 
                const bool use_als, const PointCloudXYZI::Ptr &als_map, const pcl::KdTreeFLANN<PointType>::Ptr &als_tree,       //prior ALS map
                const bool use_se3, const Sophus::SE3 &gnss_se3, const V3D &se3_std_pos_m, const V3D &se3_std_rot_deg,          //absolute SE3 meas,
                bool use_se3_rel, const Sophus::SE3 &se3_rel, const V3D &se3_rel_std_pos_m, const V3D &se3_rel_std_rot_deg,     //relative SE3 meas,
                const Sophus::SE3 &prev_X);
    
    
    int update(int maximum_iter, bool extrinsic_est, PointCloudXYZI::Ptr &feats_down_body, const PointCloudXYZI::Ptr &map);

private:
    int effct_feat_num;
    pcl::KdTreeFLANN<PointType>::Ptr localKdTree_map;
};


#endif

