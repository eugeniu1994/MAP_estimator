
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
#include <tbb/tbb.h>

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

const int state_size = 24;
const int noise_size = 12;

const int used_state_size = 12;// 6;// 12;

typedef Eigen::Matrix<double, state_size, state_size> cov;     // 24X24 covariance matrix
typedef Eigen::Matrix<double, state_size, 1> vectorized_state; // 24X1 vector

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

class Estimator
{
public:

    Estimator() {};
    ~Estimator() {};

    state get_x();
    cov get_P();

    void set_x(state &input_state);
    void set_P(cov &input_cov);

    void predict(double &dt, Eigen::Matrix<double, noise_size, noise_size> &Q, const input &i_in);

    state propagete_NO_gravity(const double &dt, const input &i_in);
    
    state boxplus(state x, vectorized_state f_);
protected:
    state x_;
    cov P_ = cov::Identity(); // 24X24

    
    vectorized_state boxminus(state x1, state x2);
};

#endif

