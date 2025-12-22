#include "Estimator.hpp"

// Initialization of noise covariance Q
Eigen::Matrix<double, noise_size, noise_size> process_noise_cov()
{
    // gyroscope variance, acceleration variance, bias gyro var, bias acc var
    Eigen::Matrix<double, noise_size, noise_size> Q = Eigen::MatrixXd::Zero(noise_size, noise_size);
    Q.block<3, 3>(G_VAR_ID, G_VAR_ID) = 0.0001 * Eye3d;    // cov gyro
    Q.block<3, 3>(A_VAR_ID, A_VAR_ID) = 0.0001 * Eye3d;    // cov acc
    Q.block<3, 3>(BG_VAR_ID, BG_VAR_ID) = 0.00001 * Eye3d; // cov_bias_gyro
    Q.block<3, 3>(BA_VAR_ID, BA_VAR_ID) = 0.00001 * Eye3d; // cov_bias_acc

    /*
    imu_std = np.array([0.01,     # gyro (rad/s)
                    0.05,     # accelerometer (m/s^2)
                    0.000001, # gyro bias (rad/s^2)
                    0.0001])  # accelerometer bias (m/s^3)

    imu_std[0]**2 is the same as var ,  squared std is var,  the other is cov

    Q = block_diag(imu_std[0]**2*np.eye(3), imu_std[1]**2*np.eye(3),
    imu_std[2]**2*np.eye(3), imu_std[3]**2*np.eye(3))
    */
    // the Q for kitti
    // Q.block<3, 3>(0, 0) = 0.0001 * Eigen::Matrix3d::Identity(); //cov gyro - var gyro
    // Q.block<3, 3>(3, 3) = 0.0025 * Eigen::Matrix3d::Identity(); //cov acc  - var acc
    // Q.block<3, 3>(6, 6) = 0.0000000001 * Eigen::Matrix3d::Identity(); //cov_bias_gyro
    // Q.block<3, 3>(9, 9) = 0.00000001 * Eigen::Matrix3d::Identity(); //cov_bias_acc

    return Q;
}

Eigen::Matrix<double, state_size, 1> f(state s, input in)
{
    Eigen::Matrix<double, state_size, 1> result = Eigen::Matrix<double, state_size, 1>::Zero();
    V3D omega = in.gyro - s.bg; // bias free angular velocity

    // bias free acceleration transform to the world frame
    V3D a_inertial = s.rot.matrix() * (in.acc - s.ba);
    // gravity is the average, negative acceleration,  when added it cancel the earth gravity
    for (int i = 0; i < 3; i++)
    {
        result(i + P_ID) = s.vel[i];                  // prev state vel (constant vel model)
        //result(i + P_ID) = s.vel[i] + a_inertial[i] * dt
        result(i + R_ID) = omega[i];                  // Angular velocity
        result(i + V_ID) = a_inertial[i] + s.grav[i]; // gravity-free acceleration
    }
    return result;
}

Eigen::Matrix<double, state_size, state_size> df_dx(state s, input in)
{
    // Fx matrix is not multiplied by dt, and the unit matrix is not added
    Eigen::Matrix<double, state_size, state_size> cov = Eigen::Matrix<double, state_size, state_size>::Zero();

    // d_position/d_vel
    cov.block<3, 3>(P_ID, V_ID) = Eye3d;
    cov.template block<3, 3>(R_ID, BG_ID) = -Eye3d; // simplified to -I try -s.rot.matrix()
    Eigen::Vector3d acc_ = in.acc - s.ba;           //  acceleration = a_m - bias
    cov.block<3, 3>(V_ID, R_ID) = -s.rot.matrix() * Sophus::SO3::hat(acc_);
    cov.block<3, 3>(V_ID, BA_ID) = -s.rot.matrix();
    cov.template block<3, 3>(V_ID, G_ID) = Eye3d;
    return cov;
}

Eigen::Matrix<double, state_size, noise_size> df_dw(state s, input in)
{
    // Fw matrix is not multiplied by dt
    Eigen::Matrix<double, state_size, noise_size> cov = Eigen::Matrix<double, state_size, noise_size>::Zero();

    cov.block<3, 3>(R_ID, G_VAR_ID) = -Eye3d; // simplified to -I
    cov.block<3, 3>(V_ID, A_VAR_ID) = -s.rot.matrix();
    cov.block<3, 3>(BG_ID, BG_VAR_ID) = Eye3d;
    cov.block<3, 3>(BA_ID, BA_VAR_ID) = Eye3d;

    return cov;
}

state Estimator::get_x()
{
    return x_;
}

cov Estimator::get_P()
{
    return P_;
}

cov Estimator::get_Fx()
{
    return Fx;
}

void Estimator::set_x(state &input_state)
{
    x_ = input_state;
}

void Estimator::set_P(cov &input_cov)
{
    P_ = input_cov;
}

// this applied right update   x = x * exp(dx)
state Estimator::boxplus(state x, vectorized_state f_)
{
    // pos, rot, extrinsic_R, extrinsic_t, vel, bg, ba, grav
    state x_r;
    
    x_r.rot = x.rot * Sophus::SO3::exp(f_.block<3, 1>(R_ID, 0));

    // rotation and translation are coupled tightly coupled
    // if (coupled_rotation_translation)
    // {
    //     // Sophus::SE3 T_updated_right = Sophus::SE3(x.rot, x.pos) * Sophus::SE3::exp(f_.block<6, 1>(P_ID, 0));
    //     // Sophus::SE3 T_updated_left = Sophus::SE3::exp(f_.block<6, 1>(P_ID, 0)) * Sophus::SE3(x.rot, x.pos);

    //     // //Sophus::SE3 T_updated = T_updated_right;
    //     // Sophus::SE3 T_updated = T_updated_left;

    //     // x_r.pos = T_updated.translation();
    //     // x_r.rot = T_updated.so3();

    //     x_r.pos = x.pos + x.rot * f_.block<3, 1>(P_ID, 0);  // Translation in body frame
    // }
    // else
    // {
        // rotation and translation are weakly coupled
        x_r.pos = x.pos + f_.block<3, 1>(P_ID, 0);
    // }

    x_r.offset_R_L_I = x.offset_R_L_I * Sophus::SO3::exp(f_.block<3, 1>(Re_ID, 0));
    x_r.offset_T_L_I = x.offset_T_L_I + f_.block<3, 1>(Te_ID, 0);

    x_r.vel = x.vel + f_.block<3, 1>(V_ID, 0);
    x_r.bg = x.bg + f_.block<3, 1>(BG_ID, 0);
    x_r.ba = x.ba + f_.block<3, 1>(BA_ID, 0);
    x_r.grav = x.grav + f_.block<3, 1>(G_ID, 0);

    return x_r;
}

vectorized_state Estimator::boxminus(state x1, state x2)
{
    // pos, rot, extrinsic_R, extrinsic_t, vel, bg, ba, grav
    vectorized_state x_r = vectorized_state::Zero();

    // rotation and translation are weakly coupled
    x_r.block<3, 1>(P_ID, 0) = x1.pos - x2.pos;
    x_r.block<3, 1>(R_ID, 0) = Sophus::SO3(x2.rot.matrix().transpose() * x1.rot.matrix()).log();

    // rotation and translation are coupled coupled
    // if (coupled_rotation_translation)
    // {
    //     Sophus::SE3 T1(x1.rot, x1.pos);
    //     Sophus::SE3 T2(x2.rot, x2.pos);
    //     Eigen::Matrix<double, 6, 1> dx = (T2.inverse() * T1).log();
    //     x_r.block<3, 1>(P_ID, 0) = dx.block<3, 1>(P_ID, 0);
    //     x_r.block<3, 1>(R_ID, 0) = dx.block<3, 1>(R_ID, 0);
    // }

    x_r.block<3, 1>(Re_ID, 0) = Sophus::SO3(x2.offset_R_L_I.matrix().transpose() * x1.offset_R_L_I.matrix()).log();
    x_r.block<3, 1>(Te_ID, 0) = x1.offset_T_L_I - x2.offset_T_L_I;

    x_r.block<3, 1>(V_ID, 0) = x1.vel - x2.vel;
    x_r.block<3, 1>(BG_ID, 0) = x1.bg - x2.bg;
    x_r.block<3, 1>(BA_ID, 0) = x1.ba - x2.ba;
    x_r.block<3, 1>(G_ID, 0) = x1.grav - x2.grav;

    return x_r;
}

void Estimator::predict(double &dt, Eigen::Matrix<double, noise_size, noise_size> &Q, const input &i_in)
{
    vectorized_state f_ = f(x_, i_in);
    Fx = df_dx(x_, i_in);
    Eigen::Matrix<double, state_size, noise_size> Fw = df_dw(x_, i_in);

    x_ = boxplus(x_, f_ * dt);
    Fx = cov::Identity() + Fx * dt; // add the missing identity and dt
    {
        // inspect this
        //  was I3,  wrong,  should be exp((w-bw) * dt)
        // Fx.block<3, 3>(R_ID, R_ID) = Sophus::SO3::exp((i_in.gyro - x_.bg) * dt).matrix();
    }
    P_ = (Fx)*P_ * (Fx).transpose() + (dt * Fw) * Q * (dt * Fw).transpose();

    // to be tested
    //  Rotation derivatives: ∂Log(R)/∂ω
    //  Using the right Jacobian of SO(3) for the exponential map
    // Eigen::Matrix3d J_r = computeRightJacobian(ang_vel_ * dt);
    // F.block<3, 3>(rot_idx_, ang_vel_idx_) = J_r * dt;

    /*
    Eigen::Matrix3d computeRightJacobian(const Eigen::Vector3d& phi) {
        // Right Jacobian of SO(3) for the exponential map
        double phi_norm = phi.norm();

        if (phi_norm < 1e-8) {
            return Eigen::Matrix3d::Identity();
        }

        Eigen::Vector3d axis = phi / phi_norm;
        double sin_phi = std::sin(phi_norm);
        double cos_phi = std::cos(phi_norm);

        Eigen::Matrix3d axis_hat = Sophus::SO3d::hat(axis);
        Eigen::Matrix3d J_r = Eigen::Matrix3d::Identity()
                            - ((1 - cos_phi) / (phi_norm)) * axis_hat
                            + ((phi_norm - sin_phi) / (phi_norm)) * axis_hat * axis_hat;

        return J_r;
    }
    */
}

state Estimator::propagete_NO_gravity(const double &dt, const input &i_in)
{
    std::cout << "\n dt:" << dt << " s" << std::endl;
    std::cout << "acc:" << i_in.acc.transpose() << std::endl;
    std::cout << "gyro:" << i_in.gyro.transpose() << std::endl;
    std::cout << "s.grav:" << x_.grav.transpose() << std::endl;
    std::cout << "s.bg:" << x_.bg.transpose() << std::endl;
    std::cout << "s.vel:" << x_.vel.transpose() << std::endl;
    std::cout << "s.rot:\n"
              << x_.rot.matrix() << std::endl;

    auto Q = process_noise_cov();
    vectorized_state f_ = f(x_, i_in);
    cov Fx = df_dx(x_, i_in);
    Eigen::Matrix<double, state_size, noise_size> Fw = df_dw(x_, i_in);
    x_ = boxplus(x_, f_ * dt);
    Fx = cov::Identity() + Fx * dt;
    P_ = (Fx)*P_ * (Fx).transpose() + (dt * Fw) * Q * (dt * Fw).transpose();

    return x_;
}