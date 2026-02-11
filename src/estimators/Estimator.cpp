#include "Estimator.hpp"

// Initialization of noise covariance Q
Eigen::Matrix<double, noise_size, noise_size> process_noise_cov()
{
    // some defualt values for Q
    //  gyroscope variance, acceleration variance, bias gyro var, bias acc var
    Eigen::Matrix<double, noise_size, noise_size> Q = Eigen::MatrixXd::Zero(noise_size, noise_size);
    Q.block<3, 3>(G_VAR_ID, G_VAR_ID) = 0.0001 * Eye3d;    // cov gyro
    Q.block<3, 3>(A_VAR_ID, A_VAR_ID) = 0.0001 * Eye3d;    // cov acc
    Q.block<3, 3>(BG_VAR_ID, BG_VAR_ID) = 0.00001 * Eye3d; // cov_bias_gyro
    Q.block<3, 3>(BA_VAR_ID, BA_VAR_ID) = 0.00001 * Eye3d; // cov_bias_acc
    return Q;
}

Eigen::Matrix<double, state_size, 1> f(state s, input in)
{
    Eigen::Matrix<double, state_size, 1> result = Eigen::Matrix<double, state_size, 1>::Zero();
    V3D omega = in.gyro - s.bg; // bias free angular velocity

    // bias free acceleration transform to the world frame
    V3D a_inertial = s.rot.matrix() * (in.acc - s.ba);
    for (int i = 0; i < 3; i++)
    {
        result(i + P_ID) = s.vel[i]; // prev state vel (constant vel model)
        // result(i + P_ID) = s.vel[i] + a_inertial[i] * dt
        result(i + R_ID) = omega[i];                  // Angular velocity
        result(i + V_ID) = a_inertial[i] + s.grav[i]; // gravity-free acceleration
    }
    return result;
}

Eigen::Matrix<double, state_size, state_size> df_dx(state s, input in)
{
    Eigen::Matrix<double, state_size, state_size> cov = Eigen::Matrix<double, state_size, state_size>::Zero();

    cov.block<3, 3>(P_ID, V_ID) = Eye3d;
    cov.block<3, 3>(R_ID, BG_ID) = -Eye3d;
    V3D acc_ = in.acc - s.ba;
    cov.block<3, 3>(V_ID, R_ID) = -s.rot.matrix() * Sophus::SO3::hat(acc_);
    cov.block<3, 3>(V_ID, BA_ID) = -s.rot.matrix();
    cov.template block<3, 3>(V_ID, G_ID) = Eye3d;
    return cov;
}

Eigen::Matrix<double, state_size, noise_size> df_dw(state s, input in)
{
    Eigen::Matrix<double, state_size, noise_size> cov = Eigen::Matrix<double, state_size, noise_size>::Zero();
    cov.block<3, 3>(R_ID, G_VAR_ID) = -Eye3d;
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

//  right update   x = x * exp(dx)
state Estimator::boxplus(state x, vectorized_state f_)
{
    state x_r;

    x_r.rot = x.rot * Sophus::SO3::exp(f_.block<3, 1>(R_ID, 0));
    x_r.pos = x.pos + f_.block<3, 1>(P_ID, 0);

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
    vectorized_state x_r = vectorized_state::Zero();

    // rotation and translation are weakly coupled
    x_r.block<3, 1>(P_ID, 0) = x1.pos - x2.pos;
    x_r.block<3, 1>(R_ID, 0) = Sophus::SO3(x2.rot.matrix().transpose() * x1.rot.matrix()).log();

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

    V3D seg_SO3 = f_.block<3, 1>(R_ID, 0) * dt;
    // M3D A = A_matrix(seg_SO3);
    Fx.block<3, 3>(R_ID, R_ID) = Sophus::SO3::exp(-1 * seg_SO3).matrix();
    Fx.block<3, 3>(R_ID, BG_ID) = -J_right(seg_SO3) * dt; // same as Jr(phi) = Jl(phi).T  // -A.transpose() * dt;
    Fw.block<3, 3>(R_ID, G_VAR_ID) = -J_right(seg_SO3);   // dt is added later // -A.transpose();

    P_ = (Fx)*P_ * (Fx).transpose() + (dt * Fw) * Q * (dt * Fw).transpose();
}

//----------------------------------------------------------------------------------------------

struct landmark
{
    V3D norm;              // the normal of the plane in global frame (normalized)
    double d;              // d parameter of the plane
    double w;              // plane weight - should be 1 over plane measurement variance
    double cost; // the residual p2plane cost value
};

constexpr int max_points = 50000; // this should be adjusted to allocate more memory for more points

// MLS globals
static std::vector<landmark> MLS_landmarks(max_points);
static std::vector<bool> MLS_valid(max_points, false);
static std::vector<PointVector> MLS_Neighbours(max_points);

// ALS globals
static std::vector<landmark> ALS_landmarks(max_points);
static std::vector<bool> ALS_valid(max_points, false);
static std::vector<PointVector> ALS_Neighbours(max_points);

double huber(double residual, double threshold = 1.0)
{
    double abs_r = std::abs(residual);
    return abs_r <= threshold ? 1.0 : threshold / abs_r;
}

void establishCorrespondences(const double R, const state &x_, const bool &update_neighbours, PointCloudXYZI::Ptr &src_frame,
                              const PointCloudXYZI::Ptr &map, const pcl::KdTreeFLANN<PointType>::Ptr &tree,
                              std::vector<bool> &global_valid, std::vector<landmark> &global_landmarks, std::vector<PointVector> &global_Neighbours)
{
    int feats_down_size = src_frame->size();

    double travelled_distance = x_.pos.norm();

#ifdef MP_EN
// #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = src_frame->points[i];
        PointType point_world;

        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos); // lidar to imu and then to world

        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        std::vector<int> pointSearchInd(NUM_MATCH_POINTS);
        std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = global_Neighbours[i];
        std::vector<double> point_weights;
        if (update_neighbours)
        {
            if (tree->nearestKSearch(point_world, NUM_MATCH_POINTS, pointSearchInd, pointSearchSqDis) >= NUM_MATCH_POINTS)
            {
                global_valid[i] = pointSearchSqDis[NUM_MATCH_POINTS - 1] > 1 ? false : true; // true if distance smaller than 1
                points_near.clear();
                for (int j = 0; j < pointSearchInd.size(); j++)
                {
                    // points_near.push_back(map->points[pointSearchInd[j]]);
                    points_near.insert(points_near.begin(), map->points[pointSearchInd[j]]);
                }
            }
            else
            {
                global_valid[i] = false;
            }
        }

        if (!global_valid[i])
            continue;

        Eigen::Matrix<float, 4, 1> pabcd; // plane coefficients [nx, ny, nz, d]
        global_valid[i] = false;
        double plane_var = 1.; // default 1
        if (ekf::esti_plane_pca(pabcd, points_near, .04, point_weights, plane_var, false))
        {
            float p2plane = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            {
                const auto &p_tgt = points_near[0];
                landmark l;
                l.norm = V3D(pabcd(0), pabcd(1), pabcd(2));
                l.d = pabcd(3);

                double base_weight = 1 / R; // Geometric weight

                // TODO: explain this - in the beginning there is no map yet, and normals are unstable
                // start using the covs when the map has at least 5m trajectory
                if (travelled_distance > 5)      // if more than 5 m travelled - start using the data-driven covs
                    base_weight = 1 / plane_var; 

                double d = p2plane / std::sqrt(plane_var);
                double kernel_weight = huber(d, 1.0); 

                l.w = base_weight * kernel_weight; 

                l.cost = -p2plane; // as in the paper r = 0 - p2plane

                global_valid[i] = true;
                global_landmarks[i] = l;
            }
        }
    }
}

void establishCorrespondences_2maps(const double R, const state &x_, const bool &update_neighbours, PointCloudXYZI::Ptr &src_frame,
                                    const PointCloudXYZI::Ptr &map1, const pcl::KdTreeFLANN<PointType>::Ptr &tree1,
                                    std::vector<bool> &global_valid1, std::vector<landmark> &global_landmarks1, std::vector<PointVector> &global_Neighbours1,
                                    const PointCloudXYZI::Ptr &map2, const pcl::KdTreeFLANN<PointType>::Ptr &tree2,
                                    std::vector<bool> &global_valid2, std::vector<landmark> &global_landmarks2, std::vector<PointVector> &global_Neighbours2)
{
    int feats_down_size = src_frame->size();

    double travelled_distance = x_.pos.norm();

#ifdef MP_EN
// #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = src_frame->points[i];
        PointType point_world;

        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos); // lidar to imu and then to world

        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        std::vector<int> pointSearchInd1(NUM_MATCH_POINTS), pointSearchInd2(NUM_MATCH_POINTS);
        std::vector<float> pointSearchSqDis1(NUM_MATCH_POINTS), pointSearchSqDis2(NUM_MATCH_POINTS);

        auto &points_near1 = global_Neighbours1[i];
        auto &points_near2 = global_Neighbours2[i];
        std::vector<double> point_weights;
        if (update_neighbours)
        {
            if (tree1->nearestKSearch(point_world, NUM_MATCH_POINTS, pointSearchInd1, pointSearchSqDis1) >= NUM_MATCH_POINTS)
            {
                global_valid1[i] = pointSearchSqDis1[NUM_MATCH_POINTS - 1] > 1 ? false : true; 
                points_near1.clear();
                for (int j = 0; j < pointSearchInd1.size(); j++)
                {
                    points_near1.insert(points_near1.begin(), map1->points[pointSearchInd1[j]]);
                }
            }
            else
            {
                global_valid1[i] = false;
            }

            if (tree2->nearestKSearch(point_world, NUM_MATCH_POINTS, pointSearchInd2, pointSearchSqDis2) >= NUM_MATCH_POINTS)
            {
                global_valid2[i] = pointSearchSqDis2[NUM_MATCH_POINTS - 1] > 1 ? false : true; 
                points_near2.clear();
                for (int j = 0; j < pointSearchInd2.size(); j++)
                {
                    points_near2.insert(points_near2.begin(), map2->points[pointSearchInd2[j]]);
                }
            }
            else
            {
                global_valid2[i] = false;
            }
        }

        if (global_valid1[i])
        {
            Eigen::Matrix<float, 4, 1> pabcd; // plane coefficients [nx, ny, nz, d]
            global_valid1[i] = false;
            double plane_var = 1.; // default 1
            if (ekf::esti_plane_pca(pabcd, points_near1, .04, point_weights, plane_var, false))
            {
                float p2plane = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
                {
                    const auto &p_tgt = points_near1[0];
                    landmark l;
                    l.norm = V3D(pabcd(0), pabcd(1), pabcd(2));
                    l.d = pabcd(3);

                    double base_weight = 1 / R; // Geometric weight

                    if (travelled_distance > 5)      // if more than 5 m travelled - start using the data-driven covs
                        base_weight = 1 / plane_var; // 1/R as in the paper

                    double d = p2plane / std::sqrt(plane_var);
                    double kernel_weight = huber(d, 1.0); // 1 sigma

                    l.w = base_weight * kernel_weight;
                    l.cost = -p2plane; // as in the paper r = 0 - p2plane

                    global_valid1[i] = true;
                    global_landmarks1[i] = l;
                }
            }
        }

        if (global_valid2[i])
        {
            Eigen::Matrix<float, 4, 1> pabcd; // plane coefficients [nx, ny, nz, d]
            global_valid2[i] = false;
            double plane_var = 1.; 
            if (ekf::esti_plane_pca(pabcd, points_near2, .04, point_weights, plane_var, false))
            {
                float p2plane = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
                {
                    // point_body.ring = 100;
                    const auto &p_tgt = points_near2[0];
                    landmark l;
                    l.norm = V3D(pabcd(0), pabcd(1), pabcd(2));
                    l.d = pabcd(3);

                    double base_weight = 1 / R; // Geometric weight

                    // TODO: explain this - in the beginning there is not map yet, and normals are unstable
                    // start using the covs when the map has at least 1m trajectory
                    if (travelled_distance > 5)      // if more than 5 m travelled - start using the data-driven covs
                        base_weight = 1 / plane_var; // 1/R as in the paper

                    double d = p2plane / std::sqrt(plane_var);
                    double kernel_weight = huber(d, 1.0); // 1 sigma

                    l.w = base_weight * kernel_weight;
                    l.cost = -p2plane; // as in the paper r = 0 - p2plane

                    global_valid2[i] = true;
                    global_landmarks2[i] = l;
                }
            }
        }
    }
}

std::tuple<cov, vectorized_state, int> BuildLinearSystem_openMP(
    const state &x_, const PointCloudXYZI::Ptr &src_frame, const bool extrinsic_est,
    std::vector<bool> &global_valid, std::vector<landmark> &global_landmarks)

{
    std::vector<cov> Hs(NUM_THREADS, cov::Zero());
    std::vector<vectorized_state> bs(NUM_THREADS, vectorized_state::Zero());
    std::vector<int> points_used(NUM_THREADS, 0);

    const int feats_down_size = src_frame->size();

    auto R = x_.rot.matrix();
    auto Rt = R.transpose();

#ifdef MP_EN
// #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
#pragma omp parallel for
#endif
    for (std::int64_t i = 0; i < feats_down_size; i++)
    {
        if (global_valid[i])
        {
            V3D point_(src_frame->points[i].x,
                       src_frame->points[i].y,
                       src_frame->points[i].z);

            M3D point_crossmat;
            point_crossmat << SKEW_SYM_MATRX(point_);
            V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;
            M3D point_I_crossmat;
            point_I_crossmat << SKEW_SYM_MATRX(point_I_);

            const landmark &lm = global_landmarks[i];

            // Rotate plane normal into IMU frame
            V3D C = Rt * lm.norm;
            V3D A = point_I_crossmat * C;

            Jacobian_plane J_r = Jacobian_plane::Zero();
            J_r.block<1, 3>(0, P_ID) = lm.norm.transpose();
            J_r.block<1, 3>(0, R_ID) = A.transpose();

            if (extrinsic_est)
            {
                V3D B(point_crossmat * x_.offset_R_L_I.matrix().transpose() * C);
                J_r.block<1, 3>(0, Re_ID) = B.transpose();
                J_r.block<1, 3>(0, Te_ID) = C.transpose();
            }

            double residual = lm.cost;
            double w = lm.w;

            cov H = J_r.transpose() * w * J_r;
            vectorized_state b = J_r.transpose() * w * residual;

            const int thread_id = omp_get_thread_num();
            Hs[thread_id] += H;
            bs[thread_id] += b;
            points_used[thread_id]++;
        }
    }

    for (int i = 1; i < NUM_THREADS; i++)
    {
        Hs[0] += Hs[i];
        bs[0] += bs[i];
        points_used[0] += points_used[i];
    }
    int da = std::max(points_used[0], 1); // number of points used in data association
    return {Hs[0], bs[0], da};
}

int MAP_::update(int maximum_iter, bool extrinsic_est, PointCloudXYZI::Ptr &feats_down_body, const PointCloudXYZI::Ptr &map)
{
    // std::cout<<"NUM_THREADS:"<<NUM_THREADS<<std::endl;
    obj_struct status;
    status.valid = true;
    status.converge = true;

    int converged_times = 0;

    if (map->points.size() < 5)
    {
        std::cerr << "Error: map Point cloud is empty! : " << map->points.size() << std::endl;
        status.valid = false;
        status.converge = false;
        return 0;
    }

    localKdTree_map->setInputCloud(map);

    state x = x_;            // Current estimate
    state x_prop = x_;       // Prior mean
    cov P_prop = P_;         // Prior covariance
    cov Pinv = P_.inverse(); // 24x24
    // prior Jacobian J_p
    cov Jp = cov::Identity();

    int iteration_finished = 0;
    for (int iter = 0; iter < maximum_iter; ++iter)
    {
        iteration_finished = iter;

        cov JTJ = cov::Zero();
        vectorized_state JTr = vectorized_state::Zero();
        establishCorrespondences(LASER_POINT_COV, x, status.converge,
                                     feats_down_body, map,
                                     localKdTree_map,
                                     MLS_valid, MLS_landmarks, MLS_Neighbours);

        const auto &[JTJ_mls, JTr_mls, da_mls] = BuildLinearSystem_openMP(x, feats_down_body, extrinsic_est, MLS_valid, MLS_landmarks);
        JTJ += JTJ_mls;
        JTr += JTr_mls;
        // std::cout << "MLS register with " << da_mls << "/" << feats_down_body->size() << " points" << std::endl;
        

        // PRIOR RESIDUAL
        vectorized_state e = boxminus(x, x_prop);
        // vectorized_state e = boxminus(x_prop, x);

        V3D phi = e.block<3, 1>(R_ID, 0);

        cov Lambda;
        vectorized_state g;

        Jp.block<3, 3>(R_ID, R_ID) = Jr_inv(phi);
        // Jp.block<3, 3>(R_ID, R_ID) = -Jl_inv(phi);

        // MAP NORMAL EQUATIONS
        Lambda = JTJ + Jp.transpose() * Pinv * Jp;
        g = JTr - Jp.transpose() * Pinv * e;
        // g = JTr + Jp.transpose() * Pinv * e;

        vectorized_state dx = Lambda.ldlt().solve(g);
        // Update state
        x = boxplus(x, dx);

        bool converged = true;
        for (int j = 0; j < state_size; ++j)
        {
            if (std::fabs(dx[j]) > ESTIMATION_THRESHOLD_)
            {
                converged = false;
                break;
            }
        }

        if (converged)
        {
            converged_times++;
        }

        if (converged > 1 || iter == maximum_iter - 1)
        {
            e = boxminus(x, x_prop);
            // e = boxminus(x_prop, x);

            phi = e.block<3, 1>(R_ID, 0);

            Jp.block<3, 3>(R_ID, R_ID) = Jr_inv(phi);
            // Jp.block<3, 3>(R_ID, R_ID) = -Jl_inv(phi);

            // Final posterior covariance
            cov Lambda_final = JTJ + Jp.transpose() * Pinv * Jp;
            P_ = Lambda_final.inverse();

            // Symmetrize
            P_ = 0.5 * (P_ + P_.transpose());

            x_ = x;
            break;
        }
    }

    // std::cout << "update in " << iteration_finished << "/" << maximum_iter << " iterations " << std::endl;

    return iteration_finished;
}


std::tuple<cov, vectorized_state> BuildLinearSystem_SE3(const state &x_, const Sophus::SE3 &measured_se3,
                                                                const V3D &std_pos_m, const V3D &std_rot_deg)
{
    cov H = cov::Zero();
    vectorized_state b = vectorized_state::Zero();

    // Convert rotation stddevs to radians
    V3D std_rot_rad = std_rot_deg * M_PI / 180.0;
    Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Identity();
    R.block<3, 3>(P_ID, P_ID) = std_pos_m.array().square().matrix().asDiagonal();   // Position covariance (diagonal with variances)
    R.block<3, 3>(R_ID, R_ID) = std_rot_rad.array().square().matrix().asDiagonal(); // Orientation covariance (diagonal with variances in radians^2)

    Vector6d r = Vector6d::Zero();
    // z - h(x)
    r.block<3, 1>(P_ID, 0) = measured_se3.translation() - x_.pos; // z.pos - x.pos
    r.block<3, 1>(R_ID, 0) = Sophus::SO3(x_.rot.matrix().transpose() * measured_se3.so3().matrix()).log(); //log(z - h(x)) = h(x).T * z
    
    double maha = std::sqrt(r.transpose() * R.inverse() * r);
    double kernel_weight = huber(maha, 1.0);
    Eigen::Matrix<double, 6, 6> W = kernel_weight * R.inverse();

    // dh/dx
    Eigen::Matrix<double, 6, state_size> J_se3 = Eigen::Matrix<double, 6, state_size>::Zero();
    J_se3.block<3, 3>(P_ID, P_ID) = Eye3d;
    J_se3.block<3, 3>(R_ID, R_ID) = Eye3d;

    H.noalias() = J_se3.transpose() * W * J_se3;
    b.noalias() = J_se3.transpose() * W * r;

    return {H, b};
}

int MAP_::update(int maximum_iter, bool extrinsic_est, PointCloudXYZI::Ptr &feats_down_body, const PointCloudXYZI::Ptr &map,
                 const bool use_als, const PointCloudXYZI::Ptr &als_map, const pcl::KdTreeFLANN<PointType>::Ptr &als_tree,   // prior ALS map
                 const bool use_se3, const Sophus::SE3 &gnss_se3, const V3D &se3_std_pos_m, const V3D &se3_std_rot_deg,      // absolute SE3 meas,
                 bool use_se3_rel, const Sophus::SE3 &se3_rel, const V3D &se3_rel_std_pos_m, const V3D &se3_rel_std_rot_deg, // relative SE3 meas,
                 const Sophus::SE3 &prev_X)
{
    obj_struct status;
    status.valid = true;
    status.converge = true;

    int converged_times = 0;

    if (map->points.size() < 5)
    {
        std::cerr << "Error: map Point cloud is empty! : " << map->points.size() << std::endl;
        status.valid = false;
        status.converge = false;
        return 0;
    }

    localKdTree_map->setInputCloud(map);

    if (use_als)
    {
        std::cout << "use_als als_map:" << als_map->size() << std::endl;
    }

    state x = x_;            // Current estimate
    state x_prop = x_;       // Prior mean
    cov P_prop = P_;         // Prior covariance
    cov Pinv = P_.inverse(); // 24x24
    // prior Jacobian J_p
    cov Jp = cov::Identity();

    int iteration_finished = 0;
    for (int iter = 0; iter < maximum_iter; ++iter)
    {
        iteration_finished = iter;

        cov JTJ = cov::Zero();
        vectorized_state JTr = vectorized_state::Zero();

        double l = 1;
        if (use_als)
        {
            establishCorrespondences_2maps(LASER_POINT_COV, x, status.converge, feats_down_body,
                                           map, localKdTree_map,
                                           MLS_valid, MLS_landmarks, MLS_Neighbours,
                                           als_map, als_tree,
                                           ALS_valid, ALS_landmarks, ALS_Neighbours);

            const auto &[JTJ_mls, JTr_mls, da_mls] = BuildLinearSystem_openMP(x, feats_down_body, extrinsic_est, MLS_valid, MLS_landmarks);
            const auto &[JTJ_als, JTr_als, da_als] = BuildLinearSystem_openMP(x, feats_down_body, extrinsic_est, ALS_valid, ALS_landmarks);

            l = std::max(static_cast<double>(da_mls), static_cast<double>(da_als));
            double alpha = std::max(l / static_cast<double>(da_mls), 1.0);
            JTJ += alpha * JTJ_mls;
            JTr += alpha * JTr_mls;

            alpha = std::max(l / static_cast<double>(da_als), 1.0);
            JTJ += alpha * JTJ_als;
            JTr += alpha * JTr_als;

            std::cout << "MLS register with " << da_mls << "/" << feats_down_body->size() << " points" << std::endl;
            std::cout << "ALS register with " << da_als << "/" << feats_down_body->size() << " points, alpha:" << alpha << std::endl;
        }
        else
        {
            establishCorrespondences(LASER_POINT_COV, x, status.converge,
                                     feats_down_body, map,
                                     localKdTree_map,
                                     MLS_valid, MLS_landmarks, MLS_Neighbours);

            const auto &[JTJ_mls, JTr_mls, da_mls] = BuildLinearSystem_openMP(x, feats_down_body, extrinsic_est, MLS_valid, MLS_landmarks);
            l = std::max(da_mls, 1);
            JTJ += JTJ_mls;
            JTr += JTr_mls;

            std::cout << "MLS register with " << da_mls << "/" << feats_down_body->size() << " points" << std::endl;
        }

        if (use_se3)
        {
            const auto &[JTJ_se3, JTr_se3] = BuildLinearSystem_SE3(x, gnss_se3, se3_std_pos_m, se3_std_rot_deg);
            JTJ += l * JTJ_se3;
            JTr += l * JTr_se3;
        }


        // PRIOR RESIDUAL
        vectorized_state e = boxminus(x, x_prop);
        // vectorized_state e = boxminus(x_prop, x);

        V3D phi = e.block<3, 1>(R_ID, 0);

        cov Lambda;
        vectorized_state g;

        Jp.block<3, 3>(R_ID, R_ID) = Jr_inv(phi);
        // Jp.block<3, 3>(R_ID, R_ID) = -Jl_inv(phi);

        // MAP NORMAL EQUATIONS
        Lambda = JTJ + Jp.transpose() * Pinv * Jp;
        g = JTr - Jp.transpose() * Pinv * e;
        // g = JTr + Jp.transpose() * Pinv * e;

        vectorized_state dx = Lambda.ldlt().solve(g);
        // Update state
        x = boxplus(x, dx);

        // CONVERGENCE CHECK
        bool converged = true;
        for (int j = 0; j < state_size; ++j)
        {
            if (std::fabs(dx[j]) > ESTIMATION_THRESHOLD_)
            {
                converged = false;
                break;
            }
        }

        if (converged)
        {
            converged_times++;
        }

        if (converged > 1 || iter == maximum_iter - 1)
        {
            e = boxminus(x, x_prop);
            // e = boxminus(x_prop, x);

            phi = e.block<3, 1>(R_ID, 0);

            Jp.block<3, 3>(R_ID, R_ID) = Jr_inv(phi);
            // Jp.block<3, 3>(R_ID, R_ID) = -Jl_inv(phi);

            // Final posterior covariance
            cov Lambda_final = JTJ + Jp.transpose() * Pinv * Jp;
            P_ = Lambda_final.inverse();

            // Symmetrize
            P_ = 0.5 * (P_ + P_.transpose());

            x_ = x;
            break;
        }
    }

    std::cout << "update in " << iteration_finished << "/" << maximum_iter << " iterations " << std::endl;

    return iteration_finished;
}