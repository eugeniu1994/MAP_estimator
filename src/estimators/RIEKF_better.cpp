#include "RIEKF.hpp"

using namespace ekf;

#define use_p2p

struct landmark
{
    int cloud_point_index; // index pf the points from the cloud
    V3D norm;              // the normal of the plane in global frame (normalized)
    double d;              // d parameter of the plane
    double w;              // plane weight - should be 1 over plane measurement variance
    V3D tgt;
    double cost; // the residual p2plane cost value
#ifdef use_p2p
    V3D p2p_cost;
#endif
};

struct ResultTuple
{
    cov JTJ;
    vectorized_state JTr;
    double squared_residual;
    int points;

    ResultTuple()
    {
        JTJ.setZero();
        JTr.setZero();
        squared_residual = 0;
        points = 0;
    }

    ResultTuple(const ResultTuple &other)
    {
        JTJ = other.JTJ;
        JTr = other.JTr;
        squared_residual = other.squared_residual;
        points = other.points;
    }

    ResultTuple &operator+=(const ResultTuple &other)
    {
        JTJ += other.JTJ;
        JTr += other.JTr;
        squared_residual += other.squared_residual;
        points += other.points;
        return *this;
    }

    ResultTuple operator+(const ResultTuple &other) const
    {
        ResultTuple out = *this;
        out += other;
        return out;
    }
};

struct AvgValue
{
    AvgValue()
    {
        squared_residual = 0.;
        points = 0.;
    }

    AvgValue operator+(const AvgValue &other)
    {
        this->squared_residual += other.squared_residual;
        this->points += other.points;
        return *this;
    }

    double squared_residual;
    int points;
};

constexpr int max_points = 50000; // this can be adjusted

// MLS globals
static std::vector<landmark> MLS_landmarks(max_points);
static std::vector<bool> MLS_valid(max_points, false);
static std::vector<PointVector> MLS_Neighbours(max_points);

// ALS globals
static std::vector<landmark> ALS_landmarks(max_points);
static std::vector<bool> ALS_valid(max_points, false);
static std::vector<PointVector> ALS_Neighbours(max_points);

// LC globals
static std::vector<landmark> LC_landmarks(max_points);
static std::vector<bool> LC_valid(max_points, false);
static std::vector<PointVector> LC_Neighbours(max_points);

// MLS globals for p2p
#ifdef use_p2p
static std::vector<landmark> MLS_landmarks_p2p(max_points);
static std::vector<bool> MLS_valid_p2p(max_points, false);
using Jacobian_point = Eigen::Matrix<double, 3, state_size>;
#endif

using Jacobian_plane = Eigen::Matrix<double, 1, state_size>;

double cauchy(double residual, double scale = 1.0)
{
    double x = residual / scale;
    return 1.0 / (1.0 + x * x);
}

double huber(double residual, double threshold = 1.0)
{
    double abs_r = std::abs(residual);
    return abs_r <= threshold ? 1.0 : threshold / abs_r;
}

// double estimateCost_tbb(const state &x_, const PointCloudXYZI::Ptr &src_frame)
// {
//     const int feats_down_size = src_frame->size();

//     const AvgValue &result = tbb::parallel_reduce(
//         tbb::blocked_range<int>(0, feats_down_size),
//         AvgValue(),
//         [&](const tbb::blocked_range<int> &r, AvgValue local_cost) -> AvgValue
//         {
//             auto &[squared_residual_private, points_private] = local_cost;
//             for (int i = r.begin(); i < r.end(); ++i)
//             {
//                 if (!global_valid[i])
//                     continue;

//                 const PointType &point_body = src_frame->points[i];
//                 const landmark &l = global_landmarks[i];

//                 // Transform point to world
//                 V3D p_body(point_body.x, point_body.y, point_body.z);
//                 V3D p_global = x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos;

//                 double residual = l.norm.dot(p_global) + l.d;

//                 squared_residual_private += l.w * residual * residual;
//                 points_private += 1;
//             }
//             return local_cost;
//         },
//         // 2nd Lambda: Parallel reduction of the private sums
//         [&](AvgValue a, const AvgValue &b) -> AvgValue
//         { return a + b; });

//     const auto &[squared_residual, n_points] = result;
//     int total_points = 1; // std::max(n_points, 1);
//     return 0.5 * squared_residual / total_points;
// }

// std::tuple<cov, vectorized_state, double> BuildLinearSystem_tbb(const state &x_, const PointCloudXYZI::Ptr &src_frame, const bool extrinsic_est)
// {
//     const int feats_down_size = src_frame->size();

//     auto R = x_.rot.matrix();

//     // const auto &[JTJ, JTr]
//     const ResultTuple &result = tbb::parallel_reduce(
//         // Range
//         tbb::blocked_range<size_t>{0, feats_down_size},
//         // Identity
//         ResultTuple(),
//         // 1st Lambda: Parallel computation
//         [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple
//         {
//             auto &[JTJ_private, JTr_private, squared_residual_private, points_private] = J;
//             for (auto i = r.begin(); i < r.end(); ++i)
//             {
//                 if (global_valid[i])
//                 {
//                     V3D point_(src_frame->points[i].x,
//                                src_frame->points[i].y,
//                                src_frame->points[i].z);

//                     M3D point_crossmat;
//                     point_crossmat << SKEW_SYM_MATRX(point_);

//                     // Transform point to IMU frame
//                     V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;
//                     M3D point_I_crossmat;
//                     point_I_crossmat << SKEW_SYM_MATRX(point_I_);

//                     const landmark &lm = global_landmarks[i];

//                     // apply right update   x = x * exp(dx)

//                     // Norm of plane expressed in world -> rotate into IMU frame
//                     V3D C = x_.rot.matrix().transpose() * lm.norm; // Rᵀ·n
//                     V3D A = point_I_crossmat * C;                  // [point_I]× · Rᵀ·n

//                     // right update [R  −R[p]×]
//                     // left update  [I  −[R*p]×]

//                     Jacobian_plane J_r = Jacobian_plane::Zero(); // 1x6 or more
//                     if (coupled_rotation_translation)
//                     {
//                         J_r.block<1, 3>(0, P_ID) = (lm.norm.transpose() * x_.rot.matrix()); // nᵀ·R
//                     }
//                     else
//                     {
//                         J_r.block<1, 3>(0, P_ID) = lm.norm.transpose(); // from Left Update
//                     }

//                     // J_r.block<1, 3>(0, R_ID) = A.transpose();          // this is from right update wtf? // ([point_I]× · Rᵀ·n)ᵀ = -nᵀ·R·[point_I]×
//                     J_r.block<1, 3>(0, R_ID) = A.transpose(); // same as   -lm.norm.transpose() * R * Sophus::SO3::hat(point_I_);

//                     // //[R −R[p]×]
//                     // Eigen::Matrix<double,3,6> J_p2p_right;J_p2p_right.setZero(); //Jacobian point to point
//                     // J_p2p_right.block<3,3>(0,P_ID) = R;                     // translation
//                     // J_p2p_right.block<3,3>(0,R_ID) = -R * point_I_crossmat; //rotation
//                     // // // Right perturbation: T_new = T_old * exp(ξ)
//                     // // J_r.block<1, 3>(0, P_ID) = (lm.norm.transpose() * R);  // nᵀ·R
//                     // // J_r.block<1, 3>(0, R_ID) = -lm.norm.transpose() * R * Sophus::SO3::hat(point_I);
//                     // J_r.block<1,6>(0,0) = lm.norm.transpose() * J_p2p_right;

//                     //[I −[R p]×]
//                     // Eigen::Matrix<double,3,6> J_p2p_left;J_p2p_left.setZero();
//                     // J_p2p_left.block<3,3>(0,P_ID) = Eye3d;                                  // translation
//                     // J_p2p_left.block<3,3>(0,R_ID) = -1.0 * Sophus::SO3::hat(R * point_I_); // rotation
//                     // Left perturbation: T_new = exp(ξ) * T_old
//                     // J_r.block<1, 3>(0, P_ID) = lm.norm.transpose();  // nᵀ
//                     // J_r.block<1, 3>(0, R_ID) = -lm.norm.transpose() * Sophus::SO3::hat(R * point_I);
//                     // J_r.block<1,6>(0,0) = lm.norm.transpose() * J_p2p_left;

//                     if (extrinsic_est)
//                     {
//                         V3D B(point_crossmat * x_.offset_R_L_I.matrix().transpose() * C);
//                         J_r.block<1, 3>(0, Re_ID) = B.transpose();
//                         J_r.block<1, 3>(0, Te_ID) = C.transpose();
//                     }

//                     // r(k) = z − h(x(k)), z is zero for p2plane
//                     // double residual = -lm.cost; // (distance to plane) negative here due to derivative chain rule
//                     double residual = lm.cost;

//                     JTJ_private.noalias() += J_r.transpose() * lm.w * J_r;
//                     JTr_private.noalias() += J_r.transpose() * lm.w * residual;
//                     squared_residual_private += lm.w * residual * residual;
//                     points_private += 1;
//                 }
//             }
//             return J;
//         },
//         // 2nd Lambda: Parallel reduction of the private Jacboians
//         [&](ResultTuple a, const ResultTuple &b) -> ResultTuple
//         { return a + b; });

//     const auto &[JTJ, JTr, squared_residual, n_points] = result;

//     std::cout << "Registration with:" << n_points << "/" << feats_down_size << std::endl;
//     int total_points =  std::max(n_points, 1);

//     return std::make_tuple(JTJ/total_points, JTr/total_points, 0.5 * squared_residual / total_points);

//              total_points = 1; // std::max(n_points, 1);

//     return std::make_tuple(JTJ, JTr, 0.5 * squared_residual / total_points);
// }

// std::vector<pcl::KdTreeFLANN<PointType>::Ptr> trees(NUM_THREADS);

double ComputeRelativeDisplacement(const Sophus::SE3 &relative_init_guess, double point_range)
{
    // given the imu relative prediction ( x^{-1}_{k-1} * x_pred_{k}  )
    // and the point range - compute the amount of potential displacement for a point 
    const double theta = Eigen::AngleAxisd(relative_init_guess.rotation_matrix()).angle();
    const double delta_rot = 2.0 * point_range * std::sin(theta / 2.0);
    const double delta_trans = relative_init_guess.translation().norm();

    return delta_trans + delta_rot; // return 1 sigma  rotation + translation part
}

void establishCorrespondences(const double R, const state &x_, const bool &update_neighbours, const PointCloudXYZI::Ptr &src_frame,
                              const PointCloudXYZI::Ptr &map, const pcl::KdTreeFLANN<PointType>::Ptr &tree,
                              std::vector<bool> &global_valid, std::vector<landmark> &global_landmarks, std::vector<PointVector> &global_Neighbours)
{
    // std::cout << "establishCorrespondences update_neighbours:" << update_neighbours << std::endl;
    int feats_down_size = src_frame->size();

    auto travelled_distance = x_.pos.norm();

#ifdef MP_EN
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
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
        if (update_neighbours) // update the nearest neighbours
        {
            // int tid = omp_get_thread_num();
            // auto &tree_i = trees[tid];

            // if (tree_i->nearestKSearch(point_world, NUM_MATCH_POINTS, pointSearchInd, pointSearchSqDis) >= NUM_MATCH_POINTS)
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
        double plane_var = 1.;
        // if (ekf::esti_plane(pabcd, points_near, .1))
        if (ekf::esti_plane_pca(pabcd, points_near, .03, point_weights, plane_var, false))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            // float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());
            float s = 1 - 0.9 * fabs(pd2) / point_body.intensity;
            if (s > 0.9)
            {
                const auto &p_tgt = points_near[0];
                landmark l;
                l.cloud_point_index = i;
                l.norm = V3D_4(pabcd(0), pabcd(1), pabcd(2));
                l.d = pabcd(3);

                double base_weight = 1 / R;

                if (travelled_distance > 5)
                    base_weight = 1 / plane_var;

                double kernel_weight = huber(pd2, 0.2); // cauchy(pd2, 0.2);
                l.w = base_weight * kernel_weight;
                // l.w = 1.0; //no adaptive robust 

                l.tgt = V3D(p_tgt.x, p_tgt.y, p_tgt.z);
                l.cost = pd2;

                global_valid[i] = true;
                global_landmarks[i] = l;
            }
        }

#ifdef use_p2p
        MLS_valid_p2p[i] = false;
        if (global_valid[i] == false && points_near.size() > 0) // not a plane & are enough points
        {
            const auto &p_tgt = points_near[0];
            double th = .2;
            if ((V3D(p_tgt.x, p_tgt.y, p_tgt.z) - p_global).norm() < th)
            {
                V3D centroid(0, 0, 0); // Compute the centroid
                M3D covariance;        // Compute covariance matrix
                covariance.setZero();

                // Regularize
                double lambda_reg = 1e-9;
                covariance = covariance + lambda_reg * Eye3d;
                const size_t neighbours = points_near.size();
                for (int j = 0; j < neighbours; j++)
                {
                    centroid(0) += points_near[j].x;
                    centroid(1) += points_near[j].y;
                    centroid(2) += points_near[j].z;
                }
                centroid /= neighbours;

                for (int j = 0; j < neighbours; j++)
                {
                    const auto &p = points_near[j];
                    V3D diff(p.x - centroid(0), p.y - centroid(1), p.z - centroid(2));
                    covariance += diff * diff.transpose();
                }
                covariance /= neighbours;

                // Compute Eigenvalues and Eigenvectors
                Eigen::SelfAdjointEigenSolver<M3D> solver(covariance);

                if (solver.info() != Eigen::Success)
                {
                    std::cerr << "Eigen solver failed!" << std::endl;
                    throw std::runtime_error("Error: Eigen solver failed for p2p covariance!");
                }

                const auto &eigenvalues = solver.eigenvalues();
                // double lambda0 = eigenvalues(0); // smallest
                // double lambda1 = eigenvalues(1);
                double var = eigenvalues(2); // biggest eigen value

                landmark l;

                l.p2p_cost = p_global - V3D(p_tgt.x, p_tgt.y, p_tgt.z); // src - tgt

                double base_weight = 1 / R;
                if (travelled_distance > 5)
                    base_weight = 1 / var;

                double kernel_weight = huber(l.p2p_cost.norm(), 0.2); // cauchy(pd2, 0.2);
                l.w = base_weight * kernel_weight;

                MLS_valid_p2p[i] = true;
                MLS_landmarks_p2p[i] = l;
            }
        }
#endif
    }
}

std::tuple<cov, vectorized_state, double, int> BuildLinearSystem_openMP(
    const state &x_, const PointCloudXYZI::Ptr &src_frame, const bool extrinsic_est,
    std::vector<bool> &global_valid, std::vector<landmark> &global_landmarks)

{
    std::vector<cov> Hs(NUM_THREADS, cov::Zero());
    std::vector<vectorized_state> bs(NUM_THREADS, vectorized_state::Zero());
    std::vector<double> es(NUM_THREADS, 0.0);
    std::vector<int> points_used(NUM_THREADS, 0);

    const int feats_down_size = src_frame->size();

    auto R = x_.rot.matrix();
    auto Rt = R.transpose();

#ifdef MP_EN
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
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

            // if (coupled_rotation_translation)
            // {
            //     J_r.block<1, 3>(0, P_ID) = (lm.norm.transpose() * R);
            // }
            // else
            // {
                J_r.block<1, 3>(0, P_ID) = lm.norm.transpose();
            // }

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
            double e = w * residual * residual;

            const int thread_id = omp_get_thread_num();
            Hs[thread_id] += H;
            bs[thread_id] += b;
            es[thread_id] += e;
            points_used[thread_id]++;
        }

#ifdef use_p2p
        if (MLS_valid_p2p[i])
        {
            V3D point_(src_frame->points[i].x,
                       src_frame->points[i].y,
                       src_frame->points[i].z);

            M3D point_crossmat;
            point_crossmat << SKEW_SYM_MATRX(point_);
            V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;
            M3D point_I_crossmat;
            point_I_crossmat << SKEW_SYM_MATRX(point_I_);

            const landmark &lm = MLS_landmarks_p2p[i];

            Jacobian_point J_r = Jacobian_point::Zero();

            // if (coupled_rotation_translation)
            // {
            //     // J_r.block<1, 3>(0, P_ID) = (lm.norm.transpose() * R);
            //     J_r.block<3, 3>(P_ID, P_ID) = R;
            // }
            // else
            // {
                // J_r.block<1, 3>(0, P_ID) = lm.norm.transpose();
                J_r.block<3, 3>(P_ID, P_ID) = Eye3d;
            // }

            // J_r.block<1, 3>(0, R_ID) = A.transpose();
            // J_r.block<3, 3>(0, R_ID) = -lm.norm.transpose() * R * Sophus::SO3::hat(point_I_);
            // J_r.block<3, 3>(0, R_ID) = -R * point_I_crossmat;
            J_r.block<3, 3>(0, R_ID) = -R * Sophus::SO3::hat(point_I_);

            V3D residual = lm.p2p_cost;
            double w = lm.w;

            cov H = J_r.transpose() * w * J_r;
            vectorized_state b = J_r.transpose() * w * residual;
            double e = w * residual.squaredNorm();

            const int thread_id = omp_get_thread_num();
            Hs[thread_id] += H;
            bs[thread_id] += b;
            es[thread_id] += e;
            points_used[thread_id]++;
        }
#endif
    }

    for (int i = 1; i < NUM_THREADS; i++)
    {
        Hs[0] += Hs[i];
        bs[0] += bs[i];
        es[0] += es[i];
        points_used[0] += points_used[i];
    }

    return {Hs[0], bs[0], es[0], points_used[0]};
}

// Compute numerical measurement Jacobian J (6x6) for:
//   residual r = log( measured.inverse() * X )
// using central differences: H[:,i] = (r(X*exp(+eps*e_i)) - r(X*exp(-eps*e_i))) / (2*eps)
// measured: measurement (Sophus::SE3d)
// X: current state pose (Sophus::SE3d)
// eps: finite-difference step (default 1e-6)
// Returns: Eigen::Matrix<double,6,6>
Eigen::Matrix<double, 6, 6> SE3numericalJacobian(const Sophus::SE3 &X, const Sophus::SE3 &measured, double eps = 1e-5)
{
    Eigen::Matrix<double, 6, 6> J;
    J.setZero();
    for (int i = 0; i < 6; ++i)
    {
        Vector6d d = Vector6d::Zero();
        d(i) = eps;

        // right perturbation: X * exp(+/- d)
        Sophus::SE3 X_plus = X * Sophus::SE3::exp(d);
        Sophus::SE3 X_minus = X * Sophus::SE3::exp(-d);

        // assumes coupled_rotation_translation
        Vector6d r_plus = (measured.inverse() * X_plus).log();
        Vector6d r_minus = (measured.inverse() * X_minus).log();

        // central difference
        J.col(i) = (r_plus - r_minus) / (2.0 * eps);
    }

    return J;
}

std::tuple<cov, vectorized_state, double> BuildLinearSystem_SE3(const state &x_, const Sophus::SE3 &measured_se3,
                                                                const V3D &std_pos_m, const V3D &std_rot_deg)
{
    cov H = cov::Zero();
    vectorized_state b = vectorized_state::Zero();
    double e = 0;

    Vector6d r = Vector6d::Zero();
    // if (coupled_rotation_translation)
    // {
    //     r = (measured_se3.inverse() * Sophus::SE3(x_.rot, x_.pos)).log();
    // }
    // else
    // {
        r.block<3, 1>(P_ID, 0) = x_.pos - measured_se3.translation();
        r.block<3, 1>(R_ID, 0) = Sophus::SO3(measured_se3.so3().matrix().transpose() * x_.rot.matrix()).log();
    // }

    double kernel_weight = huber(r.norm(), 0.2); // cauchy(pd2, 0.2);
    // double kernel_weight = 1;

    // Convert rotation stddevs to radians
    V3D std_rot_rad = std_rot_deg * M_PI / 180.0;
    Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Identity();
    R.block<3, 3>(P_ID, P_ID) = std_pos_m.array().square().matrix().asDiagonal();   // Position covariance (diagonal with variances)
    R.block<3, 3>(R_ID, R_ID) = std_rot_rad.array().square().matrix().asDiagonal(); // Orientation covariance (diagonal with variances in radians^2)
    Eigen::Matrix<double, 6, 6> W = kernel_weight * R.inverse();                    // here take the element wise inverse

    Eigen::Matrix<double, 6, state_size> J_se3 = Eigen::Matrix<double, se3_dim, state_size>::Zero();
    // APPROXIMATE WITH IDENTITY
    J_se3.block<3, 3>(P_ID, P_ID) = Eye3d;
    J_se3.block<3, 3>(R_ID, R_ID) = Eye3d;

    // this should be called only if coupled_rotation_translation is used
    //  J_se3.block<6, 6>(0, 0) = SE3numericalJacobian(Sophus::SE3(x_.rot, x_.pos), measured_se3, 1e-6);

    H.noalias() = J_se3.transpose() * W * J_se3;
    b.noalias() = J_se3.transpose() * W * r;
    e = kernel_weight * r.squaredNorm();
    return {H, b, e};
}

// to be tested
/*

VectorXd doglegStep(
    const VectorXd& r,
    const MatrixXd& J,
    double Delta)
{
    MatrixXd H = J.transpose() * J;
    VectorXd g = J.transpose() * r;

    // Gauss–Newton step
    VectorXd delta_gn = -H.ldlt().solve(g);

    if (delta_gn.norm() <= Delta)
        return delta_gn;

    // Steepest descent (Cauchy point)
    double alpha = g.squaredNorm() / (g.transpose() * H * g);
    VectorXd delta_sd = -alpha * g;

    if (delta_sd.norm() >= Delta)
        return Delta * delta_sd.normalized();

    // Dogleg interpolation
    VectorXd a = delta_sd;
    VectorXd b = delta_gn - delta_sd;

    double A = b.dot(b);
    double B = 2 * a.dot(b);
    double C = a.dot(a) - Delta * Delta;

    double beta =(-B + std::sqrt(B*B - 4*A*C)) / (2*A);

    return a + beta * b;
}

bool doglegIteration(
    Sophus::SE3& pose,
    double& Delta)
{
    // 1. Compute residuals and Jacobian
    VectorXd r;
    MatrixXd J;
    computePointToPlane(pose, r, J);

    double cost0 = 0.5 * r.squaredNorm();

    // 2. Compute dogleg step
    VectorXd delta = doglegStep(r, J, Delta);

    // 3. Apply update (SE(3))
    Sophus::SE3 pose_new = pose * Sophus::SE3::exp(delta);

    // 4. Recompute cost
    VectorXd r_new;
    MatrixXd J_new;
    computePointToPlane(pose_new, r_new, J_new);

    double cost1 = 0.5 * r_new.squaredNorm();

    // 5. Predicted reduction
    VectorXd r_pred = r + J * delta;
    double predicted = 0.5 * (r.squaredNorm() - r_pred.squaredNorm());

    double actual = cost0 - cost1;

    double rho = actual / predicted;

    // 6. Trust-region update
    if (rho < 0.25)
        Delta *= 0.25;
    else if (rho > 0.75 && std::abs(delta.norm() - Delta) < 1e-6)
        Delta = std::min(2.0 * Delta, 5.0);

    // 7. Accept / reject
    if (rho > 0)
    {
        pose = pose_new;
        return true;
    }

    return false;
}
Sophus::SE3 T = imu_predicted_pose;
double Delta = 1.0;

for (int iter = 0; iter < 10; ++iter)
{
    if (!doglegIteration(T, Delta))
        break;
}

*/

int RIEKF::update_MLS(double R, PointCloudXYZI::Ptr &feats_down_body, const PointCloudXYZI::Ptr &map, int maximum_iter, bool extrinsic_est,
                      const bool use_als, const PointCloudXYZI::Ptr &als_map, const pcl::KdTreeFLANN<PointType>::Ptr &als_tree,
                      bool use_se3, const Sophus::SE3 &gnss_se3, const V3D &gnss_std_pos_m, const V3D &gnss_std_rot_deg,
                      const bool use_lc, const PointCloudXYZI::Ptr &lc_map, const pcl::KdTreeFLANN<PointType>::Ptr &lc_tree)
{
    if (false)
    {
#pragma omp parallel
        {
#pragma omp single
            std::cout << "OpenMP threads: " << omp_get_num_threads() << std::endl;
        }
        // Check maximum available threads
        int max_threads = omp_get_max_threads();
        std::cout << "OpenMP Maximum threads available: " << max_threads << std::endl;

        // THEN use parallel regions
#pragma omp parallel
        {
#pragma omp single
            std::cout << "OpenMP threads: " << omp_get_num_threads() << std::endl;
        }

        int active_threads = tbb::global_control::active_value(
            tbb::global_control::max_allowed_parallelism);

        std::cout << "TBB Active thread limit: " << active_threads << std::endl;
        tbb::task_arena arena;
        int max_concurrency = arena.max_concurrency();
        std::cout << "Task arena max concurrency: " << max_concurrency << std::endl;
    }

    // std::cout << "Running standard update_ function" << std::endl;

    residual_struct status;
    status.valid = true;
    status.converge = true;
    int converged_times = 0;

    state x_propagated = x_;
    vectorized_state dx_new = vectorized_state::Zero(); // 24X1

    if (map->points.size() < 5)
    {
        std::cerr << "Error: map Point cloud is empty! : " << map->points.size() << std::endl;
        status.valid = false;
        status.converge = false;
        return 0;
    }

    localKdTree_map->setInputCloud(map);

    int iteration_finished = 0;
    cov P_inv = P_.inverse(); // 24x24

    if (false)
    {
        //     // different way for stop criteria
        //     int max_iter_ = 50; // MAX_NUM_ITERATIONS_

        //     vectorized_state last_dx = vectorized_state::Zero();

        //     for (int i = 0; i <= max_iter_; ++i)
        //     {
        //         status.valid = true;
        //         establishCorrespondences(R, x_, status.converge, feats_down_body, map, localKdTree_map);
        //         // JTJ = J^T W J; JTr = J^T W r
        //         const auto &[JTJ, JTr, current_cost] = BuildLinearSystem_openMP(x_, feats_down_body, extrinsic_est); // non deterministic

        //         vectorized_state dx;
        //         cov H = (JTJ + P_inv);

        //         vectorized_state dx_; // nx1 vectorized_state dx_ = K * status.innovation + (KJ - cov::Identity()) * boxminus(x_, x_propagated);
        //         cov KJ = cov::Zero(); //  matrix K * J

        //         auto H_inv = H.inverse(); // 24x24
        //         KJ = H_inv * JTJ;
        //         // dx_.noalias() = H_inv * (-JTr);       //slower
        //         dx_.noalias() = H.ldlt().solve(-JTr);                       // faster
        //         dx_ += (KJ - cov::Identity()) * boxminus(x_, x_propagated); // iterated error state kalmna filter part

        //         // this applied right update   x = x * exp(dx)
        //         x_ = boxplus(x_, dx_); // GN

        //         if ((dx_ - last_dx).norm() < 0.001 || i == (max_iter_ - 1))
        //         {
        //             P_ -= KJ * P_; // P_ = (cov::Identity() - KJ) * P_;// same as P_ = P_ - KJ*P_ = P_-= KJ*P_

        //             std::cout << "Converged with " << i << " iterations" << std::endl;
        //             break;
        //         }
        //         last_dx = dx_;

        //         status.converge = true;
        //         for (int j = 0; j < state_size; j++)
        //         {
        //             if (std::fabs(dx_[j]) > ESTIMATION_THRESHOLD_) // If dx_>ESTIMATION_THRESHOLD_ = 0.001 no convergence is considered
        //             {
        //                 status.converge = false;
        //                 break;
        //             }
        //         }
        //         if (status.converge)
        //         {
        //             // std::cout<<"Converged at i:"<<i<<std::endl;
        //             converged_times++;
        //         }

        //         if (!converged_times && i == max_iter_ - 2) // if did not converge and last iteration - force converge
        //         {
        //             status.converge = true;
        //         }
        //     }

        //     return 1;
    }

    const bool Armijo = false; // true;
    double new_cost = 0., curr_cost = 9999999999999;

    bool use_mls = false; // true;

    // if the open MP is static - this is not needed
    //  for (int t = 0; t < NUM_THREADS; t++)
    //  {
    //      trees[t].reset(new pcl::KdTreeFLANN<PointType>(*localKdTree_map));
    //  }

    if (use_mls)
    {
        std::cout << "use_mls" << std::endl;
    }
    if (use_als)
    {
        std::cout << "use_als als_map:" << als_map->size() << std::endl;
        // int n_points = als_tree->getInputCloud()->points.size();
        // std::cout << "ALS tree contains " << n_points << " points." << std::endl;
    }
    if (use_se3)
    {
        std::cout << "use_se3" << std::endl;
    }

    if (use_lc)
    {
        std::cout << "use_lc lc_map:" << lc_map->size() << std::endl;
        // int n_points = lc_tree->getInputCloud()->points.size();
        // std::cout << "LC tree contains " << n_points << " points." << std::endl;
    }

    // just a test here
    // if (!use_als)
    // {
    //     // use_mls = true; //force MLS to be true
    // }

    for (int i = -1; i < maximum_iter; i++)
    {
        iteration_finished = i + 1;
        status.valid = true;

        cov JTJ = cov::Zero();
        vectorized_state JTr = vectorized_state::Zero();
        double system_cost = 0;

        // to be done
        // if MLS and ALS toghether - do the search once for the same cloud and populate both MLS & ALS & LC globals

        // if (use_mls)
        //{
        establishCorrespondences(R, x_, status.converge, feats_down_body, map, localKdTree_map, MLS_valid, MLS_landmarks, MLS_Neighbours);

        // const auto &[JTJ_mls, JTr_mls, mls_cost] = BuildLinearSystem_tbb(x_, feats_down_body, extrinsic_est); // fast but not deterministic
        const auto &[JTJ_mls, JTr_mls, mls_cost, da_mls] = BuildLinearSystem_openMP(x_, feats_down_body, extrinsic_est, MLS_valid, MLS_landmarks);

        JTJ += JTJ_mls;
        JTr += JTr_mls;
        system_cost += mls_cost;
        std::cout << "MLS register with " << da_mls << "/" << feats_down_body->size() << " points" << std::endl;
        //}

        if (use_als)
        {
            establishCorrespondences(R, x_, status.converge, feats_down_body, als_map, als_tree, ALS_valid, ALS_landmarks, ALS_Neighbours);
            const auto &[JTJ_als, JTr_als, als_cost, da_als] = BuildLinearSystem_openMP(x_, feats_down_body, extrinsic_est, ALS_valid, ALS_landmarks);

            JTJ += JTJ_als;
            JTr += JTr_als;
            system_cost += als_cost;
        }

        if (use_lc)
        {
            establishCorrespondences(R, x_, status.converge, feats_down_body, lc_map, lc_tree, LC_valid, LC_landmarks, LC_Neighbours);
            const auto &[JTJ_lc, JTr_lc, lc_cost, da_lc] = BuildLinearSystem_openMP(x_, feats_down_body, extrinsic_est, LC_valid, LC_landmarks);

            JTJ += JTJ_lc;
            JTr += JTr_lc;
            system_cost += lc_cost;
        }

        if (use_se3)
        {
            const auto &[JTJ_se3, JTr_se3, se3_cost] = BuildLinearSystem_SE3(x_, gnss_se3, gnss_std_pos_m, gnss_std_rot_deg);
            // double alpha = std::max(da_mls, 1); // ;
            double alpha = 1;

            JTJ += alpha * JTJ_se3;
            JTr += alpha * JTr_se3;
            system_cost += alpha * se3_cost;

            // double a = .5;
            // JTJ = a*JTJ_mls + (1-a)*JTJ_se3;
            // JTr = a*JTr_mls + (1-a)*JTr_se3;
        }

        // int n_points = std::max(da_mls, 1);
        
        // JTJ /= n_points;
        // JTr /= n_points;

        vectorized_state dx;
        cov H = (JTJ + P_inv); // matrix H = JTJ + P_inv (+ lambda * H.diagonal().asDiagonal())  //JTJ - 24 x 24  JTr - 24 x 1      H += lambda * H.diagonal().asDiagonal();

        vectorized_state dx_; // nx1 vectorized_state dx_ = K * status.innovation + (KJ - cov::Identity()) * boxminus(x_, x_propagated);
        cov KJ = cov::Zero(); //  matrix K * J

        cov H_inv = H.inverse(); // 24x24
        KJ = H_inv * JTJ;        // same as (J.T * W * J + P.inv()).inv() * J.T * W * J
        // dx_.noalias() = H_inv * (-JTr);       //slower
        dx_.noalias() = H.ldlt().solve(-JTr);                       // faster
        dx_ += (KJ - cov::Identity()) * boxminus(x_, x_propagated); // iterated error state kalmna filter part

        // vectorized_state dx_ = K * residual + (KH - cov::Identity()) * dx_new; //position only ekf

        // Eigen::LDLT<cov> ldltSolver(H);
        // dx_.noalias() = ldltSolver.solve(-JTr); // Solve for dx_
        // KJ.noalias() = ldltSolver.solve(-JTJ);  // Solve for KJ (multiple right-hand sides)
        // dx_ += (KJ - cov::Identity()) * boxminus(x_, x_propagated);

        // Armijo
        if (Armijo)
        {
            // curr_cost = estimateCost_tbb(x_, feats_down_body);
            // std::cout << "curr_cost:" << curr_cost << std::endl;
            // double gTd = JTr.transpose() * dx_; // 1 x 24  *  24 x 1

            // double step = 1.0;
            // while (true)
            // {
            //     auto trial_x = boxplus(x_, step * dx_);
            //     new_cost = estimateCost_tbb(trial_x, feats_down_body);

            //     if (new_cost > curr_cost)
            //     {
            //         double d_cost = new_cost - curr_cost;
            //         std::cout << "new_cost :" << new_cost << ", d_cost:" << d_cost << ", step:" << step << std::endl;
            //     }
            //     else
            //     {
            //         std::cout << "new_cost :" << new_cost << ", step:" << step << std::endl;
            //     }

            //     if (new_cost <= curr_cost + 1e-4 * step * gTd)
            //         break; // f(x+α*δ) <= f(x)+c1 * ​α* g⊤*δ
            //     step *= 0.5;
            //     if (step < 1e-4)
            //         break; // give up
            // }
            // dx_ = step * dx_;
        }

        // this applied right update   x = x * exp(dx)
        x_ = boxplus(x_, dx_); // GN

        status.converge = true;
        for (int j = 0; j < state_size; j++)
        {
            if (std::fabs(dx_[j]) > ESTIMATION_THRESHOLD_) // If dx_>ESTIMATION_THRESHOLD_ = 0.001 no convergence is considered
            {
                status.converge = false;
                break;
            }
        }

        if (status.converge)
        {
            // std::cout<<"Converged at i:"<<i<<std::endl;
            converged_times++;
        }

        if (!converged_times && i == maximum_iter - 2) // if did not converge and last iteration - force converge
        {
            status.converge = true;
        }

        // if ((dx_ - last_dx).norm() < ESTIMATION_THRESHOLD_ || i == (maximum_iter - 1))   //ESTIMATION_THRESHOLD_ = 0.001
        // {
        //     P_ -= KJ * P_; // P_ = (cov::Identity() - KJ) * P_;// same as P_ = P_ - KJ*P_ = P_-= KJ*P_

        //     std::cout << "Converged with " << i << " iterations" << std::endl;
        //     break;
        // }
        // last_dx = dx_;

        
        // if (i == maximum_iter - 1) // last iteration
        if (converged_times > 1 || i == maximum_iter - 1) // if converged or last iteration
        {
            P_ -= KJ * P_; // P_ = (cov::Identity() - KJ) * P_;// same as P_ = P_ - KJ*P_ = P_-= KJ*P_

            break;
        }
    }

    std::cout << "update_final in " << iteration_finished << "/" << maximum_iter << " iterations " << std::endl;

    return iteration_finished;
}