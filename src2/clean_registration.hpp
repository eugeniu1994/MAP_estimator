#pragma once

#include "DataHandler_vux.hpp"
#include <pcl/kdtree/kdtree_flann.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <unordered_set>
#include <omp.h>

namespace Eigen
{
    const int state_size_ = 6;
    using Matrix6d = Eigen::Matrix<double, state_size_, state_size_>;
    using Matrix3_6d = Eigen::Matrix<double, 3, state_size_>;
    using Vector6d = Eigen::Matrix<double, state_size_, 1>;
} // namespace Eigen

namespace registration
{
    struct P2Plane_global
    {
        P2Plane_global(const V3D &curr_point_, const V3D &plane_unit_norm_,
                       double negative_OA_dot_norm_)
            : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_), negative_OA_dot_norm(negative_OA_dot_norm_) {}

        template <typename T>
        bool operator()(const T *q, const T *t, T *residual) const
        {
            Eigen::Quaternion<T> q_w(q[3], q[0], q[1], q[2]);
            Eigen::Matrix<T, 3, 1> t_w(t[0], t[1], t[2]);

            // Transform raw scanner point to world frame
            Eigen::Matrix<T, 3, 1> point_world = q_w * curr_point.template cast<T>() + t_w;

            Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
            residual[0] = norm.dot(point_world) + T(negative_OA_dot_norm);

            return true;
        }

        static ceres::CostFunction *Create(const V3D &curr_point_, const V3D &plane_unit_norm_,
                                           const double negative_OA_dot_norm_)
        {
            return (new ceres::AutoDiffCostFunction<
                    P2Plane_global, 1, 4, 3>(
                new P2Plane_global(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
        }

        V3D curr_point;
        V3D plane_unit_norm;
        double negative_OA_dot_norm;
    };

    struct P2Plane_local
    {
        P2Plane_local(const Eigen::Vector3d &src_point,
                      const Eigen::Vector3d &target_point,
                      const Eigen::Vector3d &normal)
            : src_point(src_point), target_point(target_point), normal(normal) {}

        template <typename T>
        bool operator()(const T *const q, const T *const t, T *residuals) const
        {
            Eigen::Quaternion<T> q_w(q[3], q[0], q[1], q[2]);
            Eigen::Matrix<T, 3, 1> t_w(t[0], t[1], t[2]);

            Eigen::Matrix<T, 3, 1> transformed_point = q_w * src_point.template cast<T>() + t_w;
            Eigen::Matrix<T, 3, 1> target = target_point.template cast<T>();
            Eigen::Matrix<T, 3, 1> normal_ = normal.template cast<T>();

            residuals[0] = (transformed_point - target).dot(normal_);

            return true;
        }

        static ceres::CostFunction *Create(const Eigen::Vector3d &src_point,
                                           const Eigen::Vector3d &target_point,
                                           const Eigen::Vector3d &normal)
        {
            return (new ceres::AutoDiffCostFunction<
                    P2Plane_local, 1, 4, 3>(
                new P2Plane_local(src_point, target_point, normal)));
        }

        Eigen::Vector3d src_point;
        Eigen::Vector3d target_point;
        Eigen::Vector3d normal;
    };

    struct P2Point
    {
        P2Point(const V3D &curr_point_, const V3D &closest_point_)
            : curr_point(curr_point_), closest_point(closest_point_) {}

        template <typename T>
        bool operator()(const T *const q, const T *const t, T *residual) const
        {
            Eigen::Quaternion<T> q_w(q[3], q[0], q[1], q[2]);
            Eigen::Matrix<T, 3, 1> t_w(t[0], t[1], t[2]);

            // Transform raw scanner point to world frame
            Eigen::Matrix<T, 3, 1> point_world = q_w * curr_point.template cast<T>() + t_w;

            // Compute residual as difference to the closest map point
            residual[0] = point_world.x() - T(closest_point.x());
            residual[1] = point_world.y() - T(closest_point.y());
            residual[2] = point_world.z() - T(closest_point.z());

            return true;
        }

        static ceres::CostFunction *Create(const V3D &curr_point_, const V3D &closest_point_)
        {
            return (new ceres::AutoDiffCostFunction<
                    P2Point, 3, 4, 3>(
                new P2Point(curr_point_, closest_point_)));
        }

        V3D curr_point;
        V3D closest_point;
    };

    struct LidarPlaneNormFactor_extrinsics
    {

        LidarPlaneNormFactor_extrinsics(const V3D &curr_point_, const V3D &plane_unit_norm_,
                                        double negative_OA_dot_norm_, const Sophus::SE3 &fixed_pose_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
                                                                                                        negative_OA_dot_norm(negative_OA_dot_norm_), fixed_pose(fixed_pose_) {}

        template <typename T>
        bool operator()(const T *q, const T *t, T *residual) const
        {
            // Convert extrinsic transformation (Scanner to some frame)
            Eigen::Quaternion<T> q_extrinsic(q[3], q[0], q[1], q[2]);
            Eigen::Matrix<T, 3, 1> t_extrinsic(t[0], t[1], t[2]);

            // Convert Fixed pose to appropriate type
            Eigen::Matrix<T, 3, 3> R_fixed = fixed_pose.rotation_matrix().template cast<T>();
            Eigen::Matrix<T, 3, 1> t_fixed = fixed_pose.translation().template cast<T>();

            // Transform raw scanner point to desired frame with extrinsics
            Eigen::Matrix<T, 3, 1> point_in_frame = q_extrinsic * curr_point.template cast<T>() + t_extrinsic;

            // Georeference point
            Eigen::Matrix<T, 3, 1> point_world = R_fixed * point_in_frame + t_fixed;

            Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
            residual[0] = norm.dot(point_world) + T(negative_OA_dot_norm);

            return true;
        }

        static ceres::CostFunction *Create(const V3D &curr_point_, const V3D &plane_unit_norm_,
                                           const double negative_OA_dot_norm_, const Sophus::SE3 &fixed_pose_)
        {
            return (new ceres::AutoDiffCostFunction<
                    LidarPlaneNormFactor_extrinsics, 1, 4, 3>(
                new LidarPlaneNormFactor_extrinsics(curr_point_, plane_unit_norm_, negative_OA_dot_norm_, fixed_pose_)));
        }

        V3D curr_point;
        V3D plane_unit_norm;
        double negative_OA_dot_norm;
        Sophus::SE3 fixed_pose;
    };

};

double scan2map_ceres(pcl::PointCloud<PointType>::Ptr &src,
                      const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
                      const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
                      Eigen::Quaterniond &q, V3D &t,
                      const bool &prev_segment_init, const pcl::PointCloud<PointType>::Ptr &prev_segment, const pcl::KdTreeFLANN<PointType>::Ptr &kdtree_prev_segment,
                      bool p2p = false, bool p2plane = true, bool local_error = true, double threshold_nn = 1.0)
{
    using namespace registration;

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);

    int points_used_for_registration = 0;

    // Add parameter blocks (with quaternion parameterization)
    problem.AddParameterBlock(q.coeffs().data(), 4, new ceres::EigenQuaternionParameterization());
    problem.AddParameterBlock(t.data(), 3);

    if (p2plane && p2p)
    {
        std::cout << "Perform both p2p and p2plane" << std::endl;
    }
    else if (p2plane && !p2p)
    {
        std::cout << "Perform p2plane" << std::endl;
    }
    else if (p2p && !p2plane)
    {
        std::cout << "Perform p2p" << std::endl;
    }
    for (const auto &raw_point : src->points)
    {
        V3D p_src(raw_point.x, raw_point.y, raw_point.z);
        V3D p_transformed = q * p_src + t;

        // Nearest neighbor search
        PointType search_point;
        search_point.x = p_transformed.x();
        search_point.y = p_transformed.y();
        search_point.z = p_transformed.z();

        if (p2plane && p2p)
        {
            std::vector<int> point_idx(5);
            std::vector<float> point_dist(5);
            if (refference_kdtree->nearestKSearch(search_point, 5, point_idx, point_dist) > 0)
            {
                if (point_dist[4] < threshold_nn) // not too far
                {
                    Eigen::Matrix<double, 5, 3> matA0;
                    Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                    for (int j = 0; j < 5; j++)
                    {
                        matA0(j, 0) = reference_localMap_cloud->points[point_idx[j]].x;
                        matA0(j, 1) = reference_localMap_cloud->points[point_idx[j]].y;
                        matA0(j, 2) = reference_localMap_cloud->points[point_idx[j]].z;
                    }

                    // find the norm of plane
                    V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                    double negative_OA_dot_norm = 1 / norm.norm();
                    norm.normalize();

                    bool planeValid = true;
                    for (int j = 0; j < 5; j++)
                    {
                        if (fabs(norm(0) * reference_localMap_cloud->points[point_idx[j]].x +
                                 norm(1) * reference_localMap_cloud->points[point_idx[j]].y +
                                 norm(2) * reference_localMap_cloud->points[point_idx[j]].z + negative_OA_dot_norm) > .2)
                        {
                            planeValid = false;
                            break;
                        }
                    }

                    if (planeValid)
                    {
                        if (local_error)
                        {
                            V3D target_point(
                                reference_localMap_cloud->points[point_idx[0]].x,
                                reference_localMap_cloud->points[point_idx[0]].y,
                                reference_localMap_cloud->points[point_idx[0]].z);

                            ceres::CostFunction *cost_function = P2Plane_local::Create(p_src, target_point, norm);
                            problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                        }
                        else
                        {
                            ceres::CostFunction *cost_function = P2Plane_global::Create(p_src, norm, negative_OA_dot_norm);
                            problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                        }

                        points_used_for_registration++;
                    }
                    else
                    {
                        V3D target_point(
                            reference_localMap_cloud->points[point_idx[0]].x,
                            reference_localMap_cloud->points[point_idx[0]].y,
                            reference_localMap_cloud->points[point_idx[0]].z);

                        // Add residuals to Ceres problem
                        ceres::CostFunction *cost_function = P2Point::Create(p_src, target_point);
                        problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                        points_used_for_registration++;
                    }
                }
            }
        }
        else if (p2plane && !p2p)
        {
            std::vector<int> point_idx(5);
            std::vector<float> point_dist(5);
            if (refference_kdtree->nearestKSearch(search_point, 5, point_idx, point_dist) > 0)
            {
                if (point_dist[4] < threshold_nn) // not too far
                {
                    Eigen::Matrix<double, 5, 3> matA0;
                    Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                    for (int j = 0; j < 5; j++)
                    {
                        matA0(j, 0) = reference_localMap_cloud->points[point_idx[j]].x;
                        matA0(j, 1) = reference_localMap_cloud->points[point_idx[j]].y;
                        matA0(j, 2) = reference_localMap_cloud->points[point_idx[j]].z;
                    }

                    // find the norm of plane
                    V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                    double negative_OA_dot_norm = 1 / norm.norm();
                    norm.normalize();

                    bool planeValid = true;
                    for (int j = 0; j < 5; j++)
                    {
                        if (fabs(norm(0) * reference_localMap_cloud->points[point_idx[j]].x +
                                 norm(1) * reference_localMap_cloud->points[point_idx[j]].y +
                                 norm(2) * reference_localMap_cloud->points[point_idx[j]].z + negative_OA_dot_norm) > .2)
                        {
                            planeValid = false;
                            break;
                        }
                    }

                    if (planeValid)
                    {
                        if (local_error)
                        {
                            V3D target_point(
                                reference_localMap_cloud->points[point_idx[0]].x,
                                reference_localMap_cloud->points[point_idx[0]].y,
                                reference_localMap_cloud->points[point_idx[0]].z);

                            ceres::CostFunction *cost_function = P2Plane_local::Create(p_src, target_point, norm);
                            problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                        }
                        else
                        {
                            ceres::CostFunction *cost_function = P2Plane_global::Create(p_src, norm, negative_OA_dot_norm);
                            problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                        }

                        points_used_for_registration++;
                    }
                }
            }
        }
        else if (p2p && !p2plane)
        {
            std::vector<int> point_idx(1);
            std::vector<float> point_dist(1);
            if (refference_kdtree->nearestKSearch(search_point, 1, point_idx, point_dist) > 0)
            {
                if (point_dist[0] < threshold_nn)
                {
                    V3D target_point(
                        reference_localMap_cloud->points[point_idx[0]].x,
                        reference_localMap_cloud->points[point_idx[0]].y,
                        reference_localMap_cloud->points[point_idx[0]].z);

                    // Add residuals to Ceres problem
                    ceres::CostFunction *cost_function = P2Point::Create(p_src, target_point);
                    problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                    points_used_for_registration++;
                }
            }
        }
    }

    if (prev_segment_init) // we have the prev segment too
    {
        std::cout << "add the prev segment too" << std::endl;
        for (const auto &raw_point : src->points)
        {
            V3D p_src(raw_point.x, raw_point.y, raw_point.z);
            V3D p_transformed = q * p_src + t;

            // Nearest neighbor search
            PointType search_point;
            search_point.x = p_transformed.x();
            search_point.y = p_transformed.y();
            search_point.z = p_transformed.z();

            if (p2plane && p2p)
            {
                std::vector<int> point_idx(5);
                std::vector<float> point_dist(5);
                if (kdtree_prev_segment->nearestKSearch(search_point, 5, point_idx, point_dist) > 0)
                {
                    if (point_dist[4] < threshold_nn) // not too far
                    {
                        Eigen::Matrix<double, 5, 3> matA0;
                        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                        for (int j = 0; j < 5; j++)
                        {
                            matA0(j, 0) = prev_segment->points[point_idx[j]].x;
                            matA0(j, 1) = prev_segment->points[point_idx[j]].y;
                            matA0(j, 2) = prev_segment->points[point_idx[j]].z;
                        }

                        // find the norm of plane
                        V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                        double negative_OA_dot_norm = 1 / norm.norm();
                        norm.normalize();

                        bool planeValid = true;
                        for (int j = 0; j < 5; j++)
                        {
                            if (fabs(norm(0) * prev_segment->points[point_idx[j]].x +
                                     norm(1) * prev_segment->points[point_idx[j]].y +
                                     norm(2) * prev_segment->points[point_idx[j]].z + negative_OA_dot_norm) > .2)
                            {
                                planeValid = false;
                                break;
                            }
                        }

                        if (planeValid)
                        {
                            if (local_error)
                            {
                                V3D target_point(
                                    prev_segment->points[point_idx[0]].x,
                                    prev_segment->points[point_idx[0]].y,
                                    prev_segment->points[point_idx[0]].z);

                                ceres::CostFunction *cost_function = P2Plane_local::Create(p_src, target_point, norm);
                                problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                            }
                            else
                            {
                                ceres::CostFunction *cost_function = P2Plane_global::Create(p_src, norm, negative_OA_dot_norm);
                                problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                            }

                            points_used_for_registration++;
                        }
                        else
                        {
                            V3D target_point(
                                prev_segment->points[point_idx[0]].x,
                                prev_segment->points[point_idx[0]].y,
                                prev_segment->points[point_idx[0]].z);

                            // Add residuals to Ceres problem
                            ceres::CostFunction *cost_function = P2Point::Create(p_src, target_point);
                            problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                            points_used_for_registration++;
                        }
                    }
                }
            }
            else if (p2plane && !p2p)
            {
                std::vector<int> point_idx(5);
                std::vector<float> point_dist(5);
                if (kdtree_prev_segment->nearestKSearch(search_point, 5, point_idx, point_dist) > 0)
                {
                    if (point_dist[4] < threshold_nn) // not too far
                    {
                        Eigen::Matrix<double, 5, 3> matA0;
                        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                        for (int j = 0; j < 5; j++)
                        {
                            matA0(j, 0) = prev_segment->points[point_idx[j]].x;
                            matA0(j, 1) = prev_segment->points[point_idx[j]].y;
                            matA0(j, 2) = prev_segment->points[point_idx[j]].z;
                        }

                        // find the norm of plane
                        V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                        double negative_OA_dot_norm = 1 / norm.norm();
                        norm.normalize();

                        bool planeValid = true;
                        for (int j = 0; j < 5; j++)
                        {
                            if (fabs(norm(0) * prev_segment->points[point_idx[j]].x +
                                     norm(1) * prev_segment->points[point_idx[j]].y +
                                     norm(2) * prev_segment->points[point_idx[j]].z + negative_OA_dot_norm) > .2)
                            {
                                planeValid = false;
                                break;
                            }
                        }

                        if (planeValid)
                        {
                            if (local_error)
                            {
                                V3D target_point(
                                    prev_segment->points[point_idx[0]].x,
                                    prev_segment->points[point_idx[0]].y,
                                    prev_segment->points[point_idx[0]].z);

                                ceres::CostFunction *cost_function = P2Plane_local::Create(p_src, target_point, norm);
                                problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                            }
                            else
                            {
                                ceres::CostFunction *cost_function = P2Plane_global::Create(p_src, norm, negative_OA_dot_norm);
                                problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                            }

                            points_used_for_registration++;
                        }
                    }
                }
            }
            else if (p2p && !p2plane)
            {
                std::vector<int> point_idx(1);
                std::vector<float> point_dist(1);
                if (kdtree_prev_segment->nearestKSearch(search_point, 1, point_idx, point_dist) > 0)
                {
                    if (point_dist[0] < threshold_nn)
                    {
                        V3D target_point(
                            prev_segment->points[point_idx[0]].x,
                            prev_segment->points[point_idx[0]].y,
                            prev_segment->points[point_idx[0]].z);

                        // Add residuals to Ceres problem
                        ceres::CostFunction *cost_function = P2Point::Create(p_src, target_point);
                        problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                        points_used_for_registration++;
                    }
                }
            }
        }
    }
    // Solve the problem
    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    options.linear_solver_type = ceres::DENSE_QR; // options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "Optimization complete. Points used: " << points_used_for_registration << "/" << src->size() << std::endl;
    return summary.final_cost;
}

double scan2map_GN(pcl::PointCloud<PointType>::Ptr &src,
                   const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
                   const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
                   Eigen::Quaterniond &q, V3D &t, Sophus::SE3 &T_icp,
                   bool p2p = false, bool p2plane = true, bool local_error = true, double threshold_nn = 1.0)
{
    using namespace registration;
    int points_used_for_registration = 0;

    std::cout << "Run ceres solver..." << std::endl;
    if (p2plane && p2p)
    {
        std::cout << "Perform both p2p and p2plane" << std::endl;
    }
    else if (p2plane && !p2p)
    {
        std::cout << "Perform p2plane" << std::endl;
    }
    else if (p2p && !p2plane)
    {
        std::cout << "Perform p2p" << std::endl;
    }

    // Eigen::MatrixXd H;
    // Eigen::VectorXd g;
    // int num_scans = 1;
    // H = Eigen::MatrixXd::Zero(6 * num_scans, 6 * num_scans);
    // g = Eigen::VectorXd::Zero(6 * num_scans);

    double kernel = 1.0;
    auto Weight = [&](double residual2)
    {
        return square(kernel) / square(kernel + residual2);
    };

    Eigen::Matrix6d JTJ_private; // state_size x state_size  (6x6)
    Eigen::Vector6d JTr_private; // state_size x 1           (6x1)
    JTJ_private.setZero();
    JTr_private.setZero();

    for (const auto &raw_point : src->points)
    {
        V3D p_src(raw_point.x, raw_point.y, raw_point.z);
        V3D p_transformed = q * p_src + t;

        // Nearest neighbor search
        PointType search_point;
        search_point.x = p_transformed.x();
        search_point.y = p_transformed.y();
        search_point.z = p_transformed.z();

        std::vector<int> point_idx(1);
        std::vector<float> point_dist(1);
        if (refference_kdtree->nearestKSearch(search_point, 1, point_idx, point_dist) > 0)
        {
            if (point_dist[0] < threshold_nn)
            {
                V3D target_point(
                    reference_localMap_cloud->points[point_idx[0]].x,
                    reference_localMap_cloud->points[point_idx[0]].y,
                    reference_localMap_cloud->points[point_idx[0]].z);

                const V3D residual = p_transformed - target_point;
                Eigen::Matrix3_6d J_r;
                J_r.block<3, 3>(0, 0) = Eye3d;                                  // df/dt
                J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(p_transformed); // df/dR

                double w = Weight(residual.squaredNorm());

                JTJ_private.noalias() += J_r.transpose() * w * J_r;
                JTr_private.noalias() += J_r.transpose() * w * residual;

                // points_used_for_registration++;
            }
        }
    }

    // Insert into global Hessian
    // H.block<6, 6>(0, 0) = JTJ_private;
    // g.segment<6>(0) = JTr_private;

    // Solve the system H * deltaT = -g
    // Eigen::VectorXd deltaT = H.ldlt().solve(-g); // 6*num_scans   X   1
    // Eigen::VectorXd dx = deltaT.segment<6>(0);   // Eigen::Vector6d

    const Eigen::Vector6d dx = JTJ_private.ldlt().solve(-JTr_private); // translation and rotation perturbations

    const Sophus::SE3 estimation = Sophus::SE3::exp(dx); // this is in local-align init guess to map
    T_icp = estimation * T_icp;

    q = Eigen::Quaterniond(T_icp.so3().matrix());
    t = T_icp.translation();

    return dx.norm();
}

// pcl::KdTreeFLANN<PointType>::Ptr kdtree_prev_segment(new pcl::KdTreeFLANN<PointType>());
// pcl::PointCloud<PointType>::Ptr prev_segment(new pcl::PointCloud<PointType>);
// bool prev_segment_init = false;

double scan2map_GN_omp(pcl::PointCloud<PointType>::Ptr &src,
                       const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
                       const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
                       Eigen::Quaterniond &q, V3D &t, Sophus::SE3 &T_icp,
                       const bool &prev_segment_init, const pcl::PointCloud<PointType>::Ptr &prev_segment, const pcl::KdTreeFLANN<PointType>::Ptr &kdtree_prev_segment,
                       bool p2p = false, bool p2plane = true, bool local_error = true, double threshold_nn = 1.0)
{
    using namespace registration;

    double kernel = 1.0;
    auto Weight = [&](double residual2)
    {
        return square(kernel) / square(kernel + residual2);
    };

    Eigen::Matrix6d JTJ_global = Eigen::Matrix6d::Zero();
    Eigen::Vector6d JTr_global = Eigen::Vector6d::Zero();
    double cost_total = 0.;

    int num_points = src->points.size();

#pragma omp parallel
    {
        Eigen::Matrix6d JTJ_private = Eigen::Matrix6d::Zero();
        Eigen::Vector6d JTr_private = Eigen::Vector6d::Zero();
        double cost_private = 0.;
#pragma omp for schedule(dynamic)
        for (int i = 0; i < num_points; i++)
        {
            const auto &raw_point = src->points[i];

            V3D p_src(raw_point.x, raw_point.y, raw_point.z);
            V3D p_transformed = q * p_src + t;

            // Nearest neighbor search
            PointType search_point;
            search_point.x = p_transformed.x();
            search_point.y = p_transformed.y();
            search_point.z = p_transformed.z();

            std::vector<int> point_idx(1);
            std::vector<float> point_dist(1);
            if (refference_kdtree->nearestKSearch(search_point, 1, point_idx, point_dist) > 0)
            {
                if (point_dist[0] < threshold_nn)
                {
                    V3D target_point(
                        reference_localMap_cloud->points[point_idx[0]].x,
                        reference_localMap_cloud->points[point_idx[0]].y,
                        reference_localMap_cloud->points[point_idx[0]].z);

                    const V3D residual = p_transformed - target_point;
                    Eigen::Matrix3_6d J_r;
                    J_r.block<3, 3>(0, 0) = Eye3d;                                  // df/dt
                    J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(p_transformed); // df/dR

                    double w = Weight(residual.squaredNorm());

                    JTJ_private.noalias() += J_r.transpose() * w * J_r;
                    JTr_private.noalias() += J_r.transpose() * w * residual;

                    cost_private += w*residual.squaredNorm();
                }
            }
        }

#pragma omp critical
        {
            JTJ_global += JTJ_private;
            JTr_global += JTr_private;
            cost_total += cost_private;
        }
    }

    // maybe integrate this in the prev section
    if (prev_segment_init) // we have the prev segment too
    {
        std::cout << "add the prev segment too" << std::endl;

#pragma omp parallel
        {
            Eigen::Matrix6d JTJ_private = Eigen::Matrix6d::Zero();
            Eigen::Vector6d JTr_private = Eigen::Vector6d::Zero();
            double cost_private = 0.;
            
#pragma omp for schedule(dynamic)
            for (int i = 0; i < num_points; i++)
            {
                const auto &raw_point = src->points[i];

                V3D p_src(raw_point.x, raw_point.y, raw_point.z);
                V3D p_transformed = q * p_src + t;

                // Nearest neighbor search
                PointType search_point;
                search_point.x = p_transformed.x();
                search_point.y = p_transformed.y();
                search_point.z = p_transformed.z();

                std::vector<int> point_idx(1);
                std::vector<float> point_dist(1);
                if (kdtree_prev_segment->nearestKSearch(search_point, 1, point_idx, point_dist) > 0)
                {
                    if (point_dist[0] < threshold_nn)
                    {
                        V3D target_point(
                            prev_segment->points[point_idx[0]].x,
                            prev_segment->points[point_idx[0]].y,
                            prev_segment->points[point_idx[0]].z);

                        const V3D residual = p_transformed - target_point;
                        Eigen::Matrix3_6d J_r;
                        J_r.block<3, 3>(0, 0) = Eye3d;                                  // df/dt
                        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(p_transformed); // df/dR

                        double w = Weight(residual.squaredNorm());

                        JTJ_private.noalias() += J_r.transpose() * w * J_r;
                        JTr_private.noalias() += J_r.transpose() * w * residual;

                        cost_private += w*residual.squaredNorm();
                    }
                }
            }

#pragma omp critical
            {
                JTJ_global += JTJ_private;
                JTr_global += JTr_private;
                cost_total += cost_private;
            }
        }
    }

    const Eigen::Vector6d dx = JTJ_global.ldlt().solve(-JTr_global);

    const Sophus::SE3 estimation = Sophus::SE3::exp(dx);
    T_icp = estimation * T_icp;

    q = Eigen::Quaterniond(T_icp.so3().matrix());
    t = T_icp.translation();

    return cost_total;
    //return dx.norm();
}
