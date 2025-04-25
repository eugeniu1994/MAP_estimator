#pragma once

#include "DataHandler_vux.hpp"
#include <pcl/kdtree/kdtree_flann.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <unordered_set>
#include <omp.h>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

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

    bool test_point_to_mesh = true;
    if(test_point_to_mesh)
    {
        std::cout<<"Testing point to mesh cost function "<<std::endl;
    }

    std::cout << "Run GN omp ..." << std::endl;
    if (p2plane && p2p)
    {
        std::cout << "Perform both p2p and p2plane" << std::endl;
    }
    else if (p2plane)
    {
        std::cout << "Perform p2plane" << std::endl;
    }
    else if (p2p)
    {
        std::cout << "Perform p2p" << std::endl;
    }

    int num_points = src->points.size();
    std::cout << "num_points:" << num_points << ", reference_localMap_cloud:" << reference_localMap_cloud->size() << std::endl;
    if (reference_localMap_cloud->size() == 0)
    {
        throw std::runtime_error("reference_localMap_cloud not init - no points");
    }

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
            
            // //it works----------------------------
            // if(test_point_to_mesh)
            // {
            //     std::vector<int> point_idx(3);
            //     std::vector<float> point_dist(3);
            //     if (refference_kdtree->nearestKSearch(search_point, 3, point_idx, point_dist) > 0)
            //     {
            //         if (point_dist[2] < threshold_nn) // not too far
            //         {
            //             // 3 nearest neighbors
            //             const auto &p1 = reference_localMap_cloud->points[point_idx[0]];
            //             const auto &p2 = reference_localMap_cloud->points[point_idx[1]];
            //             const auto &p3 = reference_localMap_cloud->points[point_idx[2]];

            //             Eigen::Vector3d v1(p1.x, p1.y, p1.z);
            //             Eigen::Vector3d v2(p2.x, p2.y, p2.z);
            //             Eigen::Vector3d v3(p3.x, p3.y, p3.z);

            //             // Compute normal of the triangle (mesh)
            //             V3D norm = (v2 - v1).cross(v3 - v1);
            //             double area = norm.norm() * 0.5;
            //             norm.normalize();

            //             bool planeValid = (area > 1e-6); // reject degenerate triangles

            //             //if (planeValid)
            //             { 
            //                 //use the p2plane - since projected point already uses the normal to get its value
            //                 Eigen::Vector6d J_r;                               // 6x1
            //                 J_r.block<3, 1>(0, 0) = norm;                      // df/dt
            //                 J_r.block<3, 1>(3, 0) = p_transformed.cross(norm); // df/dR

            //                 double residual = (p_transformed - v1).dot(norm);

            //                 double w = Weight(residual * residual);
            //                 JTJ_private.noalias() += J_r * w * J_r.transpose();
            //                 JTr_private.noalias() += J_r * w * residual;

            //                 cost_private += w * residual * residual; 


            //                 // Project pt onto the plane
            //                 // double distance = norm.dot(p_transformed - v1);
            //                 // V3D target_point = p_transformed - distance * norm;

            //                 // Eigen::Matrix3_6d J_r;
            //                 // const V3D residual = p_transformed - target_point;
            //                 // J_r.block<3, 3>(0, 0) = Eye3d;                                  // df/dt
            //                 // J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(p_transformed); // df/dR
            //                 // double w = Weight(residual.squaredNorm());
            //                 // JTJ_private.noalias() += J_r.transpose() * w * J_r;
            //                 // JTr_private.noalias() += J_r.transpose() * w * residual;

            //                 // cost_private += w * residual.squaredNorm();
            //             }
            //         }
            //     }
            
            // }
            
            // else
            
            if (p2p && p2plane) // both
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
                                     norm(2) * reference_localMap_cloud->points[point_idx[j]].z + negative_OA_dot_norm) > .1)
                            {
                                planeValid = false;
                                break;
                            }
                        }

                        if (planeValid)
                        {
                            V3D target_point(
                                reference_localMap_cloud->points[point_idx[0]].x,
                                reference_localMap_cloud->points[point_idx[0]].y,
                                reference_localMap_cloud->points[point_idx[0]].z);

                            Eigen::Vector6d J_r;                               // 6x1
                            J_r.block<3, 1>(0, 0) = norm;                      // df/dt
                            J_r.block<3, 1>(3, 0) = p_transformed.cross(norm); // df/dR

                            // double residual = norm(0) * p_transformed.x() + norm(1) * p_transformed.y() + norm(2) * p_transformed.z() + negative_OA_dot_norm;
                            double residual = (p_transformed - target_point).dot(norm);

                            double w = Weight(residual * residual);
                            JTJ_private.noalias() += J_r * w * J_r.transpose();
                            JTr_private.noalias() += J_r * w * residual;

                            cost_private += w * residual * residual; // Always non-negative
                        }
                        else
                        {
                            V3D target_point(
                                reference_localMap_cloud->points[point_idx[0]].x,
                                reference_localMap_cloud->points[point_idx[0]].y,
                                reference_localMap_cloud->points[point_idx[0]].z);

                            Eigen::Matrix3_6d J_r;

                            const V3D residual = p_transformed - target_point;
                            J_r.block<3, 3>(0, 0) = Eye3d;                                  // df/dt
                            J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(p_transformed); // df/dR
                            double w = Weight(residual.squaredNorm());
                            JTJ_private.noalias() += J_r.transpose() * w * J_r;
                            JTr_private.noalias() += J_r.transpose() * w * residual;

                            cost_private += w * residual.squaredNorm();
                        }
                    }
                }
            }
            else if (p2plane)
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
                                     norm(2) * reference_localMap_cloud->points[point_idx[j]].z + negative_OA_dot_norm) > .1)
                            {
                                planeValid = false;
                                break;
                            }
                        }

                        if (planeValid)
                        {
                            V3D target_point(
                                reference_localMap_cloud->points[point_idx[0]].x,
                                reference_localMap_cloud->points[point_idx[0]].y,
                                reference_localMap_cloud->points[point_idx[0]].z);

                            Eigen::Vector6d J_r;                               // 6x1
                            J_r.block<3, 1>(0, 0) = norm;                      // df/dt
                            J_r.block<3, 1>(3, 0) = p_transformed.cross(norm); // df/dR

                            // double residual = norm(0) * p_transformed.x() + norm(1) * p_transformed.y() + norm(2) * p_transformed.z() + negative_OA_dot_norm;
                            double residual = (p_transformed - target_point).dot(norm);

                            double w = Weight(residual * residual);
                            JTJ_private.noalias() += J_r * w * J_r.transpose();
                            JTr_private.noalias() += J_r * w * residual;

                            cost_private += w * residual * residual; // Always non-negative
                        }
                    }
                }
            
            }
            else if (p2p) // only p2p
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

                        Eigen::Matrix3_6d J_r;

                        const V3D residual = p_transformed - target_point;
                        J_r.block<3, 3>(0, 0) = Eye3d;                                  // df/dt
                        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(p_transformed); // df/dR
                        double w = Weight(residual.squaredNorm());
                        JTJ_private.noalias() += J_r.transpose() * w * J_r;
                        JTr_private.noalias() += J_r.transpose() * w * residual;

                        cost_private += w * residual.squaredNorm();
                    }
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

                if (p2p && p2plane) // both
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
                                         norm(2) * prev_segment->points[point_idx[j]].z + negative_OA_dot_norm) > .1)
                                {
                                    planeValid = false;
                                    break;
                                }
                            }

                            if (planeValid)
                            {
                                V3D target_point(
                                    prev_segment->points[point_idx[0]].x,
                                    prev_segment->points[point_idx[0]].y,
                                    prev_segment->points[point_idx[0]].z);

                                Eigen::Vector6d J_r;                               // 6x1
                                J_r.block<3, 1>(0, 0) = norm;                      // df/dt
                                J_r.block<3, 1>(3, 0) = p_transformed.cross(norm); // df/dR

                                // double residual = norm(0) * p_transformed.x() + norm(1) * p_transformed.y() + norm(2) * p_transformed.z() + negative_OA_dot_norm;
                                double residual = (p_transformed - target_point).dot(norm);

                                double w = Weight(residual * residual);
                                JTJ_private.noalias() += J_r * w * J_r.transpose();
                                JTr_private.noalias() += J_r * w * residual;

                                cost_private += w * residual * residual; // Always non-negative
                            }
                            else
                            {
                                V3D target_point(
                                    prev_segment->points[point_idx[0]].x,
                                    prev_segment->points[point_idx[0]].y,
                                    prev_segment->points[point_idx[0]].z);

                                Eigen::Matrix3_6d J_r;

                                const V3D residual = p_transformed - target_point;
                                J_r.block<3, 3>(0, 0) = Eye3d;                                  // df/dt
                                J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(p_transformed); // df/dR
                                double w = Weight(residual.squaredNorm());
                                JTJ_private.noalias() += J_r.transpose() * w * J_r;
                                JTr_private.noalias() += J_r.transpose() * w * residual;

                                cost_private += w * residual.squaredNorm();
                            }
                        }
                    }
                }
                else if (p2plane)
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
                                         norm(2) * prev_segment->points[point_idx[j]].z + negative_OA_dot_norm) > .1)
                                {
                                    planeValid = false;
                                    break;
                                }
                            }

                            if (planeValid)
                            {
                                V3D target_point(
                                    prev_segment->points[point_idx[0]].x,
                                    prev_segment->points[point_idx[0]].y,
                                    prev_segment->points[point_idx[0]].z);

                                Eigen::Vector6d J_r;                               // 6x1
                                J_r.block<3, 1>(0, 0) = norm;                      // df/dt
                                J_r.block<3, 1>(3, 0) = p_transformed.cross(norm); // df/dR

                                // double residual = norm(0) * p_transformed.x() + norm(1) * p_transformed.y() + norm(2) * p_transformed.z() + negative_OA_dot_norm;
                                double residual = (p_transformed - target_point).dot(norm);

                                double w = Weight(residual * residual);
                                JTJ_private.noalias() += J_r * w * J_r.transpose();
                                JTr_private.noalias() += J_r * w * residual;

                                cost_private += w * residual * residual; // Always non-negative
                            }
                        }
                    }
                }
                else if (p2p) // only p2p
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

                            Eigen::Matrix3_6d J_r;

                            const V3D residual = p_transformed - target_point;
                            J_r.block<3, 3>(0, 0) = Eye3d;                                  // df/dt
                            J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(p_transformed); // df/dR
                            double w = Weight(residual.squaredNorm());
                            JTJ_private.noalias() += J_r.transpose() * w * J_r;
                            JTr_private.noalias() += J_r.transpose() * w * residual;

                            cost_private += w * residual.squaredNorm();
                        }
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
    // return dx.norm();
}

void debug_CloudWithNormals(const pcl::PointCloud<pcl::PointNormal>::Ptr &cloud_with_normals,
                            const ros::Publisher &cloud_pub, const ros::Publisher &normals_pub)
{

    // --- 1. Publish Point Cloud ---
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud_with_normals, cloud_msg);
    cloud_msg.header.frame_id = "world";

    // --- 2. Publish Normals as Markers ---
    visualization_msgs::Marker normals_marker;
    normals_marker.header.frame_id = "world";
    normals_marker.type = visualization_msgs::Marker::LINE_LIST;
    normals_marker.action = visualization_msgs::Marker::ADD;
    normals_marker.scale.x = 0.05; // Line width
    normals_marker.color.a = 1.0;  // Full opacity
    double normal_length = 3.;     // 5.;     // Length of normal lines

    for (const auto &point : cloud_with_normals->points)
    {
        geometry_msgs::Point p1, p2;

        // Start of normal (point)
        p1.x = point.x;
        p1.y = point.y;
        p1.z = point.z;
        normals_marker.points.push_back(p1);

        // End of normal (point + normal * length)
        p2.x = point.x + normal_length * point.normal_x;
        p2.y = point.y + normal_length * point.normal_y;
        p2.z = point.z + normal_length * point.normal_z;
        normals_marker.points.push_back(p2);

        std_msgs::ColorRGBA color;
        // if (point.curvature < 2) // seen less than 10 times
        // {
        //     color.r = 1.0; // not seen enough
        //     color.g = 0.0;
        //     color.b = 0.0;
        // }
        // else
        // {
        color.r = 0.0;
        color.g = 1.0; // Green for high curvature
        color.b = 0.0;
        //}
        color.a = 1.0; // Full opacity

        normals_marker.colors.push_back(color); // Color for start point
        normals_marker.colors.push_back(color); // Color for end point
    }

    if (cloud_pub.getNumSubscribers() != 0)
    {
        cloud_pub.publish(cloud_msg);
    }

    if (normals_pub.getNumSubscribers() != 0)
    {
        normals_pub.publish(normals_marker);
    }
}

struct landmark_
{
    int map_point_index;
    V3D norm;
    double negative_OA_dot_norm;
    int seen;
    double re_proj_error;
    std::vector<int> line_idx;
    std::vector<int> scan_idx;
};

// only planes for now
std::unordered_map<int, landmark_> get_Correspondences(
    const std::deque<pcl::PointCloud<VUX_PointType>::Ptr> &lidar_lines,
    const std::deque<Sophus::SE3> &line_poses_,
    const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
    const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
    const bool &prev_segment_init,
    const pcl::PointCloud<PointType>::Ptr &prev_segment,
    const pcl::KdTreeFLANN<PointType>::Ptr &kdtree_prev_segment, double threshold_nn = 1.0, double good_plane_threshold = .1)
{
    // Containers
    std::unordered_map<int, landmark_> landmarks_map; // key is index of the point from the map

    // #define pca_norms
    //  Loop through each scan/line
    for (size_t l = 0; l < lidar_lines.size(); l++)
    {
        auto &T = line_poses_[l];
        for (int i = 0; i < lidar_lines[l]->size(); i++)
        {
            V3D p_src(lidar_lines[l]->points[i].x, lidar_lines[l]->points[i].y, lidar_lines[l]->points[i].z);
            V3D p_transformed = T * p_src;

            // Nearest neighbor search
            PointType search_point;
            search_point.x = p_transformed.x();
            search_point.y = p_transformed.y();
            search_point.z = p_transformed.z();

            std::vector<int> point_idx(5);
            std::vector<float> point_dist(5);

            if (refference_kdtree->nearestKSearch(search_point, 5, point_idx, point_dist) > 0)
            {
                if (point_dist[4] < threshold_nn)
                {
#ifndef pca_norms
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
                        double d = fabs(norm(0) * reference_localMap_cloud->points[point_idx[j]].x +
                                        norm(1) * reference_localMap_cloud->points[point_idx[j]].y +
                                        norm(2) * reference_localMap_cloud->points[point_idx[j]].z + negative_OA_dot_norm);
                        if (d > good_plane_threshold)
                        {
                            planeValid = false;
                            break;
                        }
                        point_dist[j] = d;
                    }

#else
                    // Compute the centroid
                    V3D centroid(0, 0, 0);
                    for (int j = 0; j < 5; j++)
                    {
                        centroid(0) += reference_localMap_cloud->points[point_idx[j]].x;
                        centroid(1) += reference_localMap_cloud->points[point_idx[j]].y;
                        centroid(2) += reference_localMap_cloud->points[point_idx[j]].z;
                    }
                    centroid /= 5;

                    // Compute covariance matrix
                    M3D covariance;
                    covariance.setZero();
                    for (int j = 0; j < 5; j++)
                    {
                        const auto &p = reference_localMap_cloud->points[point_idx[j]];
                        V3D diff(p.x - centroid(0), p.y - centroid(1), p.z - centroid(2));
                        covariance += diff * diff.transpose();
                    }
                    covariance /= 5;

                    // Compute Eigenvalues and Eigenvectors
                    Eigen::SelfAdjointEigenSolver<M3D> solver(covariance);
                    V3D norm = solver.eigenvectors().col(0); // Smallest eigenvector
                    double negative_OA_dot_norm = 1 / norm.norm();
                    // Normalize the normal
                    norm.normalize();

                    // Compute eigenvalue ratios to assess planarity
                    const auto &eigenvalues = solver.eigenvalues();
                    double lambda0 = eigenvalues(0); // smallest
                    double lambda1 = eigenvalues(1);
                    double lambda2 = eigenvalues(2);

                    // Planarity filter: reject unstable normals
                    // double th = .3;//allow noisy data
                    double th = .7;                                                             // good ones
                    bool planeValid = (lambda2 > 1e-6) && ((lambda1 - lambda0) / lambda2 > th); // Tunable thresholds

                    for (int j = 0; j < 5; j++)
                        point_dist[j] = lambda2;

#endif

                    if (planeValid) // found a good plane
                    {
                        // Flip normal to point consistently toward viewpoint
                        V3D point_on_the_plane(reference_localMap_cloud->points[point_idx[0]].x, reference_localMap_cloud->points[point_idx[0]].y, reference_localMap_cloud->points[point_idx[0]].z);
                        if (norm.dot(T.translation() - point_on_the_plane) < 0)
                        {
                            norm = -norm;
                        }

                        for (int j = 0; j < 5; j++) // all the points share the same normal
                        {
                            auto it = landmarks_map.find(point_idx[j]);
                            if (it == landmarks_map.end()) // does not exist - add
                            {
                                landmark_ tgt;
                                tgt.map_point_index = point_idx[j];
                                tgt.norm = norm;
                                tgt.negative_OA_dot_norm = negative_OA_dot_norm;
                                tgt.seen = 1;
                                tgt.re_proj_error = point_dist[j];
                                tgt.line_idx.push_back(l); // seen from line l
                                tgt.scan_idx.push_back(i); // and point i from this line

                                landmarks_map[point_idx[j]] = tgt;
                            }
                            else // Already exists - increment "seen"
                            {
                                it->second.seen++;                // Increment the seen count
                                it->second.line_idx.push_back(l); // seen from line l
                                it->second.scan_idx.push_back(i); // and point i from this line
                                // if a better normal was found - replace the old one
                                if (it->second.re_proj_error > point_dist[j])
                                {
                                    it->second.norm = norm;
                                    it->second.negative_OA_dot_norm = negative_OA_dot_norm;
                                    it->second.re_proj_error = point_dist[j];
                                }
                            }
                        }
                    }
                }
            }

            if (prev_segment_init) // we have the prev segment too
            {
                // std::cout << "add the prev segment too" << std::endl;
                if (kdtree_prev_segment->nearestKSearch(search_point, 5, point_idx, point_dist) > 0)
                {
                    if (point_dist[4] < threshold_nn)
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
                            double d = fabs(norm(0) * prev_segment->points[point_idx[j]].x +
                                            norm(1) * prev_segment->points[point_idx[j]].y +
                                            norm(2) * prev_segment->points[point_idx[j]].z + negative_OA_dot_norm);

                            if (d > good_plane_threshold)
                            {
                                planeValid = false;
                                break;
                            }
                            point_dist[j] = d;
                            point_idx[j] *= -1; // make it negative
                        }

                        if (planeValid) // found a good plane
                        {
                            for (int j = 0; j < 5; j++) // all the points share the same normal
                            {
                                auto it = landmarks_map.find(point_idx[j]);
                                if (it == landmarks_map.end()) // does not exist - add
                                {
                                    // std::cout << "add point with index " << point_idx[j] << std::endl;
                                    landmark_ tgt;
                                    tgt.map_point_index = point_idx[j];
                                    tgt.norm = norm;
                                    tgt.negative_OA_dot_norm = negative_OA_dot_norm;
                                    tgt.seen = 1;
                                    tgt.re_proj_error = point_dist[j];
                                    tgt.line_idx.push_back(l); // seen from line l
                                    tgt.scan_idx.push_back(i); // and point i from this line

                                    landmarks_map[point_idx[j]] = tgt;
                                }
                                else // Already exists - increment "seen"
                                {
                                    it->second.seen++;                // Increment the seen count
                                    it->second.line_idx.push_back(l); // seen from line l
                                    it->second.scan_idx.push_back(i); // and point i from this line
                                    // if a better normal was found - replace the old one
                                    if (it->second.re_proj_error > point_dist[j])
                                    {
                                        it->second.norm = norm;
                                        it->second.negative_OA_dot_norm = negative_OA_dot_norm;
                                        it->second.re_proj_error = point_dist[j];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return landmarks_map;
}

using namespace gtsam;
using gtsam::symbol_shorthand::X; // Pose symbols
using gtsam::symbol_shorthand::Z; // prev Pose symbols
using gtsam::symbol_shorthand::A; // ffor anchor
// auto robust_kernel = gtsam::noiseModel::mEstimator::Cauchy::Create(0.5);
// auto robust_kernel = gtsam::noiseModel::mEstimator::Huber::Create(.1);

double GM_robust_kernel(const double &residual2)
{
    double kernel_ = 1.0;
    return square(kernel_) / square(kernel_ + residual2);
}

// Custom factor for point-to-point constraints
class PointToPointFactor : public NoiseModelFactor1<Pose3>
{
private:
    Point3 measured_point_; // Point in sensor frame
    Point3 target_point_;   // Corresponding point in world frame
    double weight_;

    double huber_delta_ = .1;
    // Huber weight calculation
    double huberWeight(double error_norm) const
    {
        return (error_norm <= huber_delta_) ? 1.0 : huber_delta_ / error_norm;
    }

public:
    PointToPointFactor(Key pose_key,
                       const Point3 &measured_point,
                       const Point3 &target_point,
                       double weight,
                       const SharedNoiseModel &model) : NoiseModelFactor1<Pose3>(model, pose_key),
                                                        measured_point_(measured_point), target_point_(target_point), weight_(weight) {}

    // Vector evaluateError(const Pose3 &pose, boost::optional<Matrix &> H = boost::none) const override
    Vector evaluateError(const gtsam::Pose3 &pose, OptionalMatrixType H) const override
    {
        // Transform measured point to world frame
        Matrix36 H_point;
        Point3 world_point = pose.transformFrom(measured_point_, H_point); // computes ∂(world_point)/∂pose

        // Calculate error vector
        Vector3 error = world_point - target_point_;

        // auto robust_weight = GM_robust_kernel(error.squaredNorm());
        auto robust_weight = huberWeight(error.squaredNorm());

        if (H)
        {
            // Jacobian: ∂error/∂pose = ∂p_world/∂pose
            *H = H_point * robust_weight;
        }

        return robust_weight * error;

        // if (H)
        // {
        //     // // Compute Jacobian if requested
        //     // Matrix36 H_point_wrt_pose;
        //     // pose.transformFrom(measured_point_, H_point_wrt_pose);
        //     // (*H) = H_point_wrt_pose * weight_;
        // }

        /*
        // Transform landmark from world frame to LiDAR frame
        gtsam::Point3 predicted = pose.transformTo(target_point_, H);

        // Compute residual: measured - predicted
        return measured_point_ - predicted;
        */

        // return error;
    }
};

// Custom factor for point-to-plane constraints
class PointToPlaneFactor : public NoiseModelFactor1<Pose3>
{
private:
    Point3 measured_point_;       // Point in sensor frame
    Point3 plane_normal_;         // Plane normal (normalized)
    Point3 target_point_;         // A point on the plane in world frame
    double negative_OA_dot_norm_; // = 1 / norm.norm()
    double weight_;
    bool use_alternative_method_; // Flag to choose between calculation methods

    double huber_delta_ = .1;
    // Huber weight calculation
    double huberWeight(double error_norm) const
    {
        return (error_norm <= huber_delta_) ? 1.0 : huber_delta_ / error_norm;
    }

public:
    PointToPlaneFactor(Key pose_key,
                       const Point3 &measured_point,
                       const Point3 &plane_norm,
                       const Point3 &target_point,
                       double negative_OA_dot_norm,
                       double weight,
                       bool use_alternative_method,
                       const SharedNoiseModel &model) : NoiseModelFactor1<Pose3>(model, pose_key),
                                                        measured_point_(measured_point),
                                                        plane_normal_(plane_norm),
                                                        target_point_(target_point),
                                                        negative_OA_dot_norm_(negative_OA_dot_norm),
                                                        weight_(weight),
                                                        use_alternative_method_(use_alternative_method) {}

    Vector evaluateError(const Pose3 &pose, OptionalMatrixType H) const override
    {
        // Transform measured point to world frame
        // Point3 p_transformed = pose.transformFrom(measured_point_, H);
        // Transform measured point to world frame
        Matrix36 H_point_wrt_pose;
        Point3 p_transformed = pose.transformFrom(measured_point_, H ? &H_point_wrt_pose : 0);

        double error = 0.0;
        if (use_alternative_method_)
        {
            error = (p_transformed - target_point_).dot(plane_normal_);
        }
        else
        {
            // error = plane_normal_.x() * p_transformed.x() +
            //         plane_normal_.y() * p_transformed.y() +
            //         plane_normal_.z() * p_transformed.z() +
            //         negative_OA_dot_norm_;
            error = plane_normal_.dot(p_transformed) + negative_OA_dot_norm_;
        }

        //  Apply robust weighting
        // auto robust_weight = GM_robust_kernel(error * error);
        auto robust_weight = huberWeight(fabs(error));

        if (H)
        {
            // H_point_wrt_pose (3×6 matrix)
            /*
            [ ∂x/∂tx ∂x/∂ty ∂x/∂tz ∂x/∂rx ∂x/∂ry ∂x/∂rz ]
            [ ∂y/∂tx ∂y/∂ty ∂y/∂tz ∂y/∂rx ∂y/∂ry ∂y/∂rz ]
            [ ∂z/∂tx ∂z/∂ty ∂z/∂tz ∂z/∂rx ∂z/∂ry ∂z/∂rz ]
            */
            // First 3 columns: Derivatives wrt translation (simple)
            // Last 3 columns: Derivatives wrt rotation (involves cross products)
            //[ nx ny nz ] (1×3)  *  [ 3×6 Jacobian ]  =  [ 1×6 Jacobian ]
            //  Jacobian should be 1x6 (1D error, 6D pose)
            //*H = (plane_normal_.transpose() * H_point_wrt_pose); // * weight_;

            // Row Vector (1×6): [n_x, n_y, n_z, (p×n)_x, (p×n)_y, (p×n)_z]
            // Column Vector (6×1): [n_x, n_y, n_z, (p×n)_x, (p×n)_y, (p×n)_z]^T

            // Compute 6x1 column vector Jacobian
            // Eigen::Vector6d J_r;
            // J_r.block<3,1>(0,0) = plane_normal_;                     // Translation part // df/dt
            // //J_r.block<3,1>(3,0) = p_transformed.cross(plane_normal_); // Rotation part   // df/dR
            // J_r.block<3,1>(3,0) = -plane_normal_.cross(p_transformed); // Note negative sign!
            // For robust version:
            // double robust_weight = mEstimator_->weight(std::abs(error));
            // *H = J_r.transpose() * (weight_ * robust_weight);

            // Apply same weight to Jacobian
            *H = (plane_normal_.transpose() * H_point_wrt_pose) * robust_weight;
        }

        return (Vector(1) << error * robust_weight).finished();
    }
};

NonlinearFactorGraph prev_graph;
Values prev_optimized_values;
bool prev_graph_exist = false;

// auto sigma_prior = .5;// .1;
// auto sigma_odom =  .05;// 5cm;
// auto sigma_point = .1;
// auto sigma_plane = 1.;
// // noises allow (3σ) deviation in each axis

// // noises
// auto point_noise = noiseModel::Isotropic::Sigma(3, sigma_point);
// auto plane_noise = noiseModel::Isotropic::Sigma(1, sigma_plane);

auto sigma_point = 1; // 100cm standard deviation (3σ ≈ 3m allowed)
auto sigma_plane = 1; // 100cm standard deviation (3σ ≈ 30cm allowed)

// auto sigma_odom = .02; //for relative odometry - tested with skip one scan
auto sigma_odom = .01;  // for relative odometry no scan skip
auto sigma_prior = 1.0; // not sure about the prior - let it change it

auto point_noise = gtsam::noiseModel::Robust::Create(
    gtsam::noiseModel::mEstimator::Cauchy::Create(0.5), // Robust kernel for outliers
    // gtsam::noiseModel::mEstimator::Huber(.5),
    gtsam::noiseModel::Isotropic::Sigma(3, sigma_point));

auto plane_noise = gtsam::noiseModel::Robust::Create(
    gtsam::noiseModel::mEstimator::Tukey::Create(0.1), // Very aggressive outlier rejection
    // gtsam::noiseModel::mEstimator::Cauchy::Create(0.5),
    gtsam::noiseModel::Isotropic::Sigma(1, sigma_plane));

auto odometry_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector6() << gtsam::Vector3::Constant(sigma_odom), gtsam::Vector3::Constant(sigma_odom)).finished());
auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector6() << gtsam::Vector3::Constant(sigma_prior), gtsam::Vector3::Constant(sigma_prior)).finished());

auto tight_noise = noiseModel::Diagonal::Sigmas(
    (Vector6() << .0001, .0001, .0001, .001, .001, .001).finished());

double BA_refinement(
    const std::deque<pcl::PointCloud<VUX_PointType>::Ptr> &lidar_lines,
    std::deque<Sophus::SE3> &line_poses,
    const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
    const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
    const bool &prev_segment_init,
    const pcl::PointCloud<PointType>::Ptr &prev_segment,
    const pcl::KdTreeFLANN<PointType>::Ptr &kdtree_prev_segment,
    const ros::Publisher &cloud_pub, const ros::Publisher &normals_pub,
    int overlap_size = 50,
    double threshold_nn = 1.0, bool p2p = true, bool p2plane = false)
{

    std::cout << "Run BA_refinement..." << std::endl;
    if (p2plane && p2p)
    {
        std::cout << "Perform both p2p and p2plane" << std::endl;
    }
    else if (p2plane)
    {
        std::cout << "Perform p2plane" << std::endl;
    }
    else if (p2p)
    {
        std::cout << "Perform p2p" << std::endl;
    }

    double planarity = 0.1;
    std::unordered_map<int, landmark_> landmarks_map = get_Correspondences(
        lidar_lines,
        line_poses,
        refference_kdtree,
        reference_localMap_cloud,
        prev_segment_init,
        prev_segment,
        kdtree_prev_segment, threshold_nn, planarity);

    if (false)
    {
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
        // for (const auto &land : landmarks_map)
        for (const auto &[index, land] : landmarks_map)
        {
            pcl::PointNormal pt;
            pt.x = reference_localMap_cloud->points[land.map_point_index].x;
            pt.y = reference_localMap_cloud->points[land.map_point_index].y;
            pt.z = reference_localMap_cloud->points[land.map_point_index].z;

            pt.normal_x = land.norm.x();
            pt.normal_y = land.norm.y();
            pt.normal_z = land.norm.z();

            pt.curvature = land.seen;
            // pt.curvature = land.line_idx.size();

            cloud_with_normals->push_back(pt);
        }

        debug_CloudWithNormals(cloud_with_normals, cloud_pub, normals_pub);
    }

    NonlinearFactorGraph graph;
    Values initial_values;

    if (!prev_graph_exist)
    {
        // 1. Create nodes from initial odometry poses
        for (size_t i = 0; i < line_poses.size(); i++)
        {
            Pose3 gtsam_pose(line_poses[i].matrix());

            // Add to initial values
            initial_values.insert(X(i), gtsam_pose);

            // Add prior on the first pose to fix the coordinate frame
            if (i == 0)
            {
                graph.addPrior(X(0), gtsam_pose, prior_noise);
            }

            // Add odometry constraints between consecutive poses
            if (i > 0)
            {
                Sophus::SE3 relative_pose = line_poses[i - 1].inverse() * line_poses[i];
                Pose3 gtsam_relative(relative_pose.matrix());

                graph.emplace_shared<BetweenFactor<Pose3>>(X(i - 1), X(i), gtsam_relative, odometry_noise);
            }
        }
    }
    else
    {
        int prev_last_idx = line_poses.size();
        std::cout << "Use the prev values with size:" << overlap_size << ", prev_last_idx:" << prev_last_idx << std::endl;

        //  Reuse optimized poses for overlapping poses
        for (size_t i = 0; i < line_poses.size(); i++)
        {
            // Add prior and init values-------------------------------------------------------
            Sophus::SE3 pose = line_poses[i];

            Pose3 gtsam_pose(pose.matrix());
            initial_values.insert(X(i), gtsam_pose); // Use raw odometry estimate

            // the overlapping part
            if (i < overlap_size && prev_optimized_values.exists(X(prev_last_idx - overlap_size + i)))
            {
                gtsam_pose = prev_optimized_values.at<Pose3>(X(prev_last_idx - overlap_size + i));
                // initial_values.insert(X(i), gtsam_pose); // Use optimized value

                // Insert overlapping poses from the previous segment (with 'Z' keys)
                // int key = prev_last_idx - overlap_size + i;
                // std::cout<<"adding Z_"<<key<<", i:"<<i<<std::endl;
                initial_values.insert(Z(i), gtsam_pose); // Z0, Z1, ..., Z49

                // add the odometry constraints values for the last segment
                if (i == 0)
                {
                    graph.addPrior(X(i), gtsam_pose, odometry_noise); // prior_noise
                    graph.addPrior(Z(i), gtsam_pose, tight_noise);
                }
                // add the relative for Z
                if (i > 0)
                {
                    // Use the optimized values from the previous segment
                    Pose3 prev_optimized_pose = prev_optimized_values.at<Pose3>(X(prev_last_idx - overlap_size + i - 1));
                    Pose3 relative_pose = Pose3(prev_optimized_pose.between(gtsam_pose));

                    graph.emplace_shared<BetweenFactor<Pose3>>(X(i - 1), X(i), relative_pose, odometry_noise);
                    graph.emplace_shared<BetweenFactor<Pose3>>(Z(i - 1), Z(i), relative_pose, tight_noise);
                }

                // Add Identity Constraints Between Overlapping Nodes - // Identity transform (since they should align)
                graph.emplace_shared<BetweenFactor<Pose3>>(X(i), Z(i), Pose3(), tight_noise);
            }
            else
            {
                // Add prior on the first pose to fix the coordinate frame
                if (i == 0)
                {
                    graph.addPrior(X(i), gtsam_pose, prior_noise);
                }
            }

            // Add odometry constraints between consecutive poses-----------------------------
            if (i > 0) // Add Current Segment Poses (Using X)
            {
                // Use raw odometry estimate
                Pose3 gtsam_relative = Pose3((line_poses[i - 1].inverse() * line_poses[i]).matrix());
                graph.emplace_shared<BetweenFactor<Pose3>>(X(i - 1), X(i), gtsam_relative, odometry_noise);
            }
        }
    }

    // 2. Add constraints
    int added_constraints = 0;
    int constraints_to_ref_map = 0;
    int constraints_to_prev = 0;
    for (const auto &[landmark_id, land] : landmarks_map)
    {
        if (land.seen > 2)
        {
            for (int i = 0; i < land.seen; i++)
            {
                const auto &pose_idx = land.line_idx[i]; // point from line pose_idx
                const auto &p_idx = land.scan_idx[i];    // at index p_idx from that scan
                const auto &raw_point = lidar_lines[pose_idx]->points[p_idx];
                // measured_landmar_in_sensor_frame
                Point3 measured_point(raw_point.x, raw_point.y, raw_point.z);

                Point3 target_point;

                if (land.map_point_index >= 0)
                {
                    target_point = Point3(reference_localMap_cloud->points[land.map_point_index].x,
                                          reference_localMap_cloud->points[land.map_point_index].y,
                                          reference_localMap_cloud->points[land.map_point_index].z);
                    constraints_to_ref_map++;
                }
                else
                {
                    target_point = Point3(prev_segment->points[-land.map_point_index].x,
                                          prev_segment->points[-land.map_point_index].y,
                                          prev_segment->points[-land.map_point_index].z);
                    constraints_to_prev++;
                }

                if (p2plane && p2p)
                {
                    Point3 plane_norm(land.norm.x(), land.norm.y(), land.norm.z());
                    bool use_alternative_method = false;

                    // use_alternative_method = true; //see this
                    graph.emplace_shared<PointToPlaneFactor>(X(pose_idx), measured_point, plane_norm, target_point, land.negative_OA_dot_norm,
                                                             land.re_proj_error, use_alternative_method, plane_noise);

                    graph.emplace_shared<PointToPointFactor>(X(pose_idx), measured_point, target_point, land.re_proj_error, point_noise);
                }
                else if (p2plane)
                {
                    // auto weighted_plane_noise = noiseModel::Isotropic::Sigma(1, 0.1/land.weight);
                    Point3 plane_norm(land.norm.x(), land.norm.y(), land.norm.z());
                    bool use_alternative_method = false; // true;
                    graph.emplace_shared<PointToPlaneFactor>(X(pose_idx), measured_point, plane_norm, target_point, land.negative_OA_dot_norm,
                                                             land.re_proj_error, use_alternative_method, plane_noise);
                }
                else if (p2p)
                {
                    // point-to-point
                    graph.emplace_shared<PointToPointFactor>(X(pose_idx), measured_point, target_point, land.re_proj_error, point_noise);
                }
                added_constraints++;
            }
        }
    }

    std::cout << "added_constraints:" << added_constraints << std::endl;
    std::cout << "constraints_to_ref_map:" << constraints_to_ref_map << ", constraints_to_prev:" << constraints_to_prev << std::endl;

    // Set optimization parameters
    LevenbergMarquardtParams params;
    params.setMaxIterations(100);
    params.setRelativeErrorTol(1e-3);
    params.setVerbosity("ERROR");

    // Optimize the graph
    LevenbergMarquardtOptimizer optimizer(graph, initial_values, params); //
    Values optimized_values = optimizer.optimize();

    prev_graph_exist = true;
    prev_optimized_values = optimized_values;
    prev_graph = graph;

    // prev_landmarks_map = landmarks_map;

    // optimized_values.print("Optimized Results:\n");
    //  Retrieve optimized poses
    //  std::vector<Sophus::SE3> optimized_poses;
    for (size_t i = 0; i < line_poses.size(); i++)
    {
        Pose3 optimized_pose = optimized_values.at<Pose3>(X(i));
        M3D R = optimized_pose.rotation().matrix();
        V3D t = optimized_pose.translation();
        line_poses[i] = Sophus::SE3(R, t);
        // optimized_poses.emplace_back(Sophus::SE3(R, t));
    }

    return optimizer.error();
}

//----------------------------------------------------------------------------------
// the new merge graph idea

void mergeXandZGraphs(NonlinearFactorGraph &merged_graph, Values &merged_values,
                      const NonlinearFactorGraph &prev_graph, const Values &prev_values,
                      const NonlinearFactorGraph &curr_graph, const Values &curr_values,
                      const bool curr_uses_x,
                      int overlap_size = 50, int total_poses = 100,
                      Sophus::SE3 anchor_pose = Sophus::SE3(), Sophus::SE3 anchor_delta = Sophus::SE3())
{
    /*
    X-graph: x0 ─── x1 ─── ... ─── x49 ─── x50 ─── ... ─── x99
                                       │        │            │
    Identity constraints:              ≈ z0     ≈ z1         ≈ z49
    Z-graph:                          z0 ─── z1 ─── ... ─── z49 ─── z50 ─── ... ─── z99
    */

    bool prev_uses_x = !curr_uses_x;

    // Helper function to get the correct symbol
    auto symbol = [](bool use_x, int i)
    {
        return use_x ? X(i) : Z(i);
    };

    

    // 1. Add all previous graph components
    std::cout << "\nAdd all previous graph components" << std::endl;
    merged_values.insert(prev_values);
    for (const auto &factor : prev_graph)
    {
        merged_graph.push_back(factor);
    }

    // 2. Add current graph components
    std::cout << "\nAdd current graph components" << std::endl;
    merged_values.insert(curr_values);
    for (const auto &factor : curr_graph)
    {
        merged_graph.push_back(factor);
    }

    // 3. Add identity constraints between overlapping poses
    std::cout << "\nAdd identity constraints between overlapping poses" << std::endl;
    for (int i = 0; i < overlap_size; i++)
    {
        // Last 'overlap_size' poses of previous graph
        Key prev_key = symbol(prev_uses_x, total_poses - overlap_size + i);
        // First 'overlap_size' poses of current graph
        Key curr_key = symbol(curr_uses_x, i);

        // if(i==0)
        {
            // Use Symbol to get the symbol corresponding to the Key
            Symbol symbol_prev(prev_key), symbol_curr(curr_key);
            std::cout << "\nsymbol_prev: " << symbol_prev << " symbol_curr: " << symbol_curr << std::endl;
        }

        merged_graph.emplace_shared<BetweenFactor<Pose3>>(prev_key, curr_key, Pose3(), tight_noise);
    }

    // 4. Add odometry bridge between last non-overlapping poses
    std::cout << "\nAdd odometry bridge between last non-overlapping poses" << std::endl;
    /*
    Previous segment poses: [0 ... 49     50 ... 99]
                            (non-overlap) (overlap)
    Current segment poses:                [0 ... 49  50 ... 99]
                                          (overlap) (non-overlap)

    Connecting:
    - Previous segment's last non-overlap pose: 49
    - Current segment's first non-overlap pose: 50
    */
    if (overlap_size < total_poses)
    {
        auto odometry_noise = noiseModel::Diagonal::Sigmas(
            (Vector6() << 0.1, 0.1, 0.1, 0.05, 0.05, 0.05).finished());

        // Last non-overlap pose of previous graph (using correct symbol type)
        // Key last_prev = prev_uses_x ? X(total_poses - overlap_size - 1) : Z(total_poses - overlap_size - 1);

        // First non-overlap pose of current graph (using correct symbol type)
        // Key first_curr = curr_uses_x ? X(overlap_size) : Z(overlap_size);

        // std::cout<<"prev_uses_x:"<<prev_uses_x<<", curr_uses_x:"<<curr_uses_x<<std::endl;

        Key last_prev = prev_uses_x ? X(total_poses - overlap_size - 1) : Z(total_poses - overlap_size - 1);
        Key first_curr = curr_uses_x ? X(overlap_size) : Z(overlap_size);

        std::cout << "Connecting:\n";
        std::cout << "  Last prev pose: " << gtsam::DefaultKeyFormatter(last_prev) << "\n";
        std::cout << "  First curr pose: " << gtsam::DefaultKeyFormatter(first_curr) << "\n";

        // auto first_key_prev = Symbol(prev_values.begin()->key);
        // std::cout << "\nfirst_key_prev: " << first_key_prev << std::endl;

        // auto first_key_curr = Symbol(curr_values.begin()->key);
        // std::cout << "first_key_curr: " << first_key_curr << std::endl;

        // Verify keys exist before accessing
        // if (!prev_values.exists(last_prev))
        // {
        //     std::cerr << "Error: Previous pose " << gtsam::DefaultKeyFormatter(last_prev)
        //               << " does not exist\n";
        //     //continue;
        // }
        // if (!curr_values.exists(first_curr))
        // {
        //     std::cerr << "Error: Current pose " << gtsam::DefaultKeyFormatter(first_curr)
        //               << " does not exist\n";
        //     //continue;
        // }

        Pose3 last_prev_pose = prev_values.at<Pose3>(last_prev);
        Pose3 first_curr_pose = curr_values.at<Pose3>(first_curr);

        // Calculate relative transform
        Pose3 relative = last_prev_pose.inverse() * first_curr_pose;

        // Add the between factor
        merged_graph.emplace_shared<BetweenFactor<Pose3>>(last_prev, first_curr, relative, odometry_noise);
    }


    //5. Add anchor - anchor_pose (fixed prior) ──(anchor_delta)──▶ first_node
    std::cout<<"anchor_pose:"<<anchor_pose.log().transpose()<<std::endl;
    std::cout<<"anchor_delta:"<<anchor_delta.log().transpose()<<std::endl;
    // if(anchor_delta.log().norm() > 0){
    //     std::cout<<"Add first anchor..."<<std::endl;
    //     Pose3 anchor_pose(anchor_pose.matrix());
    //     Pose3 anchor_delta(anchor_delta.matrix());

    //     gtsam::Key anchor_key = A(0);  // Arbitrary high index for anchor

    //     Key first_prev = prev_uses_x ? X(0) : Z(0);

    //     std::cout << "Connecting A(0) with First prev pose: " << gtsam::DefaultKeyFormatter(first_prev) << "\n";

    //     auto anchor_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3).finished()); // tight
    //     merged_graph.add(PriorFactor<Pose3>(anchor_key, anchor_pose, anchor_noise));
    //     merged_values.insert(anchor_key, anchor_pose);
    //     merged_graph.add(BetweenFactor<Pose3>(anchor_key, first_prev, anchor_delta, anchor_noise));
    // }
}

bool useX = true; // Set this flag to false to use Z instead of X

void buildGraph(
    const std::deque<pcl::PointCloud<VUX_PointType>::Ptr> &lidar_lines,
    std::deque<Sophus::SE3> &line_poses,
    const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
    const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
    const bool &prev_segment_init,
    const pcl::PointCloud<PointType>::Ptr &prev_segment,
    const pcl::KdTreeFLANN<PointType>::Ptr &kdtree_prev_segment,
    const ros::Publisher &cloud_pub, const ros::Publisher &normals_pub,
    NonlinearFactorGraph &graph, Values &initial_values,
    int overlap_size = 50,
    double threshold_nn = 1.0, bool p2p = true, bool p2plane = false)
{
    std::cout << "Run BA_refinement_merge_graph..." << std::endl;
    

    if (p2plane && p2p)
    {
        std::cout << "Perform both p2p and p2plane" << std::endl;
    }
    else if (p2plane)
    {
        std::cout << "Perform p2plane" << std::endl;
    }
    else if (p2p)
    {
        std::cout << "Perform p2p" << std::endl;
    }

    // double planarity = 0.1;
    double planarity = .05; // 5cm allowed
    // double planarity = .02;   //2 cm allowed

    const std::unordered_map<int, landmark_> &landmarks_map = get_Correspondences(
        lidar_lines,
        line_poses,
        refference_kdtree,
        reference_localMap_cloud,
        prev_segment_init,
        prev_segment,
        kdtree_prev_segment, threshold_nn, planarity);

    if (normals_pub.getNumSubscribers() != 0)
    {
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
        for (const auto &[index, land] : landmarks_map)
        {
            pcl::PointNormal pt;
            pt.x = reference_localMap_cloud->points[land.map_point_index].x;
            pt.y = reference_localMap_cloud->points[land.map_point_index].y;
            pt.z = reference_localMap_cloud->points[land.map_point_index].z;

            pt.normal_x = land.norm.x();
            pt.normal_y = land.norm.y();
            pt.normal_z = land.norm.z();

            // pt.curvature = land.seen;
            pt.curvature = land.re_proj_error;
            // pt.curvature = land.line_idx.size();

            cloud_with_normals->push_back(pt);
        }

        debug_CloudWithNormals(cloud_with_normals, cloud_pub, normals_pub);
    }

    // 1. Create nodes from initial odometry poses
    for (size_t i = 0; i < line_poses.size(); i++)
    {
        Pose3 gtsam_pose(line_poses[i].matrix());

        // Add to initial values
        initial_values.insert(useX ? X(i) : Z(i), gtsam_pose);

        // Add prior on the first pose to fix the coordinate frame
        if (i == 0)
        {
            graph.addPrior(useX ? X(0) : Z(0), gtsam_pose, prior_noise);
        }

        // Add odometry constraints between consecutive poses
        if (i > 0)
        {
            Sophus::SE3 relative_pose = line_poses[i - 1].inverse() * line_poses[i];
            Pose3 gtsam_relative(relative_pose.matrix());

            graph.emplace_shared<BetweenFactor<Pose3>>(useX ? X(i - 1) : Z(i - 1), useX ? X(i) : Z(i), gtsam_relative, odometry_noise);
        }
    }

    // 2. Add constraints
    int added_constraints = 0;
    int constraints_to_ref_map = 0;
    int constraints_to_prev = 0;
    for (const auto &[landmark_id, land] : landmarks_map)
    {
        if (land.seen > 2)
        {
            for (int i = 0; i < land.seen; i++)
            {
                const auto &pose_idx = land.line_idx[i]; // point from line pose_idx
                const auto &p_idx = land.scan_idx[i];    // at index p_idx from that scan
                const auto &raw_point = lidar_lines[pose_idx]->points[p_idx];
                // measured_landmar_in_sensor_frame
                Point3 measured_point(raw_point.x, raw_point.y, raw_point.z);

                Point3 target_point;

                if (land.map_point_index >= 0)
                {
                    target_point = Point3(reference_localMap_cloud->points[land.map_point_index].x,
                                          reference_localMap_cloud->points[land.map_point_index].y,
                                          reference_localMap_cloud->points[land.map_point_index].z);
                    constraints_to_ref_map++;
                }
                else
                {
                    target_point = Point3(prev_segment->points[-land.map_point_index].x,
                                          prev_segment->points[-land.map_point_index].y,
                                          prev_segment->points[-land.map_point_index].z);
                    constraints_to_prev++;
                }

                if (p2plane && p2p)
                {
                    Point3 plane_norm(land.norm.x(), land.norm.y(), land.norm.z());
                    bool use_alternative_method = false; // true;
                    graph.emplace_shared<PointToPlaneFactor>(useX ? X(pose_idx) : Z(pose_idx), measured_point, plane_norm, target_point, land.negative_OA_dot_norm,
                                                             land.re_proj_error, use_alternative_method, plane_noise);

                    graph.emplace_shared<PointToPointFactor>(useX ? X(pose_idx) : Z(pose_idx), measured_point, target_point, land.re_proj_error, point_noise);
                }
                else if (p2plane)
                {
                    // auto weighted_plane_noise = noiseModel::Isotropic::Sigma(1, 0.1/land.weight);
                    Point3 plane_norm(land.norm.x(), land.norm.y(), land.norm.z());
                    bool use_alternative_method = false; // true;
                    graph.emplace_shared<PointToPlaneFactor>(useX ? X(pose_idx) : Z(pose_idx), measured_point, plane_norm, target_point, land.negative_OA_dot_norm,
                                                             land.re_proj_error, use_alternative_method, plane_noise);
                }
                else if (p2p)
                {
                    // point-to-point
                    graph.emplace_shared<PointToPointFactor>(useX ? X(pose_idx) : Z(pose_idx), measured_point, target_point, land.re_proj_error, point_noise);
                }
                added_constraints++;
            }
        }
    }

    std::cout << "added_constraints:" << added_constraints << std::endl;
    std::cout << "constraints_to_ref_map:" << constraints_to_ref_map << ", constraints_to_prev:" << constraints_to_prev << std::endl;
}

double BA_refinement_merge_graph(
    const std::deque<pcl::PointCloud<VUX_PointType>::Ptr> &lidar_lines,
    std::deque<Sophus::SE3> &line_poses,
    const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
    const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
    const bool &prev_segment_init,
    const pcl::PointCloud<PointType>::Ptr &prev_segment,
    const pcl::KdTreeFLANN<PointType>::Ptr &kdtree_prev_segment,
    const ros::Publisher &cloud_pub, const ros::Publisher &normals_pub,
    int run_iterations = 1, bool flg_exit = false,
    int overlap_size = 50,
    double threshold_nn = 1.0, bool p2p = true, bool p2plane = false,
    Sophus::SE3 anchor_pose = Sophus::SE3(), Sophus::SE3 anchor_delta = Sophus::SE3())
{
    useX = !useX;
    std::cout << "For current graph useX:" << useX << std::endl;
    std::cout << " reference_localMap_cloud:" << reference_localMap_cloud->size() << std::endl;
    if (reference_localMap_cloud->size() == 0)
    {
        throw std::runtime_error("reference_localMap_cloud not init - no points");
    }
    // Set optimization parameters
    LevenbergMarquardtParams params;
    params.setMaxIterations(100);
    params.setRelativeErrorTol(1e-3);
    params.setVerbosity("ERROR");

    bool rv = 0.;

    if (!prev_graph_exist)
    {
        NonlinearFactorGraph graph;
        Values initial_values;

        buildGraph(lidar_lines, line_poses, refference_kdtree, reference_localMap_cloud,
                   prev_segment_init, prev_segment, kdtree_prev_segment,
                   cloud_pub, normals_pub,
                   graph, initial_values,
                   overlap_size, threshold_nn, p2p, p2plane);

        // Optimize the graph
        LevenbergMarquardtOptimizer optimizer(graph, initial_values, params); //
        Values optimized_values = optimizer.optimize();

        // optimized_values.print("Optimized Results:\n");
        //  Retrieve optimized poses
        //  std::vector<Sophus::SE3> optimized_poses;
        for (size_t i = 0; i < line_poses.size(); i++)
        {
            gtsam::Key pose_key = useX ? X(i) : Z(i);
            Pose3 optimized_pose = optimized_values.at<Pose3>(pose_key);
            M3D R = optimized_pose.rotation().matrix();
            V3D t = optimized_pose.translation();
            line_poses[i] = Sophus::SE3(R, t);
            // optimized_poses.emplace_back(Sophus::SE3(R, t));
        }

        prev_graph_exist = true;
        // prev_optimized_values = initial_values; // keep the raw data
        prev_optimized_values = optimized_values; // keep the optimized values
        prev_graph = graph;

        rv = optimizer.error();

        return rv;
    }
    else
    {
        // int prev_last_idx = line_poses.size();
        // std::cout << "Use the prev values with size:" << overlap_size << ", prev_last_idx:" << prev_last_idx << std::endl;

        NonlinearFactorGraph graph;
        Values initial_values;

        for (int iter = 0; iter < run_iterations; iter++)
        {
            if (flg_exit)
                break;

            std::cout << "Runt iteration " << iter << std::endl;
            graph.resize(0);        // Clears all factors from the graph
            initial_values.clear(); // Clears all values

            buildGraph(lidar_lines, line_poses, refference_kdtree, reference_localMap_cloud,
                       prev_segment_init, prev_segment, kdtree_prev_segment,
                       cloud_pub, normals_pub,
                       graph, initial_values,
                       overlap_size, threshold_nn, p2p, p2plane);

            NonlinearFactorGraph merged_graph;
            Values merged_values;

            std::cout << "mergeXandZGraphs" << std::endl;
            int total_poses = line_poses.size();
            mergeXandZGraphs(merged_graph, merged_values,
                             prev_graph, prev_optimized_values,
                             graph, initial_values,
                             useX, overlap_size, total_poses,
                             anchor_pose, anchor_delta);

            // Optimize the graph
            LevenbergMarquardtOptimizer optimizer(merged_graph, merged_values, params); //
            Values optimized_values = optimizer.optimize();

            // optimized_values.print("Optimized Results:\n");
            //  Retrieve optimized poses

            // Values re_optimized_prev;
            for (size_t i = 0; i < total_poses; i++)
            {
                gtsam::Key curr_pose_key = useX ? X(i) : Z(i);
                if (optimized_values.exists(curr_pose_key))
                {
                    Pose3 optimized_pose = optimized_values.at<Pose3>(curr_pose_key);
                    line_poses[i] = Sophus::SE3(optimized_pose.rotation().matrix(), optimized_pose.translation());
                }
                else
                {
                    Symbol symbol_key(curr_pose_key);
                    std::cerr << "Optimized pose not found for key: " << symbol_key << std::endl;
                }

                // gtsam::Key prev_pose_key = useX ? Z(i) : X(i);
                // if (optimized_values.exists(curr_pose_key))
                // {
                //     Pose3 prev_optimized_pose = optimized_values.at<Pose3>(prev_pose_key);
                //     re_optimized_prev.insert(prev_pose_key, prev_optimized_pose);
                // }
                // else
                // {
                //     Symbol symbol_key(prev_pose_key);
                //     std::cerr << "Re-Optimized pose not found for key: " << symbol_key << std::endl;
                // }
            }
            // prev_optimized_values = re_optimized_prev;

            // std::cout << "\bIteraion:" << iter << ", error:" << optimizer.error() << std::endl;
            rv = optimizer.error();
        }

        prev_graph_exist = true;
        prev_optimized_values = initial_values; // keep the raw data
        prev_graph = graph;

        return rv;
    }
}

//------------------------------------------------------------------------------------------------------------------
#include <cassert>
#include <cmath>

//#define use_spline

#ifdef use_spline
    

// Spline basis matrix for uniform cubic B-spline
const Eigen::Matrix4d C = (Eigen::Matrix4d() << 6. / 6., 0., 0., 0.,
                           5. / 6., 3. / 6., -3. / 6., 1. / 6.,
                           1. / 6., 3. / 6., 3. / 6., -2. / 6.,
                           0., 0., 0., 1. / 6.)
                              .finished();

class CubicSpline
{
public:
    using SE3Type = Sophus::SE3;
    using SE3DerivType = Eigen::Matrix4d;

    // Constructor to initialize time step and initial time
    CubicSpline(double dt = 1.0, double t0 = 0.0) : dt_(dt), t0_(t0) {}

    // Add a knot (pose) to the spline
    void add_knot(const SE3Type &pose)
    {
        knots_.push_back(pose);
    }

    // Get a knot (pose) from the spline
    SE3Type get_knot(size_t k) const
    {
        return knots_[k];
    }

    // Evaluate the spline at time t and compute the pose and its derivatives
    void evaluate(double t, SE3Type &P) const; //, SE3DerivType &P_prim, SE3DerivType &P_bis

private:
    static int floor_(double x)
    {
        return static_cast<int>(std::floor(x));
    }

    double dt_;                  // Time step
    double t0_;                  // Initial time
    std::vector<SE3Type> knots_; // Vector of SE3 poses as knots
};

void CubicSpline::evaluate(double t, SE3Type &P) const //, SE3DerivType &P_prim, SE3DerivType &P_bis
{
    using Mat4 = Eigen::Matrix4d;
    using Vec4 = Eigen::Matrix<double, 4, 1>;

    assert(dt_ > 0.0 && "CubicSpline::evaluate: Time step (dt) must be greater than zero.");
    assert(knots_.size() >= 4 && "CubicSpline::evaluate: There must be at least 4 knots for spline interpolation.");
    assert(t >= t0_ + dt_ && "CubicSpline::evaluate: Time t must be greater than or equal to the time of the first knot.");
    assert(t < t0_ + dt_ * (knots_.size() - 2) && "CubicSpline::evaluate: Time t must be less than the time of the last knot.");


    // Compute normalized time (offset-aware)
    double s = (t - t0_) / dt_;
    int i = floor_(s);
    double u = s - i;
    int i0 = i - 1;

    double u2 = u * u, u3 = u2 * u, dt_inv = 1.0 / dt_;
    Vec4 B = C * Vec4{1.0, u, u2, u3};
    Vec4 Bd1 = C * Vec4{0.0, 1.0, 2.0 * u, 3.0 * u2} * dt_inv;
    Vec4 Bd2 = C * Vec4{0.0, 0.0, 2.0, 6.0 * u} * dt_inv * dt_inv;

    SE3Type P0 = knots_[i0]; // First knot pose
    P = P0;

    Mat4 A[3], Ad1[3], Ad2[3];

    for (int j : {1, 2, 3})
    {
        SE3Type knot1 = knots_[i0 + j - 1];
        SE3Type knot2 = knots_[i0 + j];
        auto omega = (knot1.inverse() * knot2).log();
        Mat4 omega_hat = SE3Type::hat(omega);
        SE3Type Aj = SE3Type::exp(B[j] * omega);
        P = P * Aj;
        Mat4 Ajm = Aj.matrix();
        Mat4 Ajd1 = Ajm * omega_hat * Bd1[j];
        Mat4 Ajd2 = Ajd1 * omega_hat * Bd1[j] + Ajm * omega_hat * Bd2[j];
        A[j - 1] = Ajm;
        Ad1[j - 1] = Ajd1;
        Ad2[j - 1] = Ajd2;
    }

    // // Compute the derivatives
    // Mat4 M1 = Ad1[0] * A[1] * A[2] + A[0] * Ad1[1] * A[2] + A[0] * A[1] * Ad1[2];
    // Mat4 M2 = Ad2[0] * A[1] * A[2] + A[0] * Ad2[1] * A[2] + A[0] * A[1] * Ad2[2] +
    //           2.0 * Ad1[0] * Ad1[1] * A[2] + 2.0 * Ad1[0] * A[1] * Ad1[2] +
    //           2.0 * A[0] * Ad1[1] * Ad1[2];

    // P_prim = P0.matrix() * M1;
    // P_bis = P0.matrix() * M2;
}


#endif 


